import streamlit as st
import requests
import tempfile
import os
import re
import traceback
import json
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import ast
import pandas as pd
import io

# --- Funkcje pomocnicze ---

def authenticate_user():
    """Uwierzytelnianie użytkownika"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("Edytor AI plików XML i CSV - Logowanie")
        user = st.text_input("Login")
        password = st.text_input("Hasło", type="password")
        if st.button("Zaloguj"):
            if user == st.secrets.get("APP_USER") and password == st.secrets.get("APP_PASSWORD"):
                st.session_state.authenticated = True
                st.experimental_rerun()
            else:
                st.error("Nieprawidłowy login lub hasło")
        st.stop()
    return True


def initialize_session_state():
    """Inicjalizacja zmiennych sesji"""
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""
    if "output_bytes" not in st.session_state:
        st.session_state.output_bytes = None
    if "file_info" not in st.session_state:
        st.session_state.file_info = None


def read_file_content(uploaded_file):
    """Czyta zawartość pliku z obsługą różnych kodowań"""
    if not uploaded_file:
        return None, "Nie wybrano pliku"
        
    try:
        raw_bytes = uploaded_file.read()
        file_type = uploaded_file.name.split(".")[-1].lower()
        
        if file_type not in ["xml", "csv"]:
            return None, "Nieobsługiwany typ pliku. Akceptowane formaty to XML i CSV."
        
        # Autodetekcja kodowania dla XML
        if file_type == "xml":
            encoding_declared = re.search(br'<\?xml[^>]*encoding=["\']([^"\']+)["\']', raw_bytes)
            encodings_to_try = [encoding_declared.group(1).decode('ascii')] if encoding_declared else []
        else:
            encodings_to_try = []
            
        # Lista kodowań do próbowania
        encodings_to_try += ["utf-8", "iso-8859-2", "windows-1250", "utf-16"]

        for enc in encodings_to_try:
            try:
                file_contents = raw_bytes.decode(enc)
                return {"content": file_contents, "raw_bytes": raw_bytes, 
                        "type": file_type, "encoding": enc, "name": uploaded_file.name}, None
            except UnicodeDecodeError:
                continue
        
        # Jeśli żadne kodowanie nie działa, spróbuj wczytać jako binarny
        if file_type == "csv":
            try:
                # Używając pandas do wykrycia separatora i kodowania
                buffer = io.BytesIO(raw_bytes)
                df = pd.read_csv(buffer, sep=None, engine='python')
                file_contents = df.to_csv(index=False)
                return {"content": file_contents, "raw_bytes": raw_bytes, 
                        "type": file_type, "encoding": "auto-detected", 
                        "name": uploaded_file.name, "dataframe": df}, None
            except Exception as e:
                pass
                
        return None, "Nie udało się odczytać pliku – nieznane kodowanie."
        
    except Exception as e:
        return None, f"Błąd podczas odczytu pliku: {str(e)}"


@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_code(file_contents, file_type, instruction, model, api_key):
    """Generuje kod z użyciem API z cachowaniem wyników"""
    prompt = f"""
Jesteś pomocnym asystentem, który generuje kod Python do modyfikacji plików typu {file_type.upper()}.
Użytkownik przesłał plik wejściowy. Kod powinien:
1. Wczytać plik z podanej ścieżki `input_path`
2. Zmodyfikować dane zgodnie z poniższą instrukcją
3. Zapisz wynik jako `output_path`

Dane wejściowe (fragment):
{file_contents[:1000]}

Instrukcja użytkownika:
{instruction}

Jeśli to plik CSV, użyj biblioteki pandas. Jeśli to XML, użyj xml.etree.ElementTree.

Wygeneruj kompletny kod, który:
- Otwiera plik z input_path
- Modyfikuje dane
- Zapisuje wynik do output_path

Nie dodawaj żadnych opisów ani komentarzy. Zwróć wyłącznie czysty kod Python.
"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Jesteś asystentem kodującym w Pythonie."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        with st.spinner("Generowanie kodu Python..."):
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                headers=headers, json=data, timeout=60)
            res.raise_for_status()
            
            code = res.json()["choices"][0]["message"]["content"]
            # Czyszczenie kodu
            code = re.sub(r"```(?:python)?\n", "", code)
            code = code.replace("```", "")
            code = re.sub(r"^\s*#.*$", "", code, flags=re.MULTILINE)
            code = re.sub(r"^\s*(print\(.*\)|if __name__ == .__main__.:.*)$", "", code, flags=re.MULTILINE)
            code = re.sub(r"(?i)^.*(oto kod|przykład|python).*$", "", code, flags=re.MULTILINE)
            
            return clean_and_validate_code(code)
            
    except requests.exceptions.RequestException as e:
        return f"Błąd podczas komunikacji z API: {str(e)}"


def clean_and_validate_code(code):
    """Czyści i waliduje wygenerowany kod"""
    # Usuwaj puste linie i whitespace
    code = "\n".join([line for line in code.splitlines() if line.strip()])
    
    # Upewnij się, że kod jest poprawny składniowo
    def sanitize_code(code):
        lines = code.strip().splitlines()
        while lines:
            try:
                ast.parse("\n".join(lines))
                break
            except SyntaxError:
                lines.pop()
        return "\n".join(lines)

    return sanitize_code(code).strip()


def execute_code_safely(generated_code, file_info):
    """Wykonuje wygenerowany kod w bezpiecznym środowisku"""
    with st.spinner("Wykonuję kod i przetwarzam dane..."):
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_path = os.path.join(tmpdirname, f"input.{file_info['type']}")
            output_path = os.path.join(tmpdirname, f"output.{file_info['type']}")
            
            # Zapisz plik wejściowy
            with open(input_path, "wb") as f:
                f.write(file_info["raw_bytes"])
            
            # Przygotuj kod do wykonania
            code = generated_code
            code = re.sub(r"input_path\s*=.*", "", code)
            code = re.sub(r"output_path\s*=.*", "", code)
            
            try:
                # Środowisko wykonania
                exec_globals = {
                    "__builtins__": __builtins__,
                    "pd": pd,
                    "io": io,
                    "input_path": input_path,
                    "output_path": output_path
                }
                
                exec(code, exec_globals)
                
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        output_bytes = f.read()
                    return {"success": True, "output": output_bytes}
                else:
                    return {"success": False, "error": "Nie wygenerowano pliku wynikowego."}
                    
            except Exception as e:
                return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def save_to_google_drive(output_bytes, file_info, instruction, generated_code):
    """Zapisuje wyniki do Google Drive"""
    try:
        drive_folder_id = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
        service_account_json = st.secrets.get("GOOGLE_DRIVE_CREDENTIALS_JSON")
        
        if not drive_folder_id or not service_account_json:
            return False, "Brak konfiguracji Google Drive."
            
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"history_{now}.txt"
        result_filename = f"output_{now}.{file_info['type']}"
        
        # Zapisz pliki tymczasowo
        with tempfile.NamedTemporaryFile(suffix=f".{file_info['type']}", delete=False) as temp_result:
            temp_result.write(output_bytes)
            temp_result_path = temp_result.name
            
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_log:
            log_content = f"INSTRUCTION:\n{instruction}\n\nCODE:\n{generated_code}"
            temp_log.write(log_content.encode('utf-8'))
            temp_log_path = temp_log.name
            
        # Uwierzytelnianie Google Drive
        with st.spinner("Zapisuję na Google Drive..."):
            creds_dict = json.loads(service_account_json)
            scope = ["https://www.googleapis.com/auth/drive"]
            credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            gauth = GoogleAuth()
            gauth.credentials = credentials
            drive = GoogleDrive(gauth)
            
            # Upload plików
            history_file = drive.CreateFile({"title": log_filename, "parents": [{"id": drive_folder_id}]})
            history_file.SetContentFile(temp_log_path)
            history_file.Upload()
            
            result_file = drive.CreateFile({"title": result_filename, "parents": [{"id": drive_folder_id}]})
            result_file.SetContentFile(temp_result_path)
            result_file.Upload()
            
            # Czyszczenie
            os.unlink(temp_log_path)
            os.unlink(temp_result_path)
            
            return True, "Pliki zostały zapisane na Google Drive."
            
    except Exception as e:
        return False, f"Błąd podczas zapisu na Google Drive: {str(e)}"


def main():
    """Główna funkcja aplikacji"""
    # Ustawienia strony
    st.set_page_config(page_title="Edytor XML/CSV z AI", layout="centered")
    
    # Uwierzytelnianie
    authenticate_user()
    
    # Inicjalizacja stanu sesji
    initialize_session_state()
    
    # Interfejs użytkownika - prosty layout bez sidebaru i złożonych komponentów
    st.title("Edytor AI plików XML i CSV")
    
    # Zakładki
    tab1, tab2 = st.tabs(["Edycja pliku", "Pomoc"])
    
    with tab1:
        st.markdown("""
        To narzędzie umożliwia modyfikację plików XML i CSV przy użyciu sztucznej inteligencji.
        Prześlij plik i wpisz polecenie w języku naturalnym, np.: _"Dodaj kolumnę Czas dostawy zależnie od dostępności"_.
        """)
        
        uploaded_file = st.file_uploader("Wgraj plik XML lub CSV", type=["xml", "csv"])
        
        if uploaded_file:
            file_info, error = read_file_content(uploaded_file)
            if error:
                st.error(error)
            else:
                st.success(f"Wczytano plik: {file_info['name']} ({file_info['type'].upper()}, {file_info['encoding']})")
                st.session_state.file_info = file_info
        
        instruction = st.text_area("Instrukcja modyfikacji (w języku naturalnym)")
        
        model = st.selectbox("Wybierz model LLM (OpenRouter)", [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
            "anthropic/claude-3-opus",
            "mistralai/mistral-7b-instruct",
            "google/gemini-pro"
        ])
        
        if uploaded_file and instruction:
            if st.button("Wygeneruj kod Python"):
                api_key = st.secrets["OPENROUTER_API_KEY"]
                st.session_state.generated_code = generate_ai_code(
                    st.session_state.file_info["content"], 
                    st.session_state.file_info["type"], 
                    instruction, 
                    model, 
                    api_key
                )
            
        # Wyświetl wygenerowany kod
        if st.session_state.generated_code:
            st.subheader("Wygenerowany kod:")
            st.code(st.session_state.generated_code, language="python")
            
            if st.button("Wykonaj kod i zapisz wynik"):
                result = execute_code_safely(
                    st.session_state.generated_code, 
                    st.session_state.file_info
                )
                
                if result["success"]:
                    st.session_state.output_bytes = result["output"]
                    
                    # Zapisz na Google Drive
                    success, message = save_to_google_drive(
                        st.session_state.output_bytes,
                        st.session_state.file_info,
                        instruction,
                        st.session_state.generated_code
                    )
                    
                    if success:
                        st.success(f"Dane zostały pomyślnie przetworzone! {message}")
                    else:
                        st.warning(f"Dane przetworzone, ale {message}")
                else:
                    st.error(f"Błąd: {result['error']}")
                    with st.expander("Szczegóły błędu", expanded=False):
                        st.code(result["traceback"])
        
        # Wyświetl wynik i przycisk pobierania, jeśli jest wygenerowany plik
        if st.session_state.output_bytes:
            st.subheader("Wynik przetwarzania:")
            
            try:
                # Jeśli to XML, wyświetl fragment
                if st.session_state.file_info["type"] == "xml":
                    xml_content = st.session_state.output_bytes.decode("utf-8", errors="replace")
                    st.code(xml_content[:1000] + ("..." if len(xml_content) > 1000 else ""), language="xml")
                # Jeśli to CSV, wyświetl jako tabelę
                elif st.session_state.file_info["type"] == "csv":
                    csv_content = st.session_state.output_bytes.decode("utf-8", errors="replace")
                    df = pd.read_csv(io.StringIO(csv_content))
                    st.dataframe(df.head(10))
                    if len(df) > 10:
                        st.info(f"Wyświetlono 10 z {len(df)} wierszy")
            except Exception as e:
                st.warning(f"Nie można wyświetlić podglądu: {str(e)}")
            
            # Przycisk pobierania
            file_type = st.session_state.file_info["type"]
            original_name = st.session_state.file_info["name"]
            base_name = os.path.splitext(original_name)[0]
            
            st.download_button(
                label=f"📁 Pobierz zmodyfikowany plik",
                data=st.session_state.output_bytes,
                file_name=f"{base_name}_processed.{file_type}",
                mime="text/plain"
            )
    
    with tab2:
        st.markdown("""
        ### Jak korzystać z aplikacji
        
        1. **Wgraj plik XML lub CSV** - aplikacja automatycznie wykryje kodowanie
        2. **Wpisz instrukcję** - opisz w języku naturalnym, co chcesz zmodyfikować
        3. **Wygeneruj kod** - AI stworzy kod Pythona wykonujący twoje polecenie
        4. **Wykonaj kod** - przetworzy twoje dane według instrukcji
        5. **Pobierz wynik** - zapisz przetworzony plik lokalnie
        
        ### Przykłady instrukcji
        
        - "Dodaj kolumnę z wartością TRUE jeśli cena > 1000, w przeciwnym razie FALSE"
        - "Zamień wszystkie wartości w kolumnie Status na duże litery"
        - "Usuń wszystkie elementy node gdzie atrybut type=temporary"
        - "Oblicz sumę wartości w kolumnach numerycznych i dodaj kolumnę z wynikami"
        
        ### Obsługiwane formaty
        
        - **XML** - modyfikacja struktury, atrybutów i wartości
        - **CSV** - operacje na kolumnach, filtrowanie, agregacja
        """)


if __name__ == "__main__":
    main()
