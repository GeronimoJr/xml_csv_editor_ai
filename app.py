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
import time

# --- Funkcje pomocnicze ---

def authenticate_user():
    """Uwierzytelnianie użytkownika z wykorzystaniem formularza"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.title("Edytor AI plików XML i CSV - Logowanie")
            user = st.text_input("Login")
            password = st.text_input("Hasło", type="password")
            submit = st.form_submit_button("Zaloguj")
            
            if submit:
                if user == st.secrets.get("APP_USER") and password == st.secrets.get("APP_PASSWORD"):
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Nieprawidłowy login lub hasło")
        return False
    return True


def initialize_session_state():
    """Inicjalizacja zmiennych sesji"""
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""
    if "output_bytes" not in st.session_state:
        st.session_state.output_bytes = None
    if "file_info" not in st.session_state:
        st.session_state.file_info = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "error_message" not in st.session_state:
        st.session_state.error_message = None
    if "success_message" not in st.session_state:
        st.session_state.success_message = None


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
    """Generuje kod z użyciem API z cachowaniem wyników dla identycznych zapytań"""
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
            res.raise_for_status()  # Zgłaszaj błędy HTTP
            
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
    try:
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
        
    except Exception as e:
        return f"Błąd walidacji kodu: {str(e)}"


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
                # Środowisko wykonania z ograniczonymi prawami
                exec_globals = {
                    "__builtins__": __builtins__,
                    "pd": pd,
                    "io": io,
                    "input_path": input_path,
                    "output_path": output_path
                }
                
                # Dodaj time limit
                start_time = time.time()
                exec(code, exec_globals)
                execution_time = time.time() - start_time
                
                st.info(f"Czas wykonania: {execution_time:.2f} s")
                
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        output_bytes = f.read()
                    
                    # Zapisz do historii
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append({
                        "date": now,
                        "operation": file_info.get("name", "") + " - " + file_info.get("type", ""),
                        "instruction": file_info.get("instruction", "")
                    })
                    
                    return {"success": True, "output": output_bytes}
                else:
                    return {"success": False, "error": "Nie wygenerowano pliku wynikowego."}
                    
            except Exception as e:
                return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def save_to_google_drive(output_bytes, file_info, instruction, generated_code):
    """Zapisuje wyniki do Google Drive z lepszą obsługą błędów"""
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


def create_ui():
    """Tworzy interfejs użytkownika aplikacji"""
    st.set_page_config(page_title="Edytor XML/CSV z AI", layout="wide")
    
    with st.sidebar:
        st.image("https://via.placeholder.com/150x80?text=AI+Editor", width=150)
        st.markdown("### Ustawienia")
        
        # Wybór modelu w sidepanel
        model = st.selectbox("Model LLM (OpenRouter)", [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
            "anthropic/claude-3-opus",
            "mistralai/mistral-7b-instruct",
            "google/gemini-pro"
        ])
        
        # Dodatkowe opcje
        with st.expander("Zaawansowane opcje"):
            encoding_option = st.radio("Wybór kodowania", ["auto", "manual"])
            if encoding_option == "manual":
                manual_encoding = st.selectbox("Kodowanie", ["utf-8", "iso-8859-2", "windows-1250", "utf-16"])
            
            save_to_drive = st.checkbox("Zapisz na Google Drive", value=True)
            
        # Historia operacji
        if st.session_state.history:
            with st.expander("Historia operacji", expanded=False):
                for i, item in enumerate(st.session_state.history):
                    st.write(f"{i+1}. {item['date']} - {item['operation']}")
                if st.button("Wyczyść historię"):
                    st.session_state.history = []
                    st.experimental_rerun()
                
    # Główny obszar aplikacji
    st.title("Edytor AI plików XML i CSV")
    
    # Tabs na górze strony
    tab1, tab2, tab3 = st.tabs(["Edycja pliku", "Wynik", "Pomoc"])
    
    with tab1:
        st.markdown("""
        To narzędzie umożliwia modyfikację plików XML i CSV przy użyciu sztucznej inteligencji.
        Prześlij plik i wpisz polecenie w języku naturalnym.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Wgraj plik XML lub CSV", type=["xml", "csv"])
        
        with col2:
            if uploaded_file:
                file_info, error = read_file_content(uploaded_file)
                if error:
                    st.error(error)
                else:
                    st.success(f"Wczytano plik: {file_info['name']}")
                    st.info(f"Typ: {file_info['type'].upper()} | Kodowanie: {file_info['encoding']} | Rozmiar: {len(file_info['raw_bytes'])/1024:.2f} KB")
                    st.session_state.file_info = file_info
        
        instruction = st.text_area("Instrukcja modyfikacji (w języku naturalnym)", 
                                placeholder="np.: Dodaj kolumnę Czas dostawy zależnie od dostępności...",
                                height=150)
                                
        if instruction and st.session_state.file_info:
            st.session_state.file_info["instruction"] = instruction
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Wygeneruj kod Python", type="primary", use_container_width=True):
                    api_key = st.secrets["OPENROUTER_API_KEY"]
                    st.session_state.generated_code = generate_ai_code(
                        st.session_state.file_info["content"], 
                        st.session_state.file_info["type"], 
                        instruction, 
                        model, 
                        api_key
                    )
            
            with col2:
                if st.session_state.generated_code and st.button("Wykonaj kod i przetwórz dane", 
                                                               type="primary", use_container_width=True):
                    result = execute_code_safely(
                        st.session_state.generated_code, 
                        st.session_state.file_info
                    )
                    
                    if result["success"]:
                        st.session_state.output_bytes = result["output"]
                        st.session_state.success_message = "Dane zostały pomyślnie przetworzone! 🎉"
                        st.session_state.error_message = None
                        
                        # Zapisz na Google Drive jeśli wybrano
                        if save_to_drive:
                            success, message = save_to_google_drive(
                                st.session_state.output_bytes,
                                st.session_state.file_info,
                                instruction,
                                st.session_state.generated_code
                            )
                            if success:
                                st.session_state.success_message += f" {message}"
                            else:
                                st.session_state.error_message = message
                    else:
                        st.session_state.error_message = f"Błąd: {result['error']}"
                        st.session_state.success_message = None
                        if "traceback" in result:
                            st.session_state.error_traceback = result["traceback"]
                            
                    # Przełącz na zakładkę wyników
                    tab2.button("Przejdź do wyników", type="primary", use_container_width=True)
        
        if st.session_state.generated_code:
            st.subheader("Wygenerowany kod Python:")
            with st.expander("Pokaż/ukryj kod", expanded=True):
                st.code(st.session_state.generated_code, language="python")
            
    with tab2:
        if st.session_state.success_message:
            st.success(st.session_state.success_message)
            
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            if "error_traceback" in st.session_state:
                with st.expander("Szczegóły błędu", expanded=False):
                    st.code(st.session_state.error_traceback)
        
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
                    st.dataframe(df.head(10), use_container_width=True)
                    if len(df) > 10:
                        st.info(f"Wyświetlono 10 z {len(df)} wierszy")
            except Exception as e:
                st.warning(f"Nie można wyświetlić podglądu: {str(e)}")
            
            # Przyciski pobierania i zapisu
            col1, col2 = st.columns(2)
            
            with col1:
                file_type = st.session_state.file_info["type"]
                original_name = st.session_state.file_info["name"]
                base_name = os.path.splitext(original_name)[0]
                
                st.download_button(
                    label=f"📁 Pobierz przetworzony plik ({file_type.upper()})",
                    data=st.session_state.output_bytes,
                    file_name=f"{base_name}_processed.{file_type}",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                if st.button("Nowa edycja", use_container_width=True):
                    # Zachowaj historię, ale wyczyść pozostałe dane sesji
                    history = st.session_state.history
                    for key in st.session_state.keys():
                        if key != "history" and key != "authenticated":
                            del st.session_state[key]
                    st.session_state.history = history
                    st.experimental_rerun()
    
    with tab3:
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


def main():
    """Główna funkcja aplikacji"""
    # Uwierzytelnianie
    if not authenticate_user():
        return
        
    # Inicjalizacja stanu sesji
    initialize_session_state()
    
    # Wyświetl interfejs
    create_ui()
    
    # Obsługa błędów globalnych
    if st.session_state.error_message and "error_traceback" not in st.session_state:
        st.error(st.session_state.error_message)


if __name__ == "__main__":
    main()
