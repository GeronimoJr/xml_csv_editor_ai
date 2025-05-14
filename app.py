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
                st.rerun()  # Używamy st.rerun() zamiast przestarzałego st.experimental_rerun()
            else:
                st.error("Nieprawidłowy login lub hasło")
        st.stop()
    return True


def initialize_session_state():
    """Inicjalizacja zmiennych sesji"""
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""
    if "edited_code" not in st.session_state:
        st.session_state.edited_code = ""
    if "output_bytes" not in st.session_state:
        st.session_state.output_bytes = None
    if "file_info" not in st.session_state:
        st.session_state.file_info = None
    if "show_editor" not in st.session_state:
        st.session_state.show_editor = False
    if "error_info" not in st.session_state:
        st.session_state.error_info = None
    if "code_fixed" not in st.session_state:
        st.session_state.code_fixed = False
    if "fix_requested" not in st.session_state:
        st.session_state.fix_requested = False


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


def fix_code_with_ai(code, error_message, traceback_str, file_info, instruction, model, api_key, max_attempts=2):
    """
    Próbuje naprawić kod z użyciem AI z ulepszoną skutecznością
    
    Args:
        code: Kod do naprawy
        error_message: Komunikat błędu
        traceback_str: Pełny traceback
        file_info: Informacje o pliku
        instruction: Oryginalna instrukcja użytkownika
        model: Model LLM do użycia
        api_key: Klucz API
        max_attempts: Maksymalna liczba prób naprawy
    
    Returns:
        Naprawiony kod lub None w przypadku niepowodzenia
    """
    # Wyodrębnij najważniejsze informacje z traceback
    error_type = "Unknown Error"
    error_line = "Unknown"
    problematic_code = ""
    
    # Znajdź typ błędu i linię błędu
    tb_lines = traceback_str.strip().split("\n")
    for i, line in enumerate(tb_lines):
        if "Error:" in line:
            error_type = line.strip()
        if "line " in line and ".py" in line:
            # Spróbuj wyodrębnić numer linii
            match = re.search(r"line (\d+)", line)
            if match:
                line_num = int(match.group(1))
                error_line = f"Line {line_num}"
                
                # Spróbuj znaleźć problematyczny kod
                code_lines = code.split("\n")
                if line_num <= len(code_lines):
                    start = max(0, line_num - 3)
                    end = min(len(code_lines), line_num + 2)
                    problematic_code = "\n".join(code_lines[start:end])
    
    # Przeanalizuj typ błędu, aby dostarczyć bardziej kontekstowe wskazówki
    error_hint = ""
    if "TypeError" in error_type:
        error_hint = "Sprawdź typy danych i operacje na nich. Możliwe, że próbujesz wykonać operację na niewłaściwym typie."
    elif "IndexError" in error_type or "KeyError" in error_type:
        error_hint = "Sprawdź indeksowanie list/słowników. Możliwe, że próbujesz uzyskać dostęp do nieistniejącego elementu."
    elif "AttributeError" in error_type:
        error_hint = "Sprawdź, czy obiekt posiada wywoływaną metodę/atrybut. Możliwe, że pracujesz na niewłaściwym typie obiektu."
    elif "FileNotFoundError" in error_type:
        error_hint = "Sprawdź ścieżki plików. Upewnij się, że używasz zmiennych input_path i output_path."
    elif "ValueError" in error_type and "encoding" in error_message:
        error_hint = "Sprawdź kodowanie pliku. Dodaj obsługę różnych kodowań."
    elif "SyntaxError" in error_type:
        error_hint = "Kod zawiera błąd składniowy. Sprawdź nawiasy, wcięcia, przecinki i inne elementy składni."
    
    # Pobierz fragment danych wejściowych, jeśli dostępne
    sample_data = ""
    if file_info and "content" in file_info:
        sample_data = file_info["content"][:500] + "..." if len(file_info["content"]) > 500 else file_info["content"]
    
    for attempt in range(max_attempts):
        # Różne podejścia do naprawy w zależności od numeru próby
        approach = ""
        if attempt == 0:
            approach = "Napraw tylko zidentyfikowany błąd, zachowując ogólną strukturę kodu."
        else:
            approach = "Spróbuj całkowicie przebudować kod, skupiając się na oryginalnym zadaniu."
            
        prompt = f"""
Jestem ekspertem w naprawianiu kodu Python i potrzebuję naprawić kod przetwarzający plik {file_info['type'].upper()}.

## Oryginalny kod z błędem:
```python
{code}
Szczegóły błędu:

Typ błędu: {error_type}

Lokalizacja: {error_line}

Komunikat: {error_message}

Problematyczny fragment:
{problematic_code}
Analiza błędu:
{error_hint}

Dane wejściowe (fragment):
{sample_data}
Kontekst zadania:
Ten kod ma realizować następującą instrukcję: "{instruction}"

Podejście do naprawy (próba {attempt+1}):
{approach}

Podejście naprawcze:


Znajdź i napraw główną przyczynę błędu

Upewnij się, że kod korzysta ze zmiennych input_path do wczytania i output_path do zapisu

Dostosuj kod, aby był odporny na różne formaty danych

Zachowaj podstawowe funkcjonalności zgodne z intencją oryginalnego kodu

Zwróć TYLKO poprawiony, działający kod jako blok kodu Python, bez żadnych wyjaśnień.
"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Jestem ekspertem w naprawianiu kodu Python, szczególnie do przetwarzania plików XML i CSV."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3  # Niższe temperature dla bardziej deterministycznych odpowiedzi
        }

        try:
            with st.spinner(f"Naprawianie kodu (próba {attempt+1}/{max_attempts})..."):
                res = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                    headers=headers, json=data, timeout=90)  # Dłuższy timeout
                res.raise_for_status()

                fixed_code = res.json()["choices"][0]["message"]["content"]
                # Czyszczenie kodu
                fixed_code = re.sub(r"```(?:python)?\n", "", fixed_code)
                fixed_code = fixed_code.replace("```", "")
                
                # Dodatkowe czyszczenie
                fixed_code = fixed_code.strip()

                # Walidacja kodu - sprawdzenie składni
                try:
                    ast.parse(fixed_code)
                    clean_code = clean_and_validate_code(fixed_code)

                    # Sprawdzenie czy kod zawiera wymagane elementy
                    if "input_path" not in clean_code or "output_path" not in clean_code:
                        continue  # Jeśli brakuje kluczowych elementów, spróbuj ponownie
                    
                    return clean_code
                except SyntaxError:
                    # Jeśli składnia wciąż jest niepoprawna, próbujemy ponownie
                    continue

        except requests.exceptions.RequestException as e:
            st.warning(f"Błąd podczas komunikacji z API: {str(e)}. Ponowna próba...")
            continue

    # Jeśli wszystkie próby się nie powiodły, zwróć None
    return None


def validate_code_logic(code, file_type):
    """
    Sprawdza czy kod zawiera logiczne elementy potrzebne do obsługi danego typu pliku
    
    Args:
        code: Kod do sprawdzenia
        file_type: Typ pliku (xml lub csv)
    
    Returns:
        True jeśli kod wydaje się logicznie poprawny, False w przeciwnym razie
    """
    required_elements = {
        'xml': ['ElementTree', 'parse', 'write', 'findall', 'Element'],
        'csv': ['pandas', 'pd', 'read_csv', 'to_csv', 'DataFrame']
    }

    elements_count = 0
    for element in required_elements.get(file_type, []):
        if element in code:
            elements_count += 1

    # Sprawdź czy kod zawiera przynajmniej 2 wymagane elementy dla danego typu pliku
    return elements_count >= 2


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


def execute_code_safely(code_to_execute, file_info):
    """Wykonuje wygenerowany kod w bezpiecznym środowisku"""
    with st.spinner("Wykonuję kod i przetwarzam dane..."):
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_path = os.path.join(tmpdirname, f"input.{file_info['type']}")
            output_path = os.path.join(tmpdirname, f"output.{file_info['type']}")
            
            # Zapisz plik wejściowy
            with open(input_path, "wb") as f:
                f.write(file_info["raw_bytes"])

            # Przygotuj kod do wykonania
            code = code_to_execute
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


def save_to_google_drive(output_bytes, file_info, instruction, code_executed):
    """Zapisuje wyniki do Google Drive"""
    try:
        # Sprawdź, czy mamy wszystkie potrzebne dane konfiguracyjne
        drive_folder_id = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
        credentials_json = st.secrets.get("GOOGLE_DRIVE_CREDENTIALS_JSON")
        
        if not drive_folder_id or not credentials_json:
            return False, "Brak konfiguracji Google Drive."

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"history_{now}.txt"
        result_filename = f"output_{now}.{file_info['type']}"
        
        # Utwórz tymczasowy katalog zamiast pojedynczych plików
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Zapisz pliki tymczasowo w katalogu tymczasowym
            temp_result_path = os.path.join(tmpdirname, f"output.{file_info['type']}")
            temp_log_path = os.path.join(tmpdirname, "log.txt")
            
            with open(temp_result_path, "wb") as f:
                f.write(output_bytes)
                
            log_content = f"INSTRUCTION:\n{instruction}\n\nCODE:\n{code_executed}"
            with open(temp_log_path, "w", encoding='utf-8') as f:
                f.write(log_content)
            
            # Uwierzytelnianie Google Drive
            with st.spinner("Zapisuję na Google Drive..."):
                # Przygotuj poświadczenia z JSON
                if isinstance(credentials_json, str):
                    try:
                        creds_dict = json.loads(credentials_json)
                    except json.JSONDecodeError:
                        st.error("Błąd dekodowania JSON z credentials")
                        return False, "Błąd dekodowania JSON z credentials"
                else:
                    creds_dict = credentials_json
                
                scope = ["https://www.googleapis.com/auth/drive"]
                credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                
                # Konfiguruj GoogleAuth
                gauth = GoogleAuth()
                gauth.credentials = credentials
                
                # Utwórz obiekt GoogleDrive
                drive = GoogleDrive(gauth)
                
                try:
                    # Historia operacji
                    history_file = drive.CreateFile({
                        "title": log_filename, 
                        "parents": [{"id": drive_folder_id}],
                        "mimeType": "text/plain"
                    })
                    history_file.SetContentFile(temp_log_path)
                    history_file.Upload()
                    
                    # Plik wynikowy
                    result_file = drive.CreateFile({
                        "title": result_filename, 
                        "parents": [{"id": drive_folder_id}],
                        "mimeType": f"application/{file_info['type']}"
                    })
                    result_file.SetContentFile(temp_result_path)
                    result_file.Upload()
                    
                    st.success("Pliki zostały zapisane na Google Drive.")
                    return True, "Pliki zostały zapisane na Google Drive."
                    
                except Exception as upload_error:
                    st.error(f"Błąd podczas wysyłania plików: {str(upload_error)}")
                    return False, f"Błąd podczas wysyłania plików: {str(upload_error)}"
            
    except Exception as e:
        st.error(f"Błąd podczas zapisu na Google Drive: {str(e)}")
        return False, f"Błąd podczas zapisu na Google Drive: {str(e)}"


def reset_app_state():
    """Resetuje stan aplikacji do ponownej edycji"""
    for key in list(st.session_state.keys()):
        if key != "authenticated":
            del st.session_state[key]
    initialize_session_state()
    st.rerun()


def toggle_editor():
    """Przełącza widoczność edytora kodu"""
    current_state = st.session_state.show_editor
    st.session_state.show_editor = not current_state
    if not current_state:  # tylko jeśli edytor był wcześniej ukryty
        st.session_state.edited_code = st.session_state.generated_code


def handle_fix_request():
    """Obsługuje żądanie naprawy kodu"""
    st.session_state.fix_requested = True
    st.rerun()  # Natychmiastowe odświeżenie strony


def main():
    """Główna funkcja aplikacji"""
    # Ustawienia strony
    st.set_page_config(page_title="Edytor XML/CSV z AI", layout="centered")
    # Uwierzytelnianie
    authenticate_user()

    # Inicjalizacja stanu sesji
    initialize_session_state()

    # Interfejs użytkownika - prosty layout
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
            "openai/gpt-4o-mini:floor",
            "openai/gpt-4o:floor",
            "anthropic/claude-3.5-haiku:floor",
            "anthropic/claude-3.7-sonnet:floor",
            "google/gemini-2.5-pro-preview:floor"
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
                st.session_state.edited_code = st.session_state.generated_code
                st.session_state.show_editor = False
                st.session_state.code_fixed = False
                st.session_state.error_info = None
                st.session_state.fix_requested = False
        
        # Wyświetl wygenerowany kod
        if st.session_state.generated_code:
            st.subheader("Wygenerowany kod Python:")
            st.code(st.session_state.generated_code, language="python")
            
            # NAPRAWIONO: Dodanie key do przycisku, aby uniknąć opóźnień
            if st.button("Edytuj kod" if not st.session_state.show_editor else "Ukryj edytor", key="toggle_editor_button"):
                toggle_editor()
                st.rerun()  # Natychmiastowe odświeżenie interfejsu
            
            # Wyświetl edytor kodu, jeśli tryb edycji jest aktywny
            if st.session_state.show_editor:
                st.session_state.edited_code = st.text_area(
                    "Edycja kodu",
                    value=st.session_state.edited_code,
                    height=400,
                    key="code_editor"
                )
            
            # Wyświetl informacje o naprawionym kodzie
            if st.session_state.code_fixed:
                st.info("Kod został naprawiony po wykryciu błędu.")
                with st.expander("Pokaż naprawiony kod", expanded=False):
                    st.code(st.session_state.edited_code, language="python")
            
            # Kod do wykonania
            code_to_execute = st.session_state.edited_code if st.session_state.show_editor or st.session_state.code_fixed else st.session_state.generated_code
            
            if st.button("Wykonaj kod i zapisz wynik"):
                result = execute_code_safely(
                    code_to_execute, 
                    st.session_state.file_info
                )
                
                if result["success"]:
                    st.session_state.output_bytes = result["output"]
                    st.success("Dane zostały pomyślnie przetworzone!")
                    
                    # Zapisz na Google Drive
                    try:
                        success, message = save_to_google_drive(
                            st.session_state.output_bytes,
                            st.session_state.file_info,
                            instruction,
                            code_to_execute
                        )
                        
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.warning(f"⚠️ {message}")
                    
                    except Exception as e:
                        st.error(f"Błąd podczas zapisywania na Google Drive: {str(e)}")
                else:
                    # NAPRAWIONO: Obsługa błędów z zachowaniem kontekstu do naprawy
                    st.error(f"Błąd: {result['error']}")
                    st.session_state.error_info = result
                    
                    # Zabezpieczenie przed streamlit blokującym wyświetlanie błędów
                    st.session_state["error_data"] = {
                        "error": result['error'],
                        "traceback": result['traceback']
                    }
                    
                    # Dodaj więcej szczegółów diagnostycznych 
                    with st.expander("Szczegóły błędu", expanded=True):
                        st.code(result["traceback"])
                        
                        # Dodatkowa analiza błędu dla lepszego zrozumienia
                        error_type = "Nieokreślony"
                        error_location = "Nieokreślona"
                        
                        tb_lines = result["traceback"].strip().split("\n")
                        for line in tb_lines:
                            if "Error:" in line:
                                error_type = line.strip()
                            if "line " in line and ".py" in line:
                                error_location = line.strip()
                        
                        st.markdown(f"**Typ błędu:** {error_type}")
                        if error_location != "Nieokreślona":
                            st.markdown(f"**Lokalizacja błędu:** {error_location}")
                    
                    # NAPRAWIONO: Przyciski do naprawy z unikalnymi kluczami
                    repair_col1, repair_col2 = st.columns([1, 1])
                    with repair_col1:
                        if st.button("Napraw kod z pomocą AI", key="ai_repair_button", on_click=handle_fix_request):
                            pass
                        
                    with repair_col2:
                        if st.button("Przejdź do ręcznej edycji", key="manual_edit_button"):
                            st.session_state.show_editor = True
                            st.rerun()
                    
                    # NAPRAWIONO: Obsługa żądania naprawy kodu z pomocą AI
                    if st.session_state.fix_requested and st.session_state.error_info:
                        api_key = st.secrets["OPENROUTER_API_KEY"]
                        fixed_code = fix_code_with_ai(
                            code_to_execute,
                            result["error"],
                            result["traceback"],
                            st.session_state.file_info,
                            instruction,
                            model,
                            api_key,
                            max_attempts=2
                        )
                        
                        if fixed_code:
                            st.session_state.edited_code = fixed_code
                            st.session_state.code_fixed = True
                            st.session_state.fix_requested = False
                            st.success("Kod został naprawiony. Możesz teraz ponownie wykonać kod.")
                            st.rerun()
                        else:
                            st.error("Nie udało się naprawić kodu automatycznie. Spróbuj ręcznej edycji.")
                            st.session_state.show_editor = True
                            st.session_state.fix_requested = False
                            st.rerun()
        
        # Przycisk pobierania jeśli jest wygenerowany plik
        if st.session_state.output_bytes:
            file_type = st.session_state.file_info["type"]
            original_name = st.session_state.file_info["name"]
            base_name = os.path.splitext(original_name)[0]
            
            st.download_button(
                label=f"📁 Pobierz zmodyfikowany plik",
                data=st.session_state.output_bytes,
                file_name=f"{base_name}_processed.{file_type}",
                mime="text/plain"
            )
            
            # Dodanie przycisku "Ponowna edycja" na końcu
            if st.button("Ponowna edycja"):
                reset_app_state()

    with tab2:
        st.markdown("""
        ### Jak korzystać z aplikacji
        
        1. **Wgraj plik XML lub CSV** - aplikacja automatycznie wykryje kodowanie
        2. **Wpisz instrukcję** - opisz w języku naturalnym, co chcesz zmodyfikować
        3. **Wygeneruj kod** - AI stworzy kod Pythona wykonujący twoje polecenie
        4. **Opcjonalnie: Edytuj kod** - kliknij przycisk "Edytuj kod", aby zmodyfikować wygenerowany kod
        5. **Wykonaj kod** - przetworzy twoje dane według instrukcji
        6. **W przypadku błędu** - możesz:
           - Użyć przycisku "Napraw kod z pomocą AI" do automatycznej naprawy kodu
           - Wybrać "Przejdź do ręcznej edycji" aby samodzielnie poprawić kod
        7. **Pobierz wynik** - zapisz przetworzony plik lokalnie
        
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
