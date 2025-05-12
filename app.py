import streamlit as st
import requests
import tempfile
import os
import re
import traceback
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

st.set_page_config(page_title="Edytor XML/CSV z AI", layout="centered")
st.title("🔧 AI Edytor plików XML i CSV")

# --- Stan aplikacji ---
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "output_bytes" not in st.session_state:
    st.session_state.output_bytes = None
if "technical_prompt" not in st.session_state:
    st.session_state.technical_prompt = ""

uploaded_file = st.file_uploader("Wgraj plik XML lub CSV", type=["xml", "csv"])
instruction = st.text_area("Instrukcja modyfikacji (w języku naturalnym)")

model = st.selectbox("Wybierz model LLM (OpenRouter)", [
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "anthropic/claude-3-opus",
    "mistralai/mistral-7b-instruct",
    "google/gemini-pro"
])

api_key = st.secrets["OPENROUTER_API_KEY"]
drive_folder_id = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")  # Dodaj do secrets

if uploaded_file and instruction and api_key:
    file_contents = uploaded_file.read().decode("utf-8")
    file_type = uploaded_file.name.split(".")[-1].lower()

    if st.button("1️⃣ Stwórz prompt techniczny"):
        initial_prompt = f"""
Zamień poniższą instrukcję użytkownika na szczegółową, techniczną instrukcję dla modelu LLM, która pozwoli wygenerować kompletny kod Python:

Instrukcja użytkownika:
{instruction}

Typ pliku: {file_type.upper()}
Fragment danych:
{file_contents[:500]}
        """

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": initial_prompt}]
        }
        with st.spinner("Generowanie promptu technicznego..."):
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            st.session_state.technical_prompt = res.json()["choices"][0]["message"]["content"]

    if st.session_state.technical_prompt:
        st.subheader("Prompt techniczny:")
        st.code(st.session_state.technical_prompt, language="markdown")

        if st.button("2️⃣ Wygeneruj kod Python"):
            gen_prompt = f"""
Jesteś pomocnym asystentem kodującym w Pythonie. Wygeneruj kompletny kod Python zgodnie z tą instrukcją:
{st.session_state.technical_prompt}

Plik wejściowy to input_path, plik wynikowy to output_path.
Zwróć tylko czysty kod Python bez opisów.
            """
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Jesteś asystentem kodującym."},
                    {"role": "user", "content": gen_prompt}]
            }
            with st.spinner("Generowanie kodu Python..."):
                res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
                code = res.json()["choices"][0]["message"]["content"]
                code = re.sub(r"```(?:python)?\n", "", code).replace("```", "")
                st.session_state.generated_code = code

    if st.session_state.generated_code:
        st.subheader("3️⃣ Wygenerowany kod:")
        st.code(st.session_state.generated_code, language="python")

        if st.button("4️⃣ Zrecenzuj kod przed wykonaniem"):
            review_prompt = f"Zrecenzuj poniższy kod Python pod kątem błędów i popraw go, jeśli trzeba:\n\n{st.session_state.generated_code}"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Jesteś doświadczonym recenzentem kodu w Pythonie."},
                    {"role": "user", "content": review_prompt}]
            }
            with st.spinner("Sprawdzanie i optymalizacja kodu..."):
                res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
                reviewed = res.json()["choices"][0]["message"]["content"]
                reviewed = re.sub(r"```(?:python)?\n", "", reviewed).replace("```", "")
                st.session_state.generated_code = reviewed

    if st.session_state.generated_code:
        if st.button("5️⃣ Wykonaj kod i zapisz wynik"):
            with tempfile.TemporaryDirectory() as tmpdirname:
                input_path = os.path.join(tmpdirname, f"input.{file_type}")
                output_path = os.path.join(tmpdirname, f"output.{file_type}")
                with open(input_path, "w", encoding="utf-8") as f:
                    f.write(file_contents)

                code = st.session_state.generated_code
                code = re.sub(r"input_path\s*=.*", "", code)
                code = re.sub(r"output_path\s*=.*", "", code)

                try:
                    exec_globals = {
                        "__builtins__": __builtins__,
                        "input_path": input_path,
                        "output_path": output_path
                    }
                    exec(code, exec_globals)
                    if os.path.exists(output_path):
                        with open(output_path, "rb") as f:
                            st.session_state.output_bytes = f.read()

                        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        log_filename = f"history_{now}.txt"
                        result_filename = f"output_{now}.{file_type}"

                        with open(log_filename, "w", encoding="utf-8") as log:
                            log.write(f"INSTRUCTION:\n{instruction}\n\nPROMPT:\n{st.session_state.technical_prompt}\n\nCODE:\n{st.session_state.generated_code}")

                        # --- Upload to Google Drive ---
                        if drive_folder_id:
                            gauth = GoogleAuth()
                            gauth.LocalWebserverAuth()
                            drive = GoogleDrive(gauth)

                            history_file = drive.CreateFile({"title": log_filename, "parents": [{"id": drive_folder_id}]})
                            history_file.SetContentFile(log_filename)
                            history_file.Upload()

                            result_file = drive.CreateFile({"title": result_filename, "parents": [{"id": drive_folder_id}]})
                            result_file.SetContentFile(output_path)
                            result_file.Upload()

                            st.success("Pliki zapisane na Twoim Google Drive ✅")
                    else:
                        st.error("Nie znaleziono pliku wynikowego.")
                except Exception as e:
                    st.error("Błąd wykonania kodu:")
                    st.exception(traceback.format_exc())

    if st.session_state.output_bytes:
        st.download_button(
            label="📁 Pobierz zmodyfikowany plik",
            data=st.session_state.output_bytes,
            file_name=f"output.{file_type}",
            mime="text/plain"
        )
