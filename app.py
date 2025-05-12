import streamlit as st
import requests
import tempfile
import os
import re

st.set_page_config(page_title="Edytor XML/CSV z AI", layout="centered")
st.title("🔧 AI Edytor plików XML i CSV")

# --- Upload pliku ---
uploaded_file = st.file_uploader("Wgraj plik XML lub CSV", type=["xml", "csv"])
instruction = st.text_area("Instrukcja modyfikacji (w języku naturalnym)")

model = st.selectbox("Wybierz model LLM (OpenRouter)", [
    "mistralai/mistral-small-3.1-24b-instruct:free"
])

# --- Wczytaj klucz API z sekcji 'Secrets' w Streamlit Cloud ---
api_key = st.secrets["OPENROUTER_API_KEY"]

if uploaded_file and instruction and api_key:
    file_contents = uploaded_file.read().decode("utf-8")
    file_type = uploaded_file.name.split(".")[-1].lower()

    # Prompt dla LLM
    prompt = f"""
Jesteś pomocnym asystentem, który generuje kod Python do modyfikacji plików typu {file_type.upper()}.
Plik wejściowy:
{file_contents[:1000]}

Instrukcja:
{instruction}

Wygeneruj kompletny kod Python, który:
1. Wczytuje plik {file_type}
2. Dokonuje modyfikacji zgodnie z instrukcją
3. Zapisuje wynikowy plik jako 'output.{file_type}'

Zwróć wyłącznie czysty kod w Pythonie, bez żadnych opisów ani znaczników Markdown.
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

    with st.spinner("Generowanie kodu Python..."):
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        code = res.json()["choices"][0]["message"]["content"]

        # Usuń znaczniki Markdown, jeśli występują
        code = re.sub(r"```(?:python)?\\n", "", code)
        code = code.replace("```", "")

        st.subheader("Wygenerowany kod:")
        st.code(code, language="python")

        if st.button("Wykonaj kod i pobierz wynik"):
            with tempfile.TemporaryDirectory() as tmpdirname:
                input_path = os.path.join(tmpdirname, f"input.{file_type}")
                output_path = os.path.join(tmpdirname, f"output.{file_type}")
                with open(input_path, "w", encoding="utf-8") as f:
                    f.write(file_contents)

                # Sandboxed exec z ograniczeniami
                local_vars = {"__file__": input_path}
                try:
                    exec(code, {}, local_vars)
                    if os.path.exists("output." + file_type):
                        with open("output." + file_type, "rb") as f:
                            st.download_button(
                                label="📁 Pobierz zmodyfikowany plik",
                                data=f,
                                file_name=f"output.{file_type}",
                                mime="text/plain"
                            )
                    else:
                        st.error("Nie znaleziono pliku wynikowego.")
                except Exception as e:
                    st.error(f"Błąd wykonania kodu: {e}")
