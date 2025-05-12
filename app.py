import streamlit as st
import requests
import tempfile
import os
import re

st.set_page_config(page_title="Edytor XML/CSV z AI", layout="centered")
st.title("🔧 AI Edytor plików XML i CSV")

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

if uploaded_file and instruction and api_key:
    file_contents = uploaded_file.read().decode("utf-8")
    file_type = uploaded_file.name.split(".")[-1].lower()

    if st.button("Wygeneruj kod Python"):
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

        with st.spinner("Generowanie kodu Python..."):
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            code = res.json()["choices"][0]["message"]["content"]

            code = re.sub(r"```(?:python)?\n", "", code)
            code = code.replace("```", "")

            st.subheader("Wygenerowany kod:")
            st.code(code, language="python")

            if st.button("Wykonaj kod i pobierz wynik"):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    input_path = os.path.join(tmpdirname, f"input.{file_type}")
                    output_path = os.path.join(tmpdirname, f"output.{file_type}")
                    with open(input_path, "w", encoding="utf-8") as f:
                        f.write(file_contents)

                    try:
                        exec(code, {"input_path": input_path, "output_path": output_path})
                        if os.path.exists(output_path):
                            with open(output_path, "rb") as f:
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
