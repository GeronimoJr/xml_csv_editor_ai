# 🧠 AI Edytor plików XML i CSV

Aplikacja Streamlit, która modyfikuje pliki XML i CSV na podstawie instrukcji w języku naturalnym, korzystając z modeli LLM przez [OpenRouter.ai](https://openrouter.ai).

## 🔧 Funkcje
- Obsługa plików `.xml` i `.csv`
- Wybór modelu (GPT-4 Turbo, Claude 3 Opus, Mistral itd.)
- Automatyczne generowanie kodu Python
- Wykonywanie kodu i eksport wynikowego pliku

## 🚀 Jak uruchomić

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/twoj-user/xml-csv-editor-ai.git
cd xml-csv-editor-ai
```

### 2. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

### 3. Uruchom aplikację lokalnie

```bash
streamlit run app.py
```

### 4. Lub uruchom w chmurze (za darmo)

1. Wejdź na [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Zaloguj się przez GitHub
3. Wybierz to repozytorium i folder z `app.py`
4. Gotowe 🎉

### 5. Skąd wziąć OpenRouter API Key?

Zarejestruj się na [https://openrouter.ai](https://openrouter.ai), przejdź do ustawień konta i wygeneruj klucz API.

---

## 📜 Licencja
MIT
