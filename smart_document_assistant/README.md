# Smart Document Assistant — no API keys

A Streamlit document-search app that accepts PDF, DOCX and TXT files and retrieves relevant passages with TF-IDF. No OpenAI, Groq, or other AI API key is needed.

## Features
- Multiple document uploads
- Question-based retrieval with source passages and relevance scores
- Fictional sample college FAQ
- Purple Streamlit UI
- No external LLM calls

## Deploy as a separate Streamlit app
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app**.
3. Repository: `pradeeppullamsetty/ds-faq-chatbot`
4. Branch: `main`
5. Main file path: `smart_document_assistant/app.py`
6. Click **Deploy**. No secrets or API keys are required.

Streamlit will provide a public `https://....streamlit.app` URL after the deployment succeeds.

## Local run
From this folder:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## How it works
TF-IDF and cosine similarity retrieve relevant passages. This edition does not generate free-form LLM answers; it shows source text so users can verify the information.

## Privacy
Do not upload confidential files to a public demo. Uploaded content is processed in the running app session and is not sent to an external AI API.
