# AI-Powered College FAQ Chatbot Using LLM and RAG

A professional, final-year B.Tech project featuring an intelligent chatbot designed to answer college-related FAQs based exclusively on official college documents using RAG (Retrieval-Augmented Generation).

## Features
- **Retrieval-Augmented Generation (RAG):** Extracts answers directly from provided documents, eliminating hallucinations.
- **Strict Fallback:** Replies "I don't have that information. Please contact the college office" if the answer isn't in the docs.
- **Multilingual Support:** Handles English and Tamil.
- **Stunning UI:** Glassmorphism, animated backgrounds, fully responsive layout.
- **Analytics Dashboard:** Real-time metrics logging and visualization.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd college-faq-chatbot
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   Copy `.env.example` to `.env` and add your OpenAI API key (`OPENAI_API_KEY`).

   ```bash
   cp .env.example .env
   ```

4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## Adding Your Own College Data
1. Delete the sample `.txt` files in the `data/` directory.
2. Add your college's PDF or TXT documents to the `data/` directory.
3. Open the app, go to the "Knowledge Base" page, and click **Rebuild Knowledge Base Vector Index**.

## Tests & Evaluation
Run `pytest tests/` for unit tests.
Run `python evaluate.py` to generate an evaluation report based on sample questions.