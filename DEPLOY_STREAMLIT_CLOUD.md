# Streamlit Community Cloud Deploy — 10-minute checklist

## 1. Push to GitHub
```bash
# From the project root (outputs folder)
git init
git add .
git commit -m "Initial commit: DS University FAQ Chatbot"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

⚠️ **Do not** push `.env` or `.streamlit/secrets.toml` — they're in `.gitignore`.

## 2. Connect on share.streamlit.io
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app** → select your repo.
3. Branch: `main`
4. Main file path: `app.py`
5. Click **Deploy**.

## 3. Add your secret
- While the app builds (or after), open the app's **Settings → Secrets** in Streamlit Cloud.
- Paste the content of `.streamlit/secrets.toml.example` with your **real OpenAI key**:
  ```toml
  OPENAI_API_KEY = "sk-..."   # <-- your actual key
  ```
- Save. The app will auto-reload and the warning disappears.

## 4. Test the live chatbot
Open the URL Streamlit gives you (e.g., `https://your-app-name.streamlit.app`).
Ask: "What is the B.Tech eligibility?" — you should get a real answer.

## 5. Wire the Netlify landing page
- Open `netlify-preview.html` in a browser → paste the Streamlit URL into the iframe input → click **Load Chatbot**.
- If it works, push the Netlify folder (`netlify-landing/`) to another GitHub repo and deploy on Netlify (Build: `npm run build`, Publish: `dist`).

---

## Common pitfalls

| Issue | Fix |
|-------|-----|
| "OPENAI_API_KEY not found" | You forgot to add it in Streamlit Cloud **Settings → Secrets**. |
| Build fails on `sentence-transformers` | Add `packages.txt` with `libstdc++6` (already in Dockerfile; not needed for Streamlit Cloud wheels). |
| "I don't have that information" for everything | Documents not loaded. Verify `data/` files are in the repo and the app rebuilt. |
| Iframe blank on Netlify | Streamlit Cloud sets `X-Frame-Options: SAMEORIGIN` — the embed won't work. Instead, link the Netlify button to open the Streamlit URL in a new tab (`target="_blank"`). |

---

Embedding iframe: Streamlit Cloud allows embedding with `?embed=true` appended to the app URL.
