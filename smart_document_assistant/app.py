import io
import re
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Document Assistant", page_icon="📄", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}
.hero{background:linear-gradient(125deg,#3020c9,#6247ef);color:white;padding:2rem 2.2rem;border-radius:22px;margin-bottom:1.3rem}
.hero h1{font-size:2.35rem;margin:0 0 .45rem 0;color:white}
.hero p{font-size:1.05rem;opacity:.92;margin:0}
div[data-testid="stMetric"]{border:1px solid #e9e7f8;border-radius:14px;padding:14px}
.stButton button[kind="primary"]{background:#3820d4;border-color:#3820d4}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📄 Smart Document Assistant</h1><p>Upload documents. Ask questions. Find relevant answers with source passages — no API keys required.</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.markdown("**Privacy-first retrieval**")
    st.caption("No external AI API is called.")
    st.markdown("---")
    st.write("Supported files: PDF · DOCX · TXT")
    st.caption("Key-free edition • Streamlit")

def extract_text(upload):
    raw, name = upload.getvalue(), upload.name.lower()
    try:
        if name.endswith(".pdf"):
            return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
        if name.endswith(".docx"):
            return "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
        if name.endswith(".txt"):
            return raw.decode("utf-8", errors="ignore")
    except Exception as exc:
        return f"__ERROR__{exc}"
    return ""

def split_chunks(text, size=950, overlap=140):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if text[start:end].strip():
            chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks

def build_index(documents):
    chunks, metadata = [], []
    for name, content in documents:
        for idx, chunk in enumerate(split_chunks(content)):
            chunks.append(chunk)
            metadata.append({"file": name, "chunk": idx + 1})
    if not chunks:
        return None, [], []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=45000)
    matrix = vectorizer.fit_transform(chunks)
    return (vectorizer, matrix), chunks, metadata

if "documents" not in st.session_state:
    st.session_state.documents = []
if "history" not in st.session_state:
    st.session_state.history = []

left, right = st.columns([1.2, 1], gap="large")
with left:
    st.subheader("Your documents")
    uploads = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    a, b = st.columns(2)
    with a:
        if st.button("Process documents", type="primary", use_container_width=True):
            docs = []
            if uploads:
                for f in uploads:
                    value = extract_text(f)
                    if value.startswith("__ERROR__"):
                        st.error(f"Could not read {f.name}: {value[9:]}")
                    elif value.strip():
                        docs.append((f.name, value))
            if docs:
                st.session_state.documents, st.session_state.history = docs, []
                st.success(f"Processed {len(docs)} document(s).")
            else:
                st.warning("Upload at least one readable document first.")
    with b:
        if st.button("Clear workspace", use_container_width=True):
            st.session_state.documents, st.session_state.history = [], []
            st.rerun()
    st.markdown("#### Try a sample document")
    st.caption("Load a fictional FAQ to test the assistant.")
    if st.button("Load sample college FAQ", use_container_width=True):
        sample = Path(__file__).parent / "data" / "sample_college_faq.txt"
        st.session_state.documents = [(sample.name, sample.read_text(encoding="utf-8"))]
        st.session_state.history = []
        st.success("Sample FAQ loaded.")
    if st.session_state.documents:
        st.markdown("#### Loaded files")
        for name, content in st.session_state.documents:
            st.write(f"📄 **{name}** · {len(content.split()):,} words")

with right:
    st.subheader("Ask a question")
    st.caption("Key-free retrieval finds passages relevant to your question.")
    question = st.text_area("Your question", placeholder="e.g. What are the library opening hours?", height=100)
    top_k = st.slider("Number of source passages", 1, 5, 3)
    if st.button("Find answer", type="primary", use_container_width=True):
        if not st.session_state.documents:
            st.warning("Upload and process documents, or load the sample FAQ first.")
        elif not question.strip():
            st.warning("Type a question first.")
        else:
            index, chunks, metadata = build_index(st.session_state.documents)
            if not index:
                st.error("No readable text found in the selected documents.")
            else:
                vectorizer, matrix = index
                scores = cosine_similarity(vectorizer.transform([question]), matrix).ravel()
                best = scores.argsort()[::-1][:top_k]
                results = [(float(scores[i]), chunks[i], metadata[i]) for i in best if scores[i] > 0]
                if not results:
                    st.info("No strong text match found. Try different wording or check the document.")
                else:
                    st.markdown("### Most relevant passages")
                    st.caption("Extractive search: check the passages below to verify details.")
                    for rank, (score, chunk, meta) in enumerate(results, 1):
                        with st.container(border=True):
                            st.markdown(f"**Source {rank}: {meta['file']}** · Relevance {score:.0%}")
                            st.write(chunk)
                    st.session_state.history.insert(0, {"question": question.strip(), "sources": len(results)})

if st.session_state.history:
    st.markdown("---")
    st.subheader("Recent questions")
    for item in st.session_state.history[:5]:
        st.markdown(f"**Q:** {item['question']}  \n<small>Found {item['sources']} relevant passage(s)</small>", unsafe_allow_html=True)
st.markdown("---")
st.caption("Smart Document Assistant · No API keys · Verify important answers against the original document.")
