#!/usr/bin/env python3
"""RAG core for the College FAQ Chatbot (Groq LLM + local MiniLM embeddings + Chroma/Qdrant)."""
import os
import time
import hashlib
import shutil
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, urldefrag
import urllib.robotparser

import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from langchain_community.vectorstores import Qdrant as QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K,
)

FALLBACK_MESSAGE = "I don't have that information. Please contact the college office."
GROQ_MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"

_embedding_model = None
_vector_store = None
_qdrant_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedding_model


def load_documents(data_dir: str = "data") -> List[Document]:
    """Load .txt/.pdf recursively. Files starting with 'SOURCE_URL: <url>' use that URL as source."""
    documents: List[Document] = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    for root, _, files in os.walk(data_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            rel_root = os.path.relpath(root, data_dir)
            default_source = filename if rel_root == "." else os.path.join(rel_root, filename)

            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    firstline = f.readline().strip()
                    rest = f.read()
                detected_url = ""
                if firstline.startswith("SOURCE_URL:"):
                    detected_url = firstline.split(":", 1)[1].strip()
                    content = rest
                else:
                    content = (firstline + "\n" + rest).strip() if rest else firstline
                documents.append(Document(
                    page_content=content,
                    metadata={"source": detected_url or default_source, "url": detected_url, "type": "text"},
                ))
            elif ext == ".pdf":
                from pypdf import PdfReader
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text() or ""
                        if text.strip():
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": default_source, "page": page_num, "type": "pdf"},
                            ))
    print(f"Total documents loaded: {len(documents)}")
    return documents


def create_chunks(documents: List[Document], progress_callback=None) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["total_chunks"] = len(chunks)
        chunk.metadata.setdefault("source", "unknown")
    print(f"Created {len(chunks)} chunks")
    return chunks


def initialize_vector_store(chunks: List[Document], progress_callback=None) -> Chroma:
    embeddings = get_embedding_model()
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            vs = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
            if vs.similarity_search("test", k=1):
                return vs
        except Exception as e:
            print(f"Could not load existing store, rebuilding: {e}")
            shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_vector_store(chunks: List[Document], progress_callback=None) -> Chroma:
    global _vector_store
    if _vector_store is None:
        _vector_store = initialize_vector_store(chunks, progress_callback)
    return _vector_store


def initialize_qdrant_vector_store(chunks, progress_callback=None, qdrant_url=None,
                                   qdrant_api_key=None, collection_name="college_faq"):
    if not QDRANT_AVAILABLE:
        raise ImportError("Install qdrant-client and langchain-community for Qdrant.")
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", "")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key) if qdrant_api_key else QdrantClient(url=qdrant_url)

    names = [c.name for c in client.get_collections().collections]
    fresh = collection_name not in names
    if fresh:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # MiniLM = 384 dims
        )
    store = QdrantVectorStore(client=client, collection_name=collection_name,
                              embeddings=get_embedding_model())
    if fresh and chunks:
        store.add_documents(chunks)
    return store


def get_qdrant_vector_store(chunks, progress_callback=None, qdrant_url=None,
                            qdrant_api_key=None, collection_name="college_faq"):
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = initialize_qdrant_vector_store(
            chunks, progress_callback, qdrant_url, qdrant_api_key, collection_name)
    return _qdrant_client


def retrieve_relevant_chunks(vector_store, query: str, top_k: int = None) -> List[Document]:
    return vector_store.similarity_search(query, k=top_k or RETRIEVAL_TOP_K)


def format_sources(retrieved_chunks: List[Document]) -> str:
    out = []
    for chunk in retrieved_chunks:
        m = chunk.metadata
        page = f", Page {m['page']}" if "page" in m else ""
        text = chunk.page_content
        snippet = text[:100] + "..." if len(text) > 100 else text
        out.append(f"{m.get('source', 'Unknown')} (Chunk {m.get('chunk_id', 'N/A')}{page}): {snippet}")
    return "\n".join(out)


def _get_llm():
    if not GROQ_AVAILABLE:
        raise RuntimeError("langchain-groq is not installed.")
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it in Streamlit Cloud -> Settings -> Secrets.")
    return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0)


def process_query(query: str, chunks: List[Document], vector_store, language: str = "en") -> Dict[str, Any]:
    """Retrieve top-K chunks, ask the LLM to answer ONLY from them, else return exact fallback."""
    start = time.time()
    retrieved = retrieve_relevant_chunks(vector_store, query)
    sources_text = format_sources(retrieved)

    answer = FALLBACK_MESSAGE
    if retrieved:
        context = "\n\n---\n\n".join(c.page_content for c in retrieved)
        lang_rule = "Reply in Tamil." if language == "ta" else "Reply in English."
        system = (
            "You are the FAQ assistant of a college. Answer ONLY using the CONTEXT below. "
            "Do not use outside knowledge and do not guess. "
            f"If the answer is not clearly present in the CONTEXT, reply with exactly: {FALLBACK_MESSAGE} "
            "(in English, nothing added). Otherwise be concise and accurate, keep numbers/names exact. "
            + lang_rule
        )
        user = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
        llm = _get_llm()
        resp = llm.invoke([("system", system), ("human", user)])
        text = (resp.content or "").strip()
        answer = FALLBACK_MESSAGE if (not text or "I don't have that information" in text) else text

    return {
        "answer": answer,
        "sources": sources_text,
        "retrieved_chunks": len(retrieved),
        "response_time": time.time() - start,
        "language": language,
    }


def load_and_index_documents(progress_callback=None) -> tuple:
    documents = load_documents()
    chunks = create_chunks(documents, progress_callback)
    vector_store = get_vector_store(chunks, progress_callback)
    return documents, chunks, vector_store


def clear_vector_store():
    global _vector_store
    _vector_store = None
    shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)


def fetch_and_ingest_site(
    start_url: str,
    output_dir: str = "data/web",
    max_pages: int = 20,
    keywords: Optional[List[str]] = None,
    respect_robots: bool = True,
    timeout_s: int = 20,
    user_agent: str = "CollegeFAQBot/1.0"
) -> Dict[str, Any]:
    """Fetch a limited set of dsuniversity.ac.in pages and save cleaned text as .txt.

    Each saved .txt file starts with:
      SOURCE_URL: <url>
    so the loader can attribute the correct source.
    """

    if keywords is None:
        keywords = [
            "admission", "eligibility", "fee", "fees", "scholarship", "hostel",
            "placements", "placement", "course", "courses", "department",
            "contact", "about"
        ]

    os.makedirs(output_dir, exist_ok=True)

    parsed_start = urlparse(start_url)
    if not parsed_start.scheme:
        start_url = "https://" + start_url
        parsed_start = urlparse(start_url)

    base_netloc = parsed_start.netloc.lower()

    robots = None
    if respect_robots:
        try:
            robots_url = urljoin(start_url, "/robots.txt")
            robots = urllib.robotparser.RobotFileParser()
            robots.set_url(robots_url)
            robots.read()
        except Exception:
            robots = None

    def _normalize(u: str) -> str:
        u = urldefrag(u)[0]
        # Remove trailing slash (except for root)
        if u.endswith("/") and len(u) > len("https://" + base_netloc + "/"):
            u = u.rstrip("/")
        return u

    def _same_domain(u: str) -> bool:
        try:
            return urlparse(u).netloc.lower() == base_netloc
        except Exception:
            return False

    def _looks_relevant(u: str) -> bool:
        ul = u.lower()
        return any(k in ul for k in keywords)

    def _can_fetch(u: str) -> bool:
        if not respect_robots or robots is None:
            return True
        try:
            return robots.can_fetch(user_agent, u)
        except Exception:
            return True

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    visited = set()
    saved_urls: List[str] = []
    queue = [start_url]

    def _extract_text(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Drop very noisy blocks if present
        for tag in soup.select("nav, footer, header"):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        text = " ".join(text.split())
        return text

    while queue and len(saved_urls) < max_pages:
        current = queue.pop(0)
        current = _normalize(current)

        if current in visited:
            continue
        visited.add(current)

        if not _same_domain(current):
            continue
        if not _can_fetch(current):
            continue

        if len(saved_urls) > 0 and not _looks_relevant(current):
            # After the initial start URL, keep crawl focused on relevant pages.
            continue

        try:
            r = session.get(current, timeout=timeout_s)
            if r.status_code != 200 or not r.text:
                continue

            cleaned_text = _extract_text(r.text)
            if not cleaned_text or len(cleaned_text) < 200:
                continue

            url_hash = hashlib.sha256(current.encode("utf-8")).hexdigest()[:12]
            filename = f"web_{url_hash}.txt"
            path = os.path.join(output_dir, filename)

            with open(path, "w", encoding="utf-8") as f:
                f.write(f"SOURCE_URL: {current}\n")
                f.write(cleaned_text)

            saved_urls.append(current)

            # Discover next links (only a small budget)
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.select("a[href]"):
                href = a.get("href")
                if not href:
                    continue
                nxt = urljoin(current, href)
                nxt = _normalize(nxt)
                if nxt in visited:
                    continue
                if not _same_domain(nxt):
                    continue
                # Keep crawl focused
                if _looks_relevant(nxt):
                    queue.append(nxt)

            # Hard cap duplicates
            queue = queue[:max_pages * 3]

        except Exception:
            continue

    return {
        "start_url": start_url,
        "output_dir": output_dir,
        "respect_robots": respect_robots,
        "requested_max_pages": max_pages,
        "saved_pages": len(saved_urls),
        "saved_urls": saved_urls,
    }


def get_document_stats(documents: List[Document], chunks: List[Document], vector_store) -> Dict[str, Any]:
    try:
        vector_count = vector_store._collection.count() if vector_store else 0
    except Exception:
        vector_count = len(chunks)
    return {
        "total_documents": len(documents),
        "text_documents": sum(1 for d in documents if d.metadata.get("type") == "text"),
        "pdf_documents": sum(1 for d in documents if d.metadata.get("type") == "pdf"),
        "total_chunks": len(chunks),
        "unique_sources": len(set(d.metadata.get("source", "unknown") for d in documents)),
        "vector_store_entries": vector_count,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "embedding_model": EMBEDDING_MODEL,
    }
