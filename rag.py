#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) core module for the College FAQ Chatbot.

This module handles:
1. Loading PDF and TXT files from the data/ directory
2. Chunking the text into manageable pieces
3. Creating vector embeddings
4. Storing in Chroma database for efficient retrieval
5. Retrieving relevant chunks for answering queries
6. Formatting answers with source attribution

Technical implementation details:
- Uses LangChain for document loading and processing
- sentence-transformers for text embeddings (all-MiniLM-L6-v2)
- Chroma for persistent vector database storage
- Streamlit for UI integration
- Supports both Tamil and English languages
"""
import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, urldefrag
import urllib.robotparser

import requests
from bs4 import BeautifulSoup

# LangChain imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Optional: Qdrant for self-hosted vector DB
try:
    from langchain_community.vectorstores import Qdrant as QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Optional: Groq LLM (free API) for answer generation
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Configuration
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    RETRIEVAL_TOP_K,
    SUPPORTED_EXTENSIONS
)

# Global variables
_embedding_model = None
_vector_store = None
_qdrant_client = None
_processing_progress = None

def get_embedding_model():
    """Initialize and return the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"Embedding model loaded: {EMBEDDING_MODEL}")
    return _embedding_model

def load_documents(data_dir: str = "data") -> List[Document]:
    """Load supported documents (PDF/TXT) from a directory.

    Supports recursive loading (e.g., data/web/*.txt).

    For website-ingested .txt files, if the first line starts with:
      SOURCE_URL: <url>
    then we treat the URL as the document source and remove the marker line.
    """
    documents: List[Document] = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    print(f"Loading documents from '{data_dir}' directory...")

    for root, _, files in os.walk(data_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(filename)[1].lower()
            rel_root = os.path.relpath(root, data_dir)
            default_source = filename if rel_root == "." else os.path.join(rel_root, filename)

            if ext == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        firstline = f.readline().strip()
                        rest = f.read()

                    detected_url = None
                    if firstline.startswith("SOURCE_URL:"):
                        detected_url = firstline.split(":", 1)[1].strip() or None
                        content = rest
                    else:
                        content = (firstline + "\n" + rest).strip() if rest else firstline

                    source_value = detected_url or default_source

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": source_value,
                                "url": detected_url,
                                "type": "text",
                            },
                        )
                    )
                    print(f"Loaded TXT: {default_source}")
                except Exception as e:
                    print(f"Error loading TXT file {default_source}: {str(e)}")
                    raise

            elif ext == ".pdf":
                try:
                    from pypdf import PdfReader

                    with open(file_path, "rb") as f:
                        reader = PdfReader(f)
                        for page_num, page in enumerate(reader.pages, 1):
                            page_content = page.extract_text() or ""
                            if page_content.strip():
                                documents.append(
                                    Document(
                                        page_content=page_content,
                                        metadata={
                                            "source": default_source,
                                            "page": page_num,
                                            "type": "pdf",
                                        },
                                    )
                                )
                    print(f"Loaded PDF: {default_source}")
                except Exception as e:
                    print(f"Error loading PDF file {default_source}: {str(e)}")
                    raise

            else:
                print(f"Skipping unsupported file type: {default_source}")

    print(f"Total documents loaded: {len(documents)}")
    return documents

def create_chunks(documents: List[Document], progress_callback=None) -> List[Document]:
    """
    Split documents into chunks for processing.

    Args:
        documents: List of Document objects
        progress_callback: Optional callback for progress reporting

    Returns:
        List of chunked Document objects with metadata
    """
    print("Creating chunks from documents...")

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""]
    )

    # Split documents
    chunks = text_splitter.split_documents(documents)

    # Add unique IDs and enhance metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["total_chunks"] = len(chunks)
        # Preserve source from parent document
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = chunk.metadata.get("source", "unknown")

    print(f"Created {len(chunks)} chunks")
    return chunks

def initialize_vector_store(chunks: List[Document], progress_callback=None) -> Chroma:
    """
    Initialize or load the vector store with document chunks.

    Args:
        chunks: List of Document chunks
        progress_callback: Optional callback for progress reporting

    Returns:
        Initialized Chroma vector store
    """
    print(f"Initializing vector store at '{CHROMA_PERSIST_DIR}'...")

    # Create embedding function
    embeddings = get_embedding_model()

    # Check if vector store already exists and has data
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Loading existing vector store from '{CHROMA_PERSIST_DIR}'...")
        try:
            # Try to load existing vector store
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )

            # Verify we can retrieve from it
            test_query = "test"
            results = vector_store.similarity_search(test_query, k=1)
            if results:
                print(f"Successfully loaded existing vector store with {len(results)} retrievable entries")
                return vector_store
            else:
                print("Existing vector store appears empty, rebuilding...")
                raise Exception("Vector store exists but is empty")
        except Exception as e:
            print(f"Could not load existing vector store: {str(e)}")
            print("Rebuilding vector store...")

    # Create new vector store
    print("Creating new vector store...")

    # Prepare documents for Chroma
    chromadb_documents = []
    for chunk in chunks:
        chromadb_documents.append({
            "id": chunk.metadata.get("chunk_id", str(hash(chunk.page_content))),
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })

    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chromadb_documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    print(f"Vector store created and persisted at '{CHROMA_PERSIST_DIR}'")
    print(f"Total vectors stored: {len(chromadb_documents)}")

    return vector_store

def get_vector_store(chunks: List[Document], progress_callback=None) -> Chroma:
    """
    Get or create vector store for document retrieval.

    Args:
        chunks: List of Document chunks
        progress_callback: Optional callback for progress reporting

    Returns:
        Chroma vector store
    """
    global _vector_store

    if _vector_store is None:
        _vector_store = initialize_vector_store(chunks, progress_callback)

    return _vector_store


def initialize_qdrant_vector_store(chunks: List[Document], progress_callback=None,
                                 qdrant_url: str = None, qdrant_api_key: str = None,
                                 collection_name: str = "college_faq") -> Any:
    """
    Initialize or load the Qdrant vector store with document chunks.

    Args:
        chunks: List of Document chunks
        progress_callback: Optional callback for progress reporting
        qdrant_url: Qdrant server URL (defaults to env QDRANT_URL or http://localhost:6333)
        qdrant_api_key: Qdrant API key (defaults to env QDRANT_API_KEY)
        collection_name: Name of the Qdrant collection

    Returns:
        Qdrant vector store
    """
    if not QDRANT_AVAILABLE:
        raise ImportError("Qdrant dependencies not installed. Install qdrant-client and langchain-community")

    print(f"Initializing Qdrant vector store...")

    # Get configuration from environment or parameters
    import os
    if qdrant_url is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    if qdrant_api_key is None:
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

    # Initialize Qdrant client
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)

    # Check if collection exists, if not create it
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            print(f"Creating Qdrant collection: {collection_name}")
            # Create collection with proper vector configuration
            # all-MiniLM-L6-v2 produces 384-dimensional vectors
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection '{collection_name}' with 384-dim vectors")
        else:
            print(f"Using existing Qdrant collection: {collection_name}")
    except Exception as e:
        print(f"Error checking/creating Qdrant collection: {str(e)}")
        # Try to create anyway
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection '{collection_name}'")
        except Exception as create_error:
            raise Exception(f"Failed to create Qdrant collection: {str(create_error)}")

    # Create Qdrant vector store
    print("Creating Qdrant vector store from documents...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
    )

    # Add documents to the store
    if chunks:
        print(f"Adding {len(chunks)} chunks to Qdrant...")
        vector_store.add_documents(chunks)
        print("Successfully added documents to Qdrant")

    return vector_store


def get_qdrant_vector_store(chunks: List[Document], progress_callback=None,
                           qdrant_url: str = None, qdrant_api_key: str = None,
                           collection_name: str = "college_faq") -> Any:
    """
    Get or create Qdrant vector store for document retrieval.

    Args:
        chunks: List of Document chunks
        progress_callback: Optional callback for progress reporting
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key
        collection_name: Name of the Qdrant collection

    Returns:
        Qdrant vector store
    """
    global _qdrant_client

    if _qdrant_client is None:
        _qdrant_client = initialize_qdrant_vector_store(
            chunks, progress_callback, qdrant_url, qdrant_api_key, collection_name
        )

    return _qdrant_client

def retrieve_relevant_chunks(vector_store: Chroma, query: str, top_k: int = None) -> List[Document]:
    """
    Retrieve most relevant chunks for a given query.

    Args:
        vector_store: Chroma vector store
        query: Query text
        top_k: Number of results to retrieve (defaults to config)

    Returns:
        List of retrieved Document objects
    """
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    print(f"Retrieving top {top_k} relevant chunks for query: '{query[:100]}...'")

    # Perform similarity search
    results = vector_store.similarity_search(query, k=top_k)

    print(f"Retrieved {len(results)} chunks")

    return results

def format_sources(retrieved_chunks: List[Document]) -> str:
    """
    Format source information for display.

    Args:
        retrieved_chunks: List of Document objects with metadata

    Returns:
        Formatted sources string
    """
    if not retrieved_chunks:
        return ""

    sources = []

    for chunk in retrieved_chunks:
        metadata = chunk.metadata
        source = metadata.get("source", "Unknown")
        chunk_id = metadata.get("chunk_id", "N/A")

        # Handle page numbers
        page_info = ""
        if "page" in metadata:
            page_info = f", Page {metadata['page']}"

        # Create snippet (first 100 chars)
        snippet = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content

        # Create source entry
        source_entry = f"{source} (Chunk {chunk_id}{page_info}): {snippet}"
        sources.append(source_entry)

    return "\n".join(sources)

def process_query(
    query: str,
    chunks: List[Document],
    vector_store: Chroma,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Process a user query through the RAG pipeline.

    Args:
        query: User's question
        chunks: Document chunks
        vector_store: Chroma vector store
        language: Language ("en" or "ta")

    Returns:
        Dictionary with answer, sources, and metadata
    """
    start_time = time.time()

    # Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(vector_store, query)

    # Check if the answer is in the retrieved chunks
    answer = None
    sources_text = format_sources(retrieved_chunks)

    if retrieved_chunks:
        # Simple relevance check - if chunks contain key terms from query
        query_terms = query.lower().split()

        for chunk in retrieved_chunks:
            chunk_text = chunk.page_content.lower()

            # Check for meaningful overlap (at least one term from query appears in chunk)
            overlap_count = sum(1 for term in query_terms if len(term) > 3 and term in chunk_text)

            if overlap_count > 0:
                answer = chunk.page_content
                break

    # If no answer found in chunks, try using LLM (Groq free API) if available
    if not answer and GROQ_AVAILABLE:
        try:
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key:
                llm = ChatGroq(
                    model=os.getenv("DEFAULT_LLM_MODEL", "llama-3.1-8b-instant"),
                    api_key=api_key,
                    temperature=0.2,
                )
                context = "\n\n".join([c.page_content[:800] for c in retrieved_chunks[:2]])
                prompt = f"You must answer ONLY from the documentation below. If the answer is not present, reply exactly: I don't have that information. Please contact the college office.\n\nDOCUMENT EXCERPTS:\n{context}\n\nQUESTION: {query}\nANSWER:"
                answer = llm.predict(prompt)
        except Exception as e:
            print(f"LLM generation failed: {e}")

    # Final fallback
    if not answer:
        answer = "I don't have that information. Please contact the college office."

    # Calculate response time
    response_time = time.time() - start_time

    return {
        "answer": answer,
        "sources": sources_text,
        "retrieved_chunks": len(retrieved_chunks),
        "response_time": response_time,
        "language": language,
        "query_terms": query_terms if 'query_terms' in locals() else []
    }

def load_and_index_documents(progress_callback=None) -> tuple:
    """
    Load documents, create chunks, and initialize vector store.

    Args:
        progress_callback: Optional callback for progress reporting

    Returns:
        Tuple of (documents, chunks, vector_store)
    """
    print("Starting document loading and indexing process...")

    # Load documents
    documents = load_documents()

    if progress_callback:
        progress_callback(0.3, "Documents loaded, creating chunks...")

    # Create chunks
    chunks = create_chunks(documents, progress_callback)

    if progress_callback:
        progress_callback(0.6, "Chunks created, building vector store...")

    # Initialize vector store
    vector_store = get_vector_store(chunks, progress_callback)

    if progress_callback:
        progress_callback(1.0, "Document indexing complete!")

    print("Document indexing completed successfully!")
    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector store entries: {vector_store._collection.count()}")

    return documents, chunks, vector_store

def clear_vector_store():
    """Clear the existing vector store."""
    global _vector_store

    if _vector_store:
        print("Clearing vector store...")
        _vector_store._collection.delete()
        _vector_store = None

    # Remove persist directory
    import shutil
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("Vector store directory removed")

    print("Vector store cleared successfully")

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


def get_document_stats(documents: List[Document], chunks: List[Document], vector_store: Chroma) -> Dict[str, Any]:
    """
    Get statistics about the indexed documents.

    Args:
        documents: List of original documents
        chunks: List of chunks
        vector_store: Chroma vector store

    Returns:
        Dictionary with statistics
    """
    # Count by type
    text_docs = sum(1 for d in documents if d.metadata.get("type") == "text")
    pdf_docs = sum(1 for d in documents if d.metadata.get("type") == "pdf")

    # Count pages
    total_pages = sum(d.metadata.get("page", 1) for d in documents if d.metadata.get("type") == "pdf")

    # Unique sources
    unique_sources = len(set(d.metadata.get("source", "unknown") for d in documents))

    # Vector store count
    vector_count = vector_store._collection.count() if vector_store else 0

    return {
        "total_documents": len(documents),
        "text_documents": text_docs,
        "pdf_documents": pdf_docs,
        "total_chunks": len(chunks),
        "estimated_pages": total_pages,
        "unique_sources": unique_sources,
        "vector_store_entries": vector_count,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "embedding_model": EMBEDDING_MODEL
    }
