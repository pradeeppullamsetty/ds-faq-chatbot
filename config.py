"""
Configuration settings for the AI-Powered College FAQ Chatbot.
All rebrandable settings are in one place.
"""
import os

# Default college name
COLLEGE_NAME = os.getenv("COLLEGE_NAME", "Sample College")

# Tagline for the chatbot
TAGLINE = os.getenv("TAGLINE", "Your AI-Powered College Assistant")

# Primary color theme (hex without #)
PRIMARY_COLOR = os.getenv("PRIMARY_COLOR", "1e293b")  # Deep navy

# Accent color theme (hex without #)
ACCENT_COLOR = os.getenv("ACCENT_COLOR", "38bdf8")  # Cyan

# Logo path (relative to outputs folder)
LOGO_PATH = os.getenv("LOGO_PATH", "assets/logo.svg")

# OpenAI API key - read from environment / Streamlit secrets (never hardcode)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embedding model name
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chroma database path
CHROMA_PERSIST_DIR = "chroma_db"

# Chunking configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval configuration
RETRIEVAL_TOP_K = 4

# Supported file types
SUPPORTED_EXTENSIONS = [".pdf", ".txt"]

# LLM configuration
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Page for metadata
DEFAULT_PAGE = 1