"""
config.py – Central configuration for the Semantic Search Module.

This module holds all default settings and constants used across the application.
Nothing here is hard-coded to a specific dataset or model; every value serves as
a sensible default that the user can override through the GUI.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "Vector_Store")

# Chroma persist directory (inside Vector_Store/)
CHROMA_PERSIST_DIR = os.path.join(VECTOR_STORE_DIR, "chroma_db")
# FAISS local save path
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index")

# ── Default embedding model (user can change at runtime) ──────────────────────
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Suggested HuggingFace embedding models (shown in dropdown) ────────────────
SUGGESTED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]

# ── Vector database options ───────────────────────────────────────────────────
VECTOR_DB_OPTIONS = ["FAISS", "Chroma"]

# ── Text-splitting defaults ──────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# ── Retrieval defaults ────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# ── Supported file extensions ─────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf", ".txt"]

