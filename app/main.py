"""
main.py – Core logic for the Semantic Search Module (Memory System).

This module contains all backend functions that power the AI Research Assistant's
memory.  LangChain is used throughout as the orchestration layer:

  • Document Loading  – LangChain's PyPDFLoader / TextLoader read raw files.
  • Text Splitting    – RecursiveCharacterTextSplitter chunks documents so that
                        each chunk fits within the embedding model's context window.
  • Embedding Gen.    – HuggingFaceEmbeddings (via langchain-huggingface) converts
                        each chunk into a dense vector representation.
  • Vector Storage    – FAISS or Chroma (via langchain-community) indexes vectors
                        for fast approximate nearest-neighbour retrieval.
  • Semantic Search   – similarity_search_with_score retrieves the top-k most
                        relevant chunks for a user query, ranked by cosine distance.

Together these components form the *Memory System* of the assistant: documents are
ingested once, and arbitrary natural-language queries surface the most relevant
passages without keyword matching.
"""

import os
import shutil
import tempfile
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    CHROMA_PERSIST_DIR,
    FAISS_INDEX_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DOCUMENT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_docs(file_paths: List[str]) -> List[Document]:
    """
    Load documents from a list of file paths using LangChain loaders.

    LangChain's loader abstraction lets us support multiple file types with a
    uniform Document interface (page_content + metadata).

    Parameters
    ----------
    file_paths : list[str]
        Absolute paths to .pdf or .txt files uploaded by the user.

    Returns
    -------
    list[Document]
        A flat list of LangChain Document objects (one per page for PDFs,
        one per file for text files).
    """
    documents: List[Document] = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            # PyPDFLoader splits a PDF into one Document per page and embeds
            # page number + source filename in the metadata automatically.
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif ext == ".txt":
            # TextLoader reads the entire file as a single Document.
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

        else:
            # Gracefully skip unsupported formats.
            print(f"[WARN] Skipping unsupported file type: {path}")

    return documents


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TEXT SPLITTING / CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split loaded documents into smaller, overlapping chunks.

    LangChain's RecursiveCharacterTextSplitter tries multiple separators
    (paragraphs → sentences → words) to keep chunks semantically coherent.

    Parameters
    ----------
    documents : list[Document]
        Raw documents from load_docs().
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Overlap between consecutive chunks to preserve context at boundaries.

    Returns
    -------
    list[Document]
        Chunked documents, each retaining the original metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,       # record chunk position in source
    )
    return splitter.split_documents(documents)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  EMBEDDING MODEL INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """
    Instantiate a HuggingFace embedding model via LangChain.

    HuggingFaceEmbeddings wraps sentence-transformers so that every string
    is encoded into a fixed-length dense vector suitable for similarity search.

    Parameters
    ----------
    model_name : str
        A HuggingFace Hub model identifier (e.g. "sentence-transformers/all-MiniLM-L6-v2").

    Returns
    -------
    HuggingFaceEmbeddings
        A LangChain-compatible embeddings object.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  VECTOR STORE CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    vector_db: str = "FAISS",
) -> object:
    """
    Build a vector index from document chunks using LangChain's vector-store
    integrations.  This is the *write path* of the Memory System.

    • FAISS  – Facebook AI Similarity Search; in-memory by default, optionally
               saved to disk for persistence.
    • Chroma – Lightweight embedded vector DB; persists automatically to a
               local directory so the index survives restarts.

    Parameters
    ----------
    chunks : list[Document]
        Chunked documents from split_documents().
    embeddings : HuggingFaceEmbeddings
        The embedding model to encode chunks.
    vector_db : str
        Either "FAISS" or "Chroma".

    Returns
    -------
    VectorStore
        A LangChain vector-store object ready for similarity queries.
    """
    if vector_db == "FAISS":
        # FAISS stores the index in memory.  We also persist it to disk so
        # users can reload later without re-embedding.
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        return vector_store

    elif vector_db == "Chroma":
        # Clear any previous Chroma collection to avoid dimension-mismatch
        # errors when the user switches embedding models.
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        return vector_store

    else:
        raise ValueError(f"Unsupported vector database: {vector_db}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SEMANTIC RETRIEVAL (SIMILARITY SEARCH)
# ═══════════════════════════════════════════════════════════════════════════════

def search(
    vector_store,
    query: str,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Perform a semantic similarity search – the *read path* of the Memory System.

    LangChain's `similarity_search_with_score` encodes the query with the same
    embedding model used during indexing, then returns the closest chunks along
    with their distance/similarity scores.

    Parameters
    ----------
    vector_store : VectorStore
        The built FAISS or Chroma index.
    query : str
        The user's natural-language query.
    top_k : int
        Number of most-relevant results to return.

    Returns
    -------
    list[tuple[Document, float]]
        Each tuple is (Document, score).  Lower distance ≈ higher relevance
        for FAISS (L2); for Chroma the score semantics depend on the metric.
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CONVENIENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_uploaded_files(uploaded_files) -> List[str]:
    """
    Persist Streamlit UploadedFile objects to a temporary directory so that
    LangChain file-based loaders can access them by path.

    Parameters
    ----------
    uploaded_files : list[UploadedFile]
        Files from Streamlit's file_uploader widget.

    Returns
    -------
    list[str]
        Absolute paths to the saved temporary files.
    """
    tmp_dir = tempfile.mkdtemp()
    paths: List[str] = []

    for uploaded in uploaded_files:
        dest = os.path.join(tmp_dir, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        paths.append(dest)

    return paths


def get_dataset_stats(uploaded_files) -> dict:
    """
    Compute basic statistics for the uploaded dataset.

    Parameters
    ----------
    uploaded_files : list[UploadedFile]
        Streamlit uploaded file objects.

    Returns
    -------
    dict
        Keys: 'num_files', 'total_size_mb', 'file_names'.
    """
    total_bytes = sum(f.size for f in uploaded_files)
    return {
        "num_files": len(uploaded_files),
        "total_size_mb": round(total_bytes / (1024 * 1024), 4),
        "file_names": [f.name for f in uploaded_files],
    }

