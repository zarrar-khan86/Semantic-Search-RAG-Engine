"""
gui.py – Streamlit front-end for the Semantic Search Module.

This file wires the Streamlit UI to the backend functions defined in main.py.
The layout follows the assignment specification:
    • Sidebar  → Configuration (embedding model, vector DB, chunk settings, ingest)
    • Main area → File upload, dataset stats, query input, and search results

Run with:
    cd app/
    streamlit run gui.py
"""

import sys
import os
import streamlit as st

# Ensure the app/ directory is on the Python path so that sibling imports work
# regardless of where Streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_EMBEDDING_MODEL,
    SUGGESTED_MODELS,
    VECTOR_DB_OPTIONS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    SUPPORTED_EXTENSIONS,
)
from main import (
    load_docs,
    split_documents,
    get_embedding_model,
    create_vector_store,
    search,
    save_uploaded_files,
    get_dataset_stats,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Search Module",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Semantic Search Module")
st.caption("AI Research Assistant – Memory System (Phase 1)")

# ──────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────────────────────────────────────
# We store the vector index and metadata in Streamlit session state so they
# survive across re-runs (every widget interaction triggers a re-run).
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "current_db" not in st.session_state:
    st.session_state.current_db = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR – Task 2: Embedding & Vector Store Configuration
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Embedding model selection ─────────────────────────────────────────────
    st.subheader("Embedding Model")
    model_choice = st.selectbox(
        "Select or type a HuggingFace model",
        options=SUGGESTED_MODELS,
        index=0,
        help="Choose a sentence-transformers model from HuggingFace Hub.",
    )
    custom_model = st.text_input(
        "Or enter a custom model name",
        value="",
        placeholder="e.g. sentence-transformers/all-MiniLM-L6-v2",
    )
    embedding_model_name = custom_model.strip() if custom_model.strip() else model_choice

    # ── Vector database selection ─────────────────────────────────────────────
    st.subheader("Vector Database")
    vector_db_choice = st.radio(
        "Select vector store",
        options=VECTOR_DB_OPTIONS,
        index=0,
        help="FAISS: fast in-memory index. Chroma: persistent embedded DB.",
    )

    # ── Chunking parameters ──────────────────────────────────────────────────
    st.subheader("Chunking Settings")
    chunk_size = st.number_input(
        "Chunk size (characters)",
        min_value=100,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
    )
    chunk_overlap = st.number_input(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=500,
        value=DEFAULT_CHUNK_OVERLAP,
        step=10,
    )

    st.markdown("---")

    # ── Process / Ingest button ──────────────────────────────────────────────
    process_btn = st.button("🚀 Process / Ingest", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA – Task 1: GUI Data Selection
# ══════════════════════════════════════════════════════════════════════════════
st.header("📂 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload .txt or .pdf files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    help="Upload at least 10–15 text/PDF documents for meaningful retrieval.",
)

# ── Dataset statistics ───────────────────────────────────────────────────────
if uploaded_files:
    stats = get_dataset_stats(uploaded_files)
    col1, col2 = st.columns(2)
    col1.metric("Total Documents", stats["num_files"])
    col2.metric("Total Size", f"{stats['total_size_mb']} MB")

    with st.expander("📄 Uploaded Files", expanded=False):
        for name in stats["file_names"]:
            st.write(f"- {name}")
else:
    st.info("👆 Please upload documents to get started.")

# ══════════════════════════════════════════════════════════════════════════════
#  PROCESSING PIPELINE  (triggered by sidebar button)
# ══════════════════════════════════════════════════════════════════════════════
if process_btn:
    if not uploaded_files:
        st.error("❌ No files uploaded. Please upload documents first.")
    else:
        # Detect if the user changed the model or DB since last ingestion.
        config_changed = (
            st.session_state.current_model != embedding_model_name
            or st.session_state.current_db != vector_db_choice
        )

        with st.status("⏳ Processing documents...", expanded=True) as status:
            # Step 1 – Save uploaded files to temp location
            st.write("Saving uploaded files…")
            file_paths = save_uploaded_files(uploaded_files)

            # Step 2 – Load documents with LangChain loaders
            st.write("Loading documents…")
            documents = load_docs(file_paths)
            st.write(f"  ✔ Loaded **{len(documents)}** document pages/sections.")

            # Step 3 – Split into chunks
            st.write("Splitting into chunks…")
            chunks = split_documents(documents, chunk_size, chunk_overlap)
            st.session_state.chunks = chunks
            st.write(f"  ✔ Created **{len(chunks)}** chunks "
                      f"(size={chunk_size}, overlap={chunk_overlap}).")

            # Step 4 – Load embedding model
            st.write(f"Loading embedding model: `{embedding_model_name}`…")
            embeddings = get_embedding_model(embedding_model_name)
            st.write("  ✔ Embedding model ready.")

            # Step 5 – Build vector index
            st.write(f"Building **{vector_db_choice}** vector index…")
            vector_store = create_vector_store(chunks, embeddings, vector_db_choice)
            st.session_state.vector_store = vector_store
            st.session_state.index_ready = True
            st.session_state.current_db = vector_db_choice
            st.session_state.current_model = embedding_model_name
            st.write("  ✔ Vector index built and stored.")

            status.update(label="✅ Ingestion complete!", state="complete")

        st.success(
            f"Index ready → **{len(chunks)}** chunks indexed in **{vector_db_choice}** "
            f"using **{embedding_model_name}**."
        )

# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH – Task 3: Semantic Retrieval
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🔎 Semantic Search")

if not st.session_state.index_ready:
    st.warning("⚠️ Please upload documents and click **Process / Ingest** first.")
else:
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g. What are transformer attention mechanisms?",
    )

    top_k = st.slider(
        "Number of results (top-k)",
        min_value=1,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
    )

    if query:
        with st.spinner("Searching…"):
            results = search(st.session_state.vector_store, query, top_k)

        if not results:
            st.info("No results found. Try a different query.")
        else:
            st.subheader(f"Top {len(results)} Results")
            for rank, (doc, score) in enumerate(results, start=1):
                source = doc.metadata.get("source", "Unknown")
                # Show only the filename, not the full temp path
                source_name = os.path.basename(source)
                page = doc.metadata.get("page", None)
                page_info = f" | Page {page + 1}" if page is not None else ""

                with st.container():
                    st.markdown(
                        f"**Result {rank}** &nbsp;·&nbsp; "
                        f"Score: `{score:.4f}` &nbsp;·&nbsp; "
                        f"Source: `{source_name}`{page_info}"
                    )
                    st.markdown(
                        f"<div style='background-color:#f9f9f9; padding:12px; "
                        f"border-radius:6px; border-left:4px solid #4CAF50; "
                        f"margin-bottom:12px;'>{doc.page_content}</div>",
                        unsafe_allow_html=True,
                    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("CS-4015 Agentic AI · HW1 Phase 1 · Semantic Search Module")

