"""
ui/management.py — Data Management & Settings tab for the RAG QA application.

Renders the full "Data Management & Settings" tab, including:
  - System status bar (API key badge + database connection badge)
  - Document ingestion section: file uploader, ingest button, ingestion log
  - Database statistics: stat tiles, indexed file list with per-source delete buttons
  - Model parameters: temperature, max_tokens, top_k sliders → update session_state
  - Session controls: "Clear Chat History", "Reset App State"
"""

import os
import time

import streamlit as st

from core.document_processor import (
    ingest_documents,
    save_uploaded_file,
    delete_source,
)


def render(config, vectordb=None) -> None:
    """
    Render the Data Management & Settings tab.

    Args:
        config: The Config singleton (from config.py).
        vectordb: Optional loaded Chroma vectordb instance. Used to determine
                  database connection status. If None, falls back to checking
                  st.session_state.db_stats.
    """
    stats = st.session_state.get("db_stats") or {}
    api_key_set = bool(os.getenv("GROQ_API_KEY", ""))

    if vectordb is not None:
        db_loaded = True
    else:
        db_loaded = (
            st.session_state.get("db_stats") is not None
            and "error" not in (st.session_state.get("db_stats") or {})
        )

    if not stats:
        stats = {"total_chunks": 0, "sources": [], "is_empty": True}

    status_col1, status_col2, _ = st.columns([2, 2, 6])
    with status_col1:
        if api_key_set:
            st.markdown(
                '<span class="status-badge status-ok">Groq API Key</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-error">Missing API Key</span>',
                unsafe_allow_html=True,
            )
    with status_col2:
        if db_loaded:
            st.markdown(
                '<span class="status-badge status-ok">Database Connected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-warn">Database Offline</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.markdown("### Document Ingestion")
    st.markdown(
        "Upload PDF, DOCX, TXT, or CSV files. "
        "The system will extract text, chunk it, and build a searchable vector index."
    )

    upload_col, info_col = st.columns([6, 4], gap="large")

    with upload_col:
        uploaded_files = st.file_uploader(
            "Drag & drop files here",
            type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, CSV.",
            key="doc_uploader",
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        ingest_btn = st.button(
            "Process & Ingest Documents",
            use_container_width=True,
            disabled=not uploaded_files,
        )

        if ingest_btn:
            if not uploaded_files:
                st.warning("Please select at least one file first.")
            else:
                with st.spinner(
                    "Processing documents... Extracting text and generating embeddings..."
                ):
                    log_entries = []

                    saved_paths = []
                    for uf in uploaded_files:
                        path = save_uploaded_file(uf)
                        saved_paths.append(path)

                    count = ingest_documents(saved_paths)

                    if count >= 0:
                        log_entries.append(
                            f"Successfully ingested {len(saved_paths)} file(s)."
                        )
                        log_entries.append(f"Generated {count} vector embeddings.")
                    else:
                        log_entries.append("Ingestion failed. Check terminal logs.")

                    st.cache_resource.clear()
                    st.session_state.ingestion_log.extend(log_entries)
                    st.session_state.db_stats = None  # force refresh

                for entry in log_entries:
                    st.markdown(entry)
                st.success(
                    "Ingestion complete! You can now ask questions about these documents "
                    "in the Chat Assistant tab."
                )
                time.sleep(1)
                st.rerun()

        if st.session_state.get("ingestion_log"):
            with st.expander("Recent Ingestion Activity", expanded=True):
                for entry in reversed(st.session_state.ingestion_log):
                    st.markdown(f"- {entry}")

    with info_col:
        st.markdown("#### Database Statistics", unsafe_allow_html=True)

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-number">{stats["total_chunks"]}</div>'
                f'<div class="stat-label">Total Chunks</div></div>',
                unsafe_allow_html=True,
            )
        with stat_col2:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-number">{len(stats.get("sources", []))}</div>'
                f'<div class="stat-label">Documents</div></div>',
                unsafe_allow_html=True,
            )

        if stats.get("sources"):
            st.markdown("##### Indexed Files")
            for src in stats["sources"]:
                src_name = os.path.basename(src) if src else "Unknown"

                file_col, del_col = st.columns([8, 2])
                with file_col:
                    st.markdown(f"`{src_name}`")
                with del_col:
                    if st.button(
                        "Delete",
                        key=f"del_{src_name}",
                        help=f"Delete {src_name} from database",
                    ):
                        with st.spinner(f"Deleting {src_name}..."):
                            if delete_source(src):
                                st.success(f"Deleted {src_name}")
                                st.session_state.db_stats = None  # Force refresh
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Failed to delete {src_name}")

        if st.button("Refresh Database Stats", use_container_width=True):
            st.session_state.db_stats = None
            st.rerun()

    st.divider()

    st.markdown("### Model Parameters & Config")
    st.info("Adjusting generation parameters will take effect on your next question.")

    config_col1, config_col2 = st.columns(2, gap="large")

    with config_col1:
        st.markdown("#### Generation Controls")
        new_temp = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            help="Lower values are more factual and focused. Higher values are more creative.",
        )
        st.session_state.temperature = new_temp

        new_max = st.slider(
            "Max Output Tokens",
            min_value=100,
            max_value=2000,
            value=st.session_state.max_tokens,
            step=50,
            help="Maximum length of the LLM response.",
        )
        st.session_state.max_tokens = new_max

        new_top_k = st.slider(
            "Retrieval Context (Top-K)",
            min_value=1,
            max_value=15,
            value=st.session_state.top_k,
            step=1,
            help="Number of document chunks to fetch and feed to the LLM per query.",
        )
        st.session_state.top_k = new_top_k

    with config_col2:
        st.markdown("#### System Information")
        st.markdown(
            f"""
        | Component | Configured Value |
        |---|---|
        | **LLM Inference** | `{config.LLM_MODEL_NAME}` (Groq) |
        | **Embedding Model** | `{config.EMBEDDING_MODEL_NAME}` |
        | **Re-Ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
        | **Hallucination Detector** | `all-MiniLM-L6-v2` |
        | **Text Chunk Size** | `{config.CHUNK_SIZE}` chars (overlap: {config.CHUNK_OVERLAP}) |
        | **Default Top-K** | `{config.DEFAULT_TOP_K}` (fetches `{config.DEFAULT_TOP_K * 3}` before re-rank) |
        | **Vector Store** | ChromaDB (`{config.CHROMA_PERSIST_DIRECTORY}`) |
        """
        )

        st.markdown("")
        st.markdown("#### Session Controls")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = f"rag_session_{int(time.time())}"
            st.success("Chat history cleared.")

        if st.button("Reset App State", use_container_width=True):
            st.session_state.clear()
            st.rerun()
