"""
app.py — Thin entry point for the RAG QA Streamlit application.

All business logic lives in core/; all UI rendering lives in ui/.
This file is responsible only for startup orchestration.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from core import rag_engine
from core.rag_engine import load_embeddings, load_vectordb, load_model, build_langgraph_app, load_hallucination_detector, load_reranker
from core.document_processor import get_db_stats
from ui import styles, chat, management
from config import config

st.set_page_config(
    page_title="RAG QA — Academic Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()
if os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

styles.inject()


def _init_session():
    defaults = {
        "messages": [],
        "thread_id": "rag_session_001",
        "temperature": config.DEFAULT_TEMPERATURE,
        "max_tokens": config.DEFAULT_MAX_TOKENS,
        "top_k": config.DEFAULT_TOP_K,
        "db_stats": None,
        "ingestion_log": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()

_embeddings = load_embeddings()
_vectordb = load_vectordb(_embeddings)
_model = load_model(st.session_state.temperature, st.session_state.max_tokens)
rag_engine._model = _model
_lg_app = build_langgraph_app()
_detector = load_hallucination_detector()
_reranker = load_reranker()

if st.session_state.db_stats is None:
    st.session_state.db_stats = get_db_stats()

st.markdown("""
<div class="main-header">
  <h1>Academic QA Assistant</h1>
  <p>Your intelligent research companion. Upload documents and start discovering insights instantly.</p>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_manage = st.tabs(["Chat Assistant", "Data Management & Settings"])
with tab_chat:
    chat.render(_lg_app, _vectordb, config, _detector, _reranker)
with tab_manage:
    management.render(config, vectordb=_vectordb)
