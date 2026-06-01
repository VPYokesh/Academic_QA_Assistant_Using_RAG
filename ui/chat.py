"""
ui/chat.py — Chat Assistant tab UI module.

Provides:
  - render(lg_app, vectordb, config) -> None: renders the full Chat Assistant tab
  - _sanitize_query(raw: str) -> str | None: validates and cleans user input
  - _render_source_cards(sources: list[dict]) -> None: renders HTML source card expander
"""

from __future__ import annotations

import os

import streamlit as st

from core.rag_engine import run_query
from core.document_processor import get_db_stats
from core.hallucination_detector import HallucinationDetector
from core.reranker import Reranker

MAX_QUERY_LENGTH = 1000

_GENERATION_PATTERNS = [
    "write me", "write a", "write an",
    "give me 500", "give me 1000", "give me 200", "give me 300",
    "500 words", "1000 words", "200 words", "300 words",
    "create an essay", "create a paragraph", "generate a",
    "compose a", "draft a", "make up",
]


def _is_generation_request(query: str) -> bool:
    """
    Detect if the query is asking the LLM to generate new content
    rather than retrieve and answer from documents.

    Args:
        query: The sanitized user query.

    Returns:
        True if the query looks like a generation/writing task.
    """
    q_lower = query.lower()
    return any(pattern in q_lower for pattern in _GENERATION_PATTERNS)


def _sanitize_query(raw: str) -> str | None:
    """
    Clean and validate a raw user query string.

    Strips leading/trailing whitespace, enforces a maximum length,
    and removes non-printable characters.

    Args:
        raw: The unprocessed string from the chat input widget.

    Returns:
        The sanitized query string, or None if the input is invalid
        (empty, too long, or contains only non-printable characters).
    """
    query = raw.strip()

    if not query:
        return None  # Empty or whitespace-only

    if len(query) > MAX_QUERY_LENGTH:
        return None  # Too long

    query = "".join(ch for ch in query if ch.isprintable())

    return query if query else None


def _render_confidence_badge(label: str, confidence: float) -> None:
    """
    Render a colour-coded confidence badge below the answer.

    Args:
        label:      'HIGH', 'MEDIUM', or 'LOW'
        confidence: Float score 0.0–1.0
    """
    css_class = {
        "HIGH":   "confidence-high",
        "MEDIUM": "confidence-medium",
        "LOW":    "confidence-low",
    }.get(label, "confidence-low")

    icon = {"HIGH": "✅", "MEDIUM": "⚠️", "LOW": "🚨"}.get(label, "❓")

    tooltip = {
        "HIGH":   "Answer closely matches the retrieved document chunks.",
        "MEDIUM": "Answer partially matches the retrieved chunks. May include inferred details.",
        "LOW":    "Answer has low overlap with retrieved chunks — the relevant section may not have been retrieved. Try increasing Top-K in Settings.",
    }.get(label, "")

    st.markdown(
        f'<div class="confidence-badge {css_class}" title="{tooltip}">'
        f'{icon} Answer Confidence: <strong>{label}</strong> ({confidence:.0%})'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_source_cards(sources: list[dict]) -> None:
    """
    Render an expandable section of HTML source reference cards.

    Shows retrieval match % and re-rank position if available.

    Args:
        sources: List of source metadata dicts with keys:
                 document, page, score, rerank_score, rank, snippet.
    """
    if not sources:
        return

    with st.expander("Source References", expanded=False):
        for src in sources:
            score_pct = max(0, min(100, int((1 - src["score"]) * 100)))

            rerank_badge = ""
            if src.get("rank") is not None:
                rerank_badge = (
                    f'&nbsp;<span class="rerank-pill">#{src["rank"]} re-ranked</span>'
                )

            st.markdown(f"""
<div class="source-card">
  <div class="source-title">
    {os.path.basename(src['document']) if src['document'] else 'Unknown Source'}
    &nbsp;<span class="relevance-pill">{score_pct}% match</span>
    {rerank_badge}
  </div>
  <div class="source-meta">Page {src['page'] if src['page'] is not None else '-'}</div>
  <div class="source-snippet">{src['snippet'][:300]}...</div>
</div>
""", unsafe_allow_html=True)


def render(lg_app, vectordb, config, detector: HallucinationDetector | None = None, reranker: Reranker | None = None) -> None:
    """
    Render the Chat Assistant tab.

    Args:
        lg_app:    The compiled LangGraph application.
        vectordb:  The ChromaDB vector store instance (or None if unavailable).
        config:    The application Config singleton.
        detector:  Optional HallucinationDetector for scoring answer quality.
        reranker:  Optional Reranker for improving retrieval precision.
    """
    if not os.getenv("GROQ_API_KEY", ""):
        st.error("GROQ_API_KEY is not set. Please add it to your `.env` file and restart.")
        st.stop()

    stats = st.session_state.db_stats
    if stats is None:
        stats = get_db_stats()
        st.session_state.db_stats = stats

    if stats["is_empty"]:
        st.info(
            "**The knowledge base is empty.** Please navigate to the **Data Management & Settings** "
            "tab to upload your documents and build your knowledge base."
        )

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if msg.get("hallucination"):
                        _render_confidence_badge(
                            msg["hallucination"]["label"],
                            msg["hallucination"]["confidence"],
                        )
                    if msg.get("sources"):
                        _render_source_cards(msg["sources"])

    if raw_prompt := st.chat_input("Ask a question about your documents..."):
        prompt = _sanitize_query(raw_prompt)
        if prompt is None:
            st.warning(f"Please enter a question between 1 and {MAX_QUERY_LENGTH} characters.")
        elif _is_generation_request(prompt):
            st.warning(
                "This system answers questions from your uploaded documents. "
                "It cannot write essays, abstracts, or generate new content. "
                "Try asking a specific question like: *'What are the candidate's projects?'*"
            )
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    top_k = st.session_state.top_k
                    thread_id = st.session_state.thread_id
                    result = run_query(
                        prompt, vectordb, lg_app, top_k, thread_id,
                        detector=detector,
                        reranker=reranker,
                    )

                    st.markdown(result.answer)

                    if result.error is not None:
                        st.error(f"Error: {result.error}")
                    else:
                        if result.hallucination is not None:
                            _render_confidence_badge(
                                result.hallucination.label,
                                result.hallucination.confidence,
                            )
                            if result.hallucination.warning:
                                st.warning(result.hallucination.warning)

                        _render_source_cards(result.sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                        "hallucination": {
                            "label": result.hallucination.label,
                            "confidence": result.hallucination.confidence,
                        } if result.hallucination else None,
                    })
