"""
core/rag_engine.py — Framework-agnostic RAG engine.

Provides:
  - RAGState: LangGraph state schema
  - QueryResult: return type for run_query
  - Cached resource loaders: load_embeddings, load_vectordb, load_model, build_langgraph_app
  - run_query: pure query execution function (no Streamlit calls)
"""

from __future__ import annotations

import streamlit as st

from dataclasses import dataclass, field
from typing import Any

from core.hallucination_detector import HallucinationDetector, HallucinationResult
from core.reranker import Reranker, FETCH_MULTIPLIER

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config


class RAGState(MessagesState):
    """LangGraph state schema — extends MessagesState with a retrieved context field."""
    context: str


@dataclass
class QueryResult:
    """Return type of run_query."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    error: str | None = None
    hallucination: HallucinationResult | None = None  # None only on pipeline error


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Connecting to vector store...")
def load_vectordb(_embeddings) -> Chroma | None:
    """Load and cache the ChromaDB vector store. Returns None on any exception."""
    try:
        return Chroma(
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            embedding_function=_embeddings,
        )
    except Exception:
        return None


@st.cache_resource(show_spinner="Initialising LLM...")
def load_model(temperature: float, max_tokens: int) -> ChatGroq:
    """Load and cache the Groq LLM."""
    return ChatGroq(
        model=config.LLM_MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@st.cache_resource(show_spinner="Loading hallucination detector...")
def load_hallucination_detector() -> HallucinationDetector:
    """Load and cache the hallucination detector model (all-MiniLM-L6-v2, ~80MB)."""
    return HallucinationDetector()


@st.cache_resource(show_spinner="Loading re-ranker model...")
def load_reranker() -> Reranker:
    """Load and cache the cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2, ~80MB)."""
    return Reranker()


@st.cache_resource(show_spinner=False)
def build_langgraph_app() -> Any:
    """Build and cache the compiled LangGraph application."""

    def call_model(state: RAGState):
        system_prompt = (
            "You are an AI Study Assistant designed to help students learn from their academic materials. "
            "Your role is to provide clear, accurate, and educational responses based on the uploaded study documents.\n\n"
            
            "## Core Principles:\n"
            "1. **Context-Grounded Answers**: Answer ONLY using information from the Context section below. "
            "Never use external knowledge or make assumptions beyond what's provided.\n\n"
            
            "2. **Educational Approach**: When explaining concepts:\n"
            "   - Break down complex topics into simpler parts\n"
            "   - Provide examples when available in the context\n"
            "   - Highlight key terms and definitions\n"
            "   - Connect related concepts when mentioned in the documents\n\n"
            
            "3. **Study-Focused Responses**:\n"
            "   - For 'what' questions: Provide clear definitions and explanations\n"
            "   - For 'why' questions: Explain reasoning and relationships\n"
            "   - For 'how' questions: Describe processes step-by-step\n"
            "   - For comparison questions: Highlight similarities and differences\n\n"
            
            "4. **Transparency**: If the context doesn't contain enough information, respond with: "
            "'I cannot find sufficient information about this in your uploaded study materials. "
            "Please try rephrasing your question or upload additional relevant documents.'\n\n"
            
            "5. **Citation**: Always indicate which document or section your answer comes from when possible.\n\n"
            
            f"## Context from Your Study Materials:\n{state.get('context', '')}\n\n"
            
            "Now, provide a helpful, educational response based strictly on the context above."
        )
        recent_messages = state["messages"][-5:] if len(state["messages"]) > 5 else state["messages"]
        messages = [SystemMessage(content=system_prompt)] + recent_messages
        response = _model.invoke(messages)
        return {"messages": response}

    workflow = StateGraph(state_schema=RAGState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


_model: ChatGroq | None = None


def run_query(
    prompt: str,
    vectordb,
    lg_app,
    top_k: int,
    thread_id: str,
    detector: HallucinationDetector | None = None,
    reranker: Reranker | None = None,
) -> QueryResult:
    """
    Execute a RAG query and return a QueryResult.

    Pipeline:
      1. Retrieve top (top_k × FETCH_MULTIPLIER) candidates from ChromaDB.
      2. If reranker provided: re-score all candidates with cross-encoder,
         keep best top_k in ranked order.
      3. Build context from final top_k chunks.
      4. Invoke LangGraph LLM with context.
      5. Score answer with hallucination detector.

    Args:
        prompt:    The sanitized user query string.
        vectordb:  A ChromaDB Chroma instance (or None — will raise, caught below).
        lg_app:    The compiled LangGraph application.
        top_k:     Number of final chunks to pass to the LLM.
        thread_id: LangGraph conversation thread identifier.
        detector:  Optional HallucinationDetector for answer quality scoring.
        reranker:  Optional Reranker for improving retrieval precision.

    Returns:
        QueryResult with answer, sources list, hallucination score, and optional error.
    """
    try:
        fetch_k = top_k * FETCH_MULTIPLIER if reranker else top_k
        docs = vectordb.similarity_search_with_score(prompt, k=fetch_k)

        context_parts: list[str] = []
        sources: list[dict] = []

        if reranker and docs:
            ranked_chunks = reranker.rerank(query=prompt, docs=docs, top_k=top_k)
            for chunk in ranked_chunks:
                context_parts.append(chunk.content)
                sources.append({
                    "document":       chunk.metadata.get("source"),
                    "page":           chunk.metadata.get("page"),
                    "score":          chunk.retrieval_score,
                    "rerank_score":   chunk.rerank_score,
                    "rank":           chunk.rank,
                    "snippet":        chunk.content[:400],
                })
        else:
            for doc, score in docs:
                context_parts.append(doc.page_content)
                sources.append({
                    "document":     doc.metadata.get("source"),
                    "page":         doc.metadata.get("page"),
                    "score":        score,
                    "rerank_score": None,
                    "rank":         None,
                    "snippet":      doc.page_content[:400],
                })

        context_text = "\n\n".join(context_parts)

        turn_msg = HumanMessage(content=prompt)
        result = lg_app.invoke(
            {"messages": [turn_msg], "context": context_text},
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = result["messages"][-1].content

        hallucination = detector.score(context_text, answer) if detector else None

        return QueryResult(answer=answer, sources=sources, hallucination=hallucination)

    except Exception as e:
        return QueryResult(
            answer=f"An error occurred: {e}",
            sources=[],
            error=str(e),
            hallucination=None,
        )
