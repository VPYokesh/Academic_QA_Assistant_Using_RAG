"""
Unit tests for core/reranker.py

Tests cover:
  - Most relevant chunk is ranked #1
  - top_k limits the number of returned chunks
  - Empty docs list returns empty result
  - top_k larger than docs is clamped
  - RankedChunk fields are correct
  - Re-ranker improves ordering over raw retrieval
"""

import pytest
from unittest.mock import MagicMock
from core.reranker import Reranker, RankedChunk


def make_doc(content: str, source: str = "test.pdf", page: int = 0):
    """Create a minimal mock LangChain Document."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page}
    return doc


@pytest.fixture(scope="module")
def reranker():
    return Reranker()


def test_most_relevant_chunk_ranked_first(reranker):
    """
    The chunk that directly answers the query should be ranked #1.
    """
    query = "What is the capital of France?"

    docs = [
        (make_doc("The weather in Paris is usually mild in spring."), 0.4),
        (make_doc("Paris is the capital city of France."), 0.3),          # most relevant
        (make_doc("France is known for its cuisine and wine culture."), 0.5),
    ]

    ranked = reranker.rerank(query=query, docs=docs, top_k=3)

    assert len(ranked) == 3
    assert ranked[0].rank == 1
    assert "capital" in ranked[0].content.lower() or "paris" in ranked[0].content.lower(), (
        f"Expected most relevant chunk first, got: {ranked[0].content!r}"
    )


def test_top_k_limits_results(reranker):
    """
    rerank() should return exactly top_k chunks even if more are provided.
    """
    query = "machine learning algorithms"
    docs = [(make_doc(f"Document about topic {i}"), float(i) * 0.1) for i in range(6)]

    ranked = reranker.rerank(query=query, docs=docs, top_k=3)

    assert len(ranked) == 3, f"Expected 3 results, got {len(ranked)}"


def test_empty_docs_returns_empty(reranker):
    """
    Empty docs list should return an empty list without error.
    """
    ranked = reranker.rerank(query="any query", docs=[], top_k=3)
    assert ranked == []


def test_top_k_larger_than_docs_is_clamped(reranker):
    """
    If top_k > len(docs), return all docs without error.
    """
    query = "test query"
    docs = [
        (make_doc("First document content."), 0.2),
        (make_doc("Second document content."), 0.3),
    ]

    ranked = reranker.rerank(query=query, docs=docs, top_k=10)

    assert len(ranked) == 2, f"Expected 2 results (clamped), got {len(ranked)}"


def test_ranked_chunk_fields(reranker):
    """
    RankedChunk should have all expected fields populated correctly.
    """
    query = "What is Python?"
    doc = make_doc("Python is a high-level programming language.", source="python.pdf", page=5)
    docs = [(doc, 0.25)]

    ranked = reranker.rerank(query=query, docs=docs, top_k=1)

    assert len(ranked) == 1
    chunk = ranked[0]

    assert isinstance(chunk, RankedChunk)
    assert chunk.rank == 1
    assert chunk.content == "Python is a high-level programming language."
    assert chunk.metadata["source"] == "python.pdf"
    assert chunk.metadata["page"] == 5
    assert chunk.retrieval_score == 0.25
    assert isinstance(chunk.rerank_score, float)


def test_ranks_are_sequential(reranker):
    """
    Ranks should be sequential starting from 1.
    """
    query = "data science skills"
    docs = [(make_doc(f"Content {i}"), 0.1 * i) for i in range(5)]

    ranked = reranker.rerank(query=query, docs=docs, top_k=5)

    ranks = [chunk.rank for chunk in ranked]
    assert ranks == [1, 2, 3, 4, 5], f"Expected sequential ranks, got {ranks}"


def test_reranker_improves_ordering(reranker):
    """
    Re-ranker should place the directly relevant chunk above a topically
    similar but less relevant chunk, even if ChromaDB scored them differently.
    """
    query = "What programming languages does the candidate know?"

    docs = [
        (make_doc("Phone: +91 9876543210 | Email: test@email.com | LinkedIn: /in/test"), 0.2),
        (make_doc("Skills: Python, SQL, R, Tableau, Power BI, Machine Learning"), 0.35),
    ]

    ranked = reranker.rerank(query=query, docs=docs, top_k=2)

    assert "python" in ranked[0].content.lower() or "sql" in ranked[0].content.lower(), (
        f"Expected skills chunk ranked first, got: {ranked[0].content!r}"
    )
