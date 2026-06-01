"""
core/reranker.py — Cross-Encoder Re-Ranker for RAG retrieval improvement.

What is a Re-Ranker?
--------------------
ChromaDB's initial retrieval uses bi-encoder embeddings (query and document
encoded separately, then compared by cosine similarity). This is fast but
approximate — it misses semantic nuances because query and document are
never seen together during scoring.

A Cross-Encoder re-ranker fixes this by encoding the (query, document) PAIR
together in a single forward pass. This lets the model attend to interactions
between query tokens and document tokens, producing much more accurate
relevance scores.

Pipeline with Re-Ranker
-----------------------
  Step 1 — Retrieve (fast, approximate):
      ChromaDB fetches top (top_k × FETCH_MULTIPLIER) candidates
      using bi-encoder cosine similarity.

  Step 2 — Re-Rank (slower, accurate):
      CrossEncoder scores each (query, candidate) pair.
      Candidates are sorted by cross-encoder score descending.
      Only the top_k highest-scoring candidates are kept.

  Step 3 — Generate:
      The re-ranked top_k chunks are passed to the LLM as context.

Why this matters
----------------
Without re-ranking: ChromaDB might return chunks that are topically similar
but not directly relevant to the specific question.

With re-ranking: The cross-encoder identifies which chunks actually answer
the question, not just which chunks are about the same topic.

Model used
----------
  cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking dataset (real search queries)
  - ~80MB, fast inference
  - Returns a relevance score (higher = more relevant)
  - No external API — runs fully locally

No Streamlit imports — fully testable without a running Streamlit session.
"""

from __future__ import annotations


from dataclasses import dataclass
from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FETCH_MULTIPLIER: int = 3


@dataclass
class RankedChunk:
    """
    A single retrieved document chunk with its re-ranking score.

    Attributes:
        content:       The raw text content of the chunk.
        metadata:      Original LangChain document metadata (source, page, etc.).
        retrieval_score: Original ChromaDB cosine distance score (lower = more similar).
        rerank_score:  Cross-encoder relevance score (higher = more relevant).
        rank:          Final position after re-ranking (1 = most relevant).
    """
    content: str
    metadata: dict
    retrieval_score: float   # ChromaDB cosine distance (original order)
    rerank_score: float      # CrossEncoder score (re-ranked order)
    rank: int                # 1-based rank after re-ranking


class Reranker:
    """
    Cross-Encoder re-ranker for improving RAG retrieval quality.

    Takes a list of candidate chunks retrieved by ChromaDB and re-scores
    them using a cross-encoder model that jointly encodes the query and
    each chunk. Returns the top-K most relevant chunks in ranked order.

    Example
    -------
    >>> reranker = Reranker()
    >>> ranked = reranker.rerank(
    ...     query="What is the capital of France?",
    ...     docs=[(doc1, 0.3), (doc2, 0.5), (doc3, 0.4)],
    ...     top_k=2
    ... )
    >>> ranked[0].rank
    1
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        """
        Load the cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model name.
                        Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        """
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs: list[tuple],
        top_k: int,
    ) -> list[RankedChunk]:
        """
        Re-rank retrieved document chunks by cross-encoder relevance score.

        Args:
            query:  The user's query string.
            docs:   List of (LangChain Document, cosine_distance_score) tuples
                    as returned by ChromaDB's similarity_search_with_score().
            top_k:  Number of top chunks to return after re-ranking.

        Returns:
            List of RankedChunk objects sorted by rerank_score descending,
            limited to top_k items. Returns empty list if docs is empty.
        """
        if not docs:
            return []

        top_k = min(top_k, len(docs))

        pairs = [
            (query, doc.page_content[:512])
            for doc, _ in docs
        ]

        scores = self._model.predict(pairs, show_progress_bar=False)

        scored_docs = [
            (float(score), doc, retrieval_score)
            for (doc, retrieval_score), score in zip(docs, scores)
        ]

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        ranked_chunks = []
        for rank, (rerank_score, doc, retrieval_score) in enumerate(
            scored_docs[:top_k], start=1
        ):
            ranked_chunks.append(
                RankedChunk(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    retrieval_score=float(retrieval_score),
                    rerank_score=rerank_score,
                    rank=rank,
                )
            )

        return ranked_chunks
