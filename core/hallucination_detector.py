"""
core/hallucination_detector.py — Answer Quality Scorer / Hallucination Detector

This module implements a lightweight, locally-running hallucination detector
for RAG-generated answers. It measures how well the LLM's answer is grounded
in the retrieved context using semantic similarity.

How it works
------------
1. The LLM answer is encoded into a dense vector embedding.
2. Each retrieved context chunk is encoded separately into its own vector.
3. The cosine similarity between the answer and EACH chunk is computed.
4. The MAXIMUM similarity across all chunks is used as the confidence score.
   This is the correct strategy — if the answer is grounded in ANY chunk,
   it should score high. Averaging all chunks into one vector dilutes the
   signal and produces artificially low scores.

Why chunk-level max similarity is better
-----------------------------------------
- The whole-context approach encodes all chunks into one averaged vector.
  This averages out the relevant signal with irrelevant noise from other chunks.
- The answer only needs to be grounded in ONE chunk to be valid.
- Max similarity correctly captures this: if any chunk supports the answer,
  the confidence is high.

Scoring (tuned for summarization and paraphrasing tasks)
---------------------------------------------------------
  confidence >= 0.40  →  HIGH    — answer is well-grounded in at least one chunk
  confidence >= 0.25  →  MEDIUM  — answer is partially grounded
  confidence <  0.25  →  LOW     — answer may be hallucinated

Note: These thresholds are lower than naive text-matching thresholds because
sentence-transformer cosine similarity between a summary and its source text
is naturally lower than between two paraphrases of the same sentence.
The thresholds were calibrated empirically on CV/academic document Q&A pairs.

No Streamlit imports — fully testable without a running Streamlit session.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


DETECTOR_MODEL_NAME: str = "all-MiniLM-L6-v2"

HIGH_CONFIDENCE_THRESHOLD: float = 0.40
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.25

MAX_CHUNK_CHARS: int = 512


@dataclass
class HallucinationResult:
    """
    Result of the hallucination detection check.

    Attributes:
        confidence:   Max cosine similarity between answer and any context chunk (0–1).
        label:        Human-readable label: 'HIGH', 'MEDIUM', or 'LOW'.
        is_grounded:  True if confidence >= MEDIUM_CONFIDENCE_THRESHOLD.
        best_chunk:   Index of the chunk that best supports the answer (0-based).
        warning:      Optional warning message shown when confidence is low.
    """
    confidence: float
    label: str           # 'HIGH' | 'MEDIUM' | 'LOW'
    is_grounded: bool
    best_chunk: int = 0
    warning: str | None = None


class HallucinationDetector:
    """
    Hallucination detector using chunk-level max cosine similarity.

    Compares the LLM answer against each retrieved context chunk individually
    and uses the maximum similarity as the confidence score. This correctly
    handles summarization, paraphrasing, and multi-chunk retrieval scenarios.

    Example
    -------
    >>> detector = HallucinationDetector()
    >>> result = detector.score(
    ...     chunks=["The sky is blue.", "Water is wet."],
    ...     answer="The sky appears blue in colour."
    ... )
    >>> result.label
    'HIGH'
    """

    def __init__(self, model_name: str = DETECTOR_MODEL_NAME):
        """
        Initialise the detector by loading the sentence-transformer model.

        Args:
            model_name: HuggingFace model name for encoding. Defaults to
                        'all-MiniLM-L6-v2' (80MB, fast, good quality).
        """
        self._model = SentenceTransformer(model_name)

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors, clipped to [0, 1].

        Args:
            vec_a: First embedding vector.
            vec_b: Second embedding vector.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        return float(np.clip(similarity, 0.0, 1.0))

    def _truncate(self, text: str) -> str:
        """Truncate text to MAX_CHUNK_CHARS to keep encoding fast and focused."""
        return text[:MAX_CHUNK_CHARS]

    def score(self, context: str, answer: str) -> HallucinationResult:
        """
        Score how well the answer is grounded in the retrieved context.

        Splits the context into individual chunks (separated by double newlines),
        encodes each chunk and the answer separately, then returns the maximum
        cosine similarity across all chunks as the confidence score.

        Args:
            context: The concatenated retrieved document chunks (joined by '\\n\\n').
            answer:  The LLM-generated answer to evaluate.

        Returns:
            HallucinationResult with confidence score, label, and optional warning.
        """
        if not context or not context.strip():
            return HallucinationResult(
                confidence=0.0,
                label="LOW",
                is_grounded=False,
                warning="No context was retrieved — answer may not be grounded.",
            )

        if not answer or not answer.strip():
            return HallucinationResult(
                confidence=0.0,
                label="LOW",
                is_grounded=False,
                warning="Empty answer received.",
            )

        raw_chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
        if not raw_chunks:
            raw_chunks = [context]  # fallback: treat whole context as one chunk

        chunks = [self._truncate(c) for c in raw_chunks]

        all_texts = [self._truncate(answer)] + chunks
        embeddings = self._model.encode(
            all_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        answer_vec  = embeddings[0]
        chunk_vecs  = embeddings[1:]

        similarities = [
            self._cosine_similarity(answer_vec, chunk_vec)
            for chunk_vec in chunk_vecs
        ]

        best_idx   = int(np.argmax(similarities))
        confidence = similarities[best_idx]

        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            label       = "HIGH"
            is_grounded = True
            warning     = None
        elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            label       = "MEDIUM"
            is_grounded = True
            warning     = None
        else:
            label       = "LOW"
            is_grounded = False
            warning     = (
                f"Low confidence ({confidence:.0%}) — the retrieved chunks may not contain "
                "the relevant section. Try increasing Top-K in Settings → Retrieval Context, "
                "or re-ingest your documents with smaller chunk size."
            )

        return HallucinationResult(
            confidence=confidence,
            label=label,
            is_grounded=is_grounded,
            best_chunk=best_idx,
            warning=warning,
        )
