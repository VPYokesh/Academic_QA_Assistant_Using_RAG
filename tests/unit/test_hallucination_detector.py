"""
Unit tests for core/hallucination_detector.py

Tests cover:
  - Grounded answer scores HIGH
  - Unrelated answer scores LOW
  - Empty context returns LOW with warning
  - Empty answer returns LOW with warning
  - HallucinationResult fields are correct
  - Confidence is always in [0.0, 1.0]
"""

import pytest
from core.hallucination_detector import (
    HallucinationDetector,
    HallucinationResult,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)


@pytest.fixture(scope="module")
def detector():
    """Single detector instance shared across all tests in this module."""
    return HallucinationDetector()



def test_grounded_answer_scores_high(detector):
    """
    An answer that closely paraphrases the context should score HIGH confidence.
    """
    context = (
        "The mitochondria is the powerhouse of the cell. "
        "It produces ATP through cellular respiration."
    )
    answer = "The mitochondria produces ATP and is known as the powerhouse of the cell."

    result = detector.score(context, answer)

    assert result.label == "HIGH", (
        f"Expected HIGH for grounded answer, got {result.label} ({result.confidence:.2f})"
    )
    assert result.is_grounded is True
    assert result.warning is None
    assert result.confidence >= HIGH_CONFIDENCE_THRESHOLD


def test_unrelated_answer_scores_low(detector):
    """
    An answer completely unrelated to the context should score LOW confidence.
    """
    context = (
        "The mitochondria is the powerhouse of the cell. "
        "It produces ATP through cellular respiration."
    )
    answer = (
        "The French Revolution began in 1789 and led to the rise of Napoleon Bonaparte. "
        "It fundamentally changed European political structures."
    )

    result = detector.score(context, answer)

    assert result.label == "LOW", (
        f"Expected LOW for unrelated answer, got {result.label} ({result.confidence:.2f})"
    )
    assert result.is_grounded is False
    assert result.warning is not None
    assert result.confidence < MEDIUM_CONFIDENCE_THRESHOLD


def test_empty_context_returns_low(detector):
    """
    Empty context should immediately return LOW with a warning message.
    """
    result = detector.score(context="", answer="Some answer about something.")

    assert result.label == "LOW"
    assert result.is_grounded is False
    assert result.warning is not None
    assert result.confidence == 0.0


def test_empty_answer_returns_low(detector):
    """
    Empty answer should immediately return LOW with a warning message.
    """
    result = detector.score(context="Some context about something.", answer="")

    assert result.label == "LOW"
    assert result.is_grounded is False
    assert result.warning is not None
    assert result.confidence == 0.0


def test_confidence_always_in_range(detector):
    """
    Confidence score must always be in [0.0, 1.0] regardless of input.
    """
    test_pairs = [
        ("hello world", "hello world"),
        ("abc", "xyz"),
        ("", ""),
        ("a" * 1000, "b" * 1000),
    ]
    for context, answer in test_pairs:
        result = detector.score(context, answer)
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence out of range for ({context[:20]!r}, {answer[:20]!r}): "
            f"{result.confidence}"
        )


def test_hallucination_result_fields():
    """
    HallucinationResult dataclass fields should be accessible and correct.
    """
    result = HallucinationResult(
        confidence=0.82,
        label="HIGH",
        is_grounded=True,
        warning=None,
    )
    assert result.confidence == 0.82
    assert result.label == "HIGH"
    assert result.is_grounded is True
    assert result.warning is None


def test_identical_text_scores_high(detector):
    """
    Identical context and answer should produce maximum confidence.
    """
    text = "Embedded systems are used in automotive, medical, and industrial applications."
    result = detector.score(context=text, answer=text)

    assert result.label == "HIGH"
    assert result.confidence >= HIGH_CONFIDENCE_THRESHOLD
