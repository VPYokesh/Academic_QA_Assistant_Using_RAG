"""
Unit tests for core/rag_engine.py

Tests cover:
  - QueryResult: default field values (sources=[], error=None)
  - QueryResult: error field can be set
  - run_query: passing None as vectordb returns QueryResult with error set
  - load_vectordb: bad persist dir path returns None (mocking @st.cache_resource)

Requirements validated: 3.4, 3.5, 4.5
"""

from unittest.mock import patch, MagicMock


def test_query_result_defaults():
    """
    QueryResult(answer="hi") should have sources=[] and error=None by default.

    Requirements: 4.5
    """
    from core.rag_engine import QueryResult

    result = QueryResult(answer="hi")

    assert result.answer == "hi"
    assert result.sources == [], (
        f"Expected sources=[], got {result.sources!r}"
    )
    assert result.error is None, (
        f"Expected error=None, got {result.error!r}"
    )


def test_query_result_error_field():
    """
    QueryResult with error="oops" should have a non-None error field.

    Requirements: 3.5, 4.5
    """
    from core.rag_engine import QueryResult

    result = QueryResult(answer="something went wrong", error="oops")

    assert result.error is not None, "Expected error to be non-None"
    assert result.error == "oops"


def test_run_query_returns_error_result_on_bad_vectordb():
    """
    Passing None as vectordb to run_query should trigger the except branch
    and return a QueryResult with error set (since None has no
    similarity_search_with_score method).

    Requirements: 3.4, 3.5
    """
    from core.rag_engine import run_query

    result = run_query(
        prompt="test question",
        vectordb=None,
        lg_app=MagicMock(),
        top_k=3,
        thread_id="test-thread",
    )

    assert result.error is not None, (
        "Expected error to be set when vectordb is None"
    )
    assert isinstance(result.error, str), (
        f"Expected error to be a string, got {type(result.error)}"
    )
    assert result.sources == [], (
        f"Expected sources=[] on error, got {result.sources!r}"
    )


def test_load_vectordb_returns_none_on_bad_path():
    """
    load_vectordb should return None when the persist directory is invalid
    or Chroma raises an exception.

    @st.cache_resource is bypassed by patching streamlit.cache_resource with
    a no-op decorator before the module is imported/reloaded.

    Requirements: 4.5
    """
    import importlib
    import sys

    noop_decorator = lambda *args, **kwargs: (lambda fn: fn)

    mock_st = MagicMock()
    mock_st.cache_resource = noop_decorator

    for mod_name in list(sys.modules.keys()):
        if mod_name == "core.rag_engine" or mod_name == "core":
            pass  # we'll reload below

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        if "core.rag_engine" in sys.modules:
            del sys.modules["core.rag_engine"]

        import core.rag_engine as rag_engine_mod
        importlib.reload(rag_engine_mod)

        with patch.object(rag_engine_mod, "Chroma", side_effect=Exception("bad path")):
            mock_embeddings = MagicMock()
            result = rag_engine_mod.load_vectordb(mock_embeddings)

    assert result is None, (
        f"Expected None when Chroma raises an exception, got {result!r}"
    )

    if "core.rag_engine" in sys.modules:
        del sys.modules["core.rag_engine"]
