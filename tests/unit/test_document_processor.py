"""
Unit tests for core/document_processor.py

Tests cover:
  - ingest_documents: missing file returns 0
  - get_db_stats: empty/non-existent persist dir returns is_empty=True
  - save_uploaded_file: duck-typed UploadedFile writes bytes to expected path
  - delete_source: bad persist dir returns False
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.document_processor import (
    ingest_documents,
    get_db_stats,
    save_uploaded_file,
    delete_source,
)


class MockUploadedFile:
    """Minimal duck-type matching Streamlit's UploadedFile interface."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self) -> bytes:
        return self._content


def test_ingest_documents_missing_file(tmp_path):
    """
    Passing a non-existent file path to ingest_documents should return 0
    because no documents are loaded (the missing file is skipped with a warning).

    Requirements: 5.1
    """
    non_existent = str(tmp_path / "does_not_exist.pdf")
    persist_dir = str(tmp_path / "chroma")

    result = ingest_documents([non_existent], persist_dir=persist_dir)

    assert result == 0, (
        f"Expected 0 chunks when all files are missing, got {result}"
    )


def test_get_db_stats_empty(tmp_path):
    """
    Calling get_db_stats on a non-existent (or freshly created empty) persist
    directory should return a dict with is_empty=True.

    The HuggingFace embedding model is mocked to avoid network/disk I/O.
    Chroma is mocked to return an empty collection (count=0).

    Requirements: 5.2
    """
    empty_dir = str(tmp_path / "empty_chroma")

    with patch("core.document_processor._get_embeddings") as mock_embeddings, \
         patch("core.document_processor.Chroma") as mock_chroma_cls:

        mock_embeddings.return_value = MagicMock()

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma_instance = MagicMock()
        mock_chroma_instance._collection = mock_collection
        mock_chroma_cls.return_value = mock_chroma_instance

        stats = get_db_stats(persist_dir=empty_dir)

    assert "is_empty" in stats, "Stats dict must contain 'is_empty' key"
    assert stats["is_empty"] is True, (
        f"Expected is_empty=True for an empty store, got {stats}"
    )


def test_save_uploaded_file_writes_to_disk(tmp_path):
    """
    save_uploaded_file should write the bytes from uploaded_file.getbuffer()
    to <save_directory>/<uploaded_file.name> and return the full path.

    Requirements: 5.3
    """
    content = b"Hello, test PDF content!"
    mock_file = MockUploadedFile(name="test_document.pdf", content=content)
    save_dir = str(tmp_path / "uploads")

    returned_path = save_uploaded_file(mock_file, save_directory=save_dir)

    expected_path = os.path.join(save_dir, "test_document.pdf")
    assert returned_path == expected_path, (
        f"Expected returned path {expected_path!r}, got {returned_path!r}"
    )
    assert os.path.isfile(expected_path), (
        f"File was not created at {expected_path}"
    )
    assert Path(expected_path).read_bytes() == content, (
        "File content does not match the bytes from getbuffer()"
    )


def test_delete_source_returns_false_on_bad_db(tmp_path):
    """
    delete_source should return False when the persist directory is invalid
    or the Chroma collection raises an exception.

    Requirements: 5.4
    """
    bad_dir = str(tmp_path / "nonexistent_chroma_db")

    with patch("core.document_processor._get_embeddings") as mock_embeddings, \
         patch("core.document_processor.Chroma") as mock_chroma_cls:

        mock_embeddings.return_value = MagicMock()
        mock_chroma_instance = MagicMock()
        mock_chroma_instance._collection.delete.side_effect = Exception(
            "Simulated Chroma failure"
        )
        mock_chroma_cls.return_value = mock_chroma_instance

        result = delete_source("some_source.pdf", persist_dir=bad_dir)

    assert result is False, (
        f"Expected False when Chroma raises an exception, got {result!r}"
    )
