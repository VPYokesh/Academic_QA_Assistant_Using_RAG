"""
core/document_processor.py — Document ingestion pipeline for the RAG QA application.

Supported file types: .pdf, .docx, .txt, .csv

Functions are exposed for UI usage, called directly from ui/management.py.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import config

SUPPORTED_EXTENSIONS: dict[str, type] = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".csv":  CSVLoader,
}


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Returns a HuggingFace embeddings instance."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Returns a configured RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )


def _load_document(filepath: str) -> list:
    """
    Dispatch a single file to the correct LangChain loader based on extension.

    Supported extensions: .pdf, .docx, .txt, .csv

    Args:
        filepath: Absolute or relative path to the document.

    Returns:
        List of LangChain Document objects, or an empty list on failure.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(filepath).suffix.lower()
    loader_cls = SUPPORTED_EXTENSIONS.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if loader_cls is TextLoader:
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        loader = loader_cls(filepath)

    return loader.load()


def ingest_documents(file_paths: list[str], persist_dir: str = None) -> int:
    """
    Ingest a list of document file paths into ChromaDB.

    Supported file types: .pdf, .docx, .txt, .csv

    Args:
        file_paths: Absolute or relative paths to documents to ingest.
        persist_dir: ChromaDB persist directory. Defaults to config value.

    Returns:
        Number of chunks ingested, or -1 on error.
    """
    persist_dir = persist_dir or config.CHROMA_PERSIST_DIRECTORY
    os.makedirs(persist_dir, exist_ok=True)

    all_docs = []
    for filepath in file_paths:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        print(f"Loading: {filepath}")
        try:
            pages = _load_document(filepath)
            all_docs.extend(pages)
            print(f"  Loaded {len(pages)} section(s) from {Path(filepath).name}")
        except ValueError as e:
            print(f"  Skipped {filepath}: {e}")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    if not all_docs:
        print("No documents were loaded.")
        return 0

    splitter = _get_splitter()
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = _get_embeddings()
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    vectordb.add_documents(chunks)
    print(f"Persisted {len(chunks)} chunks to ChromaDB at '{persist_dir}'.")
    return len(chunks)


def get_db_stats(persist_dir: str = None) -> dict:
    """
    Returns statistics about the ChromaDB vector store.

    Queries the ChromaDB SQLite database directly to avoid loading the
    full HuggingFace embedding model (2.2GB) just to count chunks.

    Returns a dict with keys: 'total_chunks', 'sources', 'is_empty'
    """
    import sqlite3
    persist_dir = persist_dir or config.CHROMA_PERSIST_DIRECTORY
    db_path = os.path.join(persist_dir, "chroma.sqlite3")

    if not os.path.exists(db_path):
        return {"total_chunks": 0, "sources": [], "is_empty": True}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total = cursor.fetchone()[0]

        sources = []
        if total > 0:
            cursor.execute(
                "SELECT DISTINCT string_value FROM embedding_metadata "
                "WHERE key = 'source'"
            )
            rows = cursor.fetchall()
            sources = [row[0] for row in rows if row[0]]

        conn.close()
        return {"total_chunks": total, "sources": sources, "is_empty": total == 0}

    except Exception as e:
        return {"total_chunks": 0, "sources": [], "is_empty": True, "error": str(e)}


def save_uploaded_file(uploaded_file, save_directory: str = None) -> str:
    """
    Saves a Streamlit UploadedFile object to disk.

    Args:
        uploaded_file: Streamlit UploadedFile object (or any duck-type with
                       `.name` and `.getbuffer()` attributes).
        save_directory: Directory to save the file. Defaults to config.UPLOAD_DIRECTORY.

    Returns:
        Absolute path to the saved file.
    """
    save_directory = save_directory or config.UPLOAD_DIRECTORY
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    print(f"Saved uploaded file: {save_path}")
    return save_path



def delete_source(source_name: str, persist_dir: str = None) -> bool:
    """
    Deletes all chunks associated with a specific source name from ChromaDB.

    Args:
        source_name: The value of the 'source' metadata field to delete.
        persist_dir: ChromaDB persist directory. Defaults to config value.

    Returns:
        True if successful, False otherwise.
    """
    persist_dir = persist_dir or config.CHROMA_PERSIST_DIRECTORY
    try:
        embeddings = _get_embeddings()
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        vectordb._collection.delete(where={"source": source_name})
        print(f"Successfully deleted all chunks for source: {source_name}")
        return True
    except Exception as e:
        print(f"Error deleting source {source_name}: {e}")
        return False


