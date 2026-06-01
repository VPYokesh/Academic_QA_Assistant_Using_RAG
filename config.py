import os


class Config:
    """
    Configuration class to manage file paths and settings for the RAG application.
    """


    UPLOAD_DIRECTORY: str = "data/uploads"

    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma"

    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_MAX_TOKENS: int = 1024
    DEFAULT_TOP_K: int = 6

    LLM_MODEL_NAME: str = "llama-3.1-8b-instant"

    def __init__(self):
        os.makedirs(self.UPLOAD_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)


config = Config()
