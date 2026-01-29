import os
from dotenv import load_dotenv

load_dotenv()


class AIConfig:
    # PROVIDER
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

    # OPENAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_LLM_MODEL = "gpt-4o"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

    # GROQ
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_LLM_MODEL = "llama-3.1-8b-instant"

    # CHUNKING
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # PATHS
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

    @classmethod
    def validate(cls):
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY")

        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY")

        os.makedirs(cls.VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
