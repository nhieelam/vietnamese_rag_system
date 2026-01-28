import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_LLM_MODEL = "gpt-4o"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_LLM_MODEL = "llama-3.1-8b-instant"


    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY")

    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY")
    
    # CHUNKING
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    # PATHS
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

settings = Settings()
