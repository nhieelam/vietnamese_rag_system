import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("API Key missing! Please add OPENAI_API_KEY to your .env file.")


    LLM_MODEL = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # --- CHUNKING SETTINGS ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

settings = Settings()