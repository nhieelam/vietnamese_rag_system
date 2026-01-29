from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from app.config.ai_config import AIConfig


class EmbeddingService:
    _embedding = None

    @classmethod
    def get_openai_embedding(cls):
        if cls._embedding is None:
            cls._embedding = OpenAIEmbeddings(
                api_key=AIConfig.OPENAI_API_KEY,
                model=AIConfig.OPENAI_EMBEDDING_MODEL
            )
        return cls._embedding
    
    @classmethod
    def get_huggingface_embedding(cls):
        if cls._embedding is None:
            cls._embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return cls._embedding