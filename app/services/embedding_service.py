from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from app.config.ai_config import AIConfig


class EmbeddingService:
    _embedding = None

    @classmethod
    def get_huggingface_embedding(cls):
        if cls._embedding is None:
            cls._embedding = HuggingFaceEmbeddings(
                model_name= AIConfig.MODAL_NAME
            )
        return cls._embedding