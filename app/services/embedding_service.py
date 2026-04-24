from langchain_huggingface import HuggingFaceEmbeddings

from app.config import AppConfig


class EmbeddingService:
    _embedding = None

    @classmethod
    def get_huggingface_embedding(cls):
        if cls._embedding is None:
            cls._embedding = HuggingFaceEmbeddings(
                model_name=AppConfig.EMBEDDING_MODEL_NAME,
            )
        return cls._embedding