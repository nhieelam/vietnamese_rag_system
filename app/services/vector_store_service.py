from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


from app.services.session_service import SessionService
from app.utils.logger import logger


class VectorStoreService:
    """
    RAM-only Vector Store Service
    """

    @classmethod
    def build_from_chunks(cls, chunks: List[str], embedding):
        """
        Build FAISS vector store from text chunks
        """
        if not chunks:
            raise ValueError("Chunks is empty")

        logger.info(f"Building vector store from {len(chunks)} chunks")

        docs = [Document(page_content=chunk) for chunk in chunks]
        vector_store = FAISS.from_documents(docs, embedding)

        SessionService.set_vector_store(vector_store)

        logger.info("Vector store stored in session (RAM)")

        return vector_store

    @classmethod
    def get_vector_store(cls):
        """
        Get current vector store from session
        """
        return SessionService.get_vector_store()

    @classmethod
    def clear(cls):
        """
        Clear vector store from RAM
        """
        SessionService.clear_vector_store()
        logger.info("Vector store cleared from session")
