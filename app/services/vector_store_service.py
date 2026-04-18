from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


from app.services.session_service import SessionService
from app.utils.logger import logger


class VectorStoreService:

    @classmethod
    def build_from_chunks(cls, chunks: List[str], embedding, metadata: Optional[Dict[str, Any]] = None):
        """
        Build vector store from chunks with optional metadata
        
        Args:
            chunks: List of text chunks
            embedding: Embedding model
            metadata: Optional metadata dict (e.g., {'source': 'document_name'})
        """
        if not chunks:
            raise ValueError("Chunks is empty")

        logger.info(f"Building vector store from {len(chunks)} chunks")

        # Create documents with metadata
        docs = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata['chunk_id'] = i  # Thêm chunk ID
            docs.append(Document(page_content=chunk, metadata=doc_metadata))
        
        vector_store = FAISS.from_documents(docs, embedding)

        SessionService.set_vector_store(vector_store)

        logger.info("Vector store stored in session (RAM)")

        return vector_store

    @classmethod
    def get_vector_store(cls):
        return SessionService.get_vector_store()

    @classmethod
    def clear(cls):
        SessionService.clear_vector_store()
        logger.info("Vector store cleared from session")
