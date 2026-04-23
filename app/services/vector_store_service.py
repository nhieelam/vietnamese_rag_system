from typing import List, Dict, Any, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


from app.services.session_service import SessionService
from app.utils.logger import logger


class VectorStoreService:

    @classmethod
    def build_from_chunks(
        cls,
        chunks: Union[List[str], List[Dict[str, Any]]],
        embedding,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Tạo / cập nhật vector store.

        `chunks` có thể là:
            - List[str]: không có offset / trang.
            - List[dict]: {text, start, end, page}
        `metadata` chứa các trường áp dụng chung (source, document_id).
        """
        if not chunks:
            raise ValueError("Chunks is empty")

        base_meta = metadata.copy() if metadata else {}
        docs: List[Document] = []

        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                doc_meta = {
                    **base_meta,
                    "chunk_id": i,
                    "char_start": chunk.get("start"),
                    "char_end": chunk.get("end"),
                    "page": chunk.get("page"),
                }
            else:
                text = chunk
                doc_meta = {**base_meta, "chunk_id": i}

            doc_meta = {k: v for k, v in doc_meta.items() if v is not None}
            docs.append(Document(page_content=text, metadata=doc_meta))

        logger.info(f"Building vector store from {len(docs)} chunks")

        existing = SessionService.get_vector_store()
        if existing is not None:
            try:
                existing.add_documents(docs)
                vector_store = existing
                logger.info("Appended documents to existing vector store")
            except Exception:
                logger.exception("Failed to append, rebuilding vector store")
                vector_store = FAISS.from_documents(docs, embedding)
        else:
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
