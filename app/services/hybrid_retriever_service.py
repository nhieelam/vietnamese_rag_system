"""Hybrid retrieval: BM25 (keyword) + FAISS (semantic) via EnsembleRetriever.

Yêu cầu 8.2.7: kết hợp semantic search (vector) với keyword search (BM25).
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.services.session_service import SessionService
from app.utils.logger import logger


class HybridRetrieverService:

    @staticmethod
    def _filter_chunks(
        chunks: List[Document],
        doc_filter: List[str],
        type_filter: List[str],
    ) -> List[Document]:
        if not doc_filter and not type_filter:
            return chunks
        out: List[Document] = []
        for c in chunks:
            meta = c.metadata or {}
            if doc_filter and meta.get("source") not in doc_filter:
                continue
            if type_filter and meta.get("file_type") not in type_filter:
                continue
            out.append(c)
        return out

    @classmethod
    def retrieve(
        cls,
        query: str,
        k: int = 10,
        bm25_weight: Optional[float] = None,
        doc_filter: Optional[List[str]] = None,
        type_filter: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Thực hiện hybrid retrieval, trả về list (doc, relevance 0..1).

        - BM25 được build on-the-fly từ `SessionService.get_all_chunks()`.
        - Semantic retrieval từ FAISS vector store.
        - Kết hợp qua `EnsembleRetriever` (weighted RRF).
        - Relevance score chuẩn hóa từ rank order (rank 1 = 1.0).
        """
        vector_store = SessionService.get_vector_store()
        chunks = SessionService.get_all_chunks()

        doc_filter = doc_filter if doc_filter is not None else SessionService.get_doc_filter()
        type_filter = type_filter if type_filter is not None else SessionService.get_file_type_filter()
        bm25_weight = (
            bm25_weight if bm25_weight is not None else SessionService.get_bm25_weight()
        )
        bm25_weight = max(0.0, min(1.0, float(bm25_weight)))
        vec_weight = 1.0 - bm25_weight

        if not vector_store and not chunks:
            logger.warning("HybridRetriever: no vector store and no chunks")
            return []

        t0 = time.perf_counter()

        filtered_chunks = cls._filter_chunks(chunks, doc_filter, type_filter)

        try:
            from langchain_community.retrievers import BM25Retriever
        except Exception:
            try:
                from langchain.retrievers import BM25Retriever
            except Exception as e:
                logger.exception("BM25Retriever unavailable, fallback to vector only")
                return cls._vector_only(vector_store, query, k, doc_filter, type_filter)

        retrievers = []
        weights = []

        if filtered_chunks:
            try:
                bm25 = BM25Retriever.from_documents(filtered_chunks)
                bm25.k = max(k * 2, 10)
                retrievers.append(bm25)
                weights.append(bm25_weight)
            except Exception:
                logger.exception("Failed to build BM25 retriever")

        if vector_store is not None:
            try:
                vec_retriever = vector_store.as_retriever(
                    search_kwargs={"k": max(k * 2, 10)}
                )
                retrievers.append(vec_retriever)
                weights.append(vec_weight)
            except Exception:
                logger.exception("Failed to build vector retriever")

        if not retrievers:
            return []

        if len(retrievers) == 1:
            logger.info("Hybrid: only one retriever available, using it alone")
            docs = retrievers[0].invoke(query)
            results = cls._post_filter_rank(docs, k, doc_filter, type_filter)
        else:
            try:
                from langchain.retrievers import EnsembleRetriever
                ensemble = EnsembleRetriever(retrievers=retrievers, weights=weights)
                docs = ensemble.invoke(query)
                results = cls._post_filter_rank(docs, k, doc_filter, type_filter)
            except Exception:
                logger.exception("EnsembleRetriever failed, fallback to vector only")
                results = cls._vector_only(vector_store, query, k, doc_filter, type_filter)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Hybrid search done in {elapsed:.1f}ms "
            f"(bm25_w={bm25_weight:.2f}, vec_w={vec_weight:.2f}, k={k}, "
            f"returned={len(results)})"
        )
        return results

    @staticmethod
    def _post_filter_rank(
        docs: List[Document],
        k: int,
        doc_filter: List[str],
        type_filter: List[str],
    ) -> List[Tuple[Document, float]]:
        """Dedup + filter + gán score giảm dần theo rank."""
        seen = set()
        results: List[Tuple[Document, float]] = []
        for d in docs:
            meta = d.metadata or {}
            if doc_filter and meta.get("source") not in doc_filter:
                continue
            if type_filter and meta.get("file_type") not in type_filter:
                continue
            key = (
                meta.get("source"),
                meta.get("document_id"),
                meta.get("chunk_id"),
                meta.get("char_start"),
                (d.page_content or "")[:60],
            )
            if key in seen:
                continue
            seen.add(key)
            # Rank-based relevance: rank 1 -> 1.0, giảm dần tuyến tính
            rank = len(results) + 1
            rel = max(0.0, 1.0 - (rank - 1) / max(k, 1) * 0.6)
            results.append((d, rel))
            if len(results) >= k:
                break
        return results

    @classmethod
    def _vector_only(
        cls,
        vector_store,
        query: str,
        k: int,
        doc_filter: List[str],
        type_filter: List[str],
    ) -> List[Tuple[Document, float]]:
        if vector_store is None:
            return []
        fetch_k = k * 5 if (doc_filter or type_filter) else k
        try:
            pairs = vector_store.similarity_search_with_score(query, k=fetch_k)
        except Exception:
            docs = vector_store.similarity_search(query, k=fetch_k)
            pairs = [(d, 0.0) for d in docs]

        out: List[Tuple[Document, float]] = []
        for doc, dist in pairs:
            meta = doc.metadata or {}
            if doc_filter and meta.get("source") not in doc_filter:
                continue
            if type_filter and meta.get("file_type") not in type_filter:
                continue
            try:
                rel = 1.0 / (1.0 + float(dist))
            except Exception:
                rel = 0.0
            out.append((doc, rel))
            if len(out) >= k:
                break
        return out
