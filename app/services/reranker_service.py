"""Cross-encoder re-ranking (8.2.9).

Sử dụng `BAAI/bge-reranker-v2-m3` — multilingual, tốt cho tiếng Việt.
Load lazy + cache để tránh reload khi Streamlit rerun.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.utils.logger import logger


MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_MODEL_CACHE = {"model": None}


class RerankerService:

    @classmethod
    def get_model(cls):
        if _MODEL_CACHE["model"] is not None:
            return _MODEL_CACHE["model"]

        try:
            # Prefer streamlit cache if available
            import streamlit as st
            @st.cache_resource(show_spinner=False)
            def _load():
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder reranker: {MODEL_NAME}")
                return CrossEncoder(MODEL_NAME, max_length=512)
            _MODEL_CACHE["model"] = _load()
        except Exception:
            logger.exception("Streamlit cache unavailable, loading reranker directly")
            from sentence_transformers import CrossEncoder
            _MODEL_CACHE["model"] = CrossEncoder(MODEL_NAME, max_length=512)

        return _MODEL_CACHE["model"]

    @classmethod
    def rerank(
        cls,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Re-rank list (doc, prev_score) bằng cross-encoder.

        Trả về list sorted theo rerank score giảm dần, score chuẩn hóa về 0..1
        bằng sigmoid để tương thích với pipeline citations.
        """
        if not docs_with_scores:
            return []

        try:
            model = cls.get_model()
        except Exception:
            logger.exception("Failed to load reranker model, returning original order")
            return docs_with_scores[: top_k or len(docs_with_scores)]

        t0 = time.perf_counter()
        pairs = [(query, (d.page_content or "")) for d, _ in docs_with_scores]
        try:
            raw_scores = model.predict(pairs, show_progress_bar=False)
        except Exception:
            logger.exception("CrossEncoder.predict failed, returning original order")
            return docs_with_scores[: top_k or len(docs_with_scores)]

        import math
        def _sigmoid(x: float) -> float:
            try:
                return 1.0 / (1.0 + math.exp(-float(x)))
            except OverflowError:
                return 0.0 if x < 0 else 1.0

        reranked = []
        for (doc, _), s in zip(docs_with_scores, raw_scores):
            norm = _sigmoid(float(s))
            reranked.append((doc, norm))
        reranked.sort(key=lambda x: x[1], reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Reranker done in {elapsed:.1f}ms "
            f"(n={len(pairs)}, model={MODEL_NAME})"
        )

        if top_k:
            reranked = reranked[:top_k]
        return reranked
