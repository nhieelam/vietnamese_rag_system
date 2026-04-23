from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import AppConfig
from app.utils.logger import logger


class TextSplitterService:

    @classmethod
    def _resolve_params(cls) -> tuple:
        """Ưu tiên giá trị user đặt trong session, fallback AppConfig."""
        try:
            from app.services.session_service import SessionService
            params = SessionService.get_chunk_params()
            return int(params["chunk_size"]), int(params["chunk_overlap"])
        except Exception:
            return AppConfig.CHUNK_SIZE, AppConfig.CHUNK_OVERLAP

    @classmethod
    def _build_splitter(cls, add_start_index: bool = False) -> RecursiveCharacterTextSplitter:
        chunk_size, chunk_overlap = cls._resolve_params()
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 5)
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=add_start_index,
        )

    @classmethod
    def split(cls, text: str) -> List[str]:
        if not text or not text.strip():
            raise ValueError("Text is empty, cannot split")

        chunk_size, chunk_overlap = cls._resolve_params()
        chunks = cls._build_splitter().split_text(text)
        logger.info(
            f"Split text into {len(chunks)} chunks "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )
        return chunks

    @classmethod
    def split_with_offsets(
        cls,
        text: str,
        page_ranges: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chia text và giữ offset + suy ra page dựa vào page_ranges.

        Args:
            text: toàn bộ nội dung tài liệu đã ghép.
            page_ranges: list dict {page, start, end} từ FileService.

        Returns:
            list dict {text, start, end, page}
        """
        if not text or not text.strip():
            raise ValueError("Text is empty, cannot split")

        splitter = cls._build_splitter(add_start_index=True)
        docs = splitter.create_documents([text])

        result: List[Dict[str, Any]] = []
        for d in docs:
            start = int(d.metadata.get("start_index", 0))
            end = start + len(d.page_content)
            page = cls._locate_page(start, end, page_ranges) if page_ranges else None
            result.append({
                "text": d.page_content,
                "start": start,
                "end": end,
                "page": page,
            })

        chunk_size, chunk_overlap = cls._resolve_params()
        logger.info(
            f"Split text into {len(result)} chunks with offsets "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )
        return result

    @staticmethod
    def _locate_page(
        start: int, end: int, page_ranges: List[Dict[str, Any]]
    ) -> Optional[int]:
        """Tìm trang chiếm nhiều diện tích nhất trong [start, end)."""
        best_page = None
        best_overlap = 0
        for pr in page_ranges:
            overlap = max(0, min(end, pr["end"]) - max(start, pr["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = pr["page"]
        return best_page
