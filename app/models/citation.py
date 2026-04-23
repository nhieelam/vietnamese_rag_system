from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Citation:
    """Nguồn gốc của một đoạn context được dùng để trả lời."""

    source_name: str
    document_id: Optional[int] = None
    page_number: Optional[int] = None          # Trang 1-based trong PDF (nếu biết)
    char_start: Optional[int] = None           # Offset bắt đầu trong document text
    char_end: Optional[int] = None             # Offset kết thúc trong document text
    chunk_id: Optional[int] = None             # Thứ tự chunk trong tài liệu
    relevance_score: float = 0.0               # 0..1, cao hơn = liên quan hơn
    excerpt: str = ""                          # Preview ngắn để hiển thị list
    full_text: str = ""                        # Toàn bộ nội dung chunk đã dùng
    ref_index: Optional[int] = None            # Số [n] trỏ vào câu trả lời

    def display_title(self) -> str:
        parts = [self.source_name or "Unknown Document"]
        if self.page_number is not None:
            parts.append(f"trang {self.page_number}")
        if self.ref_index is not None:
            return f"[{self.ref_index}] " + " · ".join(parts)
        return " · ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        allowed = {f for f in cls.__dataclass_fields__.keys()}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def __hash__(self):
        return hash((self.source_name, self.document_id, self.chunk_id, self.char_start))

    def __eq__(self, other):
        if not isinstance(other, Citation):
            return False
        return (
            self.source_name == other.source_name
            and self.document_id == other.document_id
            and self.chunk_id == other.chunk_id
            and self.char_start == other.char_start
        )


@dataclass
class AnswerWithCitations:
    """Câu trả lời kèm danh sách nguồn trích dẫn."""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    mode: str = "RAG"
    confidence: Optional[float] = None           # 0..1 - Self-RAG self-evaluation
    rewritten_query: Optional[str] = None        # Self-RAG query rewriting output
    grounded_score: Optional[float] = None       # 0..1 - answer grounded in context
    completeness_score: Optional[float] = None   # 0..1 - answer fully addresses question
    hops: Optional[int] = None                   # số vòng multi-hop đã chạy

    def get_formatted_answer(self) -> str:
        formatted = f"{self.answer}\n\n"
        if self.citations:
            formatted += "### Sources\n"
            for i, c in enumerate(self.citations, 1):
                formatted += f"{i}. {c.display_title()}\n"
                if c.excerpt:
                    formatted += f"   > {c.excerpt}\n"
        return formatted
