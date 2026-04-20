from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Citation:
    """Model for citation/source tracking"""
    source_name: str  # Tên tài liệu
    page_number: Optional[int] = None  # Số trang (nếu có)
    chunk_id: Optional[str] = None  # ID của chunk
    relevance_score: float = 0.0  # Điểm liên quan (0-1)
    excerpt: str = ""  # Đoạn trích từ tài liệu
    
    def __str__(self) -> str:
        """Format citation thành chuỗi"""
        citation_text = f"📄 {self.source_name}"
        if self.page_number:
            citation_text += f" (Page {self.page_number})"
        if self.relevance_score:
            citation_text += f" - Relevance: {self.relevance_score:.1%}"
        return citation_text
    
    def __hash__(self):
        """Cho phép sử dụng trong set"""
        return hash((self.source_name, self.page_number, self.chunk_id))
    
    def __eq__(self, other):
        """So sánh citations"""
        if not isinstance(other, Citation):
            return False
        return (self.source_name == other.source_name and 
                self.page_number == other.page_number and 
                self.chunk_id == other.chunk_id)


@dataclass
class AnswerWithCitations:
    """Model cho câu trả lời kèm citations"""
    answer: str  # Nội dung câu trả lời
    citations: list = field(default_factory=list)  # Danh sách citations
    mode: str = "RAG"  # Mode: RAG, Co-RAG, hoặc BOTH
    
    def get_formatted_answer(self) -> str:
        """Trả về câu trả lời đã format với citations"""
        formatted = f"{self.answer}\n\n"
        if self.citations:
            formatted += "### 📚 Sources\n"
            for i, citation in enumerate(self.citations, 1):
                formatted += f"{i}. {citation}\n"
                if citation.excerpt:
                    formatted += f"   > {citation.excerpt}\n"
        return formatted
