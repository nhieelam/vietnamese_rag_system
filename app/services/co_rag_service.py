"""
Co-RAG: collaborative retrieval — decompose the user question into sub-queries,
retrieve per sub-query, merge/deduplicate chunks, then generate one grounded answer.
"""

from app.service.file_service import FileService
from app.service.embedding_service import EmbeddingService
from app.service.rag_service import RAGService
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import AIConfig
from app.services.session_service import SessionService
from app.utils.logger import logger


class CoRAGService:
    _K_PER_SUBQUERY = 6
    _MAX_SUB_QUERIES = 4
    _FALLBACK_K = 12

    @classmethod
    def _decompose_prompt(cls) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
Bạn là bộ phân tích truy vấn cho hệ thống tìm kiếm tài liệu.

Nhiệm vụ: từ MỘT câu hỏi người dùng, tạo tối đa {max_q} câu hỏi con ngắn, cụ thể để truy xuất đoạn văn liên quan trong kho tài liệu.
- Giữ nguyên ngôn ngữ của câu gốc (tiếng Việt nếu câu hỏi là tiếng Việt).
- Mỗi dòng CHỈ chứa một câu hỏi con, không đánh số, không giải thích, không tiêu đề.
- Dòng đầu tiên nên là một diễn đạt gần với câu hỏi gốc (để không mất ngữ nghĩa).
- Nếu câu hỏi đã đủ đơn giản, chỉ cần một dòng trùng ý với câu hỏi.

Câu hỏi gốc:
{question}

Câu hỏi con (mỗi dòng một câu):
""".strip()
        )

    @classmethod
    def _answer_prompt(cls) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
Bạn là một trợ lý AI hữu ích.

Hãy sử dụng thông tin trong ngữ cảnh để trả lời câu hỏi.
Bạn có thể suy luận hợp lý từ thông tin có trong tài liệu,
nhưng không được bịa ra thông tin mới.

Nếu câu trả lời không được nêu trực tiếp nhưng có thể suy ra
một cách hợp lý từ tài liệu, hãy trả lời và nói rõ là
"Dựa trên thông tin trong tài liệu, có thể suy ra rằng ...".

Nếu hoàn toàn không tìm thấy thông tin liên quan trong tài liệu,
hãy nói:
"Tôi không tìm thấy thông tin này trong tài liệu.".

Ngữ cảnh (có thể từ nhiều lần truy xuất, đã gộp):
{context}

Câu hỏi:
{question}

Trả lời:
""".strip()
        )



    @staticmethod
    def _doc_fingerprint(doc: Document) -> str:
        normalized = re.sub(r"\s+", " ", doc.page_content or "").strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def _parse_subqueries(cls, raw: str, original: str) -> List[str]:
        lines = [ln.strip() for ln in (raw or "").splitlines()]
        out: List[str] = []
        seen: Set[str] = set()
        for ln in lines:
            if not ln:
                continue
            key = ln.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(ln)
            if len(out) >= cls._MAX_SUB_QUERIES:
                break
        if not out:
            return [original.strip()]
        return out

    @classmethod
    def _generate_subqueries(cls, question: str) -> List[str]:
        llm = RAGService._init_llm()
        prompt = cls._decompose_prompt()
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke(
            {"question": question.strip(), "max_q": cls._MAX_SUB_QUERIES}
        )
        return cls._parse_subqueries(raw, question)

    @classmethod
    def _merge_unique_docs(cls, doc_lists: List[List[Document]]) -> List[Document]:
        seen_fp: Set[str] = set()
        merged: List[Document] = []
        for docs in doc_lists:
            for doc in docs:
                fp = cls._doc_fingerprint(doc)
                if fp in seen_fp:
                    continue
                seen_fp.add(fp)
                merged.append(doc)
        return merged

    @classmethod
    def _retrieve_for_queries(
        cls, retriever, queries: List[str]
    ) -> tuple[List[Document], List[List[Document]]]:
        per_query_docs: List[List[Document]] = []
        for q in queries:
            per_query_docs.append(retriever.invoke(q))
        merged = cls._merge_unique_docs(per_query_docs)
        return merged, per_query_docs

    @classmethod
    def get_answer(cls, query: str) -> Dict[str, Any]:
        if not query.strip():
            return cls._error(400, "Query is empty")

        vector_store = SessionService.get_vector_store()
        if not vector_store:
            return cls._error(400, "No documents uploaded yet")

        try:
            retriever = vector_store.as_retriever(
                search_kwargs={"k": cls._K_PER_SUBQUERY}
            )

            sub_queries = cls._generate_subqueries(query)
            docs, per_query_breakdown = cls._retrieve_for_queries(
                retriever, sub_queries
            )

            if not docs:
                wide = vector_store.as_retriever(
                    search_kwargs={"k": cls._FALLBACK_K}
                )
                docs = wide.invoke(query)
                sub_queries = [query.strip()]
                per_query_breakdown = [docs]

            if not docs:
                return cls._error(404, "No relevant documents found")

            rag_chain = (
                {
                    "context": lambda _: RAGService._format_docs(docs),
                    "question": RunnablePassthrough(),
                }
                | cls._answer_prompt()
                | RAGService._init_llm()
                | StrOutputParser()
            )

            answer = rag_chain.invoke(query)

            return {
                "status_code": 200,
                "answer": answer.strip(),
                "message": "OK",
                "metadata": {
                    "mode": "co_rag",
                    "sub_queries": sub_queries,
                    "retrieved_docs_count": len(docs),
                    "per_subquery_counts": [len(d) for d in per_query_breakdown],
                },
            }

        except Exception as e:
            logger.exception("Co-RAG failed")
            return cls._error(500, str(e))

    @staticmethod
    def _error(code: int, msg: str) -> Dict[str, Any]:
        return {
            "status_code": code,
            "answer": None,
            "message": msg,
            "metadata": {"mode": "co_rag"},
        }
