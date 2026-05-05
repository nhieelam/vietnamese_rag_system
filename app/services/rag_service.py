import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.config import AIConfig, AppConfig
from app.services.session_service import SessionService
from app.utils.logger import logger
from app.models.citation import Citation, AnswerWithCitations
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGService:

    @classmethod
    def _init_llm(cls):
        return AIConfig.get_llm_instance()

    @classmethod
    def _create_document_chain(cls):
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use ONLY the following pieces of retrieved context to answer "
            "the question. Each context block is prefixed with a marker like [1], [2]. "
            "When you use information from a block, append its marker inline, e.g. "
            "\"... theo quy định [2]\". "
            "STRICT RULES for citation markers:\n"
            "- Only use marker numbers that actually appear in the context below.\n"
            "- Never invent a number. If unsure, omit the marker.\n"
            "- Attach a marker ONLY to the exact block that contains the information.\n"
            "- Do not reuse the same marker for information from a different block.\n"
            "If the answer is not in the context, say you don't know. "
            "Answer in the same language as the question."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return create_stuff_documents_chain(cls._init_llm(), qa_prompt)

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _metadata_filters() -> Tuple[List[str], List[str]]:
        """Đọc filter từ session (doc names + file types)."""
        try:
            return (
                SessionService.get_doc_filter(),
                SessionService.get_file_type_filter(),
            )
        except Exception:
            return [], []

    @staticmethod
    def _passes_filters(
        doc: Document,
        doc_filter: List[str],
        type_filter: List[str],
        filter_source: Optional[str] = None,
    ) -> bool:
        meta = doc.metadata or {}
        src = meta.get("source")
        if filter_source and src != filter_source:
            return False
        if doc_filter and src not in doc_filter:
            return False
        if type_filter and meta.get("file_type") not in type_filter:
            return False
        return True

    @classmethod
    def _score_docs(
        cls,
        vector_store,
        query: str,
        k: Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Trả (doc, relevance 0..1). Lọc theo session filters + `filter_source`."""
        if k is None:
            k = AppConfig.DEFAULT_RETRIEVAL_K
        doc_filter, type_filter = cls._metadata_filters()
        need_overfetch = bool(filter_source or doc_filter or type_filter)
        m = AppConfig.RETRIEVAL_OVERFETCH_MULTIPLIER
        mn = AppConfig.RETRIEVAL_OVERFETCH_MIN_DOCS
        fetch_k = max(k * m, mn) if need_overfetch else k
        try:
            pairs = vector_store.similarity_search_with_score(query, k=fetch_k)
        except Exception:
            logger.exception("similarity_search_with_score failed, fallback")
            docs = vector_store.similarity_search(query, k=fetch_k)
            pairs = [(d, 0.0) for d in docs]

        results: List[Tuple[Document, float]] = []
        for doc, dist in pairs:
            if not cls._passes_filters(doc, doc_filter, type_filter, filter_source):
                continue
            try:
                rel = 1.0 / (1.0 + float(dist))
            except Exception:
                rel = 0.0
            results.append((doc, rel))
            if len(results) >= k:
                break
        return results

    @staticmethod
    def _detect_target_sources(query: str) -> List[str]:
        """Phát hiện tên file người dùng nhắc trong câu hỏi."""
        docs = SessionService.get_documents() or []
        names = [d.get("name") for d in docs if d.get("name")]
        if not names:
            return []
        q_low = query.lower()
        hits: List[str] = []
        for name in names:
            if not name:
                continue
            n_low = name.lower()
            stem = re.sub(r"\.[^.]+$", "", n_low)  # bỏ extension
            if n_low in q_low or (stem and re.search(rf"\b{re.escape(stem)}\b", q_low)):
                hits.append(name)
        return hits

    @staticmethod
    def _diversify_by_source(
        scored: List[Tuple[Document, float]],
        per_source: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Giới hạn số chunk mỗi file để nhiều nguồn cùng xuất hiện."""
        if per_source is None:
            per_source = AppConfig.DEFAULT_PER_SOURCE_K
        counts: Dict[str, int] = {}
        out: List[Tuple[Document, float]] = []
        for doc, score in scored:
            src = (doc.metadata or {}).get("source", "?")
            if counts.get(src, 0) >= per_source:
                continue
            counts[src] = counts.get(src, 0) + 1
            out.append((doc, score))
        return out

    @staticmethod
    def _docs_to_citations(scored_docs: List[Tuple[Document, float]]) -> List[Citation]:
        citations: List[Citation] = []
        seen = set()
        for idx, (doc, score) in enumerate(scored_docs, start=1):
            meta = doc.metadata or {}
            key = (
                meta.get("source"),
                meta.get("document_id"),
                meta.get("chunk_id"),
                meta.get("char_start"),
            )
            if key in seen:
                continue
            seen.add(key)

            excerpt = doc.page_content.strip().replace("\n", " ")
            cap = AppConfig.CITATION_EXCERPT_MAX_LEN
            if len(excerpt) > cap:
                excerpt = excerpt[:cap].rstrip() + "…"

            citations.append(Citation(
                source_name=meta.get("source", "Unknown Document"),
                document_id=meta.get("document_id"),
                page_number=meta.get("page"),
                char_start=meta.get("char_start"),
                char_end=meta.get("char_end"),
                chunk_id=meta.get("chunk_id"),
                relevance_score=float(score or 0.0),
                excerpt=excerpt,
                full_text=doc.page_content,
                ref_index=len(citations) + 1,
            ))
        return citations

    @staticmethod
    def _sanitize_answer_citations(
        answer: str, citations: List[Citation]
    ) -> Tuple[str, List[Citation]]:
        """Loại marker [n] không hợp lệ, chỉ giữ citations được tham chiếu,
        đánh số lại liên tục theo thứ tự xuất hiện trong answer."""
        if not answer or not citations:
            return answer or "", []

        valid_indices = {c.ref_index for c in citations if c.ref_index is not None}
        # Bỏ marker trỏ đến số không tồn tại (vd LLM bịa [7] nhưng chỉ có 5 nguồn)
        def _strip_invalid(m: "re.Match") -> str:
            try:
                n = int(m.group(1))
            except ValueError:
                return ""
            return m.group(0) if n in valid_indices else ""

        cleaned = re.sub(r"\[(\d+)\]", _strip_invalid, answer)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)

        used_order: List[int] = []
        for m in re.finditer(r"\[(\d+)\]", cleaned):
            n = int(m.group(1))
            if n in valid_indices and n not in used_order:
                used_order.append(n)

        if not used_order:
            # LLM không dùng marker nào hợp lệ → giữ toàn bộ citations, không đổi số
            return cleaned, citations

        remap = {old: new for new, old in enumerate(used_order, start=1)}
        cleaned = re.sub(
            r"\[(\d+)\]",
            lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else "",
            cleaned,
        )

        by_old = {c.ref_index: c for c in citations}
        kept: List[Citation] = []
        for old in used_order:
            c = by_old.get(old)
            if c is None:
                continue
            c.ref_index = remap[old]
            kept.append(c)
        return cleaned.strip(), kept

    @staticmethod
    def _build_indexed_context(citations: List[Citation]) -> List[Document]:
        """Gắn prefix [n] vào page_content để LLM có thể trích dẫn inline."""
        docs: List[Document] = []
        for c in citations:
            header = f"[{c.ref_index}] Source: {c.source_name}"
            if c.page_number is not None:
                header += f" · page {c.page_number}"
            text = f"{header}\n{c.full_text}"
            docs.append(Document(page_content=text, metadata={
                "ref_index": c.ref_index,
                "source": c.source_name,
                "page": c.page_number,
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
            }))
        return docs
    @classmethod
    def _query_rewrite_chain(cls):
        template = (
            "You are a query rewriting assistant for a document retrieval system.\n"
            "Rewrite the user question to make it clearer, more specific, and easier "
            "to retrieve relevant passages. Preserve the original language.\n"
            "If the question is already clear, return it unchanged.\n"
            "Return ONLY the rewritten question, no prefix, no explanation.\n\n"
            "Original question: {question}\n\n"
            "Rewritten question:"
        )
        

        return PromptTemplate.from_template(template) | cls._init_llm() | StrOutputParser()
    
    @classmethod
    def get_answer_with_citations(cls, query: str) -> AnswerWithCitations:
        if not query.strip():
            return AnswerWithCitations(
                answer="Query is empty",
                citations=[],
                mode=AppConfig.MESSAGE_MODE_RAG,
            )

        vector_store = SessionService.get_vector_store()
        if not vector_store:
            return AnswerWithCitations(
                answer="No documents uploaded yet",
                citations=[],
                mode=AppConfig.MESSAGE_MODE_RAG,
            )

        try:
            chat_history_obj = SessionService.get_chat_history("default_session")
            chat_history_messages = getattr(chat_history_obj, "messages", []) or []

            rephrased_query  = cls._query_rewrite_chain().invoke({"question": query}).strip()

            rp = SessionService.get_retrieval_params()
            k = rp["k"]
            per_source = rp["per_source_k"]

            retriever_mode = SessionService.get_retriever_mode()
            use_reranker = SessionService.get_use_reranker()
            # If reranker is on, fetch more candidates
            r_m = AppConfig.RERANK_CANDIDATE_MULTIPLIER
            r_n = AppConfig.RERANK_CANDIDATE_MIN
            fetch_k = max(k * r_m, r_n) if use_reranker else k

            # Phát hiện câu hỏi nhắc đích danh tên file → lọc theo source
            target_sources = cls._detect_target_sources(query)
            if target_sources:
                logger.info(f"Detected target sources in query: {target_sources}")
                scored: List[Tuple[Document, float]] = []
                for src in target_sources:
                    scored.extend(
                        cls._score_docs(
                            vector_store,
                            query,
                            k=per_source,
                            filter_source=src,
                        )
                    )
                scored.sort(key=lambda x: x[1], reverse=True)
            elif retriever_mode == "hybrid":
                from app.services.hybrid_retriever_service import HybridRetrieverService
                scored = HybridRetrieverService.retrieve(rephrased_query, k=fetch_k)
                scored = cls._diversify_by_source(scored, per_source=per_source)
            else:
                scored = cls._score_docs(vector_store, rephrased_query, k=fetch_k)
                scored = cls._diversify_by_source(scored, per_source=per_source)

            if use_reranker and scored:
                from app.services.reranker_service import RerankerService
                scored = RerankerService.rerank(rephrased_query, scored, top_k=k)

            citations = cls._docs_to_citations(scored)

            # 3. Gọi LLM với context đã đánh số để trả lời có [n]
            indexed_docs = cls._build_indexed_context(citations)
            qa_chain = cls._create_document_chain()
            answer = qa_chain.invoke({
                "input": rephrased_query,
                "chat_history": chat_history_messages,
                "context": indexed_docs,
            })

            cleaned_answer, kept_citations = cls._sanitize_answer_citations(
                (answer or "").strip(), citations
            )

            # 4. Cập nhật lịch sử hội thoại
            chat_history_obj.add_user_message(query)
            chat_history_obj.add_ai_message(cleaned_answer)

            return AnswerWithCitations(
                answer=cleaned_answer,
                citations=kept_citations,
                mode=AppConfig.MESSAGE_MODE_RAG,
            )
        except Exception as e:
            logger.exception("RAG with citations failed")
            return AnswerWithCitations(
                answer=f"Error generating response: {str(e)}",
                citations=[],
                mode=AppConfig.MESSAGE_MODE_RAG,
            )
