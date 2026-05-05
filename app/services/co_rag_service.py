from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set
from operator import itemgetter

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import AIConfig, AppConfig
from app.services.rag_service import RAGService
from app.services.session_service import SessionService
from app.utils.logger import logger
from app.models.citation import Citation, AnswerWithCitations


class CoRAGService:

    @classmethod
    def _init_llm(cls):
        return AIConfig.get_llm_instance()

    @classmethod
    def _create_contextualize_chain(cls) -> Runnable:
        """
        Creates a chain to rephrase the user's question based on chat history
        to form a standalone question.
        """
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        return prompt | cls._init_llm() | StrOutputParser()


    @classmethod
    def _decompose_prompt(cls) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
You are a query decomposition component for a document retrieval system.

Task: From ONE user question, generate up to {max_q} short, specific sub-queries
that help retrieve relevant passages from the document store.

Rules:
- Keep exactly the same language as the original user question.
- Each line must contain exactly one sub-query.
- Do not add numbering, explanations, headers, or extra formatting.
- The first line should be closest to the original question wording.
- If the question is already simple enough, return only one line equivalent to the original question.

Original question:
{question}

Sub-queries (one per line):
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
            if len(out) >= AppConfig.CO_RAG_MAX_SUB_QUERIES:
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
            {"question": question.strip(), "max_q": AppConfig.CO_RAG_MAX_SUB_QUERIES}
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
    def _create_synthesis_chain(cls) -> Runnable:
        """
        Creates a chain to synthesize the final answer from the retrieved context
        and the original question.
        """
        template = """
        You are a helpful AI assistant.

        Synthesize a comprehensive answer from the provided context, which was
        retrieved based on multiple sub-queries.
        The context is a collection of passages from internal documents.

        Rules:
        - Use only the information from the provided context. Do not invent facts.
        - If the context does not contain relevant information, clearly state that
          the answer cannot be found in the documents.
        - Answer in the same language as the original question.
        - Structure the answer clearly. If the question has multiple parts,
          address each one.

        Context (merged from multiple retrieval rounds):
        {context}

        Original Question:
        {question}

        Final Answer:
        """.strip()
        prompt = PromptTemplate.from_template(template)
        return prompt | cls._init_llm() | StrOutputParser()

    @classmethod
    def _run_sub_queries(
        cls, retriever: BaseRetriever, sub_queries: List[str]
    ) -> Dict[str, Any]:
        """
        Executes retrieval for each sub-query and merges the results.
        """
        merged_docs, per_query_docs = cls._retrieve_for_queries(retriever, sub_queries)
        return {
            "sub_queries": sub_queries,
            "per_query_docs": per_query_docs,
            "context": RAGService._format_docs(merged_docs),
        }

    @classmethod
    @classmethod
    def get_answer_with_citations(cls, query: str) -> AnswerWithCitations:
        """Co-RAG with rich citations (page, offset, full chunk, inline [n])."""
        if not query.strip():
            return AnswerWithCitations(answer="Query is empty", citations=[], mode="Co-RAG")

        vector_store = SessionService.get_vector_store()
        if not vector_store:
            return AnswerWithCitations(
                answer="No documents uploaded yet", citations=[], mode="Co-RAG"
            )

        try:
            contextualize_chain = cls._create_contextualize_chain()
            chat_history_obj = SessionService.get_chat_history("default_session")
            chat_history_messages = getattr(chat_history_obj, "messages", []) or []

            standalone_question = contextualize_chain.invoke({
                "question": query,
                "chat_history": chat_history_messages,
            })

            sub_queries = cls._generate_subqueries(standalone_question)

            # Nếu user nhắc tên file cụ thể → giới hạn retrieval theo source
            target_sources = RAGService._detect_target_sources(standalone_question)
            if target_sources:
                logger.info(f"[Co-RAG] Detected target sources: {target_sources}")

            doc_filter, type_filter = RAGService._metadata_filters()
            need_overfetch = bool(target_sources or doc_filter or type_filter)

            retriever_mode = SessionService.get_retriever_mode()
            use_reranker = SessionService.get_use_reranker()

            scored_map: Dict[tuple, tuple] = {}
            for sq in sub_queries:
                if retriever_mode == "hybrid" and not target_sources:
                    from app.services.hybrid_retriever_service import HybridRetrieverService
                    base_k = AppConfig.CO_RAG_K_PER_SUBQUERY
                    raw_pairs = HybridRetrieverService.retrieve(sq, k=max(base_k, 6))
                    for doc, rel in raw_pairs:
                        meta = doc.metadata or {}
                        if target_sources and meta.get("source") not in target_sources:
                            continue
                        if not RAGService._passes_filters(doc, doc_filter, type_filter):
                            continue
                        key = (
                            meta.get("source"),
                            meta.get("document_id"),
                            meta.get("chunk_id"),
                            meta.get("char_start"),
                        )
                        prev = scored_map.get(key)
                        if prev is None or rel > prev[1]:
                            scored_map[key] = (doc, rel)
                    continue

                fetch_k = (
                    max(AppConfig.CO_RAG_K_PER_SUBQUERY * 5, 20)
                    if need_overfetch else AppConfig.CO_RAG_K_PER_SUBQUERY
                )
                try:
                    pairs = vector_store.similarity_search_with_score(sq, k=fetch_k)
                except Exception:
                    logger.exception("similarity_search_with_score failed in Co-RAG")
                    docs = vector_store.similarity_search(sq, k=fetch_k)
                    pairs = [(d, 0.0) for d in docs]

                for doc, dist in pairs:
                    meta = doc.metadata or {}
                    if target_sources and meta.get("source") not in target_sources:
                        continue
                    if not RAGService._passes_filters(doc, doc_filter, type_filter):
                        continue
                    try:
                        rel = 1.0 / (1.0 + float(dist))
                    except Exception:
                        rel = 0.0
                    key = (
                        meta.get("source"),
                        meta.get("document_id"),
                        meta.get("chunk_id"),
                        meta.get("char_start"),
                    )
                    prev = scored_map.get(key)
                    if prev is None or rel > prev[1]:
                        scored_map[key] = (doc, rel)

            scored = sorted(scored_map.values(), key=lambda x: x[1], reverse=True)
            if not target_sources:
                rp = SessionService.get_retrieval_params()
                scored = RAGService._diversify_by_source(scored, per_source=rp["per_source_k"])

            if use_reranker and scored:
                from app.services.reranker_service import RerankerService
                rp = SessionService.get_retrieval_params()
                scored = RerankerService.rerank(
                    standalone_question, scored, top_k=rp["k"]
                )

            citations = RAGService._docs_to_citations(scored)

            # Tổng hợp với context đã đánh số -> prompt đòi inline [n]
            indexed_parts = []
            for c in citations:
                header = f"[{c.ref_index}] Source: {c.source_name}"
                if c.page_number is not None:
                    header += f" · page {c.page_number}"
                indexed_parts.append(f"{header}\n{c.full_text}")
            context = "\n\n".join(indexed_parts)

            synthesis_chain = cls._create_synthesis_chain_with_citations()
            answer = synthesis_chain.invoke({
                "context": context,
                "question": standalone_question,
            })

            cleaned_answer, kept_citations = RAGService._sanitize_answer_citations(
                (answer or "").strip(), citations
            )

            chat_history_obj.add_user_message(query)
            chat_history_obj.add_ai_message(cleaned_answer)

            return AnswerWithCitations(
                answer=cleaned_answer,
                citations=kept_citations,
                mode="Co-RAG",
            )
        except Exception as e:
            logger.exception("Co-RAG with citations failed")
            return AnswerWithCitations(
                answer=f"Error generating response: {str(e)}",
                citations=[],
                mode="Co-RAG",
            )

    @classmethod
    def _create_synthesis_chain_with_citations(cls) -> Runnable:
        template = """
        You are a helpful AI assistant. Use ONLY the context below.
        Each context block is prefixed with a marker like [1], [2]. When you use
        information from a block, append its marker inline, e.g. "... theo quy định [2]".

        STRICT RULES for citation markers:
        - Only use marker numbers that actually appear in the context below.
        - Never invent a number. If unsure, omit the marker.
        - Attach a marker ONLY to the exact block that contains the information.
        - Do not reuse the same marker for information from a different block.

        If the answer is not in the context, clearly say so.
        Answer in the same language as the question.

        Context (merged from multiple retrieval rounds):
        {context}

        Original Question:
        {question}

        Final Answer (với tham chiếu [n]):
        """.strip()
        prompt = PromptTemplate.from_template(template)
        return prompt | cls._init_llm() | StrOutputParser()
