from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import AppConfig
from app.services.rag_service import RAGService
from app.services.session_service import SessionService
from app.utils.logger import logger


class CoRAGService:

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

    @classmethod
    def _answer_prompt(cls) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
You are a helpful AI assistant.

Use only the information from the provided context to answer the user question.
You may make reasonable inferences from the context, but never invent new facts.
Always answer in the same language as the user question.

If the answer is not stated directly but can be inferred, clearly say that it is
an inference from the provided documents.
If the context does not contain relevant information, clearly say that the information
is not found in the provided documents.

Context (merged from multiple retrieval rounds):
{context}

Question:
{question}

Answer:
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
    def get_answer(cls, query: str) -> Dict[str, Any]:
        if not query.strip():
            return cls._error(400, "Query is empty")

        vector_store = SessionService.get_vector_store()
        if not vector_store:
            return cls._error(400, "No documents uploaded yet")

        try:
            retriever = vector_store.as_retriever(
                search_kwargs={"k": AppConfig.CO_RAG_K_PER_SUBQUERY}
            )

            sub_queries = cls._generate_subqueries(query)
            docs, per_query_breakdown = cls._retrieve_for_queries(
                retriever, sub_queries
            )

            if not docs:
                wide = vector_store.as_retriever(
                    search_kwargs={"k": AppConfig.CO_RAG_FALLBACK_K}
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
