"""Self-RAG (8.2.10).

Flow:
  1. Query rewriting  — LLM rephrases user question for better retrieval.
  2. Retrieve         — Hybrid / vector + optional rerank.
  3. Relevance grade  — LLM tags each retrieved doc yes/no.
  4. Answer generate  — LLM answers using only relevant docs (inline [n]).
  5. Self-evaluate    — LLM rates grounded-ness & completeness (0..1).
  6. Multi-hop        — if confidence < threshold and hops < max, rewrite and loop.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.config import AIConfig, AppConfig
from app.models.citation import AnswerWithCitations, Citation
from app.services.rag_service import RAGService
from app.services.session_service import SessionService
from app.utils.logger import logger


class SelfRAGService:

    @classmethod
    def _init_llm(cls):
        return AIConfig.get_llm_instance()

    # ---------------- Prompts ----------------

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
    def _relevance_grade_chain(cls):
        template = (
            "You grade whether a document passage is relevant to a question.\n"
            "Reply with exactly one word: 'yes' or 'no'.\n\n"
            "Question: {question}\n\n"
            "Passage:\n{passage}\n\n"
            "Relevant (yes/no):"
        )
        return PromptTemplate.from_template(template) | cls._init_llm() | StrOutputParser()

    @classmethod
    def _answer_chain(cls):
        template = (
            "You are a helpful AI assistant.\n"
            "Use ONLY the context below to answer the question. Each context block is "
            "prefixed with a marker like [1], [2]. When you use information from a "
            "block, append its marker inline, e.g. \"... theo quy định [2]\".\n"
            "STRICT RULES for citation markers:\n"
            "- Only use marker numbers that actually appear in the context below.\n"
            "- Never invent a number. If unsure, omit the marker.\n"
            "- Attach a marker ONLY to the exact block that contains the information.\n"
            "- Do not reuse the same marker for information from a different block.\n"
            "If the context does not contain the answer, clearly say so.\n"
            "Answer in the same language as the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer (với tham chiếu [n]):"
        )
        return PromptTemplate.from_template(template) | cls._init_llm() | StrOutputParser()

    @classmethod
    def _self_eval_chain(cls):
        template = (
            "You evaluate an AI answer against the source context.\n"
            "Return a STRICT JSON object with two numeric fields in [0,1]:\n"
            "  - grounded: how well every claim is supported by the context\n"
            "  - completeness: how fully the answer addresses the question\n"
            "No prose, no markdown, only JSON.\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:\n{answer}\n\n"
            "JSON:"
        )
        return PromptTemplate.from_template(template) | cls._init_llm() | StrOutputParser()

    @classmethod
    def _followup_chain(cls):
        template = (
            "Given an original question and a partial answer, generate ONE short "
            "follow-up sub-question that would help fill gaps or verify details. "
            "Keep the same language. Return only the sub-question, no prefix.\n\n"
            "Original question: {question}\n\n"
            "Partial answer: {answer}\n\n"
            "Follow-up question:"
        )
        return PromptTemplate.from_template(template) | cls._init_llm() | StrOutputParser()

    # ---------------- Helpers ----------------

    @staticmethod
    def _parse_yes(s: str) -> bool:
        return bool(re.search(r"\byes\b", (s or "").strip().lower()))

    @staticmethod
    def _parse_eval(s: str) -> Tuple[float, float]:
        if not s:
            return 0.0, 0.0
        # Trích JSON ra khỏi text (model có thể kèm prefix)
        m = re.search(r"\{[^{}]*\}", s, re.DOTALL)
        raw = m.group(0) if m else s
        try:
            data = json.loads(raw)
            g = float(data.get("grounded", 0.0))
            c = float(data.get("completeness", 0.0))
            return max(0.0, min(1.0, g)), max(0.0, min(1.0, c))
        except Exception:
            # Fallback: regex floats
            nums = re.findall(r"0?\.\d+|1\.0|1|0", raw)
            try:
                g = float(nums[0]) if nums else 0.0
                c = float(nums[1]) if len(nums) > 1 else 0.0
                return max(0.0, min(1.0, g)), max(0.0, min(1.0, c))
            except Exception:
                return 0.0, 0.0

    @classmethod
    def _retrieve(cls, query: str, k: int) -> List[Tuple[Document, float]]:
        """Dùng hybrid/vector + optional rerank, tôn trọng session filters."""
        vector_store = SessionService.get_vector_store()
        if vector_store is None:
            return []
        retriever_mode = SessionService.get_retriever_mode()
        use_reranker = SessionService.get_use_reranker()
        fetch_k = max(k * 3, 15) if use_reranker else k

        if retriever_mode == "hybrid":
            from app.services.hybrid_retriever_service import HybridRetrieverService
            scored = HybridRetrieverService.retrieve(query, k=fetch_k)
        else:
            scored = RAGService._score_docs(vector_store, query, k=fetch_k)

        rp = SessionService.get_retrieval_params()
        scored = RAGService._diversify_by_source(scored, per_source=rp["per_source_k"])

        if use_reranker and scored:
            from app.services.reranker_service import RerankerService
            scored = RerankerService.rerank(query, scored, top_k=k)
        else:
            scored = scored[:k]
        return scored

    @classmethod
    def _grade_relevance(
        cls, question: str, scored: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Lọc chỉ giữ doc được LLM đánh giá relevant."""
        if not scored:
            return []
        grader = cls._relevance_grade_chain()
        kept: List[Tuple[Document, float]] = []
        for doc, score in scored:
            passage = (doc.page_content or "")[:1500]
            try:
                verdict = grader.invoke({"question": question, "passage": passage})
            except Exception:
                logger.exception("Relevance grading failed, keep doc by default")
                kept.append((doc, score))
                continue
            if cls._parse_yes(verdict):
                kept.append((doc, score))
        if not kept:
            # Safety net: nếu LLM từ chối hết, giữ top 3 để tránh không có context
            logger.warning("Self-RAG: all docs graded irrelevant, fallback to top-3")
            kept = scored[:3]
        return kept

    @staticmethod
    def _build_indexed_context(citations: List[Citation]) -> str:
        parts = []
        for c in citations:
            header = f"[{c.ref_index}] Source: {c.source_name}"
            if c.page_number is not None:
                header += f" · page {c.page_number}"
            parts.append(f"{header}\n{c.full_text}")
        return "\n\n".join(parts)

    # ---------------- Main ----------------

    @classmethod
    def get_answer_with_citations(cls, query: str) -> AnswerWithCitations:
        if not query.strip():
            return AnswerWithCitations(
                answer="Query is empty", citations=[], mode="Self-RAG"
            )

        if SessionService.get_vector_store() is None:
            return AnswerWithCitations(
                answer="No documents uploaded yet", citations=[], mode="Self-RAG"
            )

        t0 = time.perf_counter()

        try:
            # 1) Query rewriting
            try:
                rewritten = cls._query_rewrite_chain().invoke({"question": query}).strip()
                if not rewritten or len(rewritten) < 3:
                    rewritten = query
            except Exception:
                logger.exception("Query rewriting failed, using original")
                rewritten = query
            logger.info(f"[Self-RAG] rewritten: {rewritten!r}")

            rp = SessionService.get_retrieval_params()
            k = rp["k"]

            # 2) Retrieve
            scored = cls._retrieve(rewritten, k=k)

            # 3) Relevance grading
            relevant = cls._grade_relevance(rewritten, scored)
            citations = RAGService._docs_to_citations(relevant)

            # 4) Generate answer
            context = cls._build_indexed_context(citations) or "(empty)"
            try:
                answer = cls._answer_chain().invoke({
                    "context": context,
                    "question": rewritten,
                }).strip()
            except Exception as e:
                logger.exception("Answer generation failed")
                return AnswerWithCitations(
                    answer=f"Error generating response: {e}",
                    citations=citations,
                    mode="Self-RAG",
                    rewritten_query=rewritten,
                )

            # 5) Self-evaluation
            try:
                eval_raw = cls._self_eval_chain().invoke({
                    "question": rewritten,
                    "context": context,
                    "answer": answer,
                })
                grounded, completeness = cls._parse_eval(eval_raw)
            except Exception:
                logger.exception("Self-evaluation failed")
                grounded, completeness = 0.5, 0.5

            mean_rel = (
                sum(s for _, s in relevant) / len(relevant) if relevant else 0.0
            )
            confidence = (grounded + completeness) / 2.0 * (0.5 + 0.5 * mean_rel)
            confidence = max(0.0, min(1.0, confidence))
            logger.info(
                f"[Self-RAG] hop=1 grounded={grounded:.2f} "
                f"completeness={completeness:.2f} mean_rel={mean_rel:.2f} "
                f"confidence={confidence:.2f}"
            )

            hops = 1
            # 6) Multi-hop if low confidence
            if (
                confidence < AppConfig.SELF_RAG_CONFIDENCE_THRESHOLD
                and AppConfig.SELF_RAG_MAX_HOPS > 1
            ):
                try:
                    followup = cls._followup_chain().invoke({
                        "question": rewritten,
                        "answer": answer,
                    }).strip()
                except Exception:
                    logger.exception("Followup generation failed")
                    followup = ""

                if followup and followup.lower() != rewritten.lower():
                    logger.info(f"[Self-RAG] hop=2 followup: {followup!r}")
                    extra_scored = cls._retrieve(followup, k=k)
                    extra_relevant = cls._grade_relevance(followup, extra_scored)

                    # Merge unique citations
                    existing_keys = {
                        (c.source_name, c.document_id, c.chunk_id, c.char_start)
                        for c in citations
                    }
                    merged = list(relevant)
                    for d, s in extra_relevant:
                        m = d.metadata or {}
                        key = (
                            m.get("source"),
                            m.get("document_id"),
                            m.get("chunk_id"),
                            m.get("char_start"),
                        )
                        if key not in existing_keys:
                            merged.append((d, s))
                            existing_keys.add(key)

                    citations = RAGService._docs_to_citations(merged)
                    context = cls._build_indexed_context(citations) or context
                    try:
                        answer = cls._answer_chain().invoke({
                            "context": context,
                            "question": rewritten,
                        }).strip()
                        eval_raw = cls._self_eval_chain().invoke({
                            "question": rewritten,
                            "context": context,
                            "answer": answer,
                        })
                        grounded, completeness = cls._parse_eval(eval_raw)
                        mean_rel = (
                            sum(s for _, s in merged) / len(merged) if merged else 0.0
                        )
                        confidence = (grounded + completeness) / 2.0 * (
                            0.5 + 0.5 * mean_rel
                        )
                        confidence = max(0.0, min(1.0, confidence))
                        hops = 2
                        logger.info(
                            f"[Self-RAG] hop=2 grounded={grounded:.2f} "
                            f"completeness={completeness:.2f} "
                            f"confidence={confidence:.2f}"
                        )
                    except Exception:
                        logger.exception("Multi-hop refinement failed, keep hop-1 answer")

            cleaned_answer, kept_citations = RAGService._sanitize_answer_citations(
                (answer or "").strip(), citations
            )

            # Cập nhật chat history
            try:
                chat_history_obj = SessionService.get_chat_history("default_session")
                chat_history_obj.add_user_message(query)
                chat_history_obj.add_ai_message(cleaned_answer)
            except Exception:
                logger.exception("Failed to update chat history in Self-RAG")

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"[Self-RAG] done in {elapsed:.0f}ms, hops={hops}")

            return AnswerWithCitations(
                answer=cleaned_answer,
                citations=kept_citations,
                mode="Self-RAG",
                confidence=confidence,
                rewritten_query=rewritten if rewritten != query else None,
                grounded_score=grounded,
                completeness_score=completeness,
                hops=hops,
            )
        except Exception as e:
            logger.exception("Self-RAG failed")
            return AnswerWithCitations(
                answer=f"Error generating response: {e}",
                citations=[],
                mode="Self-RAG",
            )
