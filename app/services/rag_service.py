from typing import Dict, Any

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.services.session_service import SessionService
from app.config import AIConfig
from app.utils.logger import logger


class RAGService:

    @classmethod
    def _init_llm(cls):
        if AIConfig.LLM_PROVIDER == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=AIConfig.OPENAI_LLM_MODEL,
                temperature=0.3,
                api_key=AIConfig.OPENAI_API_KEY,
            )

        if AIConfig.LLM_PROVIDER == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=AIConfig.GROQ_LLM_MODEL,
                api_key=AIConfig.GROQ_API_KEY,
                temperature=0.3,
            )

        raise ValueError("Unsupported LLM provider")

    @classmethod
    def _init_prompt(cls):
        return PromptTemplate.from_template(
            """
            Bạn là một trợ lý AI hữu ích.
            Hãy sử dụng thông tin trong ngữ cảnh để trả lời câu hỏi.
            Nếu không tìm thấy câu trả lời trong tài liệu, hãy nói:
            "Tôi không tìm thấy thông tin này trong tài liệu".

            Ngữ cảnh:
            {context}

            Câu hỏi:
            {question}

            Trả lời:
            """.strip()
        )

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ---------- PUBLIC ----------
    @classmethod
    def get_answer(cls, query: str) -> Dict[str, Any]:
        if not query.strip():
            return cls._error(400, "Query is empty")

        vector_store = SessionService.get_vector_store()

        if not vector_store:
            return cls._error(400, "No documents uploaded yet")

        try:
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 3}
            )

            docs = retriever.invoke(query)
            if not docs:
                return cls._error(404, "No relevant documents found")

            rag_chain = (
                {
                    "context": retriever | cls._format_docs,
                    "question": RunnablePassthrough(),
                }
                | cls._init_prompt()
                | cls._init_llm()
                | StrOutputParser()
            )

            answer = rag_chain.invoke(query)

            return {
                "status_code": 200,
                "answer": answer.strip(),
                "message": "OK",
                "metadata": {
                    "retrieved_docs_count": len(docs)
                },
            }

        except Exception as e:
            logger.exception("RAG failed")
            return cls._error(500, str(e))

    @staticmethod
    def _error(code: int, msg: str):
        return {
            "status_code": code,
            "answer": None,
            "message": msg,
            "metadata": {},
        }
