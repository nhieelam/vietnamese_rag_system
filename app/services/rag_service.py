from typing import Dict, Any, List

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.config import AIConfig
from app.utils.logger import logger


class RAGService:
    def __init__(self):
        AIConfig.validate()

        self.embedding = self._init_embedding()
        self.vector_db = self._init_vector_store()
        self.llm = self._init_llm()
        self.prompt = self._init_prompt()

    # ---------- Init components ----------

    def _init_embedding(self):
        if AIConfig.LLM_PROVIDER == "openai":
            return OpenAIEmbeddings(
                model=AIConfig.OPENAI_EMBEDDING_MODEL,
                api_key=AIConfig.OPENAI_API_KEY,
            )

        raise ValueError(f"Unsupported embedding provider: {AIConfig.LLM_PROVIDER}")

    def _init_vector_store(self):
        return Chroma(
            persist_directory=AIConfig.VECTOR_STORE_DIR,
            embedding_function=self.embedding,
        )

    def _init_llm(self):
        if AIConfig.LLM_PROVIDER == "openai":
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

        raise ValueError(f"Unsupported LLM provider: {AIConfig.LLM_PROVIDER}")

    def _init_prompt(self):
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

    # ---------- Utils ----------

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    # ---------- Public API ----------
    @classmethod
    def get_answer(cls, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return cls._error(400, "Query cannot be empty")

        try:
            retriever = cls.vector_db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)

            if not docs:
                logger.warning("No relevant documents found")
                return cls._error(
                    404,
                    "No relevant documents found. Please upload documents first.",
                    metadata={"retrieved_docs_count": 0},
                )

            rag_chain = (
                {
                    "context": retriever | cls._format_docs,
                    "question": RunnablePassthrough(),
                }
                | cls.prompt
                | cls.llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(query)

            if not answer or not answer.strip():
                return cls._error(500, "LLM returned empty answer")

            return {
                "status_code": 200,
                "answer": answer.strip(),
                "message": "Answer generated successfully",
                "metadata": {
                    "retrieved_docs_count": len(docs),
                    "answer_length": len(answer.strip()),
                },
            }

        except Exception as e:
            logger.exception("Error while generating answer")
            return cls._error(500, str(e))

    # ---------- Response helpers ----------

    @staticmethod
    def _error(status_code: int, message: str, metadata: Dict[str, Any] = None):
        return {
            "status_code": status_code,
            "answer": None,
            "message": message,
            "metadata": metadata or {},
        }
