from typing import Any, Dict

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.config import AIConfig
from app.services.session_service import SessionService
from app.utils.logger import logger
from app.models.citation import Citation, AnswerWithCitations


class RAGService:

    @classmethod
    def _init_llm(cls):
        return AIConfig.get_llm_instance()

    @classmethod
    def _create_history_aware_retriever(cls, retriever):
        """
        Creates a chain that rephrases the user's question based on chat history
        to form a standalone question for document retrieval.
        """
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return create_history_aware_retriever(
            cls._init_llm(), retriever, contextualize_q_prompt
        )

    @classmethod
    def _create_document_chain(cls):
        """
        Creates a chain that answers a question based on a given context
        (retrieved documents) and chat history.
        """
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, just say "
            "that you don't know. Use three sentences maximum and keep the "
            "answer concise."
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

    @classmethod
    def get_answer(cls, query: str) -> Dict[str, Any]:
        if not query.strip():
            return cls._error(400, "Query is empty")

        vector_store = SessionService.get_vector_store()

        if not vector_store:
            return cls._error(400, "No documents uploaded yet")

        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 10})

            # 1. Create a history-aware retriever
            history_aware_retriever = cls._create_history_aware_retriever(retriever)

            # 2. Create the main document chain for answering
            question_answer_chain = cls._create_document_chain()

            # 3. Combine them into the final retrieval chain
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # 4. Wrap the chain to manage history automatically
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                SessionService.get_chat_history,  # Use the correct history method
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            # 5. Invoke the chain with the user's query
            result = conversational_rag_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": "default_session"}}
            )
            
            answer = result.get("answer", "")
            retrieved_docs = result.get("context", [])

            return {
                "status_code": 200,
                "answer": answer.strip(),
                "message": "OK",
                "metadata": {
                    "retrieved_docs_count": len(retrieved_docs)
                },
            }

        except Exception as e:
            logger.exception("RAG failed")
            return cls._error(500, str(e))

    @classmethod
    def get_answer_with_citations(cls, query: str) -> AnswerWithCitations:
        """Get answer with source citations"""
        if not query.strip():
            return AnswerWithCitations(
                answer="Query is empty",
                citations=[],
                mode="RAG"
            )

        vector_store = SessionService.get_vector_store()

        if not vector_store:
            return AnswerWithCitations(
                answer="No documents uploaded yet",
                citations=[],
                mode="RAG"
            )

        try:
            # Tìm kiếm documents liên quan
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            history_aware_retriever = cls._create_history_aware_retriever(retriever)
            
            # Lấy chat history từ session
            chat_history_obj = SessionService.get_chat_history("default_session")
            chat_history_messages = chat_history_obj.messages if hasattr(chat_history_obj, 'messages') else []
            
            # Lấy retrieved documents
            retrieved_docs = history_aware_retriever.invoke(
                {
                    "input": query,
                    "chat_history": chat_history_messages
                }
            )
            
            # Tạo citations từ retrieved documents
            citations = []
            for doc in retrieved_docs:
                metadata = doc.metadata or {}
                source_name = metadata.get('source', 'Unknown Document')
                
                citation = Citation(
                    source_name=source_name,
                    page_number=metadata.get('page', None),
                    chunk_id=metadata.get('chunk_id', None),
                    relevance_score=metadata.get('score', 0.0),
                    excerpt=doc.page_content[:200]  # Lấy 200 ký tự đầu
                )
                citations.append(citation)
            
            # Tạo prompt kèm context từ retrieved documents
            question_answer_chain = cls._create_document_chain()
            
            rag_chain = create_retrieval_chain(
                history_aware_retriever,
                question_answer_chain
            )
            
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                SessionService.get_chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            
            result = conversational_rag_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": "default_session"}}
            )
            
            answer = result.get("answer", "").strip()
            
            return AnswerWithCitations(
                answer=answer,
                citations=citations,
                mode="RAG"
            )
        
        except Exception as e:
            logger.exception("RAG with citations failed")
            return AnswerWithCitations(
                answer=f"Error generating response: {str(e)}",
                citations=[],
                mode="RAG"
            )

    @staticmethod
    def _error(code: int, msg: str):
        return {
            "status_code": code,
            "answer": None,
            "message": msg,
            "metadata": {},
        }
