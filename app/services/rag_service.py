import os
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from app.config import AppConfig
from app.utils.logger import logger 

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

class RAGService:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.vector_db = Chroma(
            persist_directory=AppConfig.DB_DIRECTORY,
            embedding_function=self.embedding_model
        )

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # or "gpt-4"
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 5. Define the Prompt Template (Vietnamese)
        self.prompt = PromptTemplate.from_template(
            """Bạn là một trợ lý AI hữu ích. Hãy sử dụng thông tin ngữ cảnh bên dưới để trả lời câu hỏi.
            Nếu bạn không biết câu trả lời từ ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu".
            
            Ngữ cảnh:
            {context}
            
            Câu hỏi: {question}
            Trả lời:"""
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_answer(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                'status_code': 400,
                'answer': None,
                'message': "Query cannot be empty",
                'metadata': {}
            }
        
        try:
            # Retrieve relevant documents (Top 3)
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
            
            # Get retrieved documents for metadata
            retrieved_docs = retriever.get_relevant_documents(query)
            
            if not retrieved_docs:
                logger.warning("No relevant documents found in vector database")
                return {
                    'status_code': 404,
                    'answer': None,
                    'message': "No relevant documents found in the database. Please upload documents first.",
                    'metadata': {
                        'retrieved_docs_count': 0,
                        'query': query[:100]
                    }
                }
            
            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
            
            # Build RAG chain
            rag_chain = (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer = rag_chain.invoke(query)
            
            if not answer or not answer.strip():
                return {
                    'status_code': 500,
                    'answer': None,
                    'message': "Failed to generate answer from the model",
                    'metadata': {
                        'retrieved_docs_count': len(retrieved_docs),
                        'query': query[:100]
                    }
                }
            
            logger.info(f"Successfully generated answer ({len(answer)} characters)")
            
            return {
                'status_code': 200,
                'answer': answer.strip(),
                'message': "Answer generated successfully",
                'metadata': {
                    'retrieved_docs_count': len(retrieved_docs),
                    'query': query[:100],
                    'answer_length': len(answer.strip())
                }
            }
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return {
                'status_code': 500,
                'answer': None,
                'message': error_msg,
                'metadata': {'query': query[:100]}
            }

_rag_service_instance = None

def get_rag_service():
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance





if __name__ == "__main__":
    rag = RAGService()
    response = rag.get_answer("Mức phạt vi phạm an toàn là bao nhiêu?")
    print("\nStatus Code:", response['status_code'])
    print("Message:", response['message'])
    if response['answer']:
        print("\nAI Answer:\n", response['answer'])
    print("\nMetadata:", response['metadata'])