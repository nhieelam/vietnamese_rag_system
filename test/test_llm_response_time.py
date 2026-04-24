"""
Test thời gian phản hồi của mô hình LLM khi thay đổi tham số k
k: số lượng tài liệu được truy xuất từ vector store
"""

import time
import pdfplumber
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.services.text_splitter_service import TextSplitterService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.session_service import SessionService
from app.config.ai_config import AIConfig
from app.utils.logger import logger


class LLMResponseTimeTest:
    """Test lớp để đo thời gian phản hồi của LLM với các giá trị k khác nhau"""

    def __init__(self, pdf_path: str):
        """
        Khởi tạo test
        Args:
            pdf_path: Đường dẫn đến file PDF để test
        """
        self.pdf_path = pdf_path
        self.results = []
        self.vector_store = None

    def extract_text_from_pdf(self) -> str:
        """Extract text từ PDF file"""
        logger.info(f"📄 Extracting text from PDF: {self.pdf_path}")
        extracted_text = ""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                logger.info(f"   PDF có {len(pdf.pages)} trang")
                for page_num, page in enumerate(pdf.pages, 1):
                    extracted_text += page.extract_text() or ""
                    if page_num % 5 == 0:
                        logger.info(f"   ✓ Đã xử lý {page_num} trang")
        except Exception as e:
            logger.error(f"❌ Lỗi khi extract PDF: {e}")
            raise
        
        logger.info(f"✓ Extracted {len(extracted_text)} characters")
        return extracted_text

    def setup_vector_store(self) -> None:
        """Chuẩn bị vector store từ PDF"""
        logger.info("\n🔧 Chuẩn bị Vector Store...")
        
        # Extract text
        extracted_text = self.extract_text_from_pdf()
        
        # Split text into chunks
        logger.info("✂️ Chia text thành chunks...")
        chunks = TextSplitterService.split(extracted_text)
        logger.info(f"✓ Tạo {len(chunks)} chunks")
        
        # Build vector store
        logger.info("🔍 Build vector store...")
        embedding = EmbeddingService.get_huggingface_embedding()
        self.vector_store = VectorStoreService.build_from_chunks(
            chunks=chunks,
            embedding=embedding,
            metadata={"source": self.pdf_path}
        )
        logger.info("✓ Vector store sẵn sàng")

    def _init_llm(self):
        """Initialize LLM instance"""
        return AIConfig.get_llm_instance()

    def _create_history_aware_retriever(self, retriever):
        """Creates a chain that rephrases the user's question based on chat history"""
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
            self._init_llm(), retriever, contextualize_q_prompt
        )

    def _create_document_chain(self):
        """Creates a chain that answers a question based on retrieved documents"""
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
        return create_stuff_documents_chain(self._init_llm(), qa_prompt)

    def measure_response_time(self, query: str, k: int) -> Dict[str, Any]:
        """
        Đo thời gian phản hồi với một giá trị k cụ thể
        
        Args:
            query: Câu hỏi để test
            k: Số lượng tài liệu cần truy xuất
            
        Returns:
            Dict chứa thời gian và kết quả
        """
        logger.info(f"\n⏱️  Testing with k={k}: \"{query[:50]}...\"")
        
        try:
            # Thiết lập retriever với k tham số
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            
            # Đo thời gian truy xuất tài liệu
            retrieval_start = time.time()
            retrieved_docs = retriever.invoke(query)
            retrieval_time = time.time() - retrieval_start
            
            logger.info(f"   - Retrieval time: {retrieval_time:.4f}s (retrieved {len(retrieved_docs)} docs)")
            
            # Tạo history-aware retriever
            history_aware_retriever = self._create_history_aware_retriever(retriever)
            
            # Tạo document chain
            question_answer_chain = self._create_document_chain()
            
            # Tạo retrieval chain
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # Wrap với message history
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                SessionService.get_chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            
            # Đo toàn bộ thời gian phản hồi
            total_start = time.time()
            result = conversational_rag_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": "test_session"}}
            )
            total_time = time.time() - total_start
            
            answer = result.get("answer", "")
            retrieved_docs = result.get("context", [])
            
            logger.info(f"   - Total response time: {total_time:.4f}s")
            logger.info(f"   - Answer length: {len(answer)} characters")
            
            return {
                "k": k,
                "query": query,
                "retrieval_time": retrieval_time,
                "total_time": total_time,
                "answer_length": len(answer),
                "retrieved_docs_count": len(retrieved_docs),
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"   ❌ Error with k={k}: {str(e)}")
            return {
                "k": k,
                "query": query,
                "retrieval_time": None,
                "total_time": None,
                "answer_length": 0,
                "retrieved_docs_count": 0,
                "success": False,
                "error": str(e)
            }

    def run_test(
        self,
        queries: List[str],
        k_values: List[int],
        output_file: str = "llm_response_time_results.csv"
    ) -> pd.DataFrame:
        """
        Chạy test toàn bộ với nhiều câu hỏi và giá trị k khác nhau
        
        Args:
            queries: Danh sách các câu hỏi để test
            k_values: Danh sách các giá trị k cần test
            output_file: Tên file CSV để lưu kết quả
            
        Returns:
            DataFrame chứa toàn bộ kết quả
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 BẮT ĐẦU TEST THỜI GIAN PHẢN HỒI LLM")
        logger.info("="*70)
        
        # Chuẩn bị vector store
        self.setup_vector_store()
        
        # Chạy test cho mỗi k value và query
        logger.info(f"\n📊 Test Configuration:")
        logger.info(f"   - Số câu hỏi: {len(queries)}")
        logger.info(f"   - Giá trị k: {k_values}")
        logger.info(f"   - Tổng số test: {len(queries) * len(k_values)}")
        
        logger.info("\n" + "-"*70)
        logger.info("Bắt đầu chạy test...")
        logger.info("-"*70)
        
        total_tests = len(queries) * len(k_values)
        current_test = 0
        
        for query in queries:
            for k in k_values:
                current_test += 1
                logger.info(f"\n[{current_test}/{total_tests}]")
                result = self.measure_response_time(query, k)
                self.results.append(result)
        
        # Chuyển kết quả thành DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Lưu kết quả vào CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_with_timestamp = output_file.replace(".csv", f"_{timestamp}.csv")
        df_results.to_csv(output_file_with_timestamp, index=False)
        logger.info(f"\n✓ Kết quả lưu vào: {output_file_with_timestamp}")
        
        return df_results

    def analyze_results(self, df_results: pd.DataFrame) -> None:
        """
        Phân tích kết quả và in ra báo cáo
        
        Args:
            df_results: DataFrame chứa kết quả test
        """
        logger.info("\n" + "="*70)
        logger.info("📈 PHÂN TÍCH KẾT QUẢ")
        logger.info("="*70)
        
        # Lọc chỉ các test thành công
        successful_results = df_results[df_results['success'] == True].copy()
        
        if len(successful_results) == 0:
            logger.error("❌ Không có test nào thành công")
            return
        
        logger.info("\n📊 Thống kê chung:")
        logger.info(f"   - Tổng test: {len(df_results)}")
        logger.info(f"   - Thành công: {len(successful_results)}")
        logger.info(f"   - Thất bại: {len(df_results) - len(successful_results)}")
        
        # Phân tích theo k value
        logger.info("\n📋 Thống kê theo giá trị k:")
        logger.info("-"*70)
        logger.info(f"{'k':<5} {'Avg Time (s)':<15} {'Min Time (s)':<15} {'Max Time (s)':<15} {'Tests':<8}")
        logger.info("-"*70)
        
        for k_value in sorted(successful_results['k'].unique()):
            k_results = successful_results[successful_results['k'] == k_value]
            avg_time = k_results['total_time'].mean()
            min_time = k_results['total_time'].min()
            max_time = k_results['total_time'].max()
            count = len(k_results)
            
            logger.info(f"{k_value:<5} {avg_time:<15.4f} {min_time:<15.4f} {max_time:<15.4f} {count:<8}")
        
        logger.info("-"*70)
        
        # Phân tích chi tiết hơn
        logger.info("\n🔍 Phân tích chi tiết:")
        
        # Thời gian truy xuất vs thời gian LLM
        successful_results['llm_time'] = (
            successful_results['total_time'] - successful_results['retrieval_time']
        )
        
        logger.info("\nThời gian truy xuất tài liệu (retrieval):")
        for k_value in sorted(successful_results['k'].unique()):
            k_results = successful_results[successful_results['k'] == k_value]
            avg_retrieval = k_results['retrieval_time'].mean()
            logger.info(f"   k={k_value}: {avg_retrieval:.4f}s")
        
        logger.info("\nThời gian LLM xử lý (LLM_time = total_time - retrieval_time):")
        for k_value in sorted(successful_results['k'].unique()):
            k_results = successful_results[successful_results['k'] == k_value]
            avg_llm = k_results['llm_time'].mean()
            logger.info(f"   k={k_value}: {avg_llm:.4f}s")
        
        # Hiệu suất
        logger.info("\n⚡ Hiệu suất:")
        min_k_time = successful_results.groupby('k')['total_time'].mean().min()
        max_k_time = successful_results.groupby('k')['total_time'].mean().max()
        improvement = ((max_k_time - min_k_time) / max_k_time) * 100
        logger.info(f"   - Độ cải thiện từ k nhỏ nhất đến k lớn nhất: {improvement:.2f}%")
        
        logger.info("\n" + "="*70)




def run_response_time_test():
    """Chạy test thời gian phản hồi với các câu hỏi tiếng Việt"""
    
    # Đường dẫn file PDF để test
    pdf_path = "sample.pdf"  # Thay đổi theo file thực tế của bạn
    
    # Danh sách các câu hỏi test (tiếng Việt)
    test_queries = [
        "Tài liệu này nói về cái gì?",
        "Hãy tóm tắt nội dung chính của tài liệu?",
        "Đây là tài liệu về lĩnh vực nào?",
    ]
    
    # Giá trị k để test (số lượng tài liệu truy xuất)
    k_values = [3, 5, 10, 15, 20]
    
    try:
        # Tạo test instance
        tester = LLMResponseTimeTest(pdf_path=pdf_path)
        
        # Chạy test
        results_df = tester.run_test(
            queries=test_queries,
            k_values=k_values,
            output_file="llm_response_time_results.csv"
        )
        
        # Phân tích kết quả
        tester.analyze_results(results_df)
        
        logger.info("\n" + "="*70)
        logger.info("✅ TEST HOÀN TẤT")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"\n❌ Lỗi trong quá trình test: {e}", exc_info=True)


if __name__ == "__main__":
    run_response_time_test()
