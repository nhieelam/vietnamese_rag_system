import time
import pdfplumber
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.text_splitter_service import TextSplitterService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.config.ai_config import AIConfig
from app.utils.logger import logger

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text từ PDF file"""
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        raise
    return extracted_text

def test_chunk_quality(pdf_path: str, chunk_size: int, overlap: int, test_questions: list):
    """Test chất lượng trả lời với một cấu hình chunk cụ thể"""
    from app.config import AppConfig
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🧪 Testing: Chunk Size = {chunk_size}, Overlap = {overlap}")
    logger.info(f"{'='*70}")
    
    # Lưu config gốc
    original_size = AppConfig.CHUNK_SIZE
    original_overlap = AppConfig.CHUNK_OVERLAP
    
    AppConfig.CHUNK_SIZE = chunk_size
    AppConfig.CHUNK_OVERLAP = overlap
    
    try:
        # 1. Extract PDF
        logger.info("📄 Extracting PDF...")
        extracted_text = extract_text_from_pdf(pdf_path)
        logger.info(f"✓ Extracted {len(extracted_text)} characters")
        
        # 2. Split text
        logger.info("✂️ Splitting text into chunks...")
        chunks = TextSplitterService.split(extracted_text)
        logger.info(f"✓ Created {len(chunks)} chunks")
        
        # 3. Build vector store
        logger.info("🔍 Building vector store...")
        start_time = time.time()
        embeddings = EmbeddingService.get_huggingface_embedding()
        vector_store = VectorStoreService.build_from_chunks(chunks, embeddings, {"source": "test_pdf"})
        vector_store_time = time.time() - start_time
        logger.info(f"✓ Vector store created in {vector_store_time:.2f}s")
        
        # 4. Test QA
        logger.info(f"❓ Testing {len(test_questions)} questions...\n")
        
        results = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": len(chunks),
            "avg_chunk_len": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
            "vector_store_time": vector_store_time,
            "qa_results": []
        }
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Khởi tạo LLM
        llm = AIConfig.get_llm_instance()
        
        # Tạo prompt cho QA
        qa_prompt = ChatPromptTemplate.from_template("""
Based on the following context, answer the question. If you don't know, say "I don't know".
Keep your answer concise (1-2 sentences).

Context:
{context}

Question: {question}
Answer:""")
        
        chain = qa_prompt | llm | StrOutputParser()
        
        for idx, question in enumerate(test_questions, 1):
            logger.info(f"Q{idx}: {question}")
            
            start_time = time.time()
            try:
                # Retrieve relevant chunks
                retrieved_docs = retriever.invoke(question)
                
                # Tạo context
                context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
                
                # Get answer
                answer = chain.invoke({
                    "context": context,
                    "question": question
                })
                
                qa_time = time.time() - start_time
                
                logger.info(f"⏱️ Time: {qa_time:.2f}s")
                logger.info(f"📌 Retrieved {len(retrieved_docs)} chunks")
                answer_preview = answer[:200] if len(answer) > 200 else answer
                logger.info(f"💬 Answer: {answer_preview}...\n")
                
                results["qa_results"].append({
                    "question": question,
                    "answer": answer,
                    "citations": len(retrieved_docs),  # số chunks retrieved
                    "qa_time": qa_time,
                    "retrieved_chunks": len(retrieved_docs)
                })
            except Exception as e:
                logger.error(f"❌ Error: {str(e)}\n")
                results["qa_results"].append({
                    "question": question,
                    "error": str(e)
                })
        
        return results
    finally:
        AppConfig.CHUNK_SIZE = original_size
        AppConfig.CHUNK_OVERLAP = original_overlap

if __name__ == "__main__":
    PDF_PATH = "d:\\Desktop\\vietnamese_rag_system\\sample.pdf"
    
    # Các câu hỏi test (thay đổi theo nội dung PDF của bạn)
    TEST_QUESTIONS = [
        "Tài liệu này nói về cái gì?",
        "Các thông tin chính là gì?",
        "Kết luận của tài liệu là gì?",
    ]
    
    logger.info("🚀 Bắt đầu test chất lượng QA với các chunk sizes khác nhau...")
    
    all_results = []
    configs = [
        (200, 20),
        (500, 50),
        (1000, 100),
    ]
    
    for chunk_size, overlap in configs:
        try:
            result = test_chunk_quality(PDF_PATH, chunk_size, overlap, TEST_QUESTIONS)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to test chunk_size={chunk_size}: {e}")
    
    # 📊 Tóm tắt chi tiết
    logger.info(f"\n{'='*70}")
    logger.info("📊 TÓMÚM TẤT KẾT QUẢ")
    logger.info(f"{'='*70}")
    logger.info(f"{'Chunk':<10} | {'Chunks':<10} | {'Avg Len':<10} | {'Avg QA Time':<12} | {'Citations':<10}")
    logger.info(f"{'-'*70}")
    
    for result in all_results:
        if result["qa_results"]:
            avg_qa_time = sum(
                r.get("qa_time", 0) for r in result["qa_results"] 
                if "qa_time" in r
            ) / len([r for r in result["qa_results"] if "qa_time" in r])
            
            total_citations = sum(
                r.get("citations", 0) for r in result["qa_results"] 
                if "citations" in r
            )
            
            logger.info(
                f"{result['chunk_size']:<10} | {result['num_chunks']:<10} | "
                f"{result['avg_chunk_len']:<10.0f} | {avg_qa_time:<12.3f}s | {total_citations:<10}"
            )
    
    logger.info(f"{'='*70}\n")
    logger.info("💡 Lựa chọn chunk size dựa trên:")
    logger.info("   ✅ Số citations cao = context tốt")
    logger.info("   ✅ QA time thấp = xử lý nhanh")
    logger.info("   ✅ Chunks vừa phải = cân bằng hiệu năng\n")
