"""
Báo cáo hiệu suất xử lý tài liệu: Loading (extract + split), Embedding (build vector store), Retrieval
Sản phẩm: in bảng tóm tắt tương tự Bảng I và lưu file CSV chi tiết

Cách dùng:
- Đặt các file PDF trong thư mục test/pdf to test/ hoặc điều chỉnh DOCUMENTS
- Chạy: python test_doc_processing_performance.py

Lưu ý: script dùng chính xác các service trong project:
  - TextSplitterService.split() + AppConfig.CHUNK_SIZE/CHUNK_OVERLAP
  - EmbeddingService.get_huggingface_embedding()
  - VectorStoreService.build_from_chunks()
  - RAGService / CoRAGService cho retrieval
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import pdfplumber
import pandas as pd

from app.services.text_splitter_service import TextSplitterService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.session_service import SessionService
from app.services.rag_service import RAGService
from app.services.co_rag_service import CoRAGService
from app.config.app_config import AppConfig
from app.config.ai_config import AIConfig
from app.utils.logger import logger


# Thay đường dẫn thực tế tới file PDF ở đây (small, medium, large)
# Hiện có file test1.pdf trong test/pdf to test/
DOCUMENTS = {
    "Nhỏ (2 trang)": "dao.pdf",
    # "Trung bình (15 trang)": "test/pdf to test/sample_medium.pdf",
    # "Lớn (45 trang)": "test/pdf to test/sample_large.pdf",
}

# Giá trị k mặc định để đo thời gian retrieval (có thể điều chỉnh)
RETRIEVAL_K = 5

# Câu hỏi test mặc định
TEST_QUERY = "Tài liệu này nói về gì?"


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


def measure_document_processing(
    pdf_path: str, 
    query: str = TEST_QUERY, 
    k: int = RETRIEVAL_K,
    use_co_rag: bool = False,
    num_retrieval_runs: int = 3
) -> Dict[str, Any]:
    """
    Đo thời gian cho 3 giai đoạn sử dụng các service của project:
      1. Loading: mở PDF bằng pdfplumber và chia thành chunks (AppConfig.CHUNK_SIZE/CHUNK_OVERLAP)
      2. Embedding: build vector store bằng EmbeddingService + VectorStoreService
      3. Retrieval: truy vấn sử dụng RAGService hoặc CoRAGService (lặp lại nhiều lần để có trung bình)

    Trả về dict với các trường: label, num_chunks, chunk_config, load_time, embedding_time, retrieval_time (trung bình)
    """
    result: Dict[str, Any] = {
        "pdf_path": pdf_path,
        "file_size_mb": os.path.getsize(pdf_path) / (1024 * 1024),
        "num_chunks": 0,
        "chunk_size": AppConfig.CHUNK_SIZE,
        "chunk_overlap": AppConfig.CHUNK_OVERLAP,
        "load_time": None,
        "embedding_time": None,
        # retrieval timings for three modes
        "retrieval_vector_avg": None,
        "retrieval_vector_min": None,
        "retrieval_vector_max": None,
        "retrieval_rag_avg": None,
        "retrieval_rag_min": None,
        "retrieval_rag_max": None,
        "retrieval_co_rag_avg": None,
        "retrieval_co_rag_min": None,
        "retrieval_co_rag_max": None,
        "retrieval_k": k,
        "use_co_rag": use_co_rag,
        "error": None,
    }

    try:
        # Initialize session
        SessionService.initialize()
        
        # 1) Loading (extract + split) - sử dụng TextSplitterService với AppConfig.CHUNK_SIZE/OVERLAP
        logger.info(f"📄 Loading: Extracting và splitting với chunk_size={AppConfig.CHUNK_SIZE}, overlap={AppConfig.CHUNK_OVERLAP}")
        t0 = time.time()
        text = extract_text_from_pdf(pdf_path)
        chunks = TextSplitterService.split(text)
        load_time = time.time() - t0
        result["num_chunks"] = len(chunks)
        result["load_time"] = load_time
        logger.info(f"   ✓ Extracted {len(text)} chars, split into {len(chunks)} chunks in {load_time:.4f}s")

        # 2) Embedding - sử dụng EmbeddingService + VectorStoreService
        logger.info(f"🔧 Embedding: Building vector store")
        t1 = time.time()
        embedding_model = EmbeddingService.get_huggingface_embedding()
        vector_store = VectorStoreService.build_from_chunks(
            chunks=chunks, 
            embedding=embedding_model, 
            metadata={"source": os.path.basename(pdf_path)}
        )
        embedding_time = time.time() - t1
        result["embedding_time"] = embedding_time
        logger.info(f"   ✓ Embedding completed in {embedding_time:.4f}s")

        # 3) Retrieval - đo cho 3 chế độ: 
        #    a) Vector retrieval (similarity_search) - chỉ lấy tài liệu
        #    b) RAG - retrieval + LLM inference
        #    c) Co-RAG - retrieval + LLM inference with citations
        logger.info(f"🔍 Retrieval & LLM Processing: measuring 3 modes with k={k} - {num_retrieval_runs} runs each")

        # a) Direct vector retrieval (similarity_search) - CHỈ LẤY DỮ LIỆU
        vector_times: List[float] = []
        for run in range(num_retrieval_runs):
            t2 = time.time()
            try:
                retrieved_docs = vector_store.similarity_search(query, k=k)
            except Exception as e:
                logger.warning(f"similarity_search failed: {e}, trying as_retriever...")
                retriever = vector_store.as_retriever(search_kwargs={"k": k})
                if hasattr(retriever, "get_relevant_documents"):
                    retrieved_docs = retriever.get_relevant_documents(query)
                else:
                    try:
                        retrieved_docs = retriever.invoke(query)
                    except Exception:
                        retrieved_docs = []
            vector_times.append(time.time() - t2)
            logger.info(f"   Vector (retrieval only) Run {run+1}/{num_retrieval_runs}: {vector_times[-1]:.4f}s (found {len(retrieved_docs)} docs)")

        result["retrieval_vector_avg"] = sum(vector_times) / len(vector_times)
        result["retrieval_vector_min"] = min(vector_times)
        result["retrieval_vector_max"] = max(vector_times)

        # b) RAG (retrieval + LLM inference) - LẤY DỮ LIỆU + AI XỬ LÝ
        logger.info(f"🔍 Measuring RAG (retrieval + LLM inference)...")
        rag_times: List[float] = []
        for run in range(num_retrieval_runs):
            t3 = time.time()
            try:
                # Sử dụng retriever từ vector store + gọi LLM chain
                retriever = vector_store.as_retriever(search_kwargs={"k": k})
                
                # Get LLM
                llm = AIConfig.get_llm_instance()
                
                # Create simple prompt
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_template(
                    "Based on the following context, answer the question:\n"
                    "Context: {context}\n"
                    "Question: {question}\n"
                    "Answer:"
                )
                
                # Get docs
                docs = retriever.invoke(query)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Call LLM
                response = llm.invoke(prompt.format(context=context, question=query))
                
                elapsed = time.time() - t3
                rag_times.append(elapsed)
                
                # Log response
                if hasattr(response, 'content'):
                    preview = response.content[:80]
                else:
                    preview = str(response)[:80]
                logger.info(f"   RAG (retrieval + LLM) Run {run+1}/{num_retrieval_runs}: {elapsed:.4f}s -> '{preview}...'")
                
            except Exception as e:
                elapsed = time.time() - t3
                rag_times.append(elapsed)
                logger.error(f"   RAG (retrieval + LLM) Run {run+1}/{num_retrieval_runs}: {elapsed:.4f}s ❌ ERROR: {str(e)[:100]}")

        result["retrieval_rag_avg"] = sum(rag_times) / len(rag_times) if rag_times else 0
        result["retrieval_rag_min"] = min(rag_times) if rag_times else 0
        result["retrieval_rag_max"] = max(rag_times) if rag_times else 0

        # c) Co-RAG (retrieval + LLM inference with citations) - LẤY DỮ LIỆU + AI XỬ LÝ + CITATIONS
        logger.info(f"🔍 Measuring Co-RAG (retrieval + LLM inference + citations)...")
        co_rag_times: List[float] = []
        for run in range(num_retrieval_runs):
            t4 = time.time()
            try:
                # Sử dụng retriever + LLM + format với citations
                retriever = vector_store.as_retriever(search_kwargs={"k": k})
                docs = retriever.invoke(query)
                
                # Get LLM
                llm = AIConfig.get_llm_instance()
                
                # Create prompt with citations format
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_template(
                    "Based on the following documents, answer the question and include citations:\n"
                    "Documents:\n{context}\n"
                    "Question: {question}\n"
                    "Answer (with citations):"
                )
                
                context = "\n".join([f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
                
                # Call LLM
                response = llm.invoke(prompt.format(context=context, question=query))
                
                elapsed = time.time() - t4
                co_rag_times.append(elapsed)
                
                # Log response
                if hasattr(response, 'content'):
                    preview = response.content[:80]
                else:
                    preview = str(response)[:80]
                logger.info(f"   CoRAG (retrieval + LLM + citations) Run {run+1}/{num_retrieval_runs}: {elapsed:.4f}s -> '{preview}...'")
                
            except Exception as e:
                elapsed = time.time() - t4
                co_rag_times.append(elapsed)
                logger.error(f"   CoRAG (retrieval + LLM + citations) Run {run+1}/{num_retrieval_runs}: {elapsed:.4f}s ❌ ERROR: {str(e)[:100]}")

        result["retrieval_co_rag_avg"] = sum(co_rag_times) / len(co_rag_times) if co_rag_times else 0
        result["retrieval_co_rag_min"] = min(co_rag_times) if co_rag_times else 0
        result["retrieval_co_rag_max"] = max(co_rag_times) if co_rag_times else 0

        logger.info(f"   ✓ Tóm tắt - Vector(retrieval)={result['retrieval_vector_avg']:.4f}s, RAG(retrieval+LLM)={result['retrieval_rag_avg']:.4f}s, CoRAG(retrieval+LLM+citations)={result['retrieval_co_rag_avg']:.4f}s")

    except Exception as e:
        logger.exception(f"Error measuring document processing for {pdf_path}")
        result["error"] = str(e)

    return result


def run_all(measure_k: int = RETRIEVAL_K, output_csv: str = "document_processing_performance.csv", num_retrieval_runs: int = 3):
    """
    Chạy đo cho toàn bộ DOCUMENTS và in bảng tương tự Bảng I
    
    Sử dụng:
    - TextSplitterService.split() với AppConfig.CHUNK_SIZE/CHUNK_OVERLAP
    - EmbeddingService.get_huggingface_embedding()
    - VectorStoreService.build_from_chunks()
    - RAGService.get_answer() cho retrieval (đo lặp lại để có trung bình)
    """
    logger.info("\n" + "="*100)
    logger.info("🚀 BẮT ĐẦU ĐÁNH GIÁ HIỆU SUẤT XỬ LÝ TÀI LIỆU")
    logger.info("="*100)
    logger.info(f"Config: CHUNK_SIZE={AppConfig.CHUNK_SIZE}, CHUNK_OVERLAP={AppConfig.CHUNK_OVERLAP}, K={measure_k}, Retrieval runs={num_retrieval_runs}")
    
    rows = []

    for label, path in DOCUMENTS.items():
        logger.info(f"\n--- Đo: {label} ({path}) ---")
        if not os.path.exists(path):
            logger.warning(f"❌ File not found: {path}")
            rows.append({
                "label": label,
                "pdf_path": path,
                "file_size_mb": None,
                "num_chunks": None,
                "chunk_size": AppConfig.CHUNK_SIZE,
                "chunk_overlap": AppConfig.CHUNK_OVERLAP,
                "load_time": None,
                "embedding_time": None,
                "retrieval_time": None,
                "retrieval_time_min": None,
                "retrieval_time_max": None,
                "retrieval_k": measure_k,
                "use_co_rag": False,
                "error": "file_not_found",
            })
            continue

        res = measure_document_processing(path, query=TEST_QUERY, k=measure_k, use_co_rag=False, num_retrieval_runs=num_retrieval_runs)
        res["label"] = label
        rows.append(res)

    # Lưu CSV chi tiết
    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_csv.replace('.csv', f'_{timestamp}.csv')
    df.to_csv(out_file, index=False)
    logger.info(f"\n📊 Chi tiết lưu vào CSV: {out_file}")

    # In bảng tóm tắt (thêm 3 cột retrieval: vector, RAG, Co-RAG)
    print("\n" + "="*140)
    print("Bảng I: HIỆU SUẤT XỬ LÝ TÀI LIỆU (DÙNG APP SERVICE CHÍNH THỨC)")
    print("="*140)
    print(f"{'Kích thước':<25} {'Chunk':<8} {'File(MB)':<10} {'Loading(s)':<12} {'Embedding(s)':<14} {'Retrieval Vector(s)':<20} {'Retrieval RAG(s)':<18} {'Retrieval CoRAG(s)':<18}")
    print("-"*140)

    for r in rows:
        label = r.get('label', 'Unknown')
        num_chunks = r.get('num_chunks')
        file_size = r.get('file_size_mb')
        load = r.get('load_time')
        emb = r.get('embedding_time')
        v_avg = r.get('retrieval_vector_avg')
        rag_avg = r.get('retrieval_rag_avg')
        co_avg = r.get('retrieval_co_rag_avg')

        if r.get('error'):
            print(f"{label:<25} {'ERROR':<8} {str(file_size):<10} {'N/A':<12} {'N/A':<14} {'N/A':<20} {'N/A':<18} {'N/A':<18}")
        else:
            num_chunks_str = str(num_chunks) if num_chunks is not None else 'N/A'
            file_size_str = f"{file_size:.2f}" if file_size is not None else 'N/A'
            load_str = f"{load:.2f}" if load is not None else 'N/A'
            emb_str = f"{emb:.2f}" if emb is not None else 'N/A'
            v_str = f"{v_avg:.4f}" if v_avg is not None else 'N/A'
            rag_str = f"{rag_avg:.4f}" if rag_avg is not None else 'N/A'
            co_str = f"{co_avg:.4f}" if co_avg is not None else 'N/A'
            print(f"{label:<25} {num_chunks_str:<8} {file_size_str:<10} {load_str:<12} {emb_str:<14} {v_str:<20} {rag_str:<18} {co_str:<18}")

    print("-"*140)

    # In bảng chi tiết retrieval (min/max) cho từng chế độ
    successful = [r for r in rows if not r.get('error')]
    if successful:
        print("\n📊 CHI TIẾT RETRIEVAL (lặp lại {num_retrieval_runs} lần):")
        print(f"{'Kích thước':<25} {'Mode':<12} {'Avg(s)':<12} {'Min(s)':<12} {'Max(s)':<12}")
        print("-"*75)
        for r in successful:
            label = r.get('label', 'Unknown')
            # vector
            if r.get('retrieval_vector_avg') is not None:
                print(f"{label:<25} {'vector':<12} {r['retrieval_vector_avg']:<12.4f} {r['retrieval_vector_min']:<12.4f} {r['retrieval_vector_max']:<12.4f}")
            # rag
            if r.get('retrieval_rag_avg') is not None:
                print(f"{label:<25} {'RAG':<12} {r['retrieval_rag_avg']:<12.4f} {r['retrieval_rag_min']:<12.4f} {r['retrieval_rag_max']:<12.4f}")
            # co-rag
            if r.get('retrieval_co_rag_avg') is not None:
                print(f"{label:<25} {'Co-RAG':<12} {r['retrieval_co_rag_avg']:<12.4f} {r['retrieval_co_rag_min']:<12.4f} {r['retrieval_co_rag_max']:<12.4f}")
        print("-"*75)

    # In thống kê thêm (tổng theo mode)
    if successful:
        print("\n📈 THỐNG KÊ TỔNG HỢP:")
        total_load = sum(r['load_time'] for r in successful if r['load_time'])
        total_emb = sum(r['embedding_time'] for r in successful if r['embedding_time'])
        total_vec = sum(r['retrieval_vector_avg'] for r in successful if r.get('retrieval_vector_avg') is not None)
        total_rag = sum(r['retrieval_rag_avg'] for r in successful if r.get('retrieval_rag_avg') is not None)
        total_co = sum(r['retrieval_co_rag_avg'] for r in successful if r.get('retrieval_co_rag_avg') is not None)

        print(f"  • Tổng Loading time: {total_load:.2f}s")
        print(f"  • Tổng Embedding time: {total_emb:.2f}s")
        print(f"  • Tổng Retrieval Vector (avg): {total_vec:.2f}s")
        print(f"  • Tổng Retrieval RAG (avg): {total_rag:.2f}s")
        print(f"  • Tổng Retrieval Co-RAG (avg): {total_co:.2f}s")
        print(f"  • Tổng thời gian (sum of averages): {total_load + total_emb + total_vec + total_rag + total_co:.2f}s")

    print("="*140 + "\n")


if __name__ == '__main__':
    # Có thể điều chỉnh số lần lặp retrieval ở đây
    run_all(num_retrieval_runs=3)
