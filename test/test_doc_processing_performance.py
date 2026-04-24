"""
Bao cao hieu suat xu ly tai lieu: Loading (extract + split), Embedding (build vector store), Retrieval.
In bang tom tat tuong tu Bang I va luu file CSV chi tiet.

Cach dung:
    python test_doc_processing_performance.py
    python test_doc_processing_performance.py --no-llm   # bo qua RAG/Co-RAG (khong goi LLM)
    python test_doc_processing_performance.py --runs 5 --k 5

Luu y: script dung cac service chinh cua project
    - TextSplitterService.split() + AppConfig.CHUNK_SIZE/CHUNK_OVERLAP
    - EmbeddingService.get_huggingface_embedding()
    - VectorStoreService.build_from_chunks()
    - AIConfig.get_llm_instance() de do RAG / Co-RAG
"""

from __future__ import annotations

# --- Force UTF-8 stdout/stderr (Windows console mac dinh la cp1252) ---
import sys
import io
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber

from app.config.ai_config import AIConfig
from app.config.app_config import AppConfig
from app.services.embedding_service import EmbeddingService
from app.services.text_splitter_service import TextSplitterService
from app.services.vector_store_service import VectorStoreService
from app.utils.logger import logger


BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS: Dict[str, str] = {
    "Nho (1 trang)": "dao.pdf",
    "Trung binh (43 trang)": "cpp_quickstart.pdf",
    "Lon (194 trang)": "sample.pdf",
}

RETRIEVAL_K = 5
TEST_QUERY = "Tai lieu nay noi ve gi?"


def resolve_path(path_str: str) -> Optional[Path]:
    """Tim file PDF: thu duong dan tuyet doi, roi relative theo thu muc script."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    candidate = BASE_DIR / path_str
    if candidate.exists():
        return candidate
    if p.exists():
        return p.resolve()
    return None


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def _retrieve_docs(vector_store, query: str, k: int):
    """Goi similarity_search, fallback sang as_retriever neu can."""
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        logger.warning(f"similarity_search failed ({e}), fallback to as_retriever")
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        return []


def measure_document_processing(
    pdf_path: str,
    query: str = TEST_QUERY,
    k: int = RETRIEVAL_K,
    num_retrieval_runs: int = 3,
    run_llm: bool = True,
) -> Dict[str, Any]:
    """Do 3 giai doan: Loading -> Embedding -> Retrieval (vector / RAG / Co-RAG)."""
    result: Dict[str, Any] = {
        "pdf_path": pdf_path,
        "file_size_mb": os.path.getsize(pdf_path) / (1024 * 1024),
        "num_chunks": 0,
        "chunk_size": AppConfig.CHUNK_SIZE,
        "chunk_overlap": AppConfig.CHUNK_OVERLAP,
        "load_time": None,
        "embedding_time": None,
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
        "error": None,
    }

    try:
        # 1) Loading: extract + split
        logger.info(
            f"[Loading] Extract + split (chunk_size={AppConfig.CHUNK_SIZE}, "
            f"overlap={AppConfig.CHUNK_OVERLAP})"
        )
        t0 = time.time()
        text = extract_text_from_pdf(pdf_path)
        chunks = TextSplitterService.split(text)
        load_time = time.time() - t0
        result["num_chunks"] = len(chunks)
        result["load_time"] = load_time
        logger.info(
            f"  Extracted {len(text)} chars, split into {len(chunks)} chunks in {load_time:.4f}s"
        )

        # 2) Embedding: build vector store
        logger.info("[Embedding] Building vector store")
        t1 = time.time()
        embedding_model = EmbeddingService.get_huggingface_embedding()
        vector_store = VectorStoreService.build_from_chunks(
            chunks=chunks,
            embedding=embedding_model,
            metadata={"source": os.path.basename(pdf_path)},
        )
        embedding_time = time.time() - t1
        result["embedding_time"] = embedding_time
        logger.info(f"  Embedding completed in {embedding_time:.4f}s")

        # 3a) Vector retrieval (similarity_search only)
        logger.info(
            f"[Retrieval] Vector only, k={k}, runs={num_retrieval_runs}"
        )
        vector_times: List[float] = []
        for run in range(num_retrieval_runs):
            t2 = time.time()
            docs = _retrieve_docs(vector_store, query, k)
            elapsed = time.time() - t2
            vector_times.append(elapsed)
            logger.info(
                f"  Vector run {run+1}/{num_retrieval_runs}: {elapsed:.4f}s "
                f"(found {len(docs)} docs)"
            )
        result["retrieval_vector_avg"] = sum(vector_times) / len(vector_times)
        result["retrieval_vector_min"] = min(vector_times)
        result["retrieval_vector_max"] = max(vector_times)

        if not run_llm:
            logger.info("[Retrieval] Skip RAG / Co-RAG (run_llm=False)")
            return result

        # Lazy import prompt template chi khi can LLM
        from langchain_core.prompts import ChatPromptTemplate

        try:
            llm = AIConfig.get_llm_instance()
        except Exception as e:
            logger.error(f"Cannot init LLM, skip RAG / Co-RAG: {e}")
            return result

        rag_prompt = ChatPromptTemplate.from_template(
            "Based on the following context, answer the question:\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer:"
        )
        co_rag_prompt = ChatPromptTemplate.from_template(
            "Based on the following documents, answer the question and include citations:\n"
            "Documents:\n{context}\n"
            "Question: {question}\n"
            "Answer (with citations):"
        )

        # 3b) RAG = retrieval + LLM
        logger.info("[Retrieval] RAG (vector + LLM)")
        rag_times: List[float] = []
        for run in range(num_retrieval_runs):
            t3 = time.time()
            try:
                docs = _retrieve_docs(vector_store, query, k)
                context = "\n".join(d.page_content for d in docs)
                response = llm.invoke(rag_prompt.format(context=context, question=query))
                elapsed = time.time() - t3
                rag_times.append(elapsed)
                preview = getattr(response, "content", str(response))[:80]
                logger.info(
                    f"  RAG run {run+1}/{num_retrieval_runs}: {elapsed:.4f}s -> '{preview}...'"
                )
            except Exception as e:
                elapsed = time.time() - t3
                rag_times.append(elapsed)
                logger.error(
                    f"  RAG run {run+1}/{num_retrieval_runs}: {elapsed:.4f}s ERROR: {str(e)[:120]}"
                )

        if rag_times:
            result["retrieval_rag_avg"] = sum(rag_times) / len(rag_times)
            result["retrieval_rag_min"] = min(rag_times)
            result["retrieval_rag_max"] = max(rag_times)

        # 3c) Co-RAG = retrieval + LLM + citations format
        logger.info("[Retrieval] Co-RAG (vector + LLM + citations)")
        co_rag_times: List[float] = []
        for run in range(num_retrieval_runs):
            t4 = time.time()
            try:
                docs = _retrieve_docs(vector_store, query, k)
                context = "\n".join(
                    f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(docs)
                )
                response = llm.invoke(co_rag_prompt.format(context=context, question=query))
                elapsed = time.time() - t4
                co_rag_times.append(elapsed)
                preview = getattr(response, "content", str(response))[:80]
                logger.info(
                    f"  Co-RAG run {run+1}/{num_retrieval_runs}: {elapsed:.4f}s -> '{preview}...'"
                )
            except Exception as e:
                elapsed = time.time() - t4
                co_rag_times.append(elapsed)
                logger.error(
                    f"  Co-RAG run {run+1}/{num_retrieval_runs}: {elapsed:.4f}s ERROR: {str(e)[:120]}"
                )

        if co_rag_times:
            result["retrieval_co_rag_avg"] = sum(co_rag_times) / len(co_rag_times)
            result["retrieval_co_rag_min"] = min(co_rag_times)
            result["retrieval_co_rag_max"] = max(co_rag_times)

        logger.info(
            f"  Summary vector={result['retrieval_vector_avg']:.4f}s"
            + (f", RAG={result['retrieval_rag_avg']:.4f}s" if result["retrieval_rag_avg"] else "")
            + (f", Co-RAG={result['retrieval_co_rag_avg']:.4f}s" if result["retrieval_co_rag_avg"] else "")
        )

    except Exception as e:
        logger.exception(f"Error measuring {pdf_path}")
        result["error"] = str(e)

    return result


def run_all(
    measure_k: int = RETRIEVAL_K,
    output_csv: str = "document_processing_performance.csv",
    num_retrieval_runs: int = 3,
    run_llm: bool = True,
) -> None:
    logger.info("=" * 100)
    logger.info("BAT DAU DANH GIA HIEU SUAT XU LY TAI LIEU")
    logger.info("=" * 100)
    logger.info(
        f"Config: CHUNK_SIZE={AppConfig.CHUNK_SIZE}, CHUNK_OVERLAP={AppConfig.CHUNK_OVERLAP}, "
        f"K={measure_k}, runs={num_retrieval_runs}, run_llm={run_llm}"
    )

    rows: List[Dict[str, Any]] = []
    for label, path_str in DOCUMENTS.items():
        logger.info(f"--- Do: {label} ({path_str}) ---")
        pdf_path = resolve_path(path_str)
        if pdf_path is None:
            logger.warning(f"File not found: {path_str}")
            rows.append({
                "label": label,
                "pdf_path": path_str,
                "file_size_mb": None,
                "num_chunks": None,
                "chunk_size": AppConfig.CHUNK_SIZE,
                "chunk_overlap": AppConfig.CHUNK_OVERLAP,
                "load_time": None,
                "embedding_time": None,
                "retrieval_vector_avg": None,
                "retrieval_rag_avg": None,
                "retrieval_co_rag_avg": None,
                "retrieval_k": measure_k,
                "error": "file_not_found",
            })
            continue

        res = measure_document_processing(
            str(pdf_path),
            query=TEST_QUERY,
            k=measure_k,
            num_retrieval_runs=num_retrieval_runs,
            run_llm=run_llm,
        )
        res["label"] = label
        rows.append(res)

    # CSV
    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_csv.replace(".csv", f"_{timestamp}.csv")
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    logger.info(f"CSV chi tiet: {out_file}")

    # Bang tom tat
    print("\n" + "=" * 140)
    print("Bang I: HIEU SUAT XU LY TAI LIEU")
    print("=" * 140)
    header = (
        f"{'Kich thuoc':<25} {'Chunk':<8} {'File(MB)':<10} {'Loading(s)':<12} "
        f"{'Embedding(s)':<14} {'Vector(s)':<12} {'RAG(s)':<12} {'Co-RAG(s)':<12}"
    )
    print(header)
    print("-" * 140)
    for r in rows:
        label = r.get("label", "Unknown")
        if r.get("error"):
            print(f"{label:<25} {'ERROR':<8} {'-':<10} {'-':<12} {'-':<14} {'-':<12} {'-':<12} {'-':<12}")
            continue

        def _fmt(v, prec=4):
            return f"{v:.{prec}f}" if v is not None else "N/A"

        print(
            f"{label:<25} "
            f"{str(r.get('num_chunks') or 'N/A'):<8} "
            f"{_fmt(r.get('file_size_mb'), 2):<10} "
            f"{_fmt(r.get('load_time'), 2):<12} "
            f"{_fmt(r.get('embedding_time'), 2):<14} "
            f"{_fmt(r.get('retrieval_vector_avg')):<12} "
            f"{_fmt(r.get('retrieval_rag_avg')):<12} "
            f"{_fmt(r.get('retrieval_co_rag_avg')):<12}"
        )
    print("-" * 140)

    # Chi tiet min/max
    successful = [r for r in rows if not r.get("error")]
    if successful:
        print(f"\nCHI TIET RETRIEVAL (lap lai {num_retrieval_runs} lan):")
        print(f"{'Kich thuoc':<25} {'Mode':<12} {'Avg(s)':<12} {'Min(s)':<12} {'Max(s)':<12}")
        print("-" * 75)
        for r in successful:
            label = r.get("label", "Unknown")
            for mode in ("vector", "rag", "co_rag"):
                avg = r.get(f"retrieval_{mode}_avg")
                if avg is None:
                    continue
                mn = r.get(f"retrieval_{mode}_min")
                mx = r.get(f"retrieval_{mode}_max")
                print(f"{label:<25} {mode:<12} {avg:<12.4f} {mn:<12.4f} {mx:<12.4f}")
        print("-" * 75)

        total_load = sum(r["load_time"] for r in successful if r.get("load_time"))
        total_emb = sum(r["embedding_time"] for r in successful if r.get("embedding_time"))
        total_vec = sum(r["retrieval_vector_avg"] for r in successful if r.get("retrieval_vector_avg"))
        total_rag = sum(r["retrieval_rag_avg"] for r in successful if r.get("retrieval_rag_avg"))
        total_co = sum(r["retrieval_co_rag_avg"] for r in successful if r.get("retrieval_co_rag_avg"))
        print("\nTHONG KE TONG HOP:")
        print(f"  Tong Loading time    : {total_load:.2f}s")
        print(f"  Tong Embedding time  : {total_emb:.2f}s")
        print(f"  Tong Vector (avg)    : {total_vec:.2f}s")
        print(f"  Tong RAG (avg)       : {total_rag:.2f}s")
        print(f"  Tong Co-RAG (avg)    : {total_co:.2f}s")
        print(f"  Tong (sum of avgs)   : {total_load + total_emb + total_vec + total_rag + total_co:.2f}s")
    print("=" * 140 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Do hieu suat xu ly tai lieu (RAG pipeline)")
    parser.add_argument("--k", type=int, default=RETRIEVAL_K, help="So chunk lay khi retrieval")
    parser.add_argument("--runs", type=int, default=3, help="So lan lap moi che do retrieval")
    parser.add_argument("--no-llm", action="store_true", help="Bo qua RAG va Co-RAG (khong goi LLM)")
    parser.add_argument(
        "--output",
        default="document_processing_performance.csv",
        help="Ten file CSV ket qua (se chen timestamp)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(
        measure_k=args.k,
        output_csv=args.output,
        num_retrieval_runs=args.runs,
        run_llm=not args.no_llm,
    )
