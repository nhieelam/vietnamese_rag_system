from unittest import result
import streamlit as st
from datetime import datetime
from app.services import EmbeddingService
from app.services import FileService
from app.services import SessionService
from app.config import AppConfig
from app.services import VectorStoreService
from app.services import TextSplitterService

def render_sidebar():
    with st.sidebar:
        _render_answer_mode_section()
        st.divider()
        _render_retrieval_settings()
        st.divider()
        _render_upload_section()
        st.divider()
        _render_document_list()


def _render_retrieval_settings():
    st.subheader("Chunk & Retrieval")

    # Snapshot các giá trị đã ĐƯỢC ÁP DỤNG vào index hiện tại.
    # Slider dùng key để Streamlit tự đồng bộ (không cần gán tay).
    if "chunk_size_applied" not in st.session_state:
        st.session_state.chunk_size_applied = int(st.session_state.get("chunk_size", 500))
    if "chunk_overlap_applied" not in st.session_state:
        st.session_state.chunk_overlap_applied = int(st.session_state.get("chunk_overlap", 50))

    with st.expander("Advanced settings", expanded=False):
        st.slider(
            "Chunk size (characters)",
            min_value=100, max_value=2000, step=50,
            key="chunk_size",
            help="Độ dài mỗi chunk khi chia nhỏ tài liệu.",
        )
        chunk_size = int(st.session_state.chunk_size)

        max_overlap = max(0, chunk_size - 50)
        # Clamp overlap về max_overlap nếu user vừa giảm chunk_size.
        if int(st.session_state.get("chunk_overlap", 50)) > max_overlap:
            st.session_state.chunk_overlap = max_overlap
        st.slider(
            "Chunk overlap",
            min_value=0, max_value=max_overlap, step=10,
            key="chunk_overlap",
            help="Độ chồng lấn giữa 2 chunk liên tiếp. Nên < chunk_size / 3.",
        )

        st.slider(
            "Top-K chunks khi retrieval",
            min_value=2, max_value=20, step=1,
            key="retrieval_k",
            help="Số chunk được lấy cho mỗi câu hỏi.",
        )
        retrieval_k = int(st.session_state.retrieval_k)

        if int(st.session_state.get("per_source_k", 4)) > retrieval_k:
            st.session_state.per_source_k = retrieval_k
        st.slider(
            "Max chunks mỗi file",
            min_value=1, max_value=max(1, retrieval_k), step=1,
            key="per_source_k",
            help="Giới hạn số chunk từ cùng một file để giữ đa nguồn.",
        )

        changed = (
            int(st.session_state.chunk_size) != int(st.session_state.chunk_size_applied)
            or int(st.session_state.chunk_overlap) != int(st.session_state.chunk_overlap_applied)
        )
        if changed and SessionService.get_documents():
            st.warning("Chunk params đã đổi. Bấm Re-index để áp dụng cho tài liệu hiện tại.")
            if st.button("Re-index all documents", use_container_width=True):
                _reindex_all_documents()


def _reindex_all_documents():
    docs = SessionService.get_documents() or []
    if not docs:
        return
    with st.spinner("Re-indexing..."):
        try:
            VectorStoreService.clear()
            embedding = EmbeddingService.get_huggingface_embedding()
            for doc in docs:
                extracted = doc.get("text") or {}
                text = extracted.get("text") if isinstance(extracted, dict) else None
                if not text:
                    continue
                meta = (extracted.get("metadata") or {}) if isinstance(extracted, dict) else {}
                page_ranges = meta.get("page_ranges")
                if page_ranges:
                    chunks = TextSplitterService.split_with_offsets(text, page_ranges)
                else:
                    chunks = TextSplitterService.split(text)
                VectorStoreService.build_from_chunks(
                    chunks=chunks,
                    embedding=embedding,
                    metadata={"source": doc.get("name"), "document_id": doc.get("id")},
                )
            st.session_state.chunk_size_applied = int(st.session_state.get("chunk_size", 500))
            st.session_state.chunk_overlap_applied = int(st.session_state.get("chunk_overlap", 50))
            st.success(f"Re-indexed {len(docs)} document(s) with new chunk params")
            st.rerun()
        except Exception as e:
            st.error(f"Re-index failed: {e}")


def _render_answer_mode_section():
    st.subheader("Answer Mode")
    
    mode = st.session_state.get("answer_mode", AppConfig.ANSWER_MODE_DEFAULT)
    if mode not in AppConfig.ANSWER_MODE_ORDER:
        mode = AppConfig.ANSWER_MODE_DEFAULT
    
    selected_mode = st.selectbox(
        "Choose answer mode",
        options=AppConfig.ANSWER_MODE_ORDER,
        index=AppConfig.ANSWER_MODE_ORDER.index(mode),
        key="answer_mode_selectbox",
        label_visibility="collapsed"
    )
    
    st.session_state.answer_mode = selected_mode
    
    # Display mode description
    if selected_mode == AppConfig.ANSWER_MODE_RAG:
        st.info("📚 **RAG Only**: Fast responses using traditional retrieval-augmented generation")
    elif selected_mode == AppConfig.ANSWER_MODE_CO_RAG:
        st.info("🔀 **Co-RAG Only**: Generates multiple sub-queries for comprehensive retrieval")
    else:
        st.info("🤝 **RAG & Co-RAG**: Combines both approaches for the best results")


def _render_upload_section():
    uploaded_files = st.file_uploader(
        "Choose files (PDF, Image, DOCX)",
        type=AppConfig.ALLOWED_FILE_TYPES,
        help=f"Upload internal departmental documents (up to {AppConfig.MAX_FILE_SIZE_MB} pages)",
        key="file_uploader",
        accept_multiple_files=True,
    )

    if uploaded_files:
        label = (
            f"Process & Add ({len(uploaded_files)} files)"
            if len(uploaded_files) > 1 else "Process & Add"
        )
        if st.button(label, type="primary", use_container_width=True):
            _process_and_add_documents(uploaded_files)


def _process_and_add_documents(uploaded_files):
    added, skipped, failed = [], [], []
    embedding = EmbeddingService.get_huggingface_embedding()

    progress = st.progress(0.0, text="Starting...")
    total = len(uploaded_files)

    for i, uf in enumerate(uploaded_files, start=1):
        progress.progress((i - 1) / total, text=f"Processing {uf.name} ({i}/{total})")
        try:
            if SessionService.document_exists(uf.name):
                skipped.append(uf.name)
                continue

            try:
                uf.seek(0)
            except Exception:
                pass
            extracted = FileService.extract(uf)
            if extracted.get("status_code") != 200 or not extracted.get("text"):
                failed.append((uf.name, extracted.get("message", "extract failed")))
                continue

            doc_id = len(SessionService.get_documents())
            text = extracted["text"]
            meta = extracted.get("metadata", {}) or {}
            page_ranges = meta.get("page_ranges")
            pdf_bytes = meta.get("pdf_bytes")

            SessionService.add_document({
                "id": doc_id,
                "name": uf.name,
                "text": extracted,
                "size": len(text),
                "uploaded_at": datetime.now().strftime(AppConfig.UPLOAD_TIMESTAMP_FORMAT),
                "has_pdf": bool(pdf_bytes),
                "total_pages": meta.get("total_pages"),
            })

            if pdf_bytes:
                SessionService.store_pdf(doc_id, uf.name, pdf_bytes)

            if page_ranges:
                chunks = TextSplitterService.split_with_offsets(text, page_ranges)
            else:
                chunks = TextSplitterService.split(text)

            VectorStoreService.build_from_chunks(
                chunks=chunks,
                embedding=embedding,
                metadata={"source": uf.name, "document_id": doc_id},
            )
            added.append(uf.name)
        except Exception as e:
            failed.append((uf.name, str(e)))

    progress.progress(1.0, text="Done")

    if added:
        st.session_state.chunk_size_applied = int(st.session_state.get("chunk_size", 500))
        st.session_state.chunk_overlap_applied = int(st.session_state.get("chunk_overlap", 50))
        st.success(f"Added {len(added)} file(s): " + ", ".join(added))
    if skipped:
        st.warning("Already uploaded: " + ", ".join(skipped))
    if failed:
        for name, msg in failed:
            st.error(f"{name}: {msg}")

    if added:
        st.rerun()


def _render_document_list():
    st.subheader("Documents")
    
    documents = SessionService.get_documents()
    
    if documents:
        st.caption(f"**{len(documents)} document(s) in knowledge base**")
        
        for idx, doc in enumerate(documents):
            with st.expander(f"{doc['name']}", expanded=False):
                st.caption(f"Uploaded: {doc['uploaded_at']}")
                st.caption(f"Size: {doc['size']:,} characters")
                if doc.get("total_pages"):
                    st.caption(f"Pages: {doc['total_pages']}")

                if st.button("Remove", key=f"del_{idx}", use_container_width=True):
                    SessionService.remove_pdf(doc.get("id", idx))
                    SessionService.remove_document(idx)
                    st.rerun()
        
        st.divider()
        _render_action_buttons()
    else:
        st.info("No documents yet\n\nUpload documents to start asking questions!")


def _render_action_buttons():
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All", use_container_width=True):
            SessionService.clear_documents()
            st.rerun()
    
    with col2:
        if st.button("New Chat", use_container_width=True):
            SessionService.clear_chat_history()
            st.rerun()

