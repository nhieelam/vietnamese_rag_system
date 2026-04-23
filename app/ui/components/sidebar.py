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
        st.divider()
        _render_doc_filter_section()
        st.divider()
        _render_chat_history_section()
        st.divider()
        _render_danger_zone()


def _render_retrieval_settings():
    st.subheader("Chunk & Retrieval")

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

        st.markdown("---")
        st.caption("**Retriever strategy**")
        st.selectbox(
            "Retriever mode",
            options=["vector", "hybrid"],
            key="retriever_mode",
            help="vector = FAISS semantic only. hybrid = BM25 + vector ensemble.",
        )
        if st.session_state.get("retriever_mode") == "hybrid":
            st.slider(
                "BM25 weight",
                min_value=0.0, max_value=1.0, step=0.1,
                key="bm25_weight",
                help="Trọng số BM25 (vector weight = 1 - BM25 weight).",
            )

        st.checkbox(
            "Enable Cross-Encoder Re-ranking",
            key="use_reranker",
            help="Bật re-ranking với BAAI/bge-reranker-v2-m3 (chậm hơn, chất lượng cao hơn).",
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
            SessionService.clear_all_chunks()
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
                    metadata={
                        "source": doc.get("name"),
                        "document_id": doc.get("id"),
                        "file_type": doc.get("file_type"),
                        "uploaded_at": doc.get("uploaded_at"),
                        "total_pages": doc.get("total_pages"),
                    },
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
    
    if selected_mode == AppConfig.ANSWER_MODE_RAG:
        st.info("📚 **RAG Only**: Fast responses using traditional retrieval-augmented generation")
    elif selected_mode == AppConfig.ANSWER_MODE_CO_RAG:
        st.info("🔀 **Co-RAG Only**: Generates multiple sub-queries for comprehensive retrieval")
    elif selected_mode == AppConfig.ANSWER_MODE_SELF_RAG:
        st.info("🧠 **Self-RAG**: Query rewriting, relevance grading, self-evaluation & confidence scoring")
    else:
        st.info("🤝 **RAG & Co-RAG**: Combines both approaches for the best results")


def _detect_file_type(uploaded_file) -> str:
    t = getattr(uploaded_file, "type", "") or ""
    if "pdf" in t:
        return "pdf"
    if "word" in t or "docx" in t:
        return "docx"
    if "image" in t:
        return "image"
    return "unknown"


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
            file_type = _detect_file_type(uf)
            uploaded_at = datetime.now().strftime(AppConfig.UPLOAD_TIMESTAMP_FORMAT)

            SessionService.add_document({
                "id": doc_id,
                "name": uf.name,
                "text": extracted,
                "size": len(text),
                "uploaded_at": uploaded_at,
                "file_type": file_type,
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
                metadata={
                    "source": uf.name,
                    "document_id": doc_id,
                    "file_type": file_type,
                    "uploaded_at": uploaded_at,
                    "total_pages": meta.get("total_pages"),
                },
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
                if doc.get("file_type"):
                    st.caption(f"Type: {doc['file_type'].upper()}")
                if doc.get("total_pages"):
                    st.caption(f"Pages: {doc['total_pages']}")

                if st.button("Remove", key=f"del_{idx}", use_container_width=True):
                    SessionService.remove_pdf(doc.get("id", idx))
                    SessionService.remove_document(idx)
                    st.rerun()
    else:
        st.info("No documents yet\n\nUpload documents to start asking questions!")


def _render_doc_filter_section():
    """Multi-select filter: giới hạn tài liệu dùng để trả lời câu hỏi."""
    st.subheader("Document Filter")
    documents = SessionService.get_documents() or []
    if not documents:
        st.caption("Chưa có tài liệu để lọc.")
        return

    options = [d.get("name") for d in documents if d.get("name")]
    file_types = sorted({d.get("file_type", "unknown") for d in documents})

    st.multiselect(
        "Chỉ tìm trong tài liệu",
        options=options,
        key="doc_filter",
        help="Để trống = tìm trong tất cả tài liệu.",
    )
    st.multiselect(
        "Loại file",
        options=file_types,
        key="file_type_filter",
        help="Lọc theo loại file (pdf/docx/image). Để trống = tất cả.",
    )


def _render_chat_history_section():
    """Hiển thị lịch sử câu hỏi đã hỏi trong sidebar (8.2.2)."""
    st.subheader("Chat History")
    messages = SessionService.get_messages_for_ui() or []
    if not messages:
        st.caption("Chưa có lịch sử hội thoại.")
        return

    pairs = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "user":
            replies = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "assistant":
                replies.append(messages[j])
                j += 1
            pairs.append((m, replies))
            i = j
        else:
            i += 1

    total = len(pairs)
    st.caption(f"**{total} câu hỏi đã hỏi**")

    recent = pairs[-20:][::-1]
    for idx, (user_msg, replies) in enumerate(recent):
        question = (user_msg.get("content") or "").strip()
        short = question[:60] + ("..." if len(question) > 60 else "")
        ts = user_msg.get("timestamp", "")
        with st.expander(f"Q: {short}   ·   {ts}", expanded=False):
            st.markdown(f"**Question:** {question}")
            for r in replies:
                mode = r.get("mode", "Assistant")
                content = (r.get("content") or "").strip()
                preview = content[:300] + ("..." if len(content) > 300 else "")
                st.markdown(f"**{mode}:** {preview}")


def _confirm_button(action_key: str, label: str, warning_msg: str, on_confirm):
    """Render 1 nút có confirmation: bấm lần 1 -> hiện Yes/Cancel; bấm Yes -> gọi on_confirm."""
    confirm_flag = f"confirm_{action_key}"
    if st.session_state.get(confirm_flag):
        st.warning(warning_msg)
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Yes, {label}", key=f"{action_key}_yes", use_container_width=True, type="primary"):
                on_confirm()
                st.session_state[confirm_flag] = False
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"{action_key}_cancel", use_container_width=True):
                st.session_state[confirm_flag] = False
                st.rerun()
    else:
        if st.button(label, key=f"{action_key}_trigger", use_container_width=True):
            st.session_state[confirm_flag] = True
            st.rerun()


def _render_danger_zone():
    """8.2.3 - Clear History / Clear Vector Store / Clear Everything có confirmation."""
    st.subheader("Danger Zone")

    _confirm_button(
        "clear_history",
        "Clear Chat History",
        "Toàn bộ lịch sử hội thoại sẽ bị xóa. Tiếp tục?",
        SessionService.clear_chat_history,
    )

    def _clear_vector_store():
        VectorStoreService.clear()
        SessionService.clear_documents()
        SessionService.clear_all_pdfs()
        SessionService.clear_all_chunks()

    _confirm_button(
        "clear_vs",
        "Clear Vector Store",
        "Toàn bộ tài liệu đã upload + vector index sẽ bị xóa. Tiếp tục?",
        _clear_vector_store,
    )

    def _clear_everything():
        SessionService.clear_chat_history()
        _clear_vector_store()

    _confirm_button(
        "clear_all",
        "Clear Everything",
        "Xóa TẤT CẢ: lịch sử chat + tài liệu + vector store. Tiếp tục?",
        _clear_everything,
    )
