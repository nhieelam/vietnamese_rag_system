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
        _render_upload_section()
        st.divider()
        _render_document_list()


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
    
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, Image, DOCX)",
        type=AppConfig.ALLOWED_FILE_TYPES,
        help=f"Upload internal departmental documents (up to {AppConfig.MAX_FILE_SIZE_MB} pages)",
        key="file_uploader"
    )
    
    if uploaded_file:
        if st.button("Process & Add", type="primary", use_container_width=True):
            _process_and_add_document(uploaded_file)


def _process_and_add_document(uploaded_file):
    with st.spinner("Processing document..."):
        try:
            extracted_text = FileService.extract(uploaded_file)
            
            if SessionService.document_exists(uploaded_file.name):
                st.warning("Document already uploaded!")
            else:
                doc_data = {
                    "id": len(SessionService.get_documents()),
                    "name": uploaded_file.name,
                    "text": extracted_text,
                    "size": len(extracted_text),
                    "uploaded_at": datetime.now().strftime(AppConfig.UPLOAD_TIMESTAMP_FORMAT)
                }
                SessionService.add_document(doc_data)
                chunks = TextSplitterService.split(doc_data["text"]["text"])
                
                # Truyền metadata để tracking source document
                metadata = {
                    "source": uploaded_file.name,
                    "document_id": doc_data["id"]
                }
                
                VectorStoreService.build_from_chunks(
                    chunks=chunks,
                    embedding=EmbeddingService.get_huggingface_embedding(),
                    metadata=metadata
                )
                st.success(f"✅ Added: {uploaded_file.name}")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")


def _render_document_list():
    st.subheader("Documents")
    
    documents = SessionService.get_documents()
    
    if documents:
        st.caption(f"**{len(documents)} document(s) in knowledge base**")
        
        for idx, doc in enumerate(documents):
            with st.expander(f"{doc['name']}", expanded=False):
                st.caption(f"Uploaded: {doc['uploaded_at']}")
                st.caption(f"Size: {doc['size']:,} characters")
                
                if st.button("Remove", key=f"del_{idx}", use_container_width=True):
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

