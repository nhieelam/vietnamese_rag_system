import streamlit as st
from app.services.session_service import SessionService
from app.models.citation import Citation

def render_chat_header():
    st.title("💬 Vietnamese RAG Assistant")
    st.markdown("Ask questions about your uploaded documents and get comprehensive answers")
    
    documents = SessionService.get_documents()
    if documents:
        st.success(f"Knowledge base active with {len(documents)} document(s)")
    else:
        st.warning("No documents uploaded. Please upload documents from the sidebar to begin.")
    
    st.divider()
    
def render_chat_messages():
    chat_container = st.container()
    
    with chat_container:
        messages = SessionService.get_messages_for_ui() # Use the correct method for UI
        
        if not messages:
            _render_welcome_message()
        else:
            _render_message_history(messages)
        
        # Trigger rerun if message was sent
        if st.session_state.get("message_sent", False):
            st.session_state.message_sent = False
            st.rerun()


def _render_welcome_message():
    st.info("""
    I'm here to help you find information from your departmental documents.
    
    **How to start:**
    1. Upload your documents using the sidebar (←)
    2. Ask me any questions about the content
    3. I'll provide comprehensive answers based on the documents
    
    **Example questions:**
    - "What are the main policies mentioned in the document?"
    - "Summarize the key points from section 3"
    - "What are the requirements for employee onboarding?"
    """)


def _render_message_history(messages):
    """Render message history with robust side-by-side RAG vs Co-RAG comparisons."""
    index = 0

    while index < len(messages):
        message = messages[index]
        role = message.get("role")

        if role == "user":
            _render_user_message(message)
            index += 1
            continue

        if role == "assistant":
            # Kiểm tra xem có mode mới không (citations structure)
            mode = message.get("mode")
            
            # Fallback để tương thích với cấu trúc cũ
            if not mode:
                mode, _ = _extract_mode_content(message)

            if mode in ("RAG", "Co-RAG") and index + 1 < len(messages):
                next_message = messages[index + 1]
                if next_message.get("role") == "assistant":
                    next_mode = next_message.get("mode")
                    if not next_mode:
                        next_mode, _ = _extract_mode_content(next_message)
                    
                    if next_mode and {mode, next_mode} == {"RAG", "Co-RAG"}:
                        if mode == "RAG":
                            _render_comparison_pair(message, next_message)
                        else:
                            _render_comparison_pair(next_message, message)
                        index += 2
                        continue

            _render_assistant_message(message)
            index += 1
            continue

        _render_assistant_message(message)
        index += 1


def _render_user_message(message):
    """Render a user message bubble."""
    st.markdown(
        f"""<div class="message-container">
        <div style="text-align: right; margin-bottom: 4px;">
            <small style="color: #666;">👤 You</small>
        </div>
        <div class="user-message">
            {message["content"]}
        </div>
        <div style="text-align: right; margin-top: 2px;">
            <small style="color: #999;">{message.get("timestamp", "")}</small>
        </div>
        </div>""",
        unsafe_allow_html=True
    )


def _render_assistant_message(message):
    """Render an assistant message bubble with citations."""
    mode = message.get("mode", "Assistant")
    content = message.get("content", "")
    citations = message.get("citations", [])
    
    assistant_label = f"🤖 {mode}"

    st.markdown(
        f"""<div class="message-container">
        <div style="text-align: left; margin-bottom: 4px;">
            <small style="color: #666;">{assistant_label}</small>
        </div>
        <div class="assistant-message">
            {content}
        </div>
        <div style="text-align: left; margin-top: 2px;">
            <small style="color: #999;">{message.get("timestamp", "")}</small>
        </div>
        </div>""",
        unsafe_allow_html=True
    )
    
    # Hiển thị citations nếu có
    if citations and len(citations) > 0:
        _render_citations(citations)


def _extract_mode_content(message):
    """Trích xuất mode từ content format cũ [MODE]\ncontent"""
    content = message.get("content", "")
    if not content:
        return None, ""
    
    if content.startswith("[RAG]\n"):
        return "RAG", content[len("[RAG]\n"):]
    if content.startswith("[Co-RAG]\n"):
        return "Co-RAG", content[len("[Co-RAG]\n"):]
    return None, content


def _render_citations(citations: list):
    """Render citations/sources as expandable sections."""
    if not citations:
        return
    
    st.divider()
    st.subheader("📚 Sources")
    
    for i, citation in enumerate(citations, 1):
        # Tạo title cho expander
        title = f"{i}. {citation.source_name}"
        if citation.page_number:
            title += f" (Page {citation.page_number})"
        
        with st.expander(title, expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if citation.page_number:
                    st.caption(f"📖 **Page:** {citation.page_number}")
                if citation.chunk_id:
                    st.caption(f"🔖 **Chunk ID:** {citation.chunk_id}")
                if citation.excerpt:
                    st.caption("**Excerpt:**")
                    st.markdown(f"> {citation.excerpt}...")
            
            with col2:
                if citation.relevance_score > 0:
                    st.metric("Relevance", f"{citation.relevance_score:.1%}")





def _render_comparison_pair(rag_message, co_rag_message):
    rag_timestamp = rag_message.get("timestamp", "")
    co_rag_timestamp = co_rag_message.get("timestamp", "")
    rag_content = rag_message.get("content", "")
    co_rag_content = co_rag_message.get("content", "")
    rag_citations = rag_message.get("citations", [])
    co_rag_citations = co_rag_message.get("citations", [])

    st.markdown("<div class='compare-title'>RAG vs Co-RAG</div>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            f"""<div class="compare-card compare-rag">
            <div class="compare-label">🤖 RAG</div>
            <div class="assistant-message compare-message">{rag_content}</div>
            <div class="compare-time">{rag_timestamp}</div>
            </div>""",
            unsafe_allow_html=True
        )
        if rag_citations:
            with st.expander("📚 Sources (RAG)", expanded=False):
                for i, citation in enumerate(rag_citations, 1):
                    st.caption(f"{i}. {citation.source_name}")
                    if citation.page_number:
                        st.caption(f"   Page: {citation.page_number}")
                    if citation.excerpt:
                        st.caption(f"   > {citation.excerpt}...")

    with col_right:
        st.markdown(
            f"""<div class="compare-card compare-co-rag">
            <div class="compare-label">🤖 Co-RAG</div>
            <div class="assistant-message compare-message">{co_rag_content}</div>
            <div class="compare-time">{co_rag_timestamp}</div>
            </div>""",
            unsafe_allow_html=True
        )
        if co_rag_citations:
            with st.expander("📚 Sources (Co-RAG)", expanded=False):
                for i, citation in enumerate(co_rag_citations, 1):
                    st.caption(f"{i}. {citation.source_name}")
                    if citation.page_number:
                        st.caption(f"   Page: {citation.page_number}")
                    if citation.excerpt:
                        st.caption(f"   > {citation.excerpt}...")

