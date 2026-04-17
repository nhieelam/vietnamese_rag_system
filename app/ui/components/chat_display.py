import streamlit as st
from app.services.session_service import SessionService

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
            mode, _ = _extract_mode_content(message)

            if mode in ("RAG", "Co-RAG") and index + 1 < len(messages):
                next_message = messages[index + 1]
                if next_message.get("role") == "assistant":
                    next_mode, _ = _extract_mode_content(next_message)
                    if {mode, next_mode} == {"RAG", "Co-RAG"}:
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
    """Render an assistant message bubble."""
    mode, content = _extract_mode_content(message)
    assistant_label = "🤖 Assistant"

    if mode == "RAG":
        assistant_label = "🤖 RAG"
    elif mode == "Co-RAG":
        assistant_label = "🤖 Co-RAG"

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


def _extract_mode_content(message):
    content = message["content"]
    if content.startswith("[RAG]\n"):
        return "RAG", content[len("[RAG]\n"):]
    if content.startswith("[Co-RAG]\n"):
        return "Co-RAG", content[len("[Co-RAG]\n"):]
    return None, content


def _render_comparison_pair(rag_message, co_rag_message):
    rag_timestamp = rag_message.get("timestamp", "")
    co_rag_timestamp = co_rag_message.get("timestamp", "")
    _, rag_content = _extract_mode_content(rag_message)
    _, co_rag_content = _extract_mode_content(co_rag_message)

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

    with col_right:
        st.markdown(
            f"""<div class="compare-card compare-co-rag">
            <div class="compare-label">🤖 Co-RAG</div>
            <div class="assistant-message compare-message">{co_rag_content}</div>
            <div class="compare-time">{co_rag_timestamp}</div>
            </div>""",
            unsafe_allow_html=True
        )
