
import streamlit as st
from app.services.session_service import SessionService


def render_chat_messages():
    chat_container = st.container()
    
    with chat_container:
        messages = SessionService.get_messages()
        
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
    """Render the complete message history."""
    for message in messages:
        if message["role"] == "user":
            _render_user_message(message)
        else:
            _render_assistant_message(message)


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
    st.markdown(
        f"""<div class="message-container">
        <div style="text-align: left; margin-bottom: 4px;">
            <small style="color: #666;">🤖 Assistant</small>
        </div>
        <div class="assistant-message">
            {message["content"]}
        </div>
        <div style="text-align: left; margin-top: 2px;">
            <small style="color: #999;">{message.get("timestamp", "")}</small>
        </div>
        </div>""",
        unsafe_allow_html=True
    )
