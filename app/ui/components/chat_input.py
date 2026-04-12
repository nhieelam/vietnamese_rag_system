import streamlit as st
from datetime import datetime
from app.services import CoRAGService
from app.services import RAGService
from app.services import SessionService
from app.config import AppConfig

def render_chat_input():
    st.divider()

    # Add radio button for mode selection
    mode = st.radio(
        "Choose mode:",
        ("RAG", "Co-RAG"),
        horizontal=True,
        label_visibility="collapsed",
        key="rag_mode",
        disabled=len(SessionService.get_documents()) == 0
    )
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Type your question here...",
            key="chat_input",
            label_visibility="collapsed",
            disabled=len(SessionService.get_documents()) == 0
        )
    
    with col2:
        send_button = st.button(
            "Send 📤",
            type="primary",
            use_container_width=True,
            disabled=len(SessionService.get_documents()) == 0,
            on_click=_handle_send,
            args=(user_input, mode)
        )


def _handle_send(user_input: str, mode: str):
    if user_input and user_input.strip():
        _process_user_message(user_input, mode)
        st.session_state.chat_input = ""
    else:
        st.warning("Please enter a message")


def _process_user_message(user_input: str, mode: str):
    timestamp = datetime.now().strftime(AppConfig.TIMESTAMP_FORMAT)
    SessionService.add_message("user", user_input, timestamp)
    
    with st.spinner(f"🤔 Thinking with {mode}..."):
        try:
            answer = ""
            if mode == "RAG":
                result = RAGService.get_answer(user_input)
                if result.get("status_code") == 200:
                    answer = (result.get("answer") or "").strip()
                else:
                    answer = result.get("message") or "Không tạo được câu trả lời."
            elif mode == "Co-RAG":
                result = CoRAGService.get_answer(user_input)
                if result.get("status_code") == 200:
                    answer = (result.get("answer") or "").strip()
                else:
                    answer = result.get("message") or "Không tạo được câu trả lời."
            
            SessionService.add_message("assistant", answer, timestamp)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
