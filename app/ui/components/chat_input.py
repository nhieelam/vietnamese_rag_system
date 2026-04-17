import streamlit as st
from datetime import datetime
from app.services import CoRAGService
from app.services import RAGService
from app.services import SessionService
from app.config import AppConfig


def render_chat_input():
    st.divider()
    
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
        st.button(
            "Send 📤",
            type="primary",
            use_container_width=True,
            disabled=len(SessionService.get_documents()) == 0,
            on_click=_handle_send,
            args=(user_input,)
        )


def _handle_send(user_input: str):
    if user_input and user_input.strip():
        st.session_state.chat_input = ""
        _process_user_message(user_input)
    else:
        st.warning("Please enter a message")


def _process_user_message(user_input: str):
    timestamp = datetime.now().strftime(AppConfig.TIMESTAMP_FORMAT)
    SessionService.add_message("user", user_input, timestamp)
    
    # Get the selected answer mode
    mode = st.session_state.get("answer_mode", AppConfig.ANSWER_MODE_DEFAULT)
    if mode not in AppConfig.ANSWER_MODE_ORDER:
        mode = AppConfig.ANSWER_MODE_DEFAULT

    if mode == AppConfig.ANSWER_MODE_BOTH:
        spinner_msg = "🤔 Thinking with RAG and Co-RAG..."
    elif mode == AppConfig.ANSWER_MODE_RAG:
        spinner_msg = "🤔 Thinking with RAG..."
    else:
        spinner_msg = "🤔 Thinking with Co-RAG..."

    with st.spinner(spinner_msg):
        try:
            if mode == AppConfig.ANSWER_MODE_RAG:
                # Only RAG
                rag_result = RAGService.get_answer(user_input)
                if rag_result.get("status_code") == 200:
                    rag_answer = (rag_result.get("answer") or "").strip()
                else:
                    rag_answer = rag_result.get("message") or "Không tạo được câu trả lời."
                
                SessionService.add_message("assistant", f"[RAG]\n{rag_answer}", timestamp)
            
            elif mode == AppConfig.ANSWER_MODE_CO_RAG:
                # Only Co-RAG
                co_rag_result = CoRAGService.get_answer(user_input)
                if co_rag_result.get("status_code") == 200:
                    co_rag_answer = (co_rag_result.get("answer") or "").strip()
                else:
                    co_rag_answer = co_rag_result.get("message") or "Không tạo được câu trả lời."
                
                SessionService.add_message("assistant", f"[Co-RAG]\n{co_rag_answer}", timestamp)
            
            else:  # ANSWER_MODE_BOTH
                # Both RAG and Co-RAG
                rag_result = RAGService.get_answer(user_input)
                if rag_result.get("status_code") == 200:
                    rag_answer = (rag_result.get("answer") or "").strip()
                else:
                    rag_answer = rag_result.get("message") or "Không tạo được câu trả lời."

                co_rag_result = CoRAGService.get_answer(user_input)
                if co_rag_result.get("status_code") == 200:
                    co_rag_answer = (co_rag_result.get("answer") or "").strip()
                else:
                    co_rag_answer = co_rag_result.get("message") or "Không tạo được câu trả lời."

                SessionService.add_message("assistant", f"[RAG]\n{rag_answer}", timestamp)
                SessionService.add_message("assistant", f"[Co-RAG]\n{co_rag_answer}", timestamp)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
