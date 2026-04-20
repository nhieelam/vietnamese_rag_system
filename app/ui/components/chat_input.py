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
        st.session_state.message_sent = True
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
                # Only RAG with citations
                rag_result = RAGService.get_answer_with_citations(user_input)
                
                message = {
                    "role": "assistant",
                    "content": rag_result.answer,
                    "citations": rag_result.citations,
                    "mode": "RAG"
                }
                SessionService.add_message_with_citations("assistant", message, timestamp)
            
            elif mode == AppConfig.ANSWER_MODE_CO_RAG:
                # Only Co-RAG with citations
                co_rag_result = CoRAGService.get_answer_with_citations(user_input)
                
                message = {
                    "role": "assistant",
                    "content": co_rag_result.answer,
                    "citations": co_rag_result.citations,
                    "mode": "Co-RAG"
                }
                SessionService.add_message_with_citations("assistant", message, timestamp)
            
            else:  # ANSWER_MODE_BOTH
                # Both RAG and Co-RAG with citations
                rag_result = RAGService.get_answer_with_citations(user_input)
                co_rag_result = CoRAGService.get_answer_with_citations(user_input)
                
                rag_message = {
                    "role": "assistant",
                    "content": rag_result.answer,
                    "citations": rag_result.citations,
                    "mode": "RAG"
                }
                
                co_rag_message = {
                    "role": "assistant",
                    "content": co_rag_result.answer,
                    "citations": co_rag_result.citations,
                    "mode": "Co-RAG"
                }
                
                SessionService.add_message_with_citations("assistant", rag_message, timestamp)
                SessionService.add_message_with_citations("assistant", co_rag_message, timestamp)
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
