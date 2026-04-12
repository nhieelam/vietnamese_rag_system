import streamlit as st
from datetime import datetime
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
            key="user_input",
            label_visibility="collapsed",
            disabled=len(SessionService.get_documents()) == 0
        )
    
    with col2:
        send_button = st.button(
            "Send 📤",
            type="primary",
            use_container_width=True,
            disabled=len(SessionService.get_documents()) == 0
        )
    
    if send_button and user_input and user_input.strip():
        _process_user_message(user_input, mode)



def _process_user_message(user_input: str, mode: str):
    timestamp = datetime.now().strftime(AppConfig.TIMESTAMP_FORMAT)
    SessionService.add_message("user", user_input, timestamp)
    
    with st.spinner(f"🤔 Thinking with {mode}..."):
        try:
            answer = ""
            if mode == "RAG":
                answer = RAGService.get_answer(user_input)
            elif mode == "Co-RAG":
                # Assuming you have a CoRAGService with a similar interface
                # from app.services import CoRAGService 
                # answer = CoRAGService.get_answer(user_input)
                st.warning("Co-RAG service is not implemented yet. Using RAG as fallback.")
                answer = RAGService.get_answer(user_input) # Fallback to RAG for now
            
            SessionService.add_message("assistant", answer, timestamp)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
