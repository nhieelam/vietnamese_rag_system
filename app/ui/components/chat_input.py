
import streamlit as st
from datetime import datetime
from app.services import CoRAGService
from app.services import SessionService
from app.config import AppConfig

def render_chat_input():
    st.divider()
    
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
        _process_user_message(user_input)



def _process_user_message(user_input: str):
    timestamp = datetime.now().strftime(AppConfig.TIMESTAMP_FORMAT)
    SessionService.add_message("user", user_input, timestamp)
    
    with st.spinner("🤔 Thinking..."):
        try:
            result = CoRAGService.get_answer(user_input)
            if result.get("status_code") == 200:
                text = (result.get("answer") or "").strip()
            else:
                text = result.get("message") or "Không tạo được câu trả lời."

            SessionService.add_message("assistant", text, timestamp)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
