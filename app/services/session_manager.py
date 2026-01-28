import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


def _has_context():
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False



def initialize_session_state():
    if not _has_context():
        return  
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.3
    
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 800


def add_document(doc_data: dict):
    if not _has_context():
        return
    st.session_state.documents.append(doc_data)


def remove_document(index: int):
    if not _has_context():
        return
    if 0 <= index < len(st.session_state.documents):
        st.session_state.documents.pop(index)


def clear_all_documents():
    if not _has_context():
        return
    st.session_state.documents = []


def add_message(role: str, content: str, timestamp: str):
    if not _has_context():
        return
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })


def clear_chat_history():
    if not _has_context():
        return
    st.session_state.messages = []


def get_documents():
    if not _has_context():
        return [] 
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    return st.session_state.documents


def get_messages():
    if not _has_context():
        return [] 
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages


def document_exists(filename: str) -> bool:
    if not _has_context():
        return False 
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    existing_names = [doc["name"] for doc in st.session_state.documents]
    return filename in existing_names
