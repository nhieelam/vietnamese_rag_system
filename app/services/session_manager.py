import streamlit as st



def initialize_session_state():
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
    st.session_state.documents.append(doc_data)


def remove_document(index: int):
    if 0 <= index < len(st.session_state.documents):
        st.session_state.documents.pop(index)


def clear_all_documents():
    st.session_state.documents = []


def add_message(role: str, content: str, timestamp: str):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })


def clear_chat_history():
    st.session_state.messages = []


def get_documents():
    return st.session_state.documents


def get_messages():
    return st.session_state.messages


def document_exists(filename: str) -> bool:
    existing_names = [doc["name"] for doc in st.session_state.documents]
    return filename in existing_names
