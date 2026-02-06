import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


class SessionService:
    # ---------- Internal ----------
    @staticmethod
    def _has_context():
        try:
            return get_script_run_ctx() is not None
        except Exception:
            return False

    # ---------- Init ----------
    @classmethod
    def initialize(cls):
        if not cls._has_context():
            return
        
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None

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

    # ---------- Vector Store ----------
    @classmethod
    def set_vector_store(cls, vector_store):
        if cls._has_context():
            st.session_state.vector_store = vector_store

    @classmethod
    def get_vector_store(cls):
        if not cls._has_context():
            return None
        return st.session_state.get("vector_store")

    @classmethod
    def clear_vector_store(cls):
        if cls._has_context():
            st.session_state.vector_store = None

    # ---------- Documents ----------
    @classmethod
    def add_document(cls, doc_data: dict):
        if cls._has_context():
            st.session_state.documents.append(doc_data)

    @classmethod
    def remove_document(cls, index: int):
        if cls._has_context() and 0 <= index < len(st.session_state.documents):
            st.session_state.documents.pop(index)

    @classmethod
    def clear_documents(cls):
        if cls._has_context():
            st.session_state.documents = []

    @classmethod
    def get_documents(cls):
        if not cls._has_context():
            return []

        if "documents" not in st.session_state:
            st.session_state.documents = []

        return st.session_state.documents

    @classmethod
    def document_exists(cls, filename: str) -> bool:
        if not cls._has_context():
            return False

        return any(
            doc.get("name") == filename
            for doc in st.session_state.get("documents", [])
        )

    # ---------- Messages ----------
    @classmethod
    def add_message(cls, role: str, content: str, timestamp: str):
        if cls._has_context():
            st.session_state.messages.append({
                "role": role,
                "content": content,
                "timestamp": timestamp
            })

    @classmethod
    def clear_chat_history(cls):
        if cls._has_context():
            st.session_state.messages = []

    @classmethod
    def get_messages(cls):
        if not cls._has_context():
            return []

        if "messages" not in st.session_state:
            st.session_state.messages = []

        return st.session_state.messages
