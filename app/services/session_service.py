import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Store chat history objects in a simple in-memory dictionary.
# This prevents LangChain from modifying Streamlit's session_state directly.
store = {}

class SessionService:
    @staticmethod
    def _has_context():
        try:
            return get_script_run_ctx() is not None
        except Exception:
            return False

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
            # Also clear the in-memory store
            if "default_session" in store:
                store["default_session"].clear()

    @classmethod
    def get_messages_for_ui(cls):
        """Gets messages formatted as a list of dicts for UI rendering."""
        if not cls._has_context():
            return []
        return st.session_state.get("messages", [])

    @classmethod
    def get_chat_history(cls, session_id: str = "default"):
        """
        Gets the LangChain ChatMessageHistory object from an in-memory store.
        """
        if session_id not in store:
            history = ChatMessageHistory()
            for msg in cls.get_messages_for_ui():
                if msg["role"] == "user":
                    history.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    history.add_ai_message(msg["content"])
            store[session_id] = history
        return store[session_id]