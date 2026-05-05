import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from streamlit.runtime.scriptrunner import get_script_run_ctx

from app.config import AppConfig

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
            st.session_state.temperature = AppConfig.DEFAULT_TEMPERATURE

        if "max_tokens" not in st.session_state:
            st.session_state.max_tokens = AppConfig.DEFAULT_MAX_TOKENS

        if "pdf_files" not in st.session_state:
            # Map document_id -> {"name": str, "bytes": bytes}
            st.session_state.pdf_files = {}

        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = AppConfig.CHUNK_SIZE
        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = AppConfig.CHUNK_OVERLAP
        if "retrieval_k" not in st.session_state:
            st.session_state.retrieval_k = AppConfig.DEFAULT_RETRIEVAL_K
        if "per_source_k" not in st.session_state:
            st.session_state.per_source_k = AppConfig.DEFAULT_PER_SOURCE_K

        if "retriever_mode" not in st.session_state:
            st.session_state.retriever_mode = AppConfig.RETRIEVER_MODE_VECTOR
        if "bm25_weight" not in st.session_state:
            st.session_state.bm25_weight = AppConfig.DEFAULT_BM25_WEIGHT
        if "use_reranker" not in st.session_state:
            st.session_state.use_reranker = False
        if "doc_filter" not in st.session_state:
            st.session_state.doc_filter = []
        if "file_type_filter" not in st.session_state:
            st.session_state.file_type_filter = []
        if "all_chunks" not in st.session_state:
            # List[Document] — keep a mirror of all chunks for BM25 retriever
            st.session_state.all_chunks = []

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
    def add_message_with_citations(cls, role: str, message: dict, timestamp: str):
        """Lưu message, convert Citation -> dict để serialize được."""
        if not cls._has_context():
            return

        payload = dict(message)
        payload["role"] = role
        payload["timestamp"] = timestamp

        raw_citations = payload.get("citations") or []
        serialized = []
        for c in raw_citations:
            if hasattr(c, "to_dict"):
                serialized.append(c.to_dict())
            elif isinstance(c, dict):
                serialized.append(c)
        payload["citations"] = serialized

        st.session_state.messages.append(payload)

    @classmethod
    def store_pdf(cls, document_id: int, name: str, data: bytes):
        if not cls._has_context() or not data:
            return
        files = st.session_state.setdefault("pdf_files", {})
        files[int(document_id)] = {"name": name, "bytes": data}

    @classmethod
    def get_pdf(cls, document_id: int):
        if not cls._has_context():
            return None
        files = st.session_state.get("pdf_files", {})
        return files.get(int(document_id))

    @classmethod
    def remove_pdf(cls, document_id: int):
        if cls._has_context():
            st.session_state.get("pdf_files", {}).pop(int(document_id), None)

    @classmethod
    def clear_all_pdfs(cls):
        if cls._has_context():
            st.session_state.pdf_files = {}

    @classmethod
    def add_chunks(cls, docs):
        """Thêm list[Document] vào session để BM25 có thể dùng."""
        if not cls._has_context() or not docs:
            return
        existing = st.session_state.setdefault("all_chunks", [])
        existing.extend(docs)

    @classmethod
    def get_all_chunks(cls):
        if not cls._has_context():
            return []
        return st.session_state.get("all_chunks", []) or []

    @classmethod
    def clear_all_chunks(cls):
        if cls._has_context():
            st.session_state.all_chunks = []

    @classmethod
    def get_doc_filter(cls) -> list:
        if not cls._has_context():
            return []
        return list(st.session_state.get("doc_filter", []) or [])

    @classmethod
    def get_file_type_filter(cls) -> list:
        if not cls._has_context():
            return []
        return list(st.session_state.get("file_type_filter", []) or [])

    @classmethod
    def get_retriever_mode(cls) -> str:
        if not cls._has_context():
            return AppConfig.RETRIEVER_MODE_HYBRID
        return (
            st.session_state.get("retriever_mode", AppConfig.RETRIEVER_MODE_HYBRID)
            or AppConfig.RETRIEVER_MODE_HYBRID
        )

    @classmethod
    def get_bm25_weight(cls) -> float:
        if not cls._has_context():
            return AppConfig.DEFAULT_BM25_WEIGHT
        try:
            return float(
                st.session_state.get("bm25_weight", AppConfig.DEFAULT_BM25_WEIGHT)
            )
        except Exception:
            return AppConfig.DEFAULT_BM25_WEIGHT

    @classmethod
    def get_use_reranker(cls) -> bool:
        if not cls._has_context():
            return False
        return bool(st.session_state.get("use_reranker", True))

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
    def get_chunk_params(cls) -> dict:
        if not cls._has_context():
            return {
                "chunk_size": AppConfig.CHUNK_SIZE,
                "chunk_overlap": AppConfig.CHUNK_OVERLAP,
            }
        return {
            "chunk_size": int(st.session_state.get("chunk_size", AppConfig.CHUNK_SIZE)),
            "chunk_overlap": int(st.session_state.get("chunk_overlap", AppConfig.CHUNK_OVERLAP)),
        }

    @classmethod
    def get_retrieval_params(cls) -> dict:
        if not cls._has_context():
            return {
                "k": AppConfig.DEFAULT_RETRIEVAL_K,
                "per_source_k": AppConfig.DEFAULT_PER_SOURCE_K,
            }
        return {
            "k": int(
                st.session_state.get("retrieval_k", AppConfig.DEFAULT_RETRIEVAL_K)
            ),
            "per_source_k": int(
                st.session_state.get(
                    "per_source_k", AppConfig.DEFAULT_PER_SOURCE_K
                )
            ),
        }

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