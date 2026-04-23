import os
import streamlit as st

class AppConfig:

    APP_NAME = "Vietnamese RAG Assistant"
    APP_VERSION = "v1.0"
    APP_DESCRIPTION = "Ask questions about your uploaded documents and get comprehensive answers"
    
    ALLOWED_FILE_TYPES = ["pdf", "png", "jpg", "jpeg", "docx"]
    MAX_FILE_SIZE_MB = 10
    
    DEFAULT_MAX_TOKENS = 800
    MIN_MAX_TOKENS = 100
    MAX_MAX_TOKENS = 2000
    MAX_TOKENS_STEP = 100

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    CHAT_MESSAGE_MAX_WIDTH = "70%"
    USER_MESSAGE_BG_COLOR = "#007bff"
    ASSISTANT_MESSAGE_BG_COLOR = "#f1f3f4"
    
    TIMESTAMP_FORMAT = "%H:%M"
    UPLOAD_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"

    # Answer Mode Configuration
    ANSWER_MODE_RAG = "RAG Only"
    ANSWER_MODE_CO_RAG = "Co-RAG Only"
    ANSWER_MODE_BOTH = "RAG & Co-RAG"
    ANSWER_MODE_SELF_RAG = "Self-RAG"
    ANSWER_MODE_DEFAULT = ANSWER_MODE_BOTH
    ANSWER_MODE_ORDER = [
        ANSWER_MODE_RAG,
        ANSWER_MODE_CO_RAG,
        ANSWER_MODE_BOTH,
        ANSWER_MODE_SELF_RAG,
    ]

    CO_RAG_K_PER_SUBQUERY = 6
    CO_RAG_MAX_SUB_QUERIES = 4
    CO_RAG_FALLBACK_K = 12

    # Self-RAG
    SELF_RAG_CONFIDENCE_THRESHOLD = 0.5
    SELF_RAG_MAX_HOPS = 2