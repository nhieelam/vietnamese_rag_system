import os
import streamlit as st

class AppConfig:

    APP_NAME = "Vietnamese RAG Assistant"
    APP_VERSION = "v1.0"
    APP_DESCRIPTION = "Ask questions about your uploaded documents and get comprehensive answers"
    
    ALLOWED_FILE_TYPES = ["pdf", "png", "jpg", "jpeg"]
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

    DB_DIRECTORY = "./chroma_db"
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    LLM_MODEL = "gpt-3.5-turbo"  
    LLM_TEMPERATURE = 0.3