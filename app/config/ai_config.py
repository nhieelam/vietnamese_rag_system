import os
from dotenv import load_dotenv

load_dotenv()

class AIConfig:
    MODAL_NAME = "paraphrase-multilingual-mpnet-base-v2"

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")

    @classmethod
    def get_llm_instance(cls):
        """Initializes and returns the LLM instance based on the provider."""
        provider = cls.LLM_PROVIDER.strip().lower()
        
        if provider == "ollama":
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                model=cls.OLLAMA_LLM_MODEL,
                temperature=0.3,
            )
    
        elif provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=cls.GROQ_LLM_MODEL,
                api_key=cls.GROQ_API_KEY,
                temperature=0.3,
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}'. Use 'ollama' or 'groq'")
