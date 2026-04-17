import os
class AIConfig:
    MODAL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
    OLLAMA_LLM_MODEL = "qwen2.5:7b"  

    @classmethod
    def get_llm_instance(cls):
        """Initializes and returns the LLM instance based on the provider."""
        if cls.LLM_PROVIDER == "ollama":
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                model=cls.OLLAMA_LLM_MODEL,
                temperature=0.3,
            )

        raise ValueError(f"Unsupported LLM provider: {cls.LLM_PROVIDER}")