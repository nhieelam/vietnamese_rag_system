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
    # Sidebar: chunk size / overlap sliders and max overlap cap (chunk_size - margin)
    CHUNK_SLIDER_MIN = 100
    CHUNK_SLIDER_MAX = 2000
    CHUNK_SLIDER_STEP = 50
    CHUNK_OVERLAP_SLIDER_STEP = 10
    CHUNK_MAX_OVERLAP_MARGIN = 50

    CHAT_MESSAGE_MAX_WIDTH = "70%"
    USER_MESSAGE_BG_COLOR = "#007bff"
    ASSISTANT_MESSAGE_BG_COLOR = "#f1f3f4"
    
    TIMESTAMP_FORMAT = "%H:%M"
    UPLOAD_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"

    DEFAULT_TEMPERATURE = 0.3

    # HuggingFace sentence-transformers (vector embeddings)
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Session defaults: retrieval & hybrid
    DEFAULT_RETRIEVAL_K = 5
    RETRIEVAL_K_MIN = 2
    RETRIEVAL_K_MAX = 20
    DEFAULT_PER_SOURCE_K = 4
    DEFAULT_BM25_WEIGHT = 0.4
    BM25_WEIGHT_MIN = 0.0
    BM25_WEIGHT_MAX = 1.0
    BM25_WEIGHT_STEP = 0.1
    RETRIEVER_MODE_VECTOR = "vector"
    RETRIEVER_MODE_HYBRID = "hybrid"
    RETRIEVER_MODE_OPTIONS = (RETRIEVER_MODE_VECTOR, RETRIEVER_MODE_HYBRID)

    # Labels stored in chat messages (not the sidebar "Answer Mode" labels)
    MESSAGE_MODE_RAG = "RAG"
    MESSAGE_MODE_CO_RAG = "Co-RAG"
    MESSAGE_MODE_SELF_RAG = "Self-RAG"

    # RAG / retrieval tuning
    CITATION_EXCERPT_MAX_LEN = 280
    RETRIEVAL_OVERFETCH_MULTIPLIER = 5
    RETRIEVAL_OVERFETCH_MIN_DOCS = 20
    RERANK_CANDIDATE_MULTIPLIER = 3
    RERANK_CANDIDATE_MIN = 15
    LEGACY_RAG_CHAIN_RETRIEVER_K = 10
    CHUNK_OVERLAP_AUTO_DIVISOR = 5

    # PDF / scanned pages
    PDF_RENDER_DPI = 200
    MIN_IMAGE_DIMENSION_PX = 100

    # Co-RAG
    CO_RAG_HYBRID_MIN_K = 6

    # Hybrid retriever (BM25 + vector)
    HYBRID_DEFAULT_K = 10
    HYBRID_SUB_RETRIEVER_K_MULT = 2
    HYBRID_SUB_RETRIEVER_K_MIN = 10
    HYBRID_RANK_SCORE_DECAY = 0.6
    HYBRID_DEDUP_SNIPPET_LEN = 60

    # Sidebar chat history list
    CHAT_HISTORY_QUESTION_PREVIEW_LEN = 60
    CHAT_HISTORY_ANSWER_PREVIEW_LEN = 300
    CHAT_HISTORY_MAX_QUESTIONS_SHOWN = 20

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
    SELF_RAG_PASSAGE_MAX_CHARS = 1500
    SELF_RAG_EVAL_FALLBACK_SCORE = 0.5
    SELF_RAG_IRRELEVANT_FALLBACK_TOP_K = 3
    SELF_RAG_CONFIDENCE_MEAN_REL_BASE = 0.5
    SELF_RAG_CONFIDENCE_MEAN_REL_SCALE = 0.5