import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    AZURE_CHAT_API_VERSION = os.getenv("AZURE_CHAT_API_VERSION")
    AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

    AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
    AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
    AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

    PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    DB_PATH = os.getenv("DB_PATH", "chat_memory.db")

    APP_ENV = os.getenv("APP_ENV", "dev")
    SHOW_DEBUG = _get_bool("SHOW_DEBUG", True)

    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "1.10"))
    RETRIEVAL_OVERLAP_THRESHOLD = float(os.getenv("RETRIEVAL_OVERLAP_THRESHOLD", "0.10"))
    RETRIEVAL_HARD_FILTER_ENABLED = _get_bool("RETRIEVAL_HARD_FILTER_ENABLED", True)
    ALLOW_DIRECT_LLM_FALLBACK = _get_bool("ALLOW_DIRECT_LLM_FALLBACK", False)

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    MAX_CHAT_HISTORY_MESSAGES = int(os.getenv("MAX_CHAT_HISTORY_MESSAGES", "6"))
    LLM_TEMPERATURE_DETERMINISTIC = float(os.getenv("LLM_TEMPERATURE_DETERMINISTIC", "0"))
    LLM_TEMPERATURE_DEFAULT = float(os.getenv("LLM_TEMPERATURE_DEFAULT", "0"))


settings = Settings()