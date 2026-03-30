from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    APP_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    SHOW_DEBUG: bool = True
    SDK_LOG_LEVEL: str = "WARNING"

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_CHAT_DEPLOYMENT: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str
    AZURE_CHAT_API_VERSION: str = "2024-02-01"
    AZURE_EMBEDDING_API_VERSION: str = "2024-02-01"

    # Azure identity
    AZURE_TENANT_ID: str | None = None
    AZURE_CLIENT_ID: str | None = None
    AZURE_CLIENT_SECRET: str | None = None

    # Current runtime/local fallback paths
    DB_PATH: str = "chat_memory.db"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    INGEST_DATA_DIR: str = "./data"

    # Backend switches
    VECTOR_BACKEND: str = "chroma"          # chroma | azure_search
    DOCUMENT_SOURCE_BACKEND: str = "local"  # local | blob

    # Azure AI Search
    AZURE_SEARCH_ENDPOINT: str | None = None
    AZURE_SEARCH_INDEX_NAME: str = "copilot-index"
    AZURE_SEARCH_API_KEY: str | None = None
    AZURE_SEARCH_VECTOR_DIMENSIONS: int = 1536

    # Azure Blob Storage
    AZURE_BLOB_ACCOUNT_URL: str | None = None
    AZURE_BLOB_CONTAINER_NAME: str = "documents"
    AZURE_STORAGE_CONNECTION_STRING: str | None = None

    # Retrieval
    RETRIEVAL_TOP_K: int = 4
    RETRIEVAL_SCORE_THRESHOLD: float = 1.10
    GROUNDED_SCORE_THRESHOLD: float = 1.0
    RETRIEVAL_HARD_FILTER_ENABLED: bool = True

    # Ingestion
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50

    # LLM behavior
    MAX_CHAT_HISTORY_MESSAGES: int = 6
    LLM_TEMPERATURE_DETERMINISTIC: float = 0.0
    LLM_TEMPERATURE_DEFAULT: float = 0.0

    # UI
    API_BASE_URL: str = "http://backend:8000"

    @property
    def is_dev(self) -> bool:
        return self.APP_ENV.strip().lower() == "dev"

    @property
    def is_prod(self) -> bool:
        return self.APP_ENV.strip().lower() == "prod"

    @property
    def show_debug_ui(self) -> bool:
        return self.is_dev and self.SHOW_DEBUG


settings = Settings()