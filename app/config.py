import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    AZURE_CHAT_API_VERSION = os.getenv("AZURE_CHAT_API_VERSION")
    AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

    PERSIST_DIR = "./chroma_db"
    DATA_DIR = "data"


settings = Settings()