from azure.identity import AzureCliCredential, get_bearer_token_provider
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from typing import List

from app.config import settings


def get_embeddings() -> AzureOpenAIEmbeddings:
    print("🔐 Initializing embeddings with Azure AD token...")

    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return AzureOpenAIEmbeddings(
        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=settings.AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
    )


def get_vectorstore() -> Chroma:
    print("📦 Loading vector store from disk...")

    return Chroma(
        persist_directory=settings.PERSIST_DIR,
        embedding_function=get_embeddings(),
    )

def get_known_sources(vectordb) -> List[str]:
    """
    Extract known source names from the underlying Chroma collection metadata.
    Works best when metadata contains 'source'.
    """
    try:
        collection = vectordb._collection
        raw = collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas", []) or []

        sources = set()
        for row in metadatas:
            if isinstance(row, dict):
                source = row.get("source")
                if source:
                    sources.add(source)

        return sorted(sources)

    except Exception:
        return []