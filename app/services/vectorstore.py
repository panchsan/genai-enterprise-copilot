from typing import List

from langchain_chroma import Chroma

from app.config import settings
from app.services.llm import get_embeddings
from app.services.search_store import AzureSearchStore

_vectorstore = None


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        backend = settings.VECTOR_BACKEND.strip().lower()

        if backend == "azure_search":
            _vectorstore = AzureSearchStore()
        else:
            print("📦 Loading vector store from disk...")
            _vectorstore = Chroma(
                persist_directory=settings.CHROMA_PERSIST_DIR,
                embedding_function=get_embeddings(),
            )

    return _vectorstore


def get_known_sources(vectordb) -> List[str]:
    if hasattr(vectordb, "list_known_sources"):
        return vectordb.list_known_sources()

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