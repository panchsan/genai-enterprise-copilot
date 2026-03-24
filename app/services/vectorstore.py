from typing import List

from langchain_chroma import Chroma

from app.config import settings
from app.services.llm import get_embeddings


_vectorstore = None


def get_vectorstore() -> Chroma:
    global _vectorstore

    if _vectorstore is None:
        print("📦 Loading vector store from disk...")
        _vectorstore = Chroma(
            persist_directory=settings.PERSIST_DIR,
            embedding_function=get_embeddings(),
        )

    return _vectorstore


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