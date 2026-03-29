from __future__ import annotations

from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from langchain_core.documents import Document

from app.config import settings
from app.services.llm import get_embeddings
from app.services.logging_utils import get_logger

logger = get_logger("app.search_store")

_search_credential = None


def _get_search_credential():
    global _search_credential

    if _search_credential is not None:
        return _search_credential

    if settings.AZURE_SEARCH_API_KEY:
        _search_credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
    else:
        _search_credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=True
        )

    return _search_credential


def _quote(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _build_search_filter(filters: dict | None):
    if not filters:
        return None

    if "$and" in filters:
        clauses = []
        for item in filters["$and"]:
            built = _build_search_filter(item)
            if built:
                clauses.append(f"({built})")
        return " and ".join(clauses) if clauses else None

    clauses = []
    for key, value in filters.items():
        if value is None or value == "":
            continue
        clauses.append(f"{key} eq {_quote(value)}")

    return " and ".join(clauses) if clauses else None


def _normalize_search_score(score: float | None) -> float | None:
    if score is None:
        return None

    # Your retrieval pipeline currently expects lower-is-better scores,
    # like Chroma distance. Azure AI Search returns higher-is-better.
    # Convert it to pseudo-distance so retrieve.py can remain unchanged.
    return 1.0 / max(float(score), 1e-9)


class AzureSearchStore:
    def __init__(self):
        if not settings.AZURE_SEARCH_ENDPOINT:
            raise ValueError("AZURE_SEARCH_ENDPOINT is required")
        if not settings.AZURE_SEARCH_INDEX_NAME:
            raise ValueError("AZURE_SEARCH_INDEX_NAME is required")

        self._embedding_fn = get_embeddings()

        self._search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            credential=_get_search_credential(),
        )

        self._index_client = SearchIndexClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            credential=_get_search_credential(),
        )

    def ensure_index(self):
        existing_names = {name for name in self._index_client.list_index_names()}
        if settings.AZURE_SEARCH_INDEX_NAME in existing_names:
            return

        index = SearchIndex(
            name=settings.AZURE_SEARCH_INDEX_NAME,
            fields=[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="source_normalized", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="department", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="document_title", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="document_title_normalized", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="source_aliases", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="chunk_doc_ref", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="page_label", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=settings.AZURE_SEARCH_VECTOR_DIMENSIONS,
                    vector_search_profile_name="default-vector-profile",
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw",
                    )
                ],
            ),
        )

        self._index_client.create_index(index)
        logger.info(f"Created Azure AI Search index: {settings.AZURE_SEARCH_INDEX_NAME}")

    def add_documents(self, documents: list[Document], ids: list[str]):
        if len(documents) != len(ids):
            raise ValueError("documents and ids length mismatch")

        texts = [doc.page_content for doc in documents]
        vectors = self._embedding_fn.embed_documents(texts)

        payload = []
        for doc, doc_id, vector in zip(documents, ids, vectors, strict=True):
            metadata = dict(doc.metadata or {})
            payload.append(
                {
                    "id": doc_id,
                    "document_id": metadata.get("document_id"),
                    "source": metadata.get("source"),
                    "source_normalized": metadata.get("source_normalized"),
                    "doc_type": metadata.get("doc_type"),
                    "department": metadata.get("department"),
                    "document_title": metadata.get("document_title"),
                    "document_title_normalized": metadata.get("document_title_normalized"),
                    "source_aliases": metadata.get("source_aliases"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_doc_ref": metadata.get("chunk_doc_ref"),
                    "page": metadata.get("page"),
                    "page_label": metadata.get("page_label"),
                    "content": doc.page_content,
                    "content_vector": vector,
                }
            )

        results = self._search_client.upload_documents(payload)
        failed = [r.key for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(f"Failed to upload search documents: {failed}")

    def delete(self, ids: list[str]):
        if not ids:
            return

        docs = [{"id": doc_id} for doc_id in ids]
        self._search_client.delete_documents(docs)

    def get(self, where: dict | None = None, include: list[str] | None = None):
        filter_expr = _build_search_filter(where)

        results = self._search_client.search(
            search_text="*",
            filter=filter_expr,
            top=1000,
            select=["id", "document_id", "source", "page", "page_label"],
        )

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for row in results:
            ids.append(row["id"])
            metadatas.append(
                {
                    "document_id": row.get("document_id"),
                    "source": row.get("source"),
                    "page": row.get("page"),
                    "page_label": row.get("page_label"),
                }
            )

        response: dict[str, Any] = {"ids": ids}
        if include and "metadatas" in include:
            response["metadatas"] = metadatas

        return response

    def similarity_search_with_score(self, query: str, k: int = 4, filter: dict | None = None):
        query_vector = self._embedding_fn.embed_query(query)

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=k,
            fields="content_vector",
        )

        results = self._search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=_build_search_filter(filter),
            top=k,
            select=[
                "id",
                "content",
                "document_id",
                "source",
                "source_normalized",
                "doc_type",
                "department",
                "document_title",
                "document_title_normalized",
                "source_aliases",
                "chunk_id",
                "chunk_doc_ref",
                "page",
                "page_label",
            ],
        )

        output = []
        for row in results:
            metadata = {
                "id": row.get("id"),
                "document_id": row.get("document_id"),
                "source": row.get("source"),
                "source_normalized": row.get("source_normalized"),
                "doc_type": row.get("doc_type"),
                "department": row.get("department"),
                "document_title": row.get("document_title"),
                "document_title_normalized": row.get("document_title_normalized"),
                "source_aliases": row.get("source_aliases"),
                "chunk_id": row.get("chunk_id"),
                "chunk_doc_ref": row.get("chunk_doc_ref"),
                "page": row.get("page"),
                "page_label": row.get("page_label"),
            }

            output.append(
                (
                    Document(
                        page_content=row.get("content", ""),
                        metadata=metadata,
                    ),
                    _normalize_search_score(row.get("@search.score")),
                )
            )

        return output

    def list_known_sources(self) -> list[str]:
        results = self._search_client.search(
            search_text="*",
            facets=["source,count:100"],
            top=0,
        )
        facets = getattr(results, "get_facets", lambda: {})() or {}
        source_facets = facets.get("source", []) or []
        return sorted({item.get("value") for item in source_facets if item.get("value")})