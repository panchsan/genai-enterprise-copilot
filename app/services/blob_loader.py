from __future__ import annotations

import tempfile
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader

from app.config import settings
from app.services.logging_utils import get_logger

logger = get_logger("app.blob_loader")

_SUPPORTED_SUFFIXES = {".txt", ".pdf", ".csv"}

_blob_service_client = None


def get_blob_service_client() -> BlobServiceClient:
    global _blob_service_client

    if _blob_service_client is not None:
        return _blob_service_client

    if settings.AZURE_STORAGE_CONNECTION_STRING:
        _blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
    elif settings.AZURE_BLOB_ACCOUNT_URL:
        credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        _blob_service_client = BlobServiceClient(
            account_url=settings.AZURE_BLOB_ACCOUNT_URL,
            credential=credential,
        )
    else:
        raise ValueError(
            "Either AZURE_STORAGE_CONNECTION_STRING or AZURE_BLOB_ACCOUNT_URL must be set"
        )

    return _blob_service_client


def list_supported_blobs() -> list[str]:
    container_client = get_blob_service_client().get_container_client(
        settings.AZURE_BLOB_CONTAINER_NAME
    )

    names: list[str] = []
    for blob in container_client.list_blobs():
        suffix = Path(blob.name).suffix.lower()
        if suffix in _SUPPORTED_SUFFIXES:
            names.append(blob.name)

    return sorted(names)


def load_blob_document(blob_name: str):
    container_client = get_blob_service_client().get_container_client(
        settings.AZURE_BLOB_CONTAINER_NAME
    )
    blob_client = container_client.get_blob_client(blob_name)

    suffix = Path(blob_name).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported blob type: {blob_name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(blob_name).name

        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        loader = _get_loader(local_path)
        docs = loader.load()

    logger.info(f"Loaded blob={blob_name} | raw_docs={len(docs)}")
    return docs


def _get_loader(filepath: Path):
    lower = str(filepath).lower()

    if lower.endswith(".txt"):
        return TextLoader(str(filepath), encoding="utf-8")
    if lower.endswith(".pdf"):
        return PyPDFLoader(str(filepath))
    if lower.endswith(".csv"):
        return CSVLoader(str(filepath), encoding="utf-8")

    raise ValueError(f"Unsupported file type: {filepath}")