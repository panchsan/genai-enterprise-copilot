import argparse
import os
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.llm import get_embeddings
from app.services.logging_utils import get_logger
from app.services.metadata_utils import build_source_aliases, normalize_text

logger = get_logger("scripts.ingest")


def build_document_title(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(word.capitalize() for word in cleaned.split())


def get_doc_type(filename: str) -> str:
    lower_name = filename.lower()

    if "hr" in lower_name and "policy" in lower_name:
        return "policy"
    if "finance" in lower_name and "policy" in lower_name:
        return "policy"
    if "policy" in lower_name:
        return "policy"
    if "onboarding" in lower_name:
        return "onboarding"
    if lower_name.endswith(".csv"):
        return "csv"
    if lower_name.endswith(".pdf"):
        return "pdf"

    return "general"


def get_department(filename: str) -> str:
    lower_name = filename.lower()

    if "hr" in lower_name:
        return "HR"
    if "finance" in lower_name:
        return "Finance"
    if "onboarding" in lower_name:
        return "HR"

    return "General"


def build_clean_metadata(filename: str, original_metadata: dict) -> dict:
    document_title = build_document_title(filename)
    source_aliases = build_source_aliases(filename, document_title)

    clean_metadata = {
        "source": filename,
        "source_normalized": normalize_text(filename),
        "document_id": filename,
        "doc_type": get_doc_type(filename),
        "department": get_department(filename),
        "document_title": document_title,
        "document_title_normalized": normalize_text(document_title),
        "source_aliases": "|".join(source_aliases),
    }

    if "page" in original_metadata:
        clean_metadata["page"] = original_metadata["page"]

    if "page_label" in original_metadata:
        clean_metadata["page_label"] = original_metadata["page_label"]

    return clean_metadata


def get_loader(filepath: str):
    lower = filepath.lower()

    if lower.endswith(".txt"):
        return TextLoader(filepath, encoding="utf-8")
    if lower.endswith(".pdf"):
        return PyPDFLoader(filepath)
    if lower.endswith(".csv"):
        return CSVLoader(filepath, encoding="utf-8")

    raise ValueError(f"Unsupported file type: {filepath}")


def load_single_document(filepath: str):
    filename = os.path.basename(filepath)
    loader = get_loader(filepath)
    docs = loader.load()

    for doc in docs:
        doc.metadata = build_clean_metadata(filename, doc.metadata or {})

    logger.info(f"Loaded file={filename} | raw_docs={len(docs)}")
    return docs


def load_all_documents():
    logger.info(f"Loading documents from dir={settings.INGEST_DATA_DIR}")
    documents = []

    if not os.path.exists(settings.INGEST_DATA_DIR):
        raise FileNotFoundError(
            f"Data directory not found: {settings.INGEST_DATA_DIR}"
        )

    for filename in sorted(os.listdir(settings.INGEST_DATA_DIR)):
        filepath = os.path.join(settings.INGEST_DATA_DIR, filename)

        if not os.path.isfile(filepath):
            continue

        try:
            docs = load_single_document(filepath)
            documents.extend(docs)
        except ValueError as exc:
            logger.warning(f"Skipping file={filename} | reason={exc}")

    logger.info(f"Total raw documents loaded={len(documents)}")
    return documents


def split_documents(documents):
    logger.info(
        f"Splitting documents | chunk_size={settings.CHUNK_SIZE} | "
        f"chunk_overlap={settings.CHUNK_OVERLAP}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        document_id = chunk.metadata["document_id"]
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_doc_ref"] = document_id

    logger.info(f"Created chunks={len(chunks)}")
    return chunks


def get_vectorstore():
    return Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def full_reindex():
    logger.info("Running FULL reindex")

    documents = load_all_documents()
    chunks = split_documents(documents)

    vectordb = get_vectorstore()

    existing = vectordb.get()
    existing_ids = existing.get("ids", [])

    if existing_ids:
        vectordb.delete(ids=existing_ids)
        logger.info(f"Deleted existing vectors={len(existing_ids)}")

    ids = [str(uuid.uuid4()) for _ in chunks]
    vectordb.add_documents(documents=chunks, ids=ids)

    logger.info(f"Full reindex complete | indexed_chunks={len(chunks)}")


def delete_document(document_name: str):
    logger.info(f"Deleting document from vector DB | document={document_name}")

    vectordb = get_vectorstore()
    existing = vectordb.get(where={"document_id": document_name})
    ids = existing.get("ids", [])

    if not ids:
        logger.info("No indexed chunks found for this document")
        return

    vectordb.delete(ids=ids)
    logger.info(f"Deleted chunks={len(ids)} | document={document_name}")


def upsert_document(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    filename = os.path.basename(filepath)
    logger.info(f"Upserting document | file={filename}")

    vectordb = get_vectorstore()

    existing = vectordb.get(where={"document_id": filename})
    existing_ids = existing.get("ids", [])

    if existing_ids:
        vectordb.delete(ids=existing_ids)
        logger.info(f"Deleted old chunks={len(existing_ids)} | document={filename}")

    documents = load_single_document(filepath)
    chunks = split_documents(documents)

    ids = [str(uuid.uuid4()) for _ in chunks]
    vectordb.add_documents(documents=chunks, ids=ids)

    logger.info(f"Upsert complete | indexed_chunks={len(chunks)} | document={filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Document ingestion utility")

    parser.add_argument(
        "--mode",
        choices=["full", "upsert", "delete"],
        required=True,
        help="Ingestion mode",
    )

    parser.add_argument(
        "--file",
        required=False,
        help="Path to a specific file for upsert/delete modes",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "full":
        full_reindex()
        return

    if not args.file:
        raise ValueError("--file is required for upsert and delete modes")

    file_path = Path(args.file)
    filename = file_path.name

    if args.mode == "upsert":
        upsert_document(str(file_path))
        return

    if args.mode == "delete":
        delete_document(filename)
        return


if __name__ == "__main__":
    main()