import argparse
import os
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.vectorstore import get_embeddings


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
    clean_metadata = {
        "source": filename,
        "document_id": filename,
        "doc_type": get_doc_type(filename),
        "department": get_department(filename),
    }

    if "page" in original_metadata:
        clean_metadata["page"] = original_metadata["page"]

    if "page_label" in original_metadata:
        clean_metadata["page_label"] = original_metadata["page_label"]

    return clean_metadata


def get_loader(filepath: str):
    if filepath.endswith(".txt"):
        return TextLoader(filepath, encoding="utf-8")
    if filepath.endswith(".pdf"):
        return PyPDFLoader(filepath)
    if filepath.endswith(".csv"):
        return CSVLoader(filepath)
    raise ValueError(f"Unsupported file type: {filepath}")


def load_single_document(filepath: str):
    filename = os.path.basename(filepath)
    loader = get_loader(filepath)
    docs = loader.load()

    for doc in docs:
        doc.metadata = build_clean_metadata(filename, doc.metadata)

    print(f"✅ Loaded {filename} ({len(docs)} raw docs)")
    return docs


def load_all_documents():
    print("\n📄 Loading documents from data/ folder...")
    documents = []

    if not os.path.exists(settings.DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {settings.DATA_DIR}")

    for filename in os.listdir(settings.DATA_DIR):
        filepath = os.path.join(settings.DATA_DIR, filename)

        if not os.path.isfile(filepath):
            continue

        try:
            docs = load_single_document(filepath)
            documents.extend(docs)
        except ValueError as exc:
            print(f"⚠️ Skipping file: {exc}")

    print(f"\n📊 Total raw documents loaded: {len(documents)}")
    return documents


def split_documents(documents):
    print("\n✂️ Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        document_id = chunk.metadata["document_id"]
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_doc_ref"] = document_id

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def get_vectorstore():
    return Chroma(
        persist_directory=settings.PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def full_reindex():
    print("\n🚀 Running FULL reindex...")

    documents = load_all_documents()
    chunks = split_documents(documents)

    vectordb = get_vectorstore()

    existing = vectordb.get()
    existing_ids = existing.get("ids", [])
    if existing_ids:
        vectordb.delete(ids=existing_ids)
        print(f"🗑️ Deleted {len(existing_ids)} existing vectors")

    ids = [str(uuid.uuid4()) for _ in chunks]
    vectordb.add_documents(documents=chunks, ids=ids)

    print(f"✅ Full reindex complete. Indexed {len(chunks)} chunks")


def delete_document(document_name: str):
    print(f"\n🗑️ Deleting document from vector DB: {document_name}")

    vectordb = get_vectorstore()

    existing = vectordb.get(where={"document_id": document_name})
    ids = existing.get("ids", [])

    if not ids:
        print("ℹ️ No indexed chunks found for this document")
        return

    vectordb.delete(ids=ids)
    print(f"✅ Deleted {len(ids)} chunks for document: {document_name}")


def upsert_document(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    filename = os.path.basename(filepath)

    print(f"\n♻️ Upserting document: {filename}")

    vectordb = get_vectorstore()

    existing = vectordb.get(where={"document_id": filename})
    existing_ids = existing.get("ids", [])

    if existing_ids:
        vectordb.delete(ids=existing_ids)
        print(f"🗑️ Deleted {len(existing_ids)} old chunks for document: {filename}")

    documents = load_single_document(filepath)
    chunks = split_documents(documents)

    ids = [str(uuid.uuid4()) for _ in chunks]
    vectordb.add_documents(documents=chunks, ids=ids)

    print(f"✅ Upsert complete. Indexed {len(chunks)} chunks for document: {filename}")


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
    elif args.mode == "delete":
        delete_document(filename)


if __name__ == "__main__":
    main()