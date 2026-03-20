import os
import shutil
import uuid

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


def load_documents():
    print("\n📄 Loading documents from data/ folder...")

    documents = []

    if not os.path.exists(settings.DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {settings.DATA_DIR}")

    for filename in os.listdir(settings.DATA_DIR):
        filepath = os.path.join(settings.DATA_DIR, filename)

        if not os.path.isfile(filepath):
            continue

        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".csv"):
            loader = CSVLoader(filepath)
        else:
            print(f"⚠️ Skipping unsupported file: {filename}")
            continue

        docs = loader.load()

        for doc in docs:
            doc.metadata = build_clean_metadata(filename, doc.metadata)

        documents.extend(docs)
        print(f"✅ Loaded {filename} ({len(docs)} raw docs)")

    print(f"\n📊 Total raw documents loaded: {len(documents)}")
    return documents


def split_documents(documents):
    print("\n✂️ Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def recreate_vector_db(chunks):
    print("\n🧹 Rebuilding vector DB...")

    if os.path.exists(settings.PERSIST_DIR):
        shutil.rmtree(settings.PERSIST_DIR)
        print("🗑️ Old vector DB deleted")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=settings.PERSIST_DIR,
    )

    print("✅ Vector DB created successfully")


def main():
    print("\n🚀 Starting ingestion pipeline...")

    documents = load_documents()
    chunks = split_documents(documents)
    recreate_vector_db(chunks)

    print("\n🎉 Ingestion complete!")


if __name__ == "__main__":
    main()