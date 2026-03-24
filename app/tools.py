import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.llm import get_embeddings

load_dotenv()

PERSIST_DIR = "./chroma_db"


def init_vector_db() -> None:
    print("\n🚀 Initializing Vector DB...")

    if os.path.exists(PERSIST_DIR):
        print("⚠️ Vector DB already exists. Skipping ingestion.")
        return

    print("📄 Loading documents...")
    loader = TextLoader("data/sample.txt", encoding="utf-8")
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} document(s)")

    print("✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunk(s)")

    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.page_content[:150].replace("\n", " ")
        print(f"   Chunk {i}: {preview}")

    print("🧠 Creating embeddings + storing in Chroma...")
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )

    print("✅ Vector DB created and persisted!")


def get_retriever():
    print("🔍 Loading retriever from vector DB...")

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})