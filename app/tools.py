import os

from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIR = "./chroma_db"


def get_embeddings() -> AzureOpenAIEmbeddings:
    print("🔐 Initializing embeddings with Azure AD token...")

    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_ad_token_provider=token_provider,
    )


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