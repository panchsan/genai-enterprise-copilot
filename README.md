# 🚀 GEN AI ARCHITECT NOTES — PHASE 1 (RAG SYSTEM WITH LANGGRAPH + AZURE OPENAI)

---

# 1. 🎯 OBJECTIVE

Build a **production-style Retrieval-Augmented Generation (RAG) system** using:

* Azure OpenAI (LLM + Embeddings)
* LangGraph (orchestration)
* ChromaDB (vector store)
* FastAPI (serving layer)
* Azure AD authentication (enterprise-ready)

---

# 2. 🧠 CORE ARCHITECTURE

## 🔷 High-Level Flow

User Query → Retriever → Context → LLM → Response

## 🔷 Components

### 1. API Layer (FastAPI)

* Handles incoming requests (`/chat`)
* Triggers LangGraph execution
* Returns response

---

### 2. Orchestration Layer (LangGraph)

Two-step pipeline:

1. **Retrieve Node**

   * Fetch relevant chunks from vector DB

2. **Generate Node**

   * Send query + context to LLM
   * Generate final answer

---

### 3. Vector Database (Chroma)

* Stores embeddings of document chunks
* Performs similarity search
* Returns top-K relevant documents

---

### 4. Embedding Layer

* Uses Azure OpenAI embedding model
* Converts text → vectors
* Used during:

  * Ingestion
  * Query time

---

### 5. LLM Layer (Azure OpenAI)

* Uses `AzureOpenAI` client
* Generates response using:

  * User query
  * Retrieved context

---

### 6. Authentication Layer

* Azure AD via:

  * `AzureCliCredential`
  * `DefaultAzureCredential` (optional)

---

# 3. ⚙️ KEY CONCEPTS IMPLEMENTED

---

## 🔹 1. RAG (Retrieval-Augmented Generation)

* Prevents hallucination
* Grounds LLM response in real data
* Works via:

  * Embedding + similarity search

---

## 🔹 2. Chunking Strategy

```text
Chunk size: 300
Overlap: 50
```

Why important:

* Improves retrieval accuracy
* Prevents context loss

---

## 🔹 3. Embeddings

* Semantic representation of text
* Used for:

  * Indexing
  * Retrieval

---

## 🔹 4. Vector Search

* Uses cosine similarity (internally)
* Retrieves top-K similar chunks

---

## 🔹 5. LangGraph (State-Based Orchestration)

* Defines:

  * Nodes (steps)
  * Edges (flow)
* Maintains state:

```text
query → context → answer
```

---

## 🔹 6. Prompt Engineering

Initial issue:

* Too strict → model said "I don't know"

Improved prompt:

* Flexible but grounded

---

## 🔹 7. Azure OpenAI Integration

* Used:

  * Chat model (gpt-4.1-mini)
  * Embedding model (text-embedding-3-small)

---

## 🔹 8. Azure AD Authentication

Used:

```python
AzureCliCredential()
```

Token flow:

* CLI login → token → OpenAI

---

## 🔹 9. FastAPI Lifecycle

* Used `@startup` event
* Ensures:

  * Vector DB initialized first
  * Retriever created after

---

## 🔹 10. Lazy Initialization (Critical)

❌ Wrong:

```python
retriever = get_retriever()
```

✅ Correct:

* Initialize inside startup OR runtime

---

# 4. ⚠️ CHALLENGES FACED (REAL-WORLD ISSUES)

---

## ❌ 1. LangChain Import Errors

Problem:

```text
AzureOpenAIEmbeddings not found
```

Cause:

* Package split in newer LangChain versions

Fix:

```python
from langchain_openai import AzureOpenAIEmbeddings
```

---

## ❌ 2. Missing Azure Config

Errors:

* Missing endpoint
* Missing API version

Fix:

* Proper `.env` setup

---

## ❌ 3. Environment Variables Not Loading

Cause:

* `.env` not loaded

Fix:

```python
load_dotenv()
```

---

## ❌ 4. Azure AD Authentication Failure

Error:

```text
DefaultAzureCredential failed
```

Cause:

* No local auth context

Fix:

```bash
az login
```

---

## ❌ 5. Vector DB Not Populating

Issue:

```text
Vector DB already exists. Skipping ingestion.
```

Root Cause:

* Retriever initialized before ingestion

Fix:

* Move retriever to startup

---

## ❌ 6. "I don't know" Responses

Cause:

* No relevant documents retrieved

Fix:

* Re-ingestion
* Debug retrieval
* Improve prompt

---

## ❌ 7. Streaming Not Working

Cause:

* Using `invoke()` instead of `stream()`

---

## ❌ 8. Dependency Conflicts

Example:

* Anaconda + venv issues
* LangChain version mismatches

---

# 6. 🧠 ARCHITECT-LEVEL LEARNINGS

---

## 🔥 1. Initialization Order Matters

```text
Ingestion → Retriever → Graph
```

---

## 🔥 2. Never Initialize at Import Time

Causes:

* Hidden bugs
* Race conditions

---

## 🔥 3. RAG Systems Fail Silently

* No errors
* Just bad answers

---

## 🔥 4. Observability is Critical

Always log:

* Retrieved docs
* Context
* LLM input

---

## 🔥 5. Authentication Strategy Must Be Environment-Aware

| Environment | Auth              |
| ----------- | ----------------- |
| Local       | Azure CLI         |
| Dev         | Service Principal |
| Prod        | Managed Identity  |

---

## 🔥 6. Prompt Design Impacts System Behavior

* Too strict → no answers
* Too loose → hallucination

---

## 🔥 7. LangGraph = Deterministic AI Workflows

* Better than simple chains
* Production-friendly

---

# 7. 📦 FINAL SYSTEM CAPABILITIES

---

✅ RAG pipeline
✅ Azure OpenAI integration
✅ Vector DB (Chroma)
✅ LangGraph orchestration
✅ Enterprise authentication
✅ Debuggable pipeline

---

# 🏁 FINAL SUMMARY

This phase focused on:

👉 Building a **working RAG system**
👉 Understanding **end-to-end flow**
👉 Solving **real integration issues**
👉 Thinking like a **GenAI Architect**

---

# 🚀 NEXT STEPS (PHASE 2)

---

* Multi-document ingestion (PDF, CSV)
* Metadata filtering
* Tool-based agents (LangGraph agents)
* Streaming responses
* Azure deployment (AKS)
