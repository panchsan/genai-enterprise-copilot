# Enterprise GenAI Copilot (Azure-Native RAG)

Enterprise GenAI Copilot is a production-style Retrieval-Augmented Generation (RAG) assistant built with FastAPI, LangGraph, Streamlit, Azure OpenAI, Azure AI Search, and Azure Blob Storage.

It started as a local prototype and was evolved into an Azure-native enterprise architecture aligned with production deployment patterns, managed identity, and backend abstraction.

---

## Architecture Overview

### Core components

- **UI**: Streamlit
- **Backend API**: FastAPI
- **Agent orchestration**: LangGraph
- **LLM / Embeddings**: Azure OpenAI
- **Document storage**: Azure Blob Storage
- **Vector retrieval**: Azure AI Search
- **Session store**: SQLite (currently ephemeral in `/tmp`)
- **Authentication**:
  - Managed Identity / `DefaultAzureCredential`
  - Microsoft Entra ID-based access
- **Deployment target**: Azure Container Apps

---

## Current request flow

User Query  
→ Streamlit UI  
→ FastAPI Backend  
→ Session Context  
→ Query Understanding  
→ Query Rewrite  
→ Retrieval from Azure AI Search  
→ Retrieval Validation  
→ Grounded Generation via Azure OpenAI  
→ Response returned to UI

---

## Document ingestion flow

Documents  
→ Azure Blob Storage  
→ Ingestion script  
→ Chunking  
→ Embedding generation  
→ Azure AI Search indexing

---

## Supported actions

- `qa`
- `summarize_document`
- `compare_documents`
- `answer_by_source`

### Behavior notes

- Explicit UI-selected action overrides keyword inference
- Retrieval is source-aware and filter-aware
- `qa` can fall back to direct answer if no grounded content is available
- Document-focused actions are designed to prefer retrieval-backed answers
- Retrieval debug details can be returned in development mode

---

## Key enterprise design patterns used

- Azure-native RAG architecture
- LangGraph-based workflow orchestration
- Backend abstraction for vector store migration
- Source-aware retrieval
- Retrieval validation / grounding guardrails
- Managed identity for Azure service access
- Config-driven deployment
- Debuggable and observable response path

---

## Project structure

```text
app/
  agent.py
  config.py
  main.py
  prompts.py
  state.py
  nodes/
  services/

scripts/
  ingest.py

ui/
  streamlit_app.py
  api_client.py

test/
  test_actions.py
  test_retrieve_actions.py

## Build and push your images to ACR

```cmd
az login
az acr login --name acrcopilotdev

docker build -f Dockerfile.backend -t acrcopilotdev.azurecr.io/copilot/backend:1 .
docker push acrcopilotdev.azurecr.io/copilot/backend:1

docker build -f Dockerfile.ui -t acrcopilotdev.azurecr.io/copilot/ui:1 .
docker push acrcopilotdev.azurecr.io/copilot/ui:1
```  