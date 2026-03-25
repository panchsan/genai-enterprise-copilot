# Enterprise RAG Copilot

Enterprise RAG Copilot is a local-first enterprise document assistant built with FastAPI, LangGraph, Streamlit, Azure OpenAI, and Chroma.

## Current stack

- Backend: FastAPI + LangGraph
- UI: Streamlit
- LLM: Azure OpenAI
- Auth: Microsoft Entra ID via `DefaultAzureCredential`
- Vector store: Chroma (persisted locally)
- Chat/session store: SQLite
- Deployment today: local + Docker
- Next target: Azure Container Apps

## Current flow

User Query  
→ Analyze Query  
→ Apply Session Context  
→ Rewrite Query  
→ Retrieve  
→ Validate Retrieval  
→ Generate grounded answer **or** Direct answer

## Supported actions

- `qa`
- `summarize_document`
- `answer_by_source`
- `compare_documents`

Behavior:
- explicit UI action overrides keyword detection
- retrieval is internal-first
- `qa` may fall back to direct answer when no internal match exists
- document-oriented actions return controlled messages instead of broad fallback answers

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

data/
  sample local documents