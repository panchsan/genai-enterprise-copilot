from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent import build_graph
from app.services.db import init_db, create_session, save_message, get_chat_history
from app.services.vectorstore import get_vectorstore

app = FastAPI()

graph = None


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    filters: Optional[Dict[str, Any]] = None


@app.on_event("startup")
def startup():
    global graph

    print("\n🚀 App starting...")

    init_db()
    vectordb = get_vectorstore()
    graph = build_graph(vectordb)

    print("✅ App startup complete. Graph is ready.")


@app.get("/")
def root():
    return {"message": "GenAI RAG system running 🚀"}


@app.post("/chat")
def chat(request: ChatRequest):
    global graph

    if graph is None:
        raise HTTPException(status_code=500, detail="Graph is not initialized")

    print("\n📩 Incoming Query:", request.query)
    print("🧵 Session ID:", request.session_id)
    print("🎯 Request Filters:", request.filters or {})

    create_session(request.session_id)
    chat_history = get_chat_history(request.session_id)

    result = graph.invoke({
        "query": request.query,
        "session_id": request.session_id,
        "chat_history": chat_history,
        "filters": request.filters or {},
    })

    save_message(request.session_id, "user", request.query)
    save_message(request.session_id, "assistant", result.get("answer", ""))

    return {
        "route": result.get("route"),
        "retrieval_query": result.get("retrieval_query"),
        "rewritten_query": result.get("rewritten_query"),
        "applied_filters": result.get("filters", {}),
        "response": result.get("answer"),
        "context_preview": (
            result.get("context", "")[:200] if result.get("context") else ""
        ),
        "session_id": request.session_id,
        "history_length": len(get_chat_history(request.session_id)),
        "retrieval_decision": result.get("retrieval_decision"),
        "retrieved_sources": [
            doc["metadata"].get("source")
            for doc in result.get("retrieved_docs", [])
        ],
    }