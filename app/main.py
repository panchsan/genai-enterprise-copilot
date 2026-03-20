from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent import build_graph
from app.services.db import (
    init_db,
    create_session,
    save_message,
    get_chat_history,
    get_session_context,
    update_session_context,
)
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
    session_context = get_session_context(request.session_id)

    print("🗂️ Loaded Session Context:", session_context)

    result = graph.invoke({
        "query": request.query,
        "session_id": request.session_id,
        "chat_history": chat_history,
        "filters": request.filters or {},
        "session_context": session_context,
    })

    save_message(request.session_id, "user", request.query)
    save_message(request.session_id, "assistant", result.get("answer", ""))

    retrieved_sources = [
        doc["metadata"].get("source")
        for doc in result.get("retrieved_docs", [])
    ]

    active_source = retrieved_sources[0] if retrieved_sources else session_context.get("active_source")

    update_session_context(
        session_id=request.session_id,
        active_filters=result.get("filters", {}),
        active_source=active_source,
        last_route=result.get("route"),
        last_retrieval_query=(
            result.get("rewritten_query")
            or result.get("retrieval_query")
            or request.query
        ),
    )

    updated_context = get_session_context(request.session_id)

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
        "retrieved_sources": retrieved_sources,
        "session_context": updated_context,
    }