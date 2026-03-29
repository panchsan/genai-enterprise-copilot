import uuid
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
    get_all_sessions,
    get_session_title,
    update_session_title,
    delete_session,
)
from app.services.logging_utils import get_logger, log_timing

logger = get_logger("app.main")

app = FastAPI(title="Enterprise RAG Copilot")

graph = None


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    filters: Optional[Dict[str, Any]] = None
    action: Optional[str] = None


def build_session_title(query: str, max_len: int = 60) -> str:
    cleaned = " ".join((query or "").strip().split())
    if not cleaned:
        return "New Chat"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "..."


@app.on_event("startup")
def startup_event():
    global graph
    logger.info("Initializing database...")
    init_db()

    logger.info("Building LangGraph workflow...")
    graph = build_graph()

    logger.info("Application startup complete.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    global graph
    try:
        if graph is None:
            raise RuntimeError("Graph not initialized")

        _ = get_all_sessions()

        return {
            "status": "ready",
            "graph_initialized": True,
            "db_accessible": True,
        }
    except Exception as exc:
        logger.exception(f"Readiness check failed: {exc}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/")
def root():
    return {
        "message": "Enterprise RAG Copilot API is running",
        "health": "/health",
        "ready": "/ready",
        "chat": "/chat",
        "sessions": "/sessions",
    }


@app.get("/sessions")
def list_sessions():
    try:
        sessions = get_all_sessions()
        return {
            "sessions": sessions,
            "count": len(sessions),
        }
    except Exception as exc:
        logger.exception(f"Failed to list sessions: {exc}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")


@app.get("/history/{session_id}")
def history(session_id: str):
    try:
        history_items = get_chat_history(session_id)
        session_context = get_session_context(session_id)

        return {
            "session_id": session_id,
            "history": history_items,
            "history_length": len(history_items),
            "session_context": session_context,
        }
    except Exception as exc:
        logger.exception(f"Failed to fetch history for session_id={session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")


@app.post("/chat")
def chat(request: ChatRequest):
    global graph

    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    request_id = str(uuid.uuid4())

    logger.info(
        f"[request_id={request_id}] Incoming query='{request.query}' | "
        f"session_id={request.session_id} | request_action={request.action} | "
        f"request_filters={request.filters or {}}"
    )

    try:
        create_session(request.session_id)

        if not get_session_title(request.session_id):
            update_session_title(request.session_id, build_session_title(request.query))

        chat_history = get_chat_history(request.session_id)
        session_context = get_session_context(request.session_id)

        chat_history_for_graph = [
            {"role": item["role"], "content": item["content"]}
            for item in chat_history
        ]

        with log_timing(logger, "graph_invoke", request_id):
            result = graph.invoke({
                "request_id": request_id,
                "query": request.query,
                "session_id": request.session_id,
                "chat_history": chat_history_for_graph,
                "filters": request.filters or {},
                "action": request.action,
                "session_context": session_context,
            })

        answer = result.get("answer", "I’m sorry, I couldn’t generate a response.")
        applied_filters = result.get("filters", {}) or {}
        action = result.get("action")
        retrieval_query = result.get("retrieval_query")
        rewritten_query = result.get("rewritten_query")
        target_sources = result.get("target_sources", [])
        retrieval_decision = result.get("retrieval_decision")
        retrieval_status = result.get("retrieval_status")
        retrieved_docs = result.get("retrieved_docs", []) or []
        retrieval_scores = result.get("retrieval_scores", []) or []
        top_score = result.get("top_score")
        route = result.get("route")
        retrieval_debug = result.get("retrieval_debug", {}) or {}

        retrieved_sources = []
        for doc in retrieved_docs:
            source = (doc.get("metadata", {}) or {}).get("source")
            if source and source not in retrieved_sources:
                retrieved_sources.append(source)

        save_message(request.session_id, "user", request.query)

        save_message(
            request.session_id,
            "assistant",
            answer,
            grounding="grounded" if retrieval_decision == "grounded" else "ungrounded",
            sources=retrieved_sources,
            retrieval_decision=retrieval_decision,
            top_score=top_score,
        )

        update_session_context(
            session_id=request.session_id,
            active_filters=applied_filters,
            active_source=retrieved_sources[0] if retrieved_sources else None,
            last_route=route,
            last_retrieval_query=rewritten_query or retrieval_query or request.query,
        )

        updated_context = get_session_context(request.session_id)

        logger.info(
            f"[request_id={request_id}] Completed | route={route} | action={action} | "
            f"retrieval_decision={retrieval_decision} | retrieval_status={retrieval_status} | "
            f"retrieved_sources={retrieved_sources} | top_score={top_score}"
        )

        return {
            "request_id": request_id,
            "route": route,
            "action": action,
            "target_sources": target_sources,
            "retrieval_query": retrieval_query,
            "rewritten_query": rewritten_query,
            "applied_filters": applied_filters,
            "response": answer,
            "context_preview": (
                result.get("context", "")[:200] if result.get("context") else ""
            ),
            "session_id": request.session_id,
            "history_length": len(get_chat_history(request.session_id)),
            "retrieval_decision": retrieval_decision,
            "retrieval_status": retrieval_status,
            "retrieved_sources": retrieved_sources,
            "retrieval_scores": retrieval_scores,
            "top_score": top_score,
            "retrieval_debug": retrieval_debug,
            "session_context": updated_context,
        }

    except Exception as exc:
        logger.exception(f"[request_id={request_id}] Chat request failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    try:
        deleted = delete_session(session_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "status": "deleted",
            "session_id": session_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Failed to delete session_id={session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to delete session")