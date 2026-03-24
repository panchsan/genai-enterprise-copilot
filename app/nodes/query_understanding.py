import re

from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.query_understanding")


def _detect_action(query: str) -> str:
    q = (query or "").lower().strip()

    if any(word in q for word in ["compare", "difference between", "compare documents"]):
        return "compare_documents"

    if "summarize" in q or "summary" in q:
        return "summarize_document"

    if "source" in q or "document" in q or ".txt" in q or ".pdf" in q or ".csv" in q:
        return "answer_by_source"

    return "qa"


def _extract_target_sources(query: str) -> list[str]:
    q = query or ""
    matches = re.findall(r"\b[\w\-.]+\.(?:txt|pdf|csv)\b", q, flags=re.IGNORECASE)
    return list(dict.fromkeys(matches))


def analyze_query(state: AgentState):
    request_id = state.get("request_id", "-")
    query = state["query"]

    logger.info(f"[request_id={request_id}] Incoming query='{query}'")

    # Current phase: always try internal retrieval first.
    route = "retrieve"
    action = _detect_action(query)
    target_sources = _extract_target_sources(query)

    # Important: use only user/UI-provided filters for now.
    filters = state.get("filters", {}) or {}

    logger.info(
        f"[request_id={request_id}] route={route} | action={action} | "
        f"retrieval_query='{query}' | filters={filters} | target_sources={target_sources}"
    )

    return {
        "route": route,
        "action": action,
        "retrieval_query": query,
        "filters": filters,
        "target_sources": target_sources,
    }


def route_query(state: AgentState):
    request_id = state.get("request_id", "-")
    route = state.get("route", "retrieve")
    logger.info(f"[request_id={request_id}] Routing to={route}")

    if route == "retrieve":
        return "retrieve"
    if route == "direct":
        return "direct_answer"
    return "fallback"