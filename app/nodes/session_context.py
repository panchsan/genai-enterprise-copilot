from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.session_context")


FOLLOW_UP_HINTS = [
    "what about",
    "explain that",
    "explain it",
    "summarize that",
    "tell me more",
    "what does it say",
    "and finance",
    "and hr",
    "attendance",
    "leave",
    "policy",
]


def merge_filters(current_filters: dict, stored_filters: dict) -> dict:
    merged = dict(stored_filters or {})
    merged.update({
        key: value
        for key, value in (current_filters or {}).items()
        if value not in (None, "", {})
    })
    return merged


def apply_session_context(state: AgentState):
    request_id = state.get("request_id", "-")
    query = state["query"].lower().strip()
    current_filters = state.get("filters", {}) or {}
    session_context = state.get("session_context", {}) or {}

    stored_filters = session_context.get("active_filters", {}) or {}
    active_source = session_context.get("active_source")
    last_route = session_context.get("last_route")
    last_retrieval_query = session_context.get("last_retrieval_query")

    logger.info(
        f"[request_id={request_id}] Session context loaded | "
        f"stored_filters={stored_filters} | active_source={active_source} | "
        f"last_route={last_route} | last_retrieval_query={last_retrieval_query}"
    )

    should_reuse_context = (
        state.get("route") == "retrieve"
        and last_route == "retrieve"
        and (
            len(query.split()) <= 6
            or any(hint in query for hint in FOLLOW_UP_HINTS)
        )
    )

    if should_reuse_context:
        merged_filters = merge_filters(current_filters, stored_filters)
        logger.info(
            f"[request_id={request_id}] Reusing stored filters for follow-up | merged_filters={merged_filters}"
        )
        return {
            "filters": merged_filters,
            "active_source": active_source,
            "last_route": last_route,
            "last_retrieval_query": last_retrieval_query,
        }

    logger.info(f"[request_id={request_id}] No session-context reuse applied")

    return {
        "filters": current_filters,
        "active_source": active_source,
        "last_route": last_route,
        "last_retrieval_query": last_retrieval_query,
    }