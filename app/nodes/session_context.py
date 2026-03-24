from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.session_context")


def apply_session_context(state: AgentState):
    request_id = state.get("request_id", "-")
    current_filters = state.get("filters", {}) or {}
    session_context = state.get("session_context", {}) or {}

    logger.info(
        f"[request_id={request_id}] Session context loaded | "
        f"stored_filters={session_context.get('active_filters', {})} | "
        f"active_source={session_context.get('active_source')} | "
        f"last_route={session_context.get('last_route')} | "
        f"last_retrieval_query={session_context.get('last_retrieval_query')}"
    )

    logger.info(
        f"[request_id={request_id}] Using current request filters only | "
        f"filters={current_filters}"
    )

    return {
        "filters": current_filters,
        "active_source": session_context.get("active_source"),
        "last_route": session_context.get("last_route"),
        "last_retrieval_query": session_context.get("last_retrieval_query"),
    }