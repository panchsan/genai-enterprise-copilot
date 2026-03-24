from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.rewrite_query")


FOLLOW_UP_HINTS = [
    "what about",
    "explain that",
    "explain it",
    "summarize that",
    "tell me more",
    "what does it say",
    "and finance",
    "and hr",
    "what about this",
    "what about that",
]


def _is_follow_up(query: str) -> bool:
    q = (query or "").lower().strip()

    if any(hint in q for hint in FOLLOW_UP_HINTS):
        return True

    if len(q.split()) <= 4 and any(word in q for word in ["that", "this", "it", "those", "them"]):
        return True

    return False


def rewrite_query(state: AgentState):
    request_id = state.get("request_id", "-")
    query = state.get("retrieval_query") or state.get("query", "")
    action = state.get("action", "qa")

    logger.info(f"[request_id={request_id}] Original retrieval query='{query}'")

    # Keep standalone questions unchanged.
    # Only rewrite if it looks like a context-dependent follow-up.
    if action == "qa" and not _is_follow_up(query):
        logger.info(
            f"[request_id={request_id}] Standalone QA detected -> skipping rewrite"
        )
        return {"rewritten_query": query}

    # For now, keep non-QA actions also simple and unchanged.
    logger.info(
        f"[request_id={request_id}] Rewrite not needed -> using original query"
    )
    return {"rewritten_query": query}