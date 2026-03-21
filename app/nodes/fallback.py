from app.prompts import FALLBACK_RESPONSE
from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.fallback")


def fallback(state: AgentState):
    request_id = state.get("request_id", "-")
    logger.warning(
        f"[request_id={request_id}] Fallback route triggered for query={state.get('query')!r}"
    )

    return {
        "answer": FALLBACK_RESPONSE
    }