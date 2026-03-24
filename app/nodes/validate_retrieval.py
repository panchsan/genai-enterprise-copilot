from app.config import settings
from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.validate_retrieval")


def validate_retrieval(state: AgentState):
    request_id = state.get("request_id", "-")
    retrieved_docs = state.get("retrieved_docs", []) or []
    top_score = state.get("top_score")
    retrieval_scores = state.get("retrieval_scores", [])
    retrieval_status = state.get("retrieval_status")

    logger.info(
        f"[request_id={request_id}] Validation | "
        f"retrieval_status={retrieval_status} | "
        f"retrieved_docs={len(retrieved_docs)} | "
        f"top_score={top_score} | "
        f"retrieval_scores={retrieval_scores} | "
        f"grounded_threshold={settings.GROUNDED_SCORE_THRESHOLD}"
    )

    if not retrieved_docs:
        logger.info(f"[request_id={request_id}] Retrieval decision=no_docs")
        return {
            "retrieval_decision": "no_docs",
            "retrieved_sources": [],
        }

    if top_score is not None and float(top_score) <= settings.GROUNDED_SCORE_THRESHOLD:
        logger.info(f"[request_id={request_id}] Retrieval decision=grounded")
        return {
            "retrieval_decision": "grounded",
        }

    logger.info(
        f"[request_id={request_id}] Retrieved docs exist but score is too weak for grounding "
        f"(top_score={top_score}) -> no_docs"
    )
    return {
        "retrieval_decision": "no_docs",
        "retrieved_sources": [],
    }