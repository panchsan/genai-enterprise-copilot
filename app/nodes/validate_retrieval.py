from app.config import settings
from app.services.action_utils import normalize_action
from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.validate_retrieval")


def validate_retrieval(state: AgentState):
    request_id = state.get("request_id", "-")
    action = normalize_action(state.get("action", "qa"))
    retrieved_docs = state.get("retrieved_docs", []) or []
    top_score = state.get("top_score")
    retrieval_scores = state.get("retrieval_scores", [])
    retrieval_status = state.get("retrieval_status")
    filters = state.get("filters", {}) or {}

    sources = {
        doc.get("metadata", {}).get("source")
        for doc in retrieved_docs
        if doc.get("metadata", {}).get("source")
    }

    logger.info(
        f"[request_id={request_id}] Validation | action={action} | "
        f"retrieval_status={retrieval_status} | retrieved_docs={len(retrieved_docs)} | "
        f"sources={list(sources)} | top_score={top_score} | "
        f"retrieval_scores={retrieval_scores} | "
        f"grounded_threshold={settings.GROUNDED_SCORE_THRESHOLD}"
    )

    if retrieval_status in {
        "missing_required_source",
        "source_not_found",
        "insufficient_sources",
        "no_docs",
    }:
        logger.info(
            f"[request_id={request_id}] Retrieval decision=no_docs due to retrieval_status={retrieval_status}"
        )
        return {
            "retrieval_decision": "no_docs",
            "retrieved_sources": [],
        }

    if not retrieved_docs:
        logger.info(f"[request_id={request_id}] Retrieval decision=no_docs (no retrieved_docs)")
        return {
            "retrieval_decision": "no_docs",
            "retrieved_sources": [],
        }

    if action == "answer_by_source":
        requested_source = filters.get("source")
        if requested_source and len(sources) > 1:
            logger.info(
                f"[request_id={request_id}] answer_by_source got multiple sources unexpectedly -> no_docs"
            )
            return {
                "retrieval_decision": "no_docs",
                "retrieved_sources": [],
            }

    if action == "compare_documents" and len(sources) < 2:
        logger.info(
            f"[request_id={request_id}] compare_documents requires at least 2 sources -> no_docs"
        )
        return {
            "retrieval_decision": "no_docs",
            "retrieved_sources": [],
            "retrieval_status": "insufficient_sources",
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