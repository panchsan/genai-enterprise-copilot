from app.config import settings
from app.services.action_utils import normalize_action
from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.validate_retrieval")


def _is_azure_search_backend() -> bool:
    return getattr(settings, "VECTOR_BACKEND", "chroma").strip().lower() == "azure_search"


def validate_retrieval(state: AgentState):
    request_id = state.get("request_id", "-")
    action = normalize_action(state.get("action", "qa"))
    retrieval_status = state.get("retrieval_status", "no_docs")
    retrieved_docs = state.get("retrieved_docs", []) or []
    retrieved_sources = state.get("retrieved_sources", []) or []
    retrieval_scores = state.get("retrieval_scores", []) or []
    top_score = state.get("top_score")
    grounded_threshold = getattr(settings, "GROUNDED_SCORE_THRESHOLD", 1.0)

    logger.info(
        f"[request_id={request_id}] Validation | action={action} | "
        f"retrieval_status={retrieval_status} | retrieved_docs={len(retrieved_docs)} | "
        f"sources={retrieved_sources} | top_score={top_score} | "
        f"retrieval_scores={retrieval_scores} | grounded_threshold={grounded_threshold}"
    )

    # No docs / invalid retrieval states
    if retrieval_status in {"no_docs", "missing_required_source", "source_not_found"}:
        logger.info(
            f"[request_id={request_id}] Retrieval decision=no_docs due to retrieval_status={retrieval_status}"
        )
        return {
            **state,
            "retrieval_decision": "no_docs",
        }

    # Summarize: if docs exist, allow grounded generation
    if action == "summarize_document":
        if retrieved_docs:
            logger.info(
                f"[request_id={request_id}] summarize_document retrieved content -> grounded"
            )
            return {
                **state,
                "retrieval_decision": "grounded",
            }

        logger.info(
            f"[request_id={request_id}] summarize_document has no docs -> no_docs"
        )
        return {
            **state,
            "retrieval_decision": "no_docs",
        }

    # Compare: require at least 2 distinct sources
    if action == "compare_documents":
        if len(retrieved_sources) >= 2:
            logger.info(
                f"[request_id={request_id}] compare_documents found >=2 sources -> grounded"
            )
            return {
                **state,
                "retrieval_decision": "grounded",
            }

        logger.info(
            f"[request_id={request_id}] compare_documents insufficient sources -> no_docs"
        )
        return {
            **state,
            "retrieval_decision": "no_docs",
        }

    # Answer by source: if docs exist from the chosen source, allow grounded generation
    if action == "answer_by_source":
        if retrieved_docs:
            logger.info(
                f"[request_id={request_id}] answer_by_source retrieved source-backed docs -> grounded"
            )
            return {
                **state,
                "retrieval_decision": "grounded",
            }

        logger.info(
            f"[request_id={request_id}] answer_by_source has no docs -> no_docs"
        )
        return {
            **state,
            "retrieval_decision": "no_docs",
        }

    # QA
    if action == "qa":
        if not retrieved_docs:
            logger.info(
                f"[request_id={request_id}] QA has no docs -> no_docs"
            )
            return {
                **state,
                "retrieval_decision": "no_docs",
            }

        # Azure AI Search: don't use old Chroma-style grounded threshold
        if _is_azure_search_backend():
            logger.info(
                f"[request_id={request_id}] QA on azure_search with retrieved docs -> grounded"
            )
            return {
                **state,
                "retrieval_decision": "grounded",
            }

        # Chroma / distance-style validation
        if top_score is not None and float(top_score) <= grounded_threshold:
            logger.info(
                f"[request_id={request_id}] QA top_score={top_score} <= grounded_threshold={grounded_threshold} -> grounded"
            )
            return {
                **state,
                "retrieval_decision": "grounded",
            }

        logger.info(
            f"[request_id={request_id}] Retrieved docs exist but score is too weak for grounding "
            f"(top_score={top_score}) -> no_docs"
        )
        return {
            **state,
            "retrieval_decision": "no_docs",
        }

    # Safe default
    if retrieved_docs:
        logger.info(
            f"[request_id={request_id}] Default validation with retrieved docs -> grounded"
        )
        return {
            **state,
            "retrieval_decision": "grounded",
        }

    logger.info(
        f"[request_id={request_id}] Default validation without docs -> no_docs"
    )
    return {
        **state,
        "retrieval_decision": "no_docs",
    }