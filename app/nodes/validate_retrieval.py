import re

from app.config import settings
from app.services.action_utils import normalize_action
from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.validate_retrieval")

STOPWORDS = {
    "the", "and", "for", "with", "what", "why", "how", "are", "is", "was", "were",
    "this", "that", "from", "into", "about", "according", "document", "program",
    "guide", "plan", "include", "using", "only", "based", "under", "does", "should",
}


def _is_azure_search_backend() -> bool:
    return getattr(settings, "VECTOR_BACKEND", "chroma").strip().lower() == "azure_search"


def _meaningful_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _compute_alignment(state: AgentState) -> dict:
    query_terms = _meaningful_terms(state.get("retrieval_query") or state.get("query") or "")
    retrieved_docs = state.get("retrieved_docs", []) or []
    retrieved_sources = state.get("retrieved_sources", []) or []
    target_sources = state.get("target_sources", []) or []

    max_overlap = 0
    matched_terms: list[str] = []

    for doc in retrieved_docs[:4]:
        metadata = doc.get("metadata", {}) or {}
        combined_text = " ".join([
            metadata.get("source", ""),
            metadata.get("document_title", ""),
            doc.get("page_content", "")[:600],
        ])
        overlap = sorted(query_terms & _meaningful_terms(combined_text))
        if len(overlap) > max_overlap:
            max_overlap = len(overlap)
            matched_terms = overlap[:8]

    source_hint_matched = False
    if target_sources and retrieved_sources:
        target_term_pool = set()
        for source in target_sources:
            target_term_pool.update(_meaningful_terms(source))

        for retrieved_source in retrieved_sources:
            retrieved_terms = _meaningful_terms(retrieved_source)
            if target_term_pool & retrieved_terms:
                source_hint_matched = True
                break

    return {
        "query_terms": sorted(query_terms),
        "max_overlap": max_overlap,
        "matched_terms": matched_terms,
        "source_hint_matched": source_hint_matched,
    }


def validate_retrieval(state: AgentState):
    request_id = state.get("request_id", "-")
    action = normalize_action(state.get("action", "qa"))
    retrieval_status = state.get("retrieval_status", "no_docs")
    retrieved_docs = state.get("retrieved_docs", []) or []
    retrieved_sources = state.get("retrieved_sources", []) or []
    retrieval_scores = state.get("retrieval_scores", []) or []
    top_score = state.get("top_score")
    grounded_threshold = getattr(settings, "GROUNDED_SCORE_THRESHOLD", 1.0)
    alignment = _compute_alignment(state)

    logger.info(
        f"[request_id={request_id}] Validation | action={action} | "
        f"retrieval_status={retrieval_status} | retrieved_docs={len(retrieved_docs)} | "
        f"sources={retrieved_sources} | top_score={top_score} | "
        f"retrieval_scores={retrieval_scores} | grounded_threshold={grounded_threshold} | "
        f"max_overlap={alignment['max_overlap']} | matched_terms={alignment['matched_terms']} | "
        f"source_hint_matched={alignment['source_hint_matched']}"
    )

    if retrieval_status in {"no_docs", "missing_required_source", "source_not_found"}:
        logger.info(
            f"[request_id={request_id}] Retrieval decision=no_docs due to retrieval_status={retrieval_status}"
        )
        return {
            **state,
            "retrieval_decision": "no_docs",
        }

    if action == "summarize_document":
        return {**state, "retrieval_decision": "grounded" if retrieved_docs else "no_docs"}

    if action == "compare_documents":
        if len(retrieved_sources) >= 2:
            return {**state, "retrieval_decision": "grounded"}
        return {**state, "retrieval_decision": "no_docs"}

    if action == "answer_by_source":
        if retrieved_docs:
            return {**state, "retrieval_decision": "grounded"}
        return {**state, "retrieval_decision": "no_docs"}

    if action == "qa":
        if not retrieved_docs:
            logger.info(f"[request_id={request_id}] QA has no docs -> no_docs")
            return {**state, "retrieval_decision": "no_docs"}

        if _is_azure_search_backend():
            has_soft_signal = alignment["max_overlap"] >= 2
            if state.get("target_sources"):
                has_soft_signal = has_soft_signal or alignment["source_hint_matched"]

            if has_soft_signal:
                logger.info(
                    f"[request_id={request_id}] QA on azure_search passed soft alignment checks -> grounded"
                )
                return {**state, "retrieval_decision": "grounded"}

            logger.info(
                f"[request_id={request_id}] QA on azure_search retrieved docs but alignment is weak -> weak_match"
            )
            return {
                **state,
                "retrieval_status": "weak_match",
                "retrieval_decision": "no_docs",
            }

        if top_score is not None and float(top_score) <= grounded_threshold:
            return {**state, "retrieval_decision": "grounded"}

        return {**state, "retrieval_decision": "no_docs"}

    return {**state, "retrieval_decision": "grounded" if retrieved_docs else "no_docs"}