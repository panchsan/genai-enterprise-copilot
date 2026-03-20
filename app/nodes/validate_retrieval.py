import re

from app.config import settings
from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.validate_retrieval")

STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "for",
    "to", "and", "does", "do", "about", "tell", "me", "are",
    "this", "that", "with", "from", "our",
    "explain", "simply", "simple", "summarize", "summary",
    "more", "please", "terms"
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def validate_retrieval(state: AgentState):
    request_id = state.get("request_id", "-")
    query = (
        state.get("rewritten_query")
        or state.get("retrieval_query")
        or state.get("query", "")
    )
    context = state.get("context", "")
    top_score = state.get("top_score")
    retrieval_scores = state.get("retrieval_scores", [])

    logger.info(
        f"[request_id={request_id}] Validation query='{query}' | scores={retrieval_scores} | top_score={top_score}"
    )

    if not context.strip():
        logger.warning(f"[request_id={request_id}] Empty context -> ungrounded")
        return {"retrieval_decision": "ungrounded"}

    query_terms = tokenize(query)
    context_terms = tokenize(context)

    overlap = query_terms.intersection(context_terms)
    overlap_score = len(overlap) / max(len(query_terms), 1)

    strong_vector_match = (
        top_score is not None and top_score <= settings.RETRIEVAL_SCORE_THRESHOLD
    )
    strong_overlap = overlap_score >= settings.RETRIEVAL_OVERLAP_THRESHOLD

    logger.info(
        f"[request_id={request_id}] overlap_terms={overlap} | overlap_score={round(overlap_score, 2)} | "
        f"strong_vector_match={strong_vector_match} | strong_overlap={strong_overlap}"
    )

    if strong_vector_match and strong_overlap:
        decision = "grounded"
    elif strong_vector_match and overlap_score >= 0.15:
        decision = "grounded"
    else:
        decision = "ungrounded"

    logger.info(f"[request_id={request_id}] Retrieval decision={decision}")
    return {"retrieval_decision": decision}