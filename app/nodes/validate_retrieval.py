import re

from app.state import AgentState


STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "for",
    "to", "and", "does", "do", "about", "tell", "me", "are",
    "this", "that", "with", "from", "our"
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def validate_retrieval(state: AgentState):
    query = state.get("query", "")
    context = state.get("context", "")

    print("\n🧪 [VALIDATE] Checking retrieval relevance...")

    query_terms = tokenize(query)
    context_terms = tokenize(context)

    overlap = query_terms.intersection(context_terms)
    overlap_score = len(overlap) / max(len(query_terms), 1)

    print("📝 Query Terms:", query_terms)
    print("🔗 Overlap Terms:", overlap)
    print("📊 Overlap Score:", round(overlap_score, 2))

    if overlap_score >= 0.3:
        decision = "grounded"
    else:
        decision = "ungrounded"

    print("✅ [VALIDATE] Decision:", decision)

    return {"retrieval_decision": decision}