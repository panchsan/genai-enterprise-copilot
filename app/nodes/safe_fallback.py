from app.state import AgentState


def safe_fallback(state: AgentState):
    query = (
        state.get("rewritten_query")
        or state.get("retrieval_query")
        or state.get("query")
        or "your request"
    )

    return {
        "answer": (
            f"I could not find a reliable answer for '{query}' in the indexed enterprise "
            "documents. Please try rephrasing the request, specifying a document/source, "
            "or removing restrictive filters."
        ),
        "retrieval_decision": "ungrounded",
        "retrieved_docs": [],
        "retrieved_sources": [],
        "retrieval_scores": [],
        "top_score": None,
    }