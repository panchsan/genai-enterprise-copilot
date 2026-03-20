from app.config import settings
from app.state import AgentState


def build_chroma_filter(filters: dict | None):
    if not filters:
        return None

    cleaned_filters = {
        key: value
        for key, value in filters.items()
        if value is not None and value != ""
    }

    if not cleaned_filters:
        return None

    if len(cleaned_filters) == 1:
        key, value = next(iter(cleaned_filters.items()))
        return {key: value}

    return {
        "$and": [{key: value} for key, value in cleaned_filters.items()]
    }


def retrieve(state: AgentState, vectordb):
    query = (
        state.get("rewritten_query")
        or state.get("retrieval_query")
        or state["query"]
    )
    filters = state.get("filters", {})

    chroma_filter = build_chroma_filter(filters)

    print("\n🔎 [RETRIEVE] Query:", query)
    print("🎯 Raw Filters:", filters)
    print("🧱 Chroma Filter:", chroma_filter)

    results = vectordb.similarity_search_with_score(
        query=query,
        k=settings.RETRIEVAL_TOP_K,
        filter=chroma_filter,
    )

    print("📚 Retrieved Docs Count:", len(results))

    if not results:
        print("❌ No documents retrieved!")
        return {
            "context": "",
            "retrieved_docs": [],
            "retrieval_scores": [],
            "top_score": None,
        }

    context_parts = []
    retrieved_docs = []
    scores = []

    for i, (doc, score) in enumerate(results, start=1):
        metadata = getattr(doc, "metadata", {})
        content = doc.page_content[:300]
        numeric_score = float(score)

        print(f"\n--- Doc {i} ---")
        print("Score:", numeric_score)
        print("Metadata:", metadata)
        print(content)

        context_parts.append(doc.page_content)
        retrieved_docs.append({
            "page_content": doc.page_content,
            "metadata": metadata,
        })
        scores.append(numeric_score)

    context = "\n".join(context_parts)
    top_score = min(scores) if scores else None

    print("📊 Retrieval Scores:", scores)
    print("🏅 Top Score:", top_score)

    return {
        "context": context,
        "retrieved_docs": retrieved_docs,
        "retrieval_scores": scores,
        "top_score": top_score,
    }