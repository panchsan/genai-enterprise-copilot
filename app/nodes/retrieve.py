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

    docs = vectordb.similarity_search(
        query=query,
        k=3,
        filter=chroma_filter
    )

    print("📚 Retrieved Docs Count:", len(docs))

    if not docs:
        print("❌ No documents retrieved!")
        return {
            "context": "",
            "retrieved_docs": []
        }

    context_parts = []
    retrieved_docs = []

    for i, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {})
        content = doc.page_content[:300]

        print(f"\n--- Doc {i} ---")
        print("Metadata:", metadata)
        print(content)

        context_parts.append(doc.page_content)
        retrieved_docs.append({
            "page_content": doc.page_content,
            "metadata": metadata
        })

    context = "\n".join(context_parts)

    return {
        "context": context,
        "retrieved_docs": retrieved_docs
    }