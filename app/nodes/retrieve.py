from app.config import settings
from app.services.logging_utils import get_logger, log_timing
from app.state import AgentState

logger = get_logger("app.retrieve")


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
    request_id = state.get("request_id", "-")
    query = (
        state.get("rewritten_query")
        or state.get("retrieval_query")
        or state["query"]
    )
    filters = state.get("filters", {})

    chroma_filter = build_chroma_filter(filters)

    logger.info(
        f"[request_id={request_id}] Retrieve query='{query}' | raw_filters={filters} | chroma_filter={chroma_filter}"
    )

    try:
        with log_timing(logger, "vector_retrieval", request_id):
            results = vectordb.similarity_search_with_score(
                query=query,
                k=settings.RETRIEVAL_TOP_K,
                filter=chroma_filter,
            )
    except Exception as exc:
        logger.error(f"[request_id={request_id}] Retrieval failed: {exc}")
        return {
            "context": "",
            "retrieved_docs": [],
            "retrieval_scores": [],
            "top_score": None,
        }

    logger.info(f"[request_id={request_id}] Retrieved docs count={len(results)}")

    if not results:
        logger.warning(f"[request_id={request_id}] No documents retrieved")
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
        numeric_score = float(score)

        logger.info(
            f"[request_id={request_id}] Doc {i} | score={numeric_score} | source={metadata.get('source')} | metadata={metadata}"
        )

        context_parts.append(doc.page_content)
        retrieved_docs.append({
            "page_content": doc.page_content,
            "metadata": metadata,
        })
        scores.append(numeric_score)

    context = "\n".join(context_parts)
    top_score = min(scores) if scores else None

    logger.info(f"[request_id={request_id}] Retrieval scores={scores} | top_score={top_score}")

    return {
        "context": context,
        "retrieved_docs": retrieved_docs,
        "retrieval_scores": scores,
        "top_score": top_score,
    }