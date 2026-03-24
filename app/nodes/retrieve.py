from copy import deepcopy

from app.config import settings
from app.services.logging_utils import get_logger, log_timing
from app.services.metadata_utils import resolve_target_source
from app.services.vectorstore import get_known_sources
from app.state import AgentState

logger = get_logger("app.retrieve")


def build_chroma_filter(filters: dict | None):
    if not filters:
        return None

    cleaned_filters = {k: v for k, v in filters.items() if v is not None and v != ""}
    if not cleaned_filters:
        return None

    if len(cleaned_filters) == 1:
        key, value = next(iter(cleaned_filters.items()))
        return {key: value}

    return {"$and": [{k: v} for k, v in cleaned_filters.items()]}


def _run_search(vectordb, query: str, action: str, chroma_filter):
    return vectordb.similarity_search_with_score(
        query=query,
        k=settings.RETRIEVAL_TOP_K if action != "compare_documents" else 6,
        filter=chroma_filter,
    )


def _apply_score_threshold(results):
    if not getattr(settings, "RETRIEVAL_HARD_FILTER_ENABLED", True):
        return results

    threshold = getattr(settings, "RETRIEVAL_SCORE_THRESHOLD", None)
    if threshold is None:
        return results

    filtered = []
    for doc, score in results:
        if score is None:
            continue

        score = float(score)
        # Chroma distance-like score: lower is better
        if score <= threshold:
            filtered.append((doc, score))

    return filtered


def _format_results(results, retrieval_status: str = "found"):
    if not results:
        return {
            "context": "",
            "retrieved_docs": [],
            "retrieved_sources": [],
            "retrieval_scores": [],
            "top_score": None,
            "retrieval_status": retrieval_status,
        }

    context_parts = []
    retrieved_docs = []
    sources = []
    scores = []

    for doc, score in results:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source")

        context_parts.append(
            f"[SOURCE: {source or 'unknown'}]\n{doc.page_content}"
        )
        retrieved_docs.append({
            "page_content": doc.page_content,
            "metadata": metadata,
        })

        if source:
            sources.append(source)

        scores.append(float(score) if score is not None else None)

    numeric_scores = [s for s in scores if s is not None]

    return {
        "context": "\n\n".join(context_parts),
        "retrieved_docs": retrieved_docs,
        "retrieved_sources": sources,
        "retrieval_scores": numeric_scores,
        "top_score": min(numeric_scores) if numeric_scores else None,
        "retrieval_status": retrieval_status,
    }


def retrieve(state: AgentState, vectordb):
    request_id = state.get("request_id", "-")
    query = state.get("rewritten_query") or state.get("retrieval_query") or state["query"]
    filters = deepcopy(state.get("filters", {}) or {})
    target_sources = state.get("target_sources", []) or []
    action = state.get("action", "qa")

    inferred_single_source = (
        action in {"summarize_document", "answer_by_source"}
        and len(target_sources) == 1
        and "source" not in filters
    )

    strict_filters = deepcopy(filters)
    resolved_source = None

    if inferred_single_source:
        known_sources = get_known_sources(vectordb)
        guessed_source = target_sources[0]
        resolved_source = resolve_target_source(guessed_source, known_sources)

        if resolved_source:
            strict_filters["source"] = resolved_source
            logger.info(
                f"[request_id={request_id}] Resolved inferred target source "
                f"'{guessed_source}' -> '{resolved_source}'"
            )

    strict_chroma_filter = build_chroma_filter(strict_filters)

    logger.info(
        f"[request_id={request_id}] Retrieve query='{query}' | "
        f"action={action} | raw_filters={strict_filters} | "
        f"target_sources={target_sources} | resolved_source={resolved_source} | "
        f"chroma_filter={strict_chroma_filter}"
    )

    try:
        with log_timing(logger, "vector_retrieval", request_id):
            raw_results = _run_search(
                vectordb=vectordb,
                query=query,
                action=action,
                chroma_filter=strict_chroma_filter,
            )
    except Exception as exc:
        logger.exception(
            f"[request_id={request_id}] Retrieval failed on strict search: {exc}"
        )
        return _format_results([], retrieval_status="no_docs")

    results = _apply_score_threshold(raw_results)

    logger.info(
        f"[request_id={request_id}] Strict search raw_count={len(raw_results)} | "
        f"filtered_count={len(results)} | "
        f"threshold={getattr(settings, 'RETRIEVAL_SCORE_THRESHOLD', None)}"
    )

    # Retry without filters only if strict filters were used and nothing survived.
    if not results and strict_chroma_filter is not None:
        logger.warning(
            f"[request_id={request_id}] No documents retrieved with filters. "
            f"Retrying without filters."
        )

        try:
            with log_timing(logger, "vector_retrieval_no_filters", request_id):
                raw_results = _run_search(
                    vectordb=vectordb,
                    query=query,
                    action=action,
                    chroma_filter=None,
                )
        except Exception as exc:
            logger.exception(
                f"[request_id={request_id}] Retrieval failed on no-filter retry: {exc}"
            )
            return _format_results([], retrieval_status="no_docs")

        results = _apply_score_threshold(raw_results)

        logger.info(
            f"[request_id={request_id}] No-filter search raw_count={len(raw_results)} | "
            f"filtered_count={len(results)} | "
            f"threshold={getattr(settings, 'RETRIEVAL_SCORE_THRESHOLD', None)}"
        )

    if not results:
        logger.warning(f"[request_id={request_id}] No documents retrieved after thresholding")
        return _format_results([], retrieval_status="no_docs")

    return _format_results(results, retrieval_status="found")