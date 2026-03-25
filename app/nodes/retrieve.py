from copy import deepcopy

from app.config import settings
from app.services.action_utils import (
    dedupe_keep_order,
    get_action_config,
    normalize_action,
)
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
        "retrieved_sources": dedupe_keep_order(sources),
        "retrieval_scores": numeric_scores,
        "top_score": min(numeric_scores) if numeric_scores else None,
        "retrieval_status": retrieval_status,
    }


def _run_search(vectordb, query: str, k: int, chroma_filter):
    return vectordb.similarity_search_with_score(
        query=query,
        k=k,
        filter=chroma_filter,
    )


def _search(vectordb, query: str, k: int, filters: dict | None, request_id: str, timer_name: str):
    chroma_filter = build_chroma_filter(filters)
    logger.info(
        f"[request_id={request_id}] Search | query='{query}' | k={k} | "
        f"filters={filters or {}} | chroma_filter={chroma_filter}"
    )

    with log_timing(logger, timer_name, request_id):
        raw_results = _run_search(
            vectordb=vectordb,
            query=query,
            k=k,
            chroma_filter=chroma_filter,
        )

    results = _apply_score_threshold(raw_results)

    logger.info(
        f"[request_id={request_id}] Search result | raw_count={len(raw_results)} | "
        f"filtered_count={len(results)} | "
        f"threshold={getattr(settings, 'RETRIEVAL_SCORE_THRESHOLD', None)}"
    )
    return results


def _resolve_effective_sources(state: AgentState, vectordb, filters: dict) -> list[str]:
    explicit_source = filters.get("source")
    target_sources = state.get("target_sources", []) or []

    known_sources = get_known_sources(vectordb)
    resolved = []

    if explicit_source:
        matched = resolve_target_source(explicit_source, known_sources)
        if matched:
            resolved.append(matched)
        else:
            # keep original value so the caller can decide how to fail
            resolved.append(explicit_source)

    for src in target_sources:
        matched = resolve_target_source(src, known_sources)
        if matched:
            resolved.append(matched)

    return dedupe_keep_order(resolved)


def _retrieve_for_qa(query: str, filters: dict, vectordb, request_id: str, config: dict):
    try:
        results = _search(
            vectordb=vectordb,
            query=query,
            k=config["top_k"],
            filters=filters,
            request_id=request_id,
            timer_name="vector_retrieval_qa",
        )
    except Exception as exc:
        logger.exception(f"[request_id={request_id}] QA retrieval failed: {exc}")
        return _format_results([], retrieval_status="no_docs")

    if results:
        return _format_results(results, retrieval_status="found")

    if filters and config["allow_filter_relaxation"]:
        logger.warning(
            f"[request_id={request_id}] QA strict retrieval returned no results. "
            f"Retrying without filters."
        )
        try:
            results = _search(
                vectordb=vectordb,
                query=query,
                k=config["top_k"],
                filters=None,
                request_id=request_id,
                timer_name="vector_retrieval_qa_no_filters",
            )
        except Exception as exc:
            logger.exception(f"[request_id={request_id}] QA retry without filters failed: {exc}")
            return _format_results([], retrieval_status="no_docs")

        if results:
            return _format_results(results, retrieval_status="found")

    return _format_results([], retrieval_status="no_docs")


def _retrieve_for_summary(
    query: str,
    filters: dict,
    effective_sources: list[str],
    vectordb,
    request_id: str,
    config: dict,
):
    strict_filters = deepcopy(filters)

    if "source" not in strict_filters and effective_sources:
        strict_filters["source"] = effective_sources[0]

    try:
        results = _search(
            vectordb=vectordb,
            query=query,
            k=config["top_k"],
            filters=strict_filters,
            request_id=request_id,
            timer_name="vector_retrieval_summary",
        )
    except Exception as exc:
        logger.exception(f"[request_id={request_id}] Summary retrieval failed: {exc}")
        return _format_results([], retrieval_status="no_docs")

    if not results:
        return _format_results([], retrieval_status="no_docs")

    return _format_results(results, retrieval_status="found")


def _retrieve_for_answer_by_source(
    query: str,
    filters: dict,
    effective_sources: list[str],
    vectordb,
    request_id: str,
    config: dict,
):
    known_sources = get_known_sources(vectordb)

    source = None
    if filters.get("source"):
        source = filters["source"]
    elif effective_sources:
        source = effective_sources[0]

    if not source:
        logger.warning(f"[request_id={request_id}] answer_by_source missing source")
        return _format_results([], retrieval_status="missing_required_source")

    resolved_source = resolve_target_source(source, known_sources)
    if not resolved_source:
        logger.warning(
            f"[request_id={request_id}] answer_by_source source not found | source={source}"
        )
        return _format_results([], retrieval_status="source_not_found")

    strict_filters = deepcopy(filters)
    strict_filters["source"] = resolved_source

    try:
        results = _search(
            vectordb=vectordb,
            query=query,
            k=config["top_k"],
            filters=strict_filters,
            request_id=request_id,
            timer_name="vector_retrieval_answer_by_source",
        )
    except Exception as exc:
        logger.exception(f"[request_id={request_id}] answer_by_source retrieval failed: {exc}")
        return _format_results([], retrieval_status="source_not_found")

    if not results:
        return _format_results([], retrieval_status="source_not_found")

    return _format_results(results, retrieval_status="found")


def _retrieve_for_compare(
    query: str,
    filters: dict,
    effective_sources: list[str],
    vectordb,
    request_id: str,
    config: dict,
):
    all_results = []

    if len(effective_sources) >= 2:
        per_source_k = max(3, config["top_k"] // min(len(effective_sources), 4))

        for source in effective_sources[:4]:
            strict_filters = deepcopy(filters)
            strict_filters["source"] = source

            try:
                source_results = _search(
                    vectordb=vectordb,
                    query=query,
                    k=per_source_k,
                    filters=strict_filters,
                    request_id=request_id,
                    timer_name=f"vector_retrieval_compare_{source}",
                )
                all_results.extend(source_results)
            except Exception as exc:
                logger.exception(
                    f"[request_id={request_id}] compare retrieval failed for source={source}: {exc}"
                )

    else:
        try:
            all_results = _search(
                vectordb=vectordb,
                query=query,
                k=config["top_k"],
                filters=filters,
                request_id=request_id,
                timer_name="vector_retrieval_compare",
            )
        except Exception as exc:
            logger.exception(f"[request_id={request_id}] compare retrieval failed: {exc}")
            return _format_results([], retrieval_status="no_docs")

    if not all_results:
        return _format_results([], retrieval_status="no_docs")

    distinct_sources = dedupe_keep_order([
        (doc.metadata or {}).get("source")
        for doc, _ in all_results
        if (doc.metadata or {}).get("source")
    ])

    if len(distinct_sources) < 2:
        return _format_results(all_results, retrieval_status="insufficient_sources")

    return _format_results(all_results, retrieval_status="found")


def retrieve(state: AgentState, vectordb):
    request_id = state.get("request_id", "-")
    query = state.get("rewritten_query") or state.get("retrieval_query") or state["query"]
    filters = deepcopy(state.get("filters", {}) or {})
    action = normalize_action(state.get("action", "qa"))
    config = get_action_config(action)

    effective_sources = _resolve_effective_sources(state, vectordb, filters)

    logger.info(
        f"[request_id={request_id}] Retrieve start | action={action} | "
        f"query='{query}' | filters={filters} | effective_sources={effective_sources}"
    )

    if action == "answer_by_source":
        return _retrieve_for_answer_by_source(
            query=query,
            filters=filters,
            effective_sources=effective_sources,
            vectordb=vectordb,
            request_id=request_id,
            config=config,
        )

    if action == "summarize_document":
        return _retrieve_for_summary(
            query=query,
            filters=filters,
            effective_sources=effective_sources,
            vectordb=vectordb,
            request_id=request_id,
            config=config,
        )

    if action == "compare_documents":
        return _retrieve_for_compare(
            query=query,
            filters=filters,
            effective_sources=effective_sources,
            vectordb=vectordb,
            request_id=request_id,
            config=config,
        )

    return _retrieve_for_qa(
        query=query,
        filters=filters,
        vectordb=vectordb,
        request_id=request_id,
        config=config,
    )