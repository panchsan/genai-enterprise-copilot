from app.config import settings
from app.prompts import (
    ANSWER_BY_SOURCE_SYSTEM_PROMPT,
    COMPARE_DOCUMENTS_SYSTEM_PROMPT,
    GROUNDED_GENERATION_SYSTEM_PROMPT,
    GROUNDING_FAILURE_RESPONSE,
    SUMMARIZE_DOCUMENT_SYSTEM_PROMPT,
)
from app.services.action_utils import normalize_action
from app.services.llm import get_azure_openai_client, safe_chat_completion
from app.services.logging_utils import get_logger, log_timing
from app.state import AgentState

client = get_azure_openai_client()
logger = get_logger("app.generate")


def build_user_message(state: AgentState, context: str) -> str:
    action = normalize_action(state.get("action", "qa"))
    query = state["query"]
    target_sources = state.get("target_sources", []) or []
    retrieved_docs = state.get("retrieved_docs", []) or []

    available_sources = []
    for doc in retrieved_docs:
        source = doc.get("metadata", {}).get("source")
        if source and source not in available_sources:
            available_sources.append(source)

    if action == "summarize_document":
        source_line = f"Matched source(s): {', '.join(available_sources)}" if available_sources else "Matched source(s): none"
        return (
            f"User request: {query}\n\n"
            f"{source_line}\n\n"
            f"Context:\n{context}"
        )

    if action == "answer_by_source":
        source_line = f"Selected source(s): {', '.join(available_sources or target_sources)}"
        return (
            f"{source_line}\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}"
        )

    if action == "compare_documents":
        source_line = f"Documents to compare: {', '.join(available_sources)}"
        return (
            f"{source_line}\n\n"
            f"User request: {query}\n\n"
            f"Context:\n{context}"
        )

    return f"Question: {query}\n\nContext:\n{context}"


def _choose_system_prompt(action: str) -> str:
    if action == "summarize_document":
        return SUMMARIZE_DOCUMENT_SYSTEM_PROMPT
    if action == "answer_by_source":
        return ANSWER_BY_SOURCE_SYSTEM_PROMPT
    if action == "compare_documents":
        return COMPARE_DOCUMENTS_SYSTEM_PROMPT
    return GROUNDED_GENERATION_SYSTEM_PROMPT


def _build_action_failure_response(action: str, retrieval_status: str) -> str | None:
    if action == "answer_by_source":
        if retrieval_status == "missing_required_source":
            return "Please select or specify a source document for 'Answer by Source'."
        if retrieval_status == "source_not_found":
            return "I could not find the selected source in the indexed documents."

    if action == "compare_documents" and retrieval_status == "insufficient_sources":
        return "I need at least two matching documents or sources to compare."

    if action == "summarize_document" and retrieval_status == "no_docs":
        return "I could not find document content to summarize."

    return None


def generate(state: AgentState):
    request_id = state.get("request_id", "-")
    context = state.get("context", "")
    chat_history = state.get("chat_history", [])
    retrieved_docs = state.get("retrieved_docs", [])
    action = normalize_action(state.get("action", "qa"))
    retrieval_status = state.get("retrieval_status", "found")

    logger.info(
        f"[request_id={request_id}] Generating grounded response | "
        f"action={action} | retrieval_status={retrieval_status} | "
        f"context_chars={len(context)} | retrieved_docs={len(retrieved_docs)}"
    )

    action_failure = _build_action_failure_response(action, retrieval_status)
    if action_failure:
        logger.info(
            f"[request_id={request_id}] Returning action-specific failure response | "
            f"action={action} | retrieval_status={retrieval_status}"
        )
        return {
            "answer": action_failure,
            "retrieval_decision": "no_docs",
        }

    system_prompt = _choose_system_prompt(action)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    for msg in chat_history[-settings.MAX_CHAT_HISTORY_MESSAGES:]:
        if isinstance(msg, dict) and msg.get("role") in {"user", "assistant", "system"}:
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg.get("content", ""),
                }
            )

    messages.append(
        {
            "role": "user",
            "content": build_user_message(state, context),
        }
    )

    try:
        with log_timing(logger, "generate_llm", request_id):
            response = safe_chat_completion(
                client,
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE_DEFAULT,
            )

        answer = response.choices[0].message.content or "I don't know"
    except Exception as exc:
        logger.error(f"[request_id={request_id}] Grounded generation failed: {exc}")
        answer = GROUNDING_FAILURE_RESPONSE

    logger.info(
        f"[request_id={request_id}] Grounded answer generated | "
        f"action={action} | answer_preview={answer[:120]!r}"
    )

    return {"answer": answer, "retrieval_decision": "grounded"}