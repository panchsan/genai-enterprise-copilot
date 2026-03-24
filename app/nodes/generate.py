from app.config import settings
from app.prompts import (
    DOCUMENT_ACTION_SYSTEM_PROMPT,
    GROUNDED_GENERATION_SYSTEM_PROMPT,
    GROUNDING_FAILURE_RESPONSE,
)
from app.services.llm import get_azure_openai_client, safe_chat_completion
from app.services.logging_utils import get_logger, log_timing
from app.state import AgentState

client = get_azure_openai_client()
logger = get_logger("app.generate")


def build_user_message(state: AgentState, context: str) -> str:
    action = state.get("action", "qa")
    query = state["query"]
    target_sources = state.get("target_sources", []) or []

    if action == "summarize_document":
        if target_sources:
            return (
                f"Summarize the following document(s): {', '.join(target_sources)}.\n\n"
                f"User request: {query}\n\nContext:\n{context}"
            )
        return f"Summarize this document based on the context.\n\nUser request: {query}\n\nContext:\n{context}"

    if action == "answer_by_source":
        if target_sources:
            return (
                f"Answer the user's question using only these source(s): {', '.join(target_sources)}.\n\n"
                f"Question: {query}\n\nContext:\n{context}"
            )
        return f"Question: {query}\n\nContext:\n{context}"

    if action == "compare_documents":
        return (
            f"Compare the relevant documents or sources based on the user's request.\n\n"
            f"User request: {query}\n\nContext:\n{context}"
        )

    return f"Question: {query}\n\nContext:\n{context}"


def generate(state: AgentState):
    request_id = state.get("request_id", "-")
    context = state.get("context", "")
    chat_history = state.get("chat_history", [])
    retrieved_docs = state.get("retrieved_docs", [])
    action = state.get("action", "qa")

    logger.info(
        f"[request_id={request_id}] Generating grounded response | "
        f"action={action} | context_chars={len(context)} | retrieved_docs={len(retrieved_docs)}"
    )

    system_prompt = (
        DOCUMENT_ACTION_SYSTEM_PROMPT
        if action in {"summarize_document", "answer_by_source", "compare_documents"}
        else GROUNDED_GENERATION_SYSTEM_PROMPT
    )

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