from app.config import settings
from app.prompts import DIRECT_ANSWER_SYSTEM_PROMPT, DIRECT_FAILURE_RESPONSE
from app.services.llm import get_azure_openai_client, safe_chat_completion
from app.services.logging_utils import get_logger, log_timing
from app.state import AgentState

client = get_azure_openai_client()
logger = get_logger("app.direct_answer")


def direct_answer(state: AgentState):
    request_id = state.get("request_id", "-")
    chat_history = state.get("chat_history", [])

    if (
        state.get("retrieval_decision") == "ungrounded"
        and not getattr(settings, "ALLOW_DIRECT_LLM_FALLBACK", False)
    ):
        logger.warning(
            f"[request_id={request_id}] Direct-answer blocked for ungrounded retrieval"
        )
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

    logger.info(f"[request_id={request_id}] Answering via direct path")

    messages = [
        {
            "role": "system",
            "content": DIRECT_ANSWER_SYSTEM_PROMPT,
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
            "content": state["query"],
        }
    )

    try:
        with log_timing(logger, "direct_answer_llm", request_id):
            response = safe_chat_completion(
                client,
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE_DEFAULT,
            )

        answer = response.choices[0].message.content or "I don't know"
    except Exception as exc:
        logger.error(f"[request_id={request_id}] Direct answer generation failed: {exc}")
        answer = DIRECT_FAILURE_RESPONSE

    logger.info(
        f"[request_id={request_id}] Direct answer generated | "
        f"answer_preview={answer[:120]!r}"
    )

    return {"answer": answer}