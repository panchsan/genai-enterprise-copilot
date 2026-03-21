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

    logger.info(f"[request_id={request_id}] Answering via direct path")

    messages = [
        {
            "role": "system",
            "content": DIRECT_ANSWER_SYSTEM_PROMPT,
        }
    ]

    for msg in chat_history[-settings.MAX_CHAT_HISTORY_MESSAGES:]:
        messages.append(msg)

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