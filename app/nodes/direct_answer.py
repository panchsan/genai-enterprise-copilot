from app.config import settings
from app.services.llm import get_azure_openai_client
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
            "content": "You are a helpful assistant. Answer briefly and clearly.",
        }
    ]

    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append(
        {
            "role": "user",
            "content": state["query"],
        }
    )

    with log_timing(logger, "direct_answer_llm", request_id):
        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=messages,
        )

    answer = response.choices[0].message.content or "I don't know"

    logger.info(
        f"[request_id={request_id}] Direct answer generated | "
        f"answer_preview={answer[:120]!r}"
    )

    return {"answer": answer}