from app.config import settings
from app.services.llm import get_azure_openai_client
from app.services.logging_utils import get_logger, log_timing
from app.services.llm import get_azure_openai_client, safe_chat_completion
from app.state import AgentState

client = get_azure_openai_client()
logger = get_logger("app.generate")


def generate(state: AgentState):
    request_id = state.get("request_id", "-")
    context = state.get("context", "")
    chat_history = state.get("chat_history", [])
    retrieved_docs = state.get("retrieved_docs", [])

    logger.info(
        f"[request_id={request_id}] Generating grounded response | "
        f"context_chars={len(context)} | retrieved_docs={len(retrieved_docs)}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Use the provided context to answer the question. "
                "If the answer is not in the context, say 'I don't know'."
            ),
        }
    ]

    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append(
        {
            "role": "user",
            "content": f"Question: {state['query']}\n\nContext:\n{context}",
        }
    )

    try:
        with log_timing(logger, "generate_llm", request_id):
            response = safe_chat_completion(
                client,
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=messages,
            )

        answer = response.choices[0].message.content or "I don't know"
    except Exception as exc:
        logger.error(f"[request_id={request_id}] Grounded generation failed: {exc}")
        answer = "I’m sorry, I couldn’t generate a grounded answer right now."

    logger.info(
        f"[request_id={request_id}] Grounded answer generated | "
        f"answer_preview={answer[:120]!r}"
    )

    return {"answer": answer}