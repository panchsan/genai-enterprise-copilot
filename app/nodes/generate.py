from app.config import settings
from app.services.llm import get_azure_openai_client
from app.services.logging_utils import get_logger, log_timing
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

    with log_timing(logger, "generate_llm", request_id):
        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=messages,
        )

    answer = response.choices[0].message.content or "I don't know"

    logger.info(
        f"[request_id={request_id}] Grounded answer generated | "
        f"answer_preview={answer[:120]!r}"
    )

    return {"answer": answer}