from app.config import settings
from app.prompts import REWRITE_QUERY_PROMPT
from app.services.llm import get_azure_openai_client, safe_chat_completion
from app.services.logging_utils import get_logger, log_timing
from app.state import AgentState

client = get_azure_openai_client()
logger = get_logger("app.rewrite_query")


def rewrite_query(state: AgentState):
    request_id = state.get("request_id", "-")
    original_query = state.get("retrieval_query") or state["query"]
    chat_history = state.get("chat_history", [])

    logger.info(f"[request_id={request_id}] Original retrieval query='{original_query}'")

    messages = [
        {"role": "system", "content": REWRITE_QUERY_PROMPT},
    ]

    if chat_history:
        history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history[-settings.MAX_CHAT_HISTORY_MESSAGES:]
        )
        messages.append(
            {
                "role": "system",
                "content": f"Recent conversation:\n{history_text}",
            }
        )

    messages.append(
        {
            "role": "user",
            "content": original_query,
        }
    )

    try:
        with log_timing(logger, "rewrite_query_llm", request_id):
            response = safe_chat_completion(
                client,
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE_DETERMINISTIC,
            )

        rewritten_query = (response.choices[0].message.content or original_query).strip()
    except Exception as exc:
        logger.error(f"[request_id={request_id}] Rewrite query failed: {exc}")
        rewritten_query = original_query

    if not rewritten_query:
        rewritten_query = original_query
        logger.warning(f"[request_id={request_id}] Empty rewritten query; using original query")

    logger.info(f"[request_id={request_id}] Rewritten query='{rewritten_query}'")

    return {"rewritten_query": rewritten_query}