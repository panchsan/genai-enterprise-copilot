from app.config import settings
from app.services.llm import get_azure_openai_client
from app.services.logging_utils import get_logger, log_timing
from app.state import AgentState

client = get_azure_openai_client()
logger = get_logger("app.rewrite_query")


SYSTEM_PROMPT = """
You are a query rewriting component for an enterprise RAG system.

Your job is to rewrite the user's latest message into a standalone retrieval query using recent chat history.

Rules:
1. Keep the meaning exactly the same.
2. Resolve references like "that", "it", "this", "those", "the previous one".
3. Preserve enterprise context such as HR policy, finance policy, onboarding, or a specific document.
4. Make the rewritten query concise and retrieval-friendly.
5. If the user's latest message is already standalone, return it unchanged.
6. Return ONLY the rewritten query text. No explanation. No quotes.

Examples:
History:
User: What does HR policy say about leave?
Assistant: ...
User: Explain that simply
Output:
Explain the HR leave policy in simple terms

History:
User: Summarize onboarding document
Assistant: ...
User: What about finance?
Output:
Summarize the finance policy document

History:
User: What does the company policy say about working hours?
Assistant: ...
User: What does it say about attendance?
Output:
What does the company policy say about attendance?
""".strip()


def rewrite_query(state: AgentState):
    request_id = state.get("request_id", "-")
    original_query = state.get("retrieval_query") or state["query"]
    chat_history = state.get("chat_history", [])

    logger.info(f"[request_id={request_id}] Original retrieval query='{original_query}'")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if chat_history:
        history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history[-6:]
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

    with log_timing(logger, "rewrite_query_llm", request_id):
        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0,
        )

    rewritten_query = (response.choices[0].message.content or original_query).strip()

    if not rewritten_query:
        rewritten_query = original_query
        logger.warning(f"[request_id={request_id}] Empty rewritten query; using original query")

    logger.info(f"[request_id={request_id}] Rewritten query='{rewritten_query}'")

    return {"rewritten_query": rewritten_query}