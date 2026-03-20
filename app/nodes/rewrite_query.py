from app.config import settings
from app.services.llm import get_azure_openai_client
from app.state import AgentState

client = get_azure_openai_client()


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
    original_query = state.get("retrieval_query") or state["query"]
    chat_history = state.get("chat_history", [])

    print("\n📝 [REWRITE] Original Retrieval Query:", original_query)

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

    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0,
    )

    rewritten_query = (response.choices[0].message.content or original_query).strip()

    if not rewritten_query:
        rewritten_query = original_query

    print("✏️ [REWRITE] Rewritten Query:", rewritten_query)

    return {"rewritten_query": rewritten_query}