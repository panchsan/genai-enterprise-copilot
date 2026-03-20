from app.config import settings
from app.services.llm import get_azure_openai_client
from app.state import AgentState

client = get_azure_openai_client()


def direct_answer(state: AgentState):
    print("\n⚡ [DIRECT] Answering without retrieval...")

    chat_history = state.get("chat_history", [])

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

    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
    )

    answer = response.choices[0].message.content or "I don't know"
    print("💡 [DIRECT] Final Answer:", answer)

    return {"answer": answer}