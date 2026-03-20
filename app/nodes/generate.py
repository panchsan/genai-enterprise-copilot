from app.config import settings
from app.services.llm import get_azure_openai_client
from app.state import AgentState

client = get_azure_openai_client()


def generate(state: AgentState):
    print("\n🤖 [GENERATE] Creating grounded response...")

    context = state.get("context", "")
    chat_history = state.get("chat_history", [])

    print("📦 Context preview:")
    print(context[:500] if context else "No context")

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

    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
    )

    answer = response.choices[0].message.content or "I don't know"
    print("💡 [GENERATE] Final Answer:", answer)

    return {"answer": answer}