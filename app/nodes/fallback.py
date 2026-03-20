from app.state import AgentState


def fallback(state: AgentState):
    print("\n🛑 [FALLBACK] Unsupported or out-of-scope query.")
    return {
        "answer": "I cannot answer that from the current knowledge base."
    }