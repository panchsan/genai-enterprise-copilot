from app.state import AgentState


def analyze_query(state: AgentState):
    query = state["query"].lower().strip()

    print("\n🧭 [ANALYZE] Incoming query:", query)

    retrieve_keywords = [
        "document",
        "policy",
        "company",
        "internal",
        "this file",
        "this document",
        "knowledge base",
        "according to the document",
        "based on the file",
        "in our data",
        "in the policy",
        "summarize document",
        "from the document",
        "hr policy",
        "finance policy",
        "onboarding"
    ]

    fallback_keywords = [
        "weather",
        "stock price",
        "sports score",
        "live score"
    ]

    if any(keyword in query for keyword in fallback_keywords):
        route = "fallback"
    elif any(keyword in query for keyword in retrieve_keywords):
        route = "retrieve"
    else:
        route = "direct"

    print("🛣️ [ANALYZE] Selected route:", route)
    return {"route": route}


def route_query(state: AgentState):
    route = state.get("route", "fallback")
    print("🔀 [ROUTER] Routing to:", route)

    if route == "retrieve":
        return "retrieve"
    if route == "direct":
        return "direct_answer"
    return "fallback"