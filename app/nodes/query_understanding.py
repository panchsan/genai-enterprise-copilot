import json

from app.config import settings
from app.services.llm import get_azure_openai_client
from app.state import AgentState

client = get_azure_openai_client()


SYSTEM_PROMPT = """
You are a query analysis component for an enterprise RAG assistant.

Return ONLY valid JSON:

{
  "route": "retrieve" | "direct" | "fallback",
  "retrieval_query": "<string>",
  "filters": {
    "doc_type": "<optional string>",
    "department": "<optional string>",
    "source": "<optional string>"
  }
}

Rules:

1. If the query is a FOLLOW-UP (e.g. "explain that", "what about", "summarize that", "tell me more"):
   -> ALWAYS use route = "retrieve"
   -> Use previous conversation context to infer meaning

2. Use "retrieve" when:
   - query relates to company docs, policies, onboarding, internal knowledge
   - OR follow-up to a previous retrieval-based question

3. Use "direct" only when:
   - general knowledge question
   - standalone question unrelated to enterprise docs

4. Use "fallback" for:
   - weather, stock price, sports score, live external data

5. Infer filters:
   - "HR policy" -> doc_type=policy, department=HR
   - "finance policy" -> doc_type=policy, department=Finance
   - "onboarding" -> doc_type=onboarding

6. retrieval_query:
   - clean and concise
   - include inferred context for follow-ups

Return ONLY JSON.
""".strip()


FOLLOW_UP_KEYWORDS = [
    "explain that",
    "explain it",
    "summarize that",
    "tell me more",
    "what about",
    "simplify",
    "in simple terms",
    "explain simply",
    "explain that simply",
    "what does it say",
    "what about this",
    "and finance",
    "and hr",
]


def is_follow_up_query(query: str, chat_history: list[dict]) -> bool:
    query_lower = query.lower().strip()

    if not chat_history:
        return False

    if any(keyword in query_lower for keyword in FOLLOW_UP_KEYWORDS):
        return True

    short_follow_up_patterns = {
        "why",
        "how",
        "what about",
        "explain",
        "summarize",
        "simplify",
        "details",
        "more",
    }

    if query_lower in short_follow_up_patterns:
        return True

    if len(query_lower.split()) <= 4 and any(
        pronoun in query_lower for pronoun in ["that", "it", "this", "those", "them"]
    ):
        return True

    return False


def analyze_query(state: AgentState):
    query = state["query"]
    chat_history = state.get("chat_history", [])

    print("\n🧠 [QUERY UNDERSTANDING] Incoming query:", query)

    if is_follow_up_query(query, chat_history):
        print("⚠️ [QUERY UNDERSTANDING] Detected follow-up query -> forcing retrieve route")
        return {
            "route": "retrieve",
            "retrieval_query": query,
            "filters": state.get("filters", {}) or {},
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if chat_history:
        messages.append(
            {
                "role": "system",
                "content": f"Recent chat history: {json.dumps(chat_history[-6:])}",
            }
        )

    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0,
    )

    raw_output = response.choices[0].message.content or "{}"
    print("🧾 [QUERY UNDERSTANDING] Raw model output:", raw_output)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse model output. Falling back to safe defaults.")
        parsed = {
            "route": "direct",
            "retrieval_query": query,
            "filters": {},
        }

    route = parsed.get("route", "direct")
    retrieval_query = parsed.get("retrieval_query", query)
    filters = parsed.get("filters", {}) or {}

    allowed_routes = {"retrieve", "direct", "fallback"}
    if route not in allowed_routes:
        route = "direct"

    print("🛣️ [QUERY UNDERSTANDING] Route:", route)
    print("🔎 [QUERY UNDERSTANDING] Retrieval Query:", retrieval_query)
    print("🎯 [QUERY UNDERSTANDING] Filters:", filters)

    return {
        "route": route,
        "retrieval_query": retrieval_query,
        "filters": filters,
    }


def route_query(state: AgentState):
    route = state.get("route", "direct")
    print("🔀 [ROUTER] Routing to:", route)

    if route == "retrieve":
        return "retrieve"
    if route == "direct":
        return "direct_answer"
    return "fallback"