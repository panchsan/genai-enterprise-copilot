import json

from app.config import settings
from app.services.llm import get_azure_openai_client
from app.services.logging_utils import get_logger, log_timing
from app.services.llm import get_azure_openai_client, safe_chat_completion
from app.state import AgentState

client = get_azure_openai_client()
logger = get_logger("app.query_understanding")

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
        "why", "how", "what about", "explain", "summarize",
        "simplify", "details", "more",
    }

    if query_lower in short_follow_up_patterns:
        return True

    if len(query_lower.split()) <= 4 and any(
        pronoun in query_lower for pronoun in ["that", "it", "this", "those", "them"]
    ):
        return True

    return False


def analyze_query(state: AgentState):
    request_id = state.get("request_id", "-")
    query = state["query"]
    chat_history = state.get("chat_history", [])

    logger.info(f"[request_id={request_id}] Incoming query='{query}'")

    if is_follow_up_query(query, chat_history):
        logger.info(f"[request_id={request_id}] Follow-up detected -> forcing retrieve route")
        return {
            "route": "retrieve",
            "retrieval_query": query,
            "filters": state.get("filters", {}) or {},
        }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.append(
            {
                "role": "system",
                "content": f"Recent chat history: {json.dumps(chat_history[-6:])}",
            }
        )

    messages.append({"role": "user", "content": query})

    try:
        with log_timing(logger, "query_understanding_llm", request_id):
            response = safe_chat_completion(
                client,
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=messages,
                temperature=0,
            )

        raw_output = response.choices[0].message.content or "{}"
        logger.info(f"[request_id={request_id}] Raw model output={raw_output}")

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning(f"[request_id={request_id}] Failed to parse model output; using safe defaults")
            parsed = {
                "route": "direct",
                "retrieval_query": query,
                "filters": {},
            }
    except Exception as exc:
        logger.error(f"[request_id={request_id}] Query understanding failed: {exc}")
        parsed = {
            "route": "direct",
            "retrieval_query": query,
            "filters": {},
        }
    route = parsed.get("route", "direct")
    retrieval_query = parsed.get("retrieval_query", query)
    filters = parsed.get("filters", {}) or {}

    if route not in {"retrieve", "direct", "fallback"}:
        route = "direct"

    logger.info(
        f"[request_id={request_id}] route={route} | retrieval_query='{retrieval_query}' | filters={filters}"
    )

    return {
        "route": route,
        "retrieval_query": retrieval_query,
        "filters": filters,
    }


def route_query(state: AgentState):
    request_id = state.get("request_id", "-")
    route = state.get("route", "direct")
    logger.info(f"[request_id={request_id}] Routing to={route}")

    if route == "retrieve":
        return "retrieve"
    if route == "direct":
        return "direct_answer"
    return "fallback"