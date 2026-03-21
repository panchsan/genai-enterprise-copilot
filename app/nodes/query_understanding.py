import json

from app.config import settings
from app.prompts import QUERY_UNDERSTANDING_PROMPT
from app.services.llm import get_azure_openai_client, safe_chat_completion
from app.services.logging_utils import get_logger, log_timing
from app.state import AgentState

client = get_azure_openai_client()
logger = get_logger("app.query_understanding")


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
            "action": "qa",
            "retrieval_query": query,
            "filters": state.get("filters", {}) or {},
            "target_sources": [],
        }

    messages = [{"role": "system", "content": QUERY_UNDERSTANDING_PROMPT}]

    if chat_history:
        messages.append(
            {
                "role": "system",
                "content": f"Recent chat history: {json.dumps(chat_history[-settings.MAX_CHAT_HISTORY_MESSAGES:])}",
            }
        )

    messages.append({"role": "user", "content": query})

    try:
        with log_timing(logger, "query_understanding_llm", request_id):
            response = safe_chat_completion(
                client,
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE_DETERMINISTIC,
            )

        raw_output = response.choices[0].message.content or "{}"
        logger.info(f"[request_id={request_id}] Raw model output={raw_output}")

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning(f"[request_id={request_id}] Failed to parse model output; using safe defaults")
            parsed = {
                "route": "direct",
                "action": "qa",
                "retrieval_query": query,
                "filters": {},
                "target_sources": [],
            }
    except Exception as exc:
        logger.error(f"[request_id={request_id}] Query understanding failed: {exc}")
        parsed = {
            "route": "direct",
            "action": "qa",
            "retrieval_query": query,
            "filters": {},
            "target_sources": [],
        }

    route = parsed.get("route", "direct")
    action = parsed.get("action", "qa")
    retrieval_query = parsed.get("retrieval_query", query)
    filters = parsed.get("filters", {}) or {}
    target_sources = parsed.get("target_sources", []) or []

    if route not in {"retrieve", "direct", "fallback"}:
        route = "direct"

    if action not in {"qa", "summarize_document", "answer_by_source", "compare_documents"}:
        action = "qa"

    logger.info(
        f"[request_id={request_id}] route={route} | action={action} | "
        f"retrieval_query='{retrieval_query}' | filters={filters} | target_sources={target_sources}"
    )

    return {
        "route": route,
        "action": action,
        "retrieval_query": retrieval_query,
        "filters": filters,
        "target_sources": target_sources,
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