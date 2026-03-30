import re

from app.services.action_utils import normalize_action
from app.services.logging_utils import get_logger
from app.state import AgentState

logger = get_logger("app.query_understanding")

SOFT_SOURCE_PATTERNS = [
    r"according to (?P<source>.+?)(?:\?|$)",
    r"in (?P<source>.+?)(?:\?|$)",
    r"from (?P<source>.+?)(?:\?|$)",
]

STRICT_SOURCE_PATTERNS = [
    r"(?:using|based on|from) only (?P<source>[\w\-. &()]+\.(?:txt|pdf|csv|docx))",
    r"(?:using|based on|from) (?P<source>[\w\-. &()]+\.(?:txt|pdf|csv|docx))",
]

COMPARE_PATTERNS = [
    r"compare (?P<left>.+?) (?:and|with|to) (?P<right>.+)$",
    r"difference between (?P<left>.+?) and (?P<right>.+)$",
    r"how does (?P<left>.+?) differ from (?P<right>.+)$",
]

DIRECT_ANSWER_PATTERNS = [
    r"\bwrite\b.*\bpython\b",
    r"\bpython program\b",
    r"\bpython code\b",
    r"\bcode\b",
    r"\bexample code\b",
    r"\bsample code\b",
    r"\bprogram to\b",
    r"\bscript to\b",
    r"\bwhat is\b.*\bprotocol\b",
    r"\bwhat is\b.*\bagent to agent\b",
    r"\bexplain\b",
]


def _clean_source_phrase(value: str) -> str:
    value = (value or "").strip().strip("?.,:;\"'")
    value = re.sub(r"^(the|this|that)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value


def _is_direct_answer_query(query: str) -> bool:
    q = (query or "").lower().strip()

    for pattern in DIRECT_ANSWER_PATTERNS:
        if re.search(pattern, q, flags=re.IGNORECASE):
            return True

    return False


def _detect_action(query: str) -> str:
    q = (query or "").lower().strip()

    if any(word in q for word in ["compare", "difference between", "differ from", "compare documents"]):
        return "compare_documents"

    if "summarize" in q or "summary" in q:
        return "summarize_document"

    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in STRICT_SOURCE_PATTERNS):
        return "answer_by_source"

    return "qa"


def _extract_explicit_filenames(query: str) -> list[str]:
    matches = re.findall(
        r"[\w&()\- ]+\.(?:txt|pdf|csv|docx)",
        query or "",
        flags=re.IGNORECASE,
    )
    cleaned = [_clean_source_phrase(match) for match in matches]
    return list(dict.fromkeys([item for item in cleaned if item]))


def _extract_compare_sources(query: str) -> list[str]:
    for pattern in COMPARE_PATTERNS:
        match = re.search(pattern, query or "", flags=re.IGNORECASE)
        if match:
            left = _clean_source_phrase(match.group("left"))
            right = _clean_source_phrase(match.group("right"))
            return [item for item in [left, right] if item]
    return []


def _extract_source_hints(query: str, action: str) -> list[str]:
    hints = _extract_explicit_filenames(query)

    if action == "compare_documents":
        hints.extend(_extract_compare_sources(query))

    patterns = STRICT_SOURCE_PATTERNS if action == "answer_by_source" else SOFT_SOURCE_PATTERNS

    for pattern in patterns:
        match = re.search(pattern, query or "", flags=re.IGNORECASE)
        if not match:
            continue

        source = _clean_source_phrase(match.group("source"))
        if source:
            hints.append(source)
            break

    return list(dict.fromkeys([hint for hint in hints if hint]))


def analyze_query(state: AgentState):
    query = (state.get("query") or "").strip()
    requested_action = state.get("action")

    logger.info(f"[request_id={state.get('request_id', '-')}] Incoming query={query!r}")

    filters = state.get("filters", {}) or {}

    if requested_action:
        action = normalize_action(requested_action)
        action_source = "requested"
        route = "retrieve"
    else:
        if _is_direct_answer_query(query):
            action = "qa"
            action_source = "detected_direct"
            route = "direct"
        else:
            detected_action = _detect_action(query)
            action = normalize_action(detected_action)
            action_source = "detected"
            route = "retrieve"

    target_sources = _extract_source_hints(query, action)

    if action == "answer_by_source" and not target_sources and not filters.get("source"):
        action = "qa"

    logger.info(
        f"[request_id={state.get('request_id', '-')}] "
        f"route={route} | action={action} (source={action_source}) | "
        f"retrieval_query={query!r} | filters={filters} | target_sources={target_sources}"
    )

    return {
        **state,
        "route": route,
        "action": action,
        "retrieval_query": query,
        "target_sources": target_sources,
        "filters": filters,
    }


def route_query(state: AgentState):
    route = state.get("route", "retrieve")
    logger.info(f"[request_id={state.get('request_id', '-')}] Routing to={route}")

    if route == "retrieve":
        return "retrieve"
    if route == "direct":
        return "direct_answer"
    return "fallback"