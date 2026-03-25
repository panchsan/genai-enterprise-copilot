from typing import List, Dict


VALID_ACTIONS = {
    "qa",
    "summarize_document",
    "answer_by_source",
    "compare_documents",
}


ACTION_RETRIEVAL_CONFIG: Dict[str, Dict] = {
    "qa": {
        "top_k": 4,
        "allow_filter_relaxation": True,
        "require_source": False,
        "require_multiple_sources": False,
    },
    "summarize_document": {
        "top_k": 10,
        "allow_filter_relaxation": False,
        "require_source": False,
        "require_multiple_sources": False,
    },
    "answer_by_source": {
        "top_k": 8,
        "allow_filter_relaxation": False,
        "require_source": True,
        "require_multiple_sources": False,
    },
    "compare_documents": {
        "top_k": 12,
        "allow_filter_relaxation": False,
        "require_source": False,
        "require_multiple_sources": True,
    },
}


VALID_RETRIEVAL_STATUS = {
    "found",
    "no_docs",
    "missing_required_source",
    "source_not_found",
    "insufficient_sources",
}


def normalize_action(action: str | None) -> str:
    if action in VALID_ACTIONS:
        return action
    return "qa"


def get_action_config(action: str) -> Dict:
    return ACTION_RETRIEVAL_CONFIG.get(normalize_action(action), ACTION_RETRIEVAL_CONFIG["qa"])


def is_source_required(action: str) -> bool:
    return get_action_config(action)["require_source"]


def requires_multiple_sources(action: str) -> bool:
    return get_action_config(action)["require_multiple_sources"]


def dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result