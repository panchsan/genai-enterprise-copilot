from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    request_id: str

    query: str
    retrieval_query: str
    rewritten_query: str
    route: str
    action: str
    context: str
    answer: str

    session_id: str
    chat_history: List[Dict[str, Any]]

    filters: Dict[str, Any]
    session_context: Dict[str, Any]
    target_sources: List[str]

    retrieved_docs: List[Dict[str, Any]]
    retrieved_sources: List[str]
    retrieval_decision: str
    retrieval_status: str

    active_source: Optional[str]
    last_route: Optional[str]
    last_retrieval_query: Optional[str]

    retrieval_scores: List[float]
    top_score: Optional[float]