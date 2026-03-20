from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    query: str
    retrieval_query: str
    rewritten_query: str
    route: str
    context: str
    answer: str

    session_id: str
    chat_history: List[Dict[str, Any]]

    filters: Dict[str, Any]
    session_context: Dict[str, Any]

    retrieved_docs: List[Dict[str, Any]]
    retrieval_decision: str

    active_source: Optional[str]
    last_route: Optional[str]
    last_retrieval_query: Optional[str]