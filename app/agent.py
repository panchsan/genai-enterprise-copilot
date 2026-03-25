from langgraph.graph import END, StateGraph

from app.nodes.direct_answer import direct_answer
from app.nodes.generate import generate
from app.nodes.query_understanding import analyze_query, route_query
from app.nodes.retrieve import retrieve
from app.nodes.rewrite_query import rewrite_query
from app.nodes.session_context import apply_session_context
from app.nodes.validate_retrieval import validate_retrieval
from app.services.logging_utils import get_logger
from app.services.vectorstore import get_vectorstore
from app.state import AgentState

logger = get_logger("app.agent")


def route_after_validation(state: AgentState):
    decision = state.get("retrieval_decision", "no_docs")
    action = state.get("action", "qa")
    retrieval_status = state.get("retrieval_status", "no_docs")

    logger.info(
        f"[POST-VALIDATION ROUTER] decision={decision} | "
        f"action={action} | retrieval_status={retrieval_status}"
    )

    if decision == "grounded":
        return "generate"

    # For document-oriented actions, do not fall back to broad direct answer.
    # Let generate() return a controlled failure / guidance message.
    if action in {"summarize_document", "answer_by_source", "compare_documents"}:
        return "generate"

    return "direct_answer"


def build_graph():
    vectordb = get_vectorstore()

    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("session_context", apply_session_context)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", lambda state: retrieve(state, vectordb))
    workflow.add_node("validate_retrieval", validate_retrieval)
    workflow.add_node("generate", generate)
    workflow.add_node("direct_answer", direct_answer)

    workflow.set_entry_point("analyze_query")

    workflow.add_conditional_edges(
        "analyze_query",
        route_query,
        {
            "retrieve": "session_context",
            "direct_answer": "direct_answer",
            "fallback": "direct_answer",
        },
    )

    workflow.add_edge("session_context", "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "validate_retrieval")

    workflow.add_conditional_edges(
        "validate_retrieval",
        route_after_validation,
        {
            "generate": "generate",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("direct_answer", END)

    graph = workflow.compile()

    logger.info("LangGraph workflow compiled successfully.")
    return graph