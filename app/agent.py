from langgraph.graph import END, StateGraph

from app.config import settings
from app.nodes.direct_answer import direct_answer
from app.nodes.fallback import fallback
from app.nodes.generate import generate
from app.nodes.query_understanding import analyze_query, route_query
from app.nodes.retrieve import retrieve
from app.nodes.rewrite_query import rewrite_query
from app.nodes.safe_fallback import safe_fallback
from app.nodes.session_context import apply_session_context
from app.nodes.validate_retrieval import validate_retrieval
from app.state import AgentState


def build_graph(vectordb):
    graph = StateGraph(AgentState)

    def retrieve_node(state: AgentState):
        return retrieve(state, vectordb)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("apply_session_context", apply_session_context)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("validate_retrieval", validate_retrieval)
    graph.add_node("generate", generate)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("safe_fallback", safe_fallback)
    graph.add_node("fallback", fallback)

    graph.set_entry_point("analyze_query")

    graph.add_conditional_edges(
        "analyze_query",
        route_query,
        {
            "retrieve": "apply_session_context",
            "direct_answer": "direct_answer",
            "fallback": "fallback",
        },
    )

    graph.add_edge("apply_session_context", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "validate_retrieval")

    def route_after_validation(state: AgentState):
        decision = state.get("retrieval_decision", "ungrounded")
        print("🔀 [POST-VALIDATION ROUTER] Decision:", decision)

        if decision == "grounded":
            return "generate"

        if getattr(settings, "ALLOW_DIRECT_LLM_FALLBACK", False):
            return "direct_answer"

        return "safe_fallback"

    graph.add_conditional_edges(
        "validate_retrieval",
        route_after_validation,
        {
            "generate": "generate",
            "direct_answer": "direct_answer",
            "safe_fallback": "safe_fallback",
        },
    )

    graph.add_edge("generate", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("safe_fallback", END)
    graph.add_edge("fallback", END)

    return graph.compile()