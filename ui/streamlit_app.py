import os
import uuid
from datetime import datetime

import requests
import streamlit as st

from api_client import RAGApiClient


st.set_page_config(
    page_title="Enterprise Copilot",
    page_icon="💬",
    layout="wide",
)

DEFAULT_API_URL = "http://127.0.0.1:8000"
APP_ENV = os.getenv("APP_ENV", "dev").strip().lower()
SHOW_DEBUG_DEFAULT = os.getenv("SHOW_DEBUG", "true").strip().lower() in {
    "1", "true", "yes", "y", "on"
}

ACTION_OPTIONS = {
    "Auto": None,
    "Q&A": "qa",
    "Summarize Document": "summarize_document",
    "Compare Documents": "compare_documents",
    "Answer by Source": "answer_by_source",
}


def new_session_id() -> str:
    return str(uuid.uuid4())


def init_state():
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = new_session_id()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sessions" not in st.session_state:
        st.session_state.sessions = []

    if "selected_action_label" not in st.session_state:
        st.session_state.selected_action_label = "Auto"

    if "filters" not in st.session_state:
        st.session_state.filters = {
            "doc_type": "",
            "department": "",
            "source": "",
        }

    if "show_debug" not in st.session_state:
        st.session_state.show_debug = SHOW_DEBUG_DEFAULT

    if "chat_search" not in st.session_state:
        st.session_state.chat_search = ""


def build_grounding_from_history(item: dict) -> dict | None:
    role = item.get("role")
    if role != "assistant":
        return None

    retrieval_decision = item.get("retrieval_decision")
    sources = item.get("sources", []) or []

    if retrieval_decision == "grounded":
        return {
            "label": "Grounded",
            "icon": "🟢",
            "help": "Answer is based on retrieved source content.",
        }

    if sources:
        return {
            "label": "Partially grounded",
            "icon": "🟡",
            "help": "Some source evidence was retrieved, but grounding was not confirmed strongly.",
        }

    return {
        "label": "No source match",
        "icon": "🔴",
        "help": "No strong source-backed retrieval was found for this answer.",
    }


def build_retrieval_summary_from_history(item: dict) -> str | None:
    role = item.get("role")
    if role != "assistant":
        return None

    sources = item.get("sources", []) or []
    unique_sources = list(dict.fromkeys([s for s in sources if s]))
    top_score = item.get("top_score")

    if unique_sources and top_score is not None:
        return (
            f"Based on {len(sources)} retrieved chunks from "
            f"{len(unique_sources)} source(s). Best score: {float(top_score):.3f}"
        )

    if unique_sources:
        return (
            f"Based on {len(sources)} retrieved chunks from "
            f"{len(unique_sources)} source(s)."
        )

    return "No retrieved source evidence was attached to this answer."


def build_source_cards_from_history(item: dict) -> list[dict]:
    sources = item.get("sources", []) or []
    if not sources:
        return []

    return [{"source": s, "score": None} for s in sources if s]


def normalize_history_item(item: dict) -> dict:
    return {
        "role": item.get("role", "assistant"),
        "content": item.get("content", ""),
        "debug": None,
        "sources": build_source_cards_from_history(item),
        "grounding": build_grounding_from_history(item),
        "retrieval_summary": build_retrieval_summary_from_history(item),
    }


def load_session_history(client: RAGApiClient, session_id: str):
    result = client.get_history(session_id)
    history = result.get("history", [])

    st.session_state.current_session_id = session_id
    st.session_state.messages = [normalize_history_item(item) for item in history]


def refresh_sessions(client: RAGApiClient):
    result = client.list_sessions()
    st.session_state.sessions = result.get("sessions", [])


def start_new_chat():
    st.session_state.current_session_id = new_session_id()
    st.session_state.messages = []


def clear_ui_only():
    st.session_state.messages = []


def delete_chat(client: RAGApiClient, session_id: str):
    client.delete_session(session_id)

    if session_id == st.session_state.current_session_id:
        start_new_chat()

    refresh_sessions(client)


def format_session_label(session: dict) -> str:
    title = (session.get("title") or "").strip()
    if title:
        return title

    session_id = session.get("session_id", "")
    return f"Chat {session_id[:8]}" if session_id else "Untitled chat"


def format_session_subtitle(session: dict) -> str:
    updated_at = session.get("updated_at") or session.get("created_at")
    if not updated_at:
        return ""

    try:
        dt = datetime.fromisoformat(str(updated_at).replace("Z", ""))
        return dt.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return str(updated_at)


def filter_sessions(sessions: list[dict], query: str) -> list[dict]:
    query = query.strip().lower()
    if not query:
        return sessions

    filtered = []
    for session in sessions:
        title = (session.get("title") or "").lower()
        session_id = (session.get("session_id") or "").lower()
        if query in title or query in session_id:
            filtered.append(session)
    return filtered


def build_filters() -> dict:
    filters = {}

    if st.session_state.filters["doc_type"].strip():
        filters["doc_type"] = st.session_state.filters["doc_type"].strip()

    if st.session_state.filters["department"].strip():
        filters["department"] = st.session_state.filters["department"].strip()

    if st.session_state.filters["source"].strip():
        filters["source"] = st.session_state.filters["source"].strip()

    selected_action_value = ACTION_OPTIONS[st.session_state.selected_action_label]
    if selected_action_value:
        filters["action"] = selected_action_value

    return filters


def render_sidebar_sessions(client: RAGApiClient):
    st.markdown("### Chats")

    st.session_state.chat_search = st.text_input(
        "Search chats",
        value=st.session_state.chat_search,
        placeholder="Search by title...",
    )

    if not st.session_state.sessions:
        try:
            refresh_sessions(client)
        except Exception:
            st.caption("Unable to load chats.")
            return

    sessions_to_show = filter_sessions(
        st.session_state.sessions,
        st.session_state.chat_search,
    )

    if not sessions_to_show:
        st.caption("No chats found.")
        return

    for session in sessions_to_show:
        session_id = session.get("session_id")
        label = format_session_label(session)
        subtitle = format_session_subtitle(session)
        is_active = session_id == st.session_state.current_session_id

        container = st.container()
        with container:
            col1, col2 = st.columns([6, 1])

            with col1:
                button_label = f"● {label}" if is_active else label
                if st.button(
                    button_label,
                    key=f"session_{session_id}",
                    use_container_width=True,
                ):
                    try:
                        load_session_history(client, session_id)
                    except Exception as exc:
                        st.error(f"Failed to load chat: {exc}")

            with col2:
                if st.button("🗑️", key=f"delete_{session_id}", use_container_width=True):
                    try:
                        delete_chat(client, session_id)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

            if subtitle:
                st.caption(subtitle)


def build_grounding_payload(result: dict) -> dict:
    retrieval_decision = result.get("retrieval_decision")
    sources = result.get("retrieved_sources") or []

    if retrieval_decision == "grounded":
        return {
            "label": "Grounded",
            "icon": "🟢",
            "help": "Answer is based on retrieved source content.",
        }

    if sources:
        return {
            "label": "Partially grounded",
            "icon": "🟡",
            "help": "Some source evidence was retrieved, but grounding was not confirmed strongly.",
        }

    return {
        "label": "No source match",
        "icon": "🔴",
        "help": "No strong source-backed retrieval was found for this answer.",
    }


def build_retrieval_summary(result: dict) -> str:
    sources = result.get("retrieved_sources") or []
    unique_sources = list(dict.fromkeys([s for s in sources if s]))
    chunk_count = len(sources)
    top_score = result.get("top_score")

    if unique_sources and top_score is not None:
        return (
            f"Based on {chunk_count} retrieved chunks from "
            f"{len(unique_sources)} source(s). Best score: {top_score:.3f}"
        )

    if unique_sources:
        return (
            f"Based on {chunk_count} retrieved chunks from "
            f"{len(unique_sources)} source(s)."
        )

    return "No retrieved source evidence was attached to this answer."


def build_source_cards(result: dict) -> list[dict]:
    sources = result.get("retrieved_sources") or []
    scores = result.get("retrieval_scores") or []

    cards = []
    for idx, source in enumerate(sources):
        score = scores[idx] if idx < len(scores) else None
        cards.append({
            "source": source or "Unknown source",
            "score": score,
        })

    return cards


def render_grounding_block(grounding: dict | None, retrieval_summary: str | None):
    if not grounding:
        return

    st.info(f"{grounding['icon']} **{grounding['label']}** — {grounding['help']}")
    if retrieval_summary:
        st.caption(retrieval_summary)


def render_sources_block(source_cards: list[dict]):
    if not source_cards:
        return

    st.markdown("**Sources**")
    best_scores = {}

    for card in source_cards:
        source = card["source"]
        score = card["score"]

        if source not in best_scores:
            best_scores[source] = score
        else:
            existing = best_scores[source]
            if score is not None and (existing is None or score < existing):
                best_scores[source] = score

    for source, score in best_scores.items():
        if score is not None:
            st.markdown(f"- 📄 `{source}` · best score `{score:.3f}`")
        else:
            st.markdown(f"- 📄 `{source}`")


init_state()

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1250px;
    }

    div[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    div[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    .app-shell-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-shell-subtitle {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.2rem;
        margin-bottom: 0.25rem;
    }

    .main-subtitle {
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .empty-state {
        border: 1px dashed #d1d5db;
        border-radius: 12px;
        padding: 1.25rem;
        background: #fafafa;
        margin-top: 1rem;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

client = RAGApiClient(st.session_state.api_url)

with st.sidebar:
    st.markdown('<div class="app-shell-title">Enterprise Copilot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-shell-subtitle">Internal enterprise document assistant</div>',
        unsafe_allow_html=True,
    )

    st.text_input(
        "FastAPI Base URL",
        key="api_url",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("New Chat", use_container_width=True):
            start_new_chat()
            st.rerun()

    with col2:
        if st.button("Refresh", use_container_width=True):
            try:
                refresh_sessions(client)
                st.success("Chat list refreshed")
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")

    st.divider()

    render_sidebar_sessions(client)

    st.divider()

    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown("#### Action")
        st.session_state.selected_action_label = st.selectbox(
            "Mode",
            options=list(ACTION_OPTIONS.keys()),
            index=list(ACTION_OPTIONS.keys()).index(st.session_state.selected_action_label),
            label_visibility="collapsed",
        )

        st.markdown("#### Filters")
        st.session_state.filters["doc_type"] = st.text_input(
            "Document type",
            value=st.session_state.filters["doc_type"],
            placeholder="e.g. policy",
        )
        st.session_state.filters["department"] = st.text_input(
            "Department",
            value=st.session_state.filters["department"],
            placeholder="e.g. HR",
        )
        st.session_state.filters["source"] = st.text_input(
            "Source",
            value=st.session_state.filters["source"],
            placeholder="e.g. hr_policy.txt",
        )

        if APP_ENV != "prod":
            st.session_state.show_debug = st.toggle(
                "Show debug details",
                value=st.session_state.show_debug,
            )

        if st.button("Clear current window", use_container_width=True):
            clear_ui_only()
            st.rerun()

st.markdown('<div class="main-title">Enterprise Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Ask questions across your indexed enterprise documents.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size:1.05rem; font-weight:600; margin-bottom:0.4rem;">
                Start a new conversation
            </div>
            <div class="small-note">
                Try asking things like:
                <br>• Summarize the hybrid work policy
                <br>• Compare the leave policy and finance policy
                <br>• Tell me about the AI Contract Analyzer product
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

filters = build_filters()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_grounding_block(
                message.get("grounding"),
                message.get("retrieval_summary"),
            )
            render_sources_block(message.get("sources", []))

            if message.get("debug") and st.session_state.show_debug and APP_ENV != "prod":
                with st.expander("Debug details"):
                    st.json(message["debug"])

user_prompt = st.chat_input("Ask a question about your documents...")

if user_prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt,
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = client.chat(
                    query=user_prompt,
                    session_id=st.session_state.current_session_id,
                    filters=filters,
                )

                answer = result.get("response", "No response returned.")
                debug_payload = {
                    "request_id": result.get("request_id"),
                    "route": result.get("route"),
                    "action": result.get("action"),
                    "target_sources": result.get("target_sources"),
                    "retrieval_query": result.get("retrieval_query"),
                    "rewritten_query": result.get("rewritten_query"),
                    "applied_filters": result.get("applied_filters"),
                    "retrieval_decision": result.get("retrieval_decision"),
                    "retrieved_sources": result.get("retrieved_sources"),
                    "retrieval_scores": result.get("retrieval_scores"),
                    "top_score": result.get("top_score"),
                    "history_length": result.get("history_length"),
                    "session_context": result.get("session_context"),
                }

                grounding_payload = build_grounding_payload(result)
                retrieval_summary = build_retrieval_summary(result)
                source_cards = build_source_cards(result)

                st.markdown(answer)
                render_grounding_block(grounding_payload, retrieval_summary)
                render_sources_block(source_cards)

                if st.session_state.show_debug and APP_ENV != "prod":
                    with st.expander("Debug details"):
                        st.json(debug_payload)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "debug": debug_payload,
                    "sources": source_cards,
                    "grounding": grounding_payload,
                    "retrieval_summary": retrieval_summary,
                })

                try:
                    refresh_sessions(client)
                except Exception:
                    pass

            except requests.exceptions.ConnectionError:
                error_text = (
                    "Could not connect to the FastAPI backend. "
                    "Please check that the API is running and the URL is correct."
                )
                st.error(error_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_text,
                    "debug": None,
                    "sources": [],
                    "grounding": {
                        "label": "No source match",
                        "icon": "🔴",
                        "help": "Backend connection failed, so no source-backed answer could be produced.",
                    },
                    "retrieval_summary": None,
                })

            except Exception as exc:
                error_text = f"Error: {str(exc)}"
                st.error(error_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_text,
                    "debug": None,
                    "sources": [],
                    "grounding": {
                        "label": "No source match",
                        "icon": "🔴",
                        "help": "An internal error occurred before source-backed output could be shown.",
                    },
                    "retrieval_summary": None,
                })