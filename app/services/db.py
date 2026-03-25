import os
import json
import sqlite3
from typing import List, Dict, Any, Optional

from app.services.logging_utils import get_logger

logger = get_logger("app.db")

DB_PATH = os.getenv("DB_PATH", "chat_memory.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            grounding TEXT,
            sources TEXT,
            retrieval_decision TEXT,
            top_score REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_context (
            session_id TEXT PRIMARY KEY,
            active_filters TEXT,
            active_source TEXT,
            last_route TEXT,
            last_retrieval_query TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        _ensure_schema_updates(cursor, conn)
        logger.info("Database initialized")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_schema_updates(cursor, conn):
    cursor.execute("PRAGMA table_info(sessions)")
    session_columns = [row[1] for row in cursor.fetchall()]
    if "title" not in session_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN title TEXT")

    cursor.execute("PRAGMA table_info(messages)")
    message_columns = [row[1] for row in cursor.fetchall()]

    required_message_columns = {
        "grounding": "TEXT",
        "sources": "TEXT",
        "retrieval_decision": "TEXT",
        "top_score": "REAL",
    }

    for column_name, column_type in required_message_columns.items():
        if column_name not in message_columns:
            cursor.execute(f"ALTER TABLE messages ADD COLUMN {column_name} {column_type}")

    conn.commit()


def create_session(session_id: str, title: Optional[str] = None):
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT OR IGNORE INTO sessions (session_id, title)
        VALUES (?, ?)
        """, (session_id, title))

        cursor.execute("""
        INSERT OR IGNORE INTO session_context (
            session_id, active_filters, active_source, last_route, last_retrieval_query
        )
        VALUES (?, ?, ?, ?, ?)
        """, (session_id, "{}", None, None, None))

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_message(
    session_id: str,
    role: str,
    content: str,
    grounding: Optional[str] = None,
    sources: Optional[List[str]] = None,
    retrieval_decision: Optional[str] = None,
    top_score: Optional[float] = None,
):
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO messages (
            session_id, role, content, grounding, sources, retrieval_decision, top_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            role,
            content,
            grounding,
            json.dumps(sources or []),
            retrieval_decision,
            top_score,
        ))

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT role, content, grounding, sources, retrieval_decision, top_score
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """, (session_id,))

        rows = cursor.fetchall()

        history = []
        for row in rows:
            try:
                parsed_sources = json.loads(row[3]) if row[3] else []
            except json.JSONDecodeError:
                parsed_sources = []

            history.append({
                "role": row[0],
                "content": row[1],
                "grounding": row[2],
                "sources": parsed_sources,
                "retrieval_decision": row[4],
                "top_score": row[5],
            })

        return history
    finally:
        conn.close()


def get_session_context(session_id: str) -> Dict[str, Any]:
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT active_filters, active_source, last_route, last_retrieval_query
        FROM session_context
        WHERE session_id = ?
        """, (session_id,))

        row = cursor.fetchone()
        if not row:
            return {
                "active_filters": {},
                "active_source": None,
                "last_route": None,
                "last_retrieval_query": None,
            }

        active_filters_raw, active_source, last_route, last_retrieval_query = row

        try:
            active_filters = json.loads(active_filters_raw) if active_filters_raw else {}
        except json.JSONDecodeError:
            active_filters = {}

        return {
            "active_filters": active_filters,
            "active_source": active_source,
            "last_route": last_route,
            "last_retrieval_query": last_retrieval_query,
        }
    finally:
        conn.close()


def update_session_context(
    session_id: str,
    active_filters: Optional[Dict[str, Any]] = None,
    active_source: Optional[str] = None,
    last_route: Optional[str] = None,
    last_retrieval_query: Optional[str] = None,
):
    existing = get_session_context(session_id)

    final_filters = existing.get("active_filters", {})
    if active_filters is not None:
        final_filters = active_filters

    final_source = existing.get("active_source")
    if active_source is not None:
        final_source = active_source

    final_last_route = existing.get("last_route")
    if last_route is not None:
        final_last_route = last_route

    final_last_retrieval_query = existing.get("last_retrieval_query")
    if last_retrieval_query is not None:
        final_last_retrieval_query = last_retrieval_query

    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO session_context (
            session_id, active_filters, active_source, last_route, last_retrieval_query, updated_at
        )
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(session_id) DO UPDATE SET
            active_filters = excluded.active_filters,
            active_source = excluded.active_source,
            last_route = excluded.last_route,
            last_retrieval_query = excluded.last_retrieval_query,
            updated_at = CURRENT_TIMESTAMP
        """, (
            session_id,
            json.dumps(final_filters or {}),
            final_source,
            final_last_route,
            final_last_retrieval_query,
        ))

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_session_title(session_id: str, title: str):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE sessions
        SET title = ?
        WHERE session_id = ?
        """, (title, session_id))
        conn.commit()
    finally:
        conn.close()


def get_session_title(session_id: str) -> Optional[str]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT title
        FROM sessions
        WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_all_sessions() -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT
            s.session_id,
            COALESCE(NULLIF(s.title, ''), 'New Chat') AS title,
            s.created_at,
            sc.updated_at
        FROM sessions s
        LEFT JOIN session_context sc
            ON s.session_id = sc.session_id
        ORDER BY COALESCE(sc.updated_at, s.created_at) DESC
        """)

        rows = cursor.fetchall()

        return [
            {
                "session_id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]
    finally:
        conn.close()


def delete_session(session_id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
            (session_id,),
        )
        exists = cursor.fetchone() is not None

        if not exists:
            return False

        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM session_context WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()