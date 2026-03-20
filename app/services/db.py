import json
import sqlite3
from typing import List, Dict, Any, Optional

from app.services.logging_utils import get_logger

logger = get_logger("app.db")

DB_PATH = "chat_memory.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
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
        logger.info("Database initialized")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_session(session_id: str):
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT OR IGNORE INTO sessions (session_id)
        VALUES (?)
        """, (session_id,))

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


def save_message(session_id: str, role: str, content: str):
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO messages (session_id, role, content)
        VALUES (?, ?, ?)
        """, (session_id, role, content))

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """, (session_id,))

        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]
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