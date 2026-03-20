import sqlite3
from typing import List, Dict


DB_PATH = "chat_memory.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
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

    conn.commit()
    conn.close()

    print("✅ Database initialized")


def create_session(session_id: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO sessions (session_id)
    VALUES (?)
    """, (session_id,))

    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO messages (session_id, role, content)
    VALUES (?, ?, ?)
    """, (session_id, role, content))

    conn.commit()
    conn.close()


def get_chat_history(session_id: str) -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT role, content
    FROM messages
    WHERE session_id = ?
    ORDER BY id ASC
    """, (session_id,))

    rows = cursor.fetchall()
    conn.close()

    return [{"role": r[0], "content": r[1]} for r in rows]