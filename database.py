"""
Database module for the Incede contact bot.
Handles SQLite3 database initialization, table creation, and all CRUD operations
for sessions, conversations, errors, and contact details.
"""

import sqlite3
import uuid
import json
from datetime import datetime
from contextlib import contextmanager

# Path to the SQLite database file
DATABASE_PATH = "incede_bot.db"


@contextmanager
def get_db_connection():
    """
    Context manager that provides a database connection with row_factory set.
    Ensures the connection is properly closed after use and handles exceptions.

    Yields:
        sqlite3.Connection: An active database connection with row factory enabled.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """
    Initialize the database by creating all required tables if they do not exist.
    Creates four tables: sessions, conversations, errors, and contact_details.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Table for tracking bot sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                contact_collected INTEGER NOT NULL DEFAULT 0
            )
        """)

        # Table for storing each message in a conversation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Table for storing errors that occur during bot execution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                traceback TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Table for storing collected contact details
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contact_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                description TEXT,
                collected_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        conn.commit()


# --------------------------------------------------------------------------
# Session operations
# --------------------------------------------------------------------------

def create_session() -> str:
    """
    Create a new bot session and persist it in the database.

    Returns:
        str: The unique session ID for the newly created session.
    """
    session_id = str(uuid.uuid4())
    started_at = datetime.utcnow().isoformat()

    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (id, started_at, status) VALUES (?, ?, ?)",
            (session_id, started_at, "active")
        )

    return session_id


def end_session(session_id: str, contact_collected: bool = False):
    """
    Mark a session as ended and update its status and end timestamp.

    Args:
        session_id: The unique identifier for the session to end.
        contact_collected: Whether contact details were successfully collected in this session.
    """
    ended_at = datetime.utcnow().isoformat()
    status = "completed" if contact_collected else "abandoned"

    with get_db_connection() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ?, status = ?, contact_collected = ? WHERE id = ?",
            (ended_at, status, int(contact_collected), session_id)
        )


def get_session(session_id: str) -> dict | None:
    """
    Retrieve a session record by its ID.

    Args:
        session_id: The unique identifier of the session to retrieve.

    Returns:
        A dictionary with session fields or None if not found.
    """
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def get_paginated_sessions(page: int = 1, per_page: int = 10) -> dict:
    """
    Retrieve a paginated list of sessions ordered by start time descending.

    Args:
        page: The page number to retrieve (1-indexed).
        per_page: The number of records per page.

    Returns:
        A dictionary with keys: items (list of dicts), total, page, per_page, total_pages.
    """
    offset = (page - 1) * per_page

    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (per_page, offset)
        ).fetchall()

    items = [dict(row) for row in rows]
    total_pages = (total + per_page - 1) // per_page

    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    }


# --------------------------------------------------------------------------
# Conversation operations
# --------------------------------------------------------------------------

def save_message(session_id: str, role: str, message: str):
    """
    Save a single message to the conversations table.

    Args:
        session_id: The session this message belongs to.
        role: The speaker role, either 'user' or 'assistant'.
        message: The text content of the message.
    """
    timestamp = datetime.utcnow().isoformat()

    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, message, timestamp)
        )


def get_conversation(session_id: str) -> list[dict]:
    """
    Retrieve all messages for a given session in chronological order.

    Args:
        session_id: The session ID whose conversation to retrieve.

    Returns:
        A list of message dictionaries with role, message, and timestamp keys.
    """
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT role, message, timestamp FROM conversations WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        ).fetchall()

    return [dict(row) for row in rows]


# --------------------------------------------------------------------------
# Error operations
# --------------------------------------------------------------------------

def log_error(session_id: str, error_type: str, error_message: str, traceback: str = None):
    """
    Log an error that occurred during bot processing to the errors table.

    Args:
        session_id: The session during which the error occurred.
        error_type: The class name or category of the error (e.g., 'ValueError', 'LLMError').
        error_message: A human-readable description of the error.
        traceback: Optional full traceback string for debugging.
    """
    timestamp = datetime.utcnow().isoformat()

    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO errors (session_id, error_type, error_message, traceback, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, error_type, error_message, traceback, timestamp)
        )


def get_errors_for_session(session_id: str) -> list[dict]:
    """
    Retrieve all errors logged for a specific session.

    Args:
        session_id: The session ID to look up errors for.

    Returns:
        A list of error dictionaries, or an empty list if none exist.
    """
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT error_type, error_message, traceback, timestamp FROM errors WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        ).fetchall()

    return [dict(row) for row in rows]


# --------------------------------------------------------------------------
# Contact detail operations
# --------------------------------------------------------------------------

def save_contact_detail(session_id: str, name: str, email: str, phone: str, description: str = None):
    """
    Save the collected contact details for a completed session.

    Args:
        session_id: The session ID associated with these contact details.
        name: The full name of the contact.
        email: The email address of the contact.
        phone: The phone number of the contact.
        description: Optional description or message from the contact.
    """
    collected_at = datetime.utcnow().isoformat()

    with get_db_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO contact_details (session_id, name, email, phone, description, collected_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, name, email, phone, description, collected_at)
        )


def get_contact_detail(session_id: str) -> dict | None:
    """
    Retrieve the contact detail record for a given session.

    Args:
        session_id: The session ID to look up contact details for.

    Returns:
        A dictionary of contact details or None if not found.
    """
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM contact_details WHERE session_id = ?", (session_id,)
        ).fetchone()

    return dict(row) if row else None
