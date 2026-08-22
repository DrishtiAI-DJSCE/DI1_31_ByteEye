"""
database.py
===========
All SQLite interaction for DRISHTI MVP.

Architecture rule: event-level storage ONLY.
- We store ONE row per confirmed anomaly event.
- We do NOT store every frame.
- We do NOT store permanent student identity.
- snapshot_path points to the evidence image saved by evidence.py.

Status lifecycle: NEW → REVIEWED | DISMISSED
"""

import os
import sqlite3
import config


def _get_conn() -> sqlite3.Connection:
    """Open the database, creating the parent directory if needed."""
    db_dir = os.path.dirname(config.DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn


def init_db() -> None:
    """
    Create the events table if it does not exist.
    Safe to call every time the app starts — CREATE TABLE IF NOT EXISTS
    means it is a no-op when the table already exists.
    """
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     TEXT,
            timestamp      REAL    NOT NULL,
            event_type     TEXT    NOT NULL,
            severity       TEXT    NOT NULL,
            confidence     REAL    DEFAULT 0.0,
            snapshot_path  TEXT    DEFAULT '',
            status         TEXT    NOT NULL DEFAULT 'NEW',
            cooldown_group TEXT
        )
    """)
    try:
        conn.execute("ALTER TABLE events ADD COLUMN session_id TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def insert_event(event: dict, session_id: str = None) -> int:
    """
    Insert one confirmed anomaly event.

    `event` must contain:
        timestamp, event_type, severity
    `event` may contain:
        confidence, snapshot_path, bbox (ignored at DB level)
    """
    if not session_id:
        return -1
        
    conn = _get_conn()
    cur = conn.execute(
        """
        INSERT INTO events
            (session_id, timestamp, event_type, severity, confidence, snapshot_path,
             status, cooldown_group)
        VALUES (?, ?, ?, ?, ?, ?, 'NEW', ?)
        """,
        (
            session_id,
            event["timestamp"],
            event["event_type"],
            event.get("severity", "LOW"),
            event.get("confidence", 0.0),
            event.get("snapshot_path", ""),
            event["event_type"],   # cooldown_group same as event_type
        ),
    )
    event_id = cur.lastrowid
    conn.commit()
    conn.close()
    return event_id


def get_events(status: str = None, severity: str = None, session_id: str = None) -> list[dict]:
    """
    Return all events, optionally filtered, newest first.
    Pass status="NEW" or severity="HIGH" to filter.
    "All" is treated the same as None (no filter).
    """
    conn = _get_conn()
    query = "SELECT * FROM events WHERE 1=1"
    params: list = []

    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)

    if status and status != "All":
        query += " AND status = ?"
        params.append(status)
    if severity and severity != "All":
        query += " AND severity = ?"
        params.append(severity)

    query += " ORDER BY timestamp DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_event_status(event_id: int, new_status: str) -> None:
    """Change an event's status. new_status must be REVIEWED or DISMISSED."""
    if new_status not in ("NEW", "REVIEWED", "DISMISSED"):
        raise ValueError(f"Invalid status: {new_status}")
    conn = _get_conn()
    conn.execute(
        "UPDATE events SET status = ? WHERE event_id = ?",
        (new_status, event_id),
    )
    conn.commit()
    conn.close()

def delete_session_events(session_id: str) -> None:
    """Delete all events associated with a specific session_id."""
    if not session_id:
        return
    conn = _get_conn()
    conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
