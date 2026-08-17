"""
FeedLab — Database Schema & Helpers

Creates (or opens) `reels.db` and ensures the `accounts` and `reels`
tables exist.
"""

import sqlite3
import os
import asyncio

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reels.db")


def get_connection() -> sqlite3.Connection:
    """Return a connection to the FeedLab SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Create the accounts and reels tables if they do not already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            username            TEXT PRIMARY KEY,
            password            TEXT NOT NULL,
            status              TEXT NOT NULL
                                    CHECK (status IN (
                                        'WARMING_UP', 'ACTIVE',
                                        'COOLDOWN', 'BANNED',
                                        'NEEDS_VERIFICATION'
                                    )),
            daily_scroll_minutes INTEGER DEFAULT 0,
            last_used_timestamp  TEXT
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reels (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP,
            reel_url            TEXT,
            caption             TEXT,
            transcript          TEXT,
            alignment_score     REAL,
            like_count          INTEGER,
            comment_count       INTEGER,
            action_taken        TEXT
        );
        """
    )

    conn.commit()
    conn.close()
    print("✅  Database initialised — reels.db is ready.")


def get_next_available_account() -> dict | None:
    """Return the next ACTIVE account, or None if no account is available."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT username, password, status
        FROM accounts
        WHERE status = 'ACTIVE'
        ORDER BY last_used_timestamp ASC NULLS FIRST
        LIMIT 1
        """
    )
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return None

    username, password, status = row

    cursor.execute(
        "UPDATE accounts SET last_used_timestamp = datetime('now') WHERE username = ?",
        (username,),
    )
    conn.commit()
    conn.close()

    return {"username": username, "password": password, "status": status}


def log_reel_event(
    reel_url: str,
    caption: str,
    transcript: str,
    alignment_score: float,
    like_count: int,
    comment_count: int,
    action_taken: str,
) -> None:
    """Insert a reel evaluation event into the reels table synchronously."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO reels (reel_url, caption, transcript,
                           alignment_score, like_count,
                           comment_count, action_taken)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            reel_url, caption, transcript,
            alignment_score, like_count, comment_count,
            action_taken,
        ),
    )
    conn.commit()
    conn.close()


async def log_reel_event_async(
    reel_url: str,
    caption: str,
    transcript: str,
    alignment_score: float,
    like_count: int,
    comment_count: int,
    action_taken: str,
) -> None:
    """Asynchronously log reel evaluation event in background thread."""
    await asyncio.to_thread(
        log_reel_event,
        reel_url,
        caption,
        transcript,
        alignment_score,
        like_count,
        comment_count,
        action_taken,
    )


def update_account_status(username: str, new_status: str) -> None:
    """Update the status of an account (WARMING_UP, ACTIVE, COOLDOWN, BANNED)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE accounts SET status = ? WHERE username = ?",
        (new_status, username),
    )
    conn.commit()
    conn.close()


def reset_cooldown_accounts() -> int:
    """Reset all COOLDOWN accounts back to ACTIVE."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE accounts SET status = 'ACTIVE' WHERE status = 'COOLDOWN'"
    )
    count = cursor.rowcount
    conn.commit()
    conn.close()
    if count > 0:
        print(f"  🔄  Reset {count} account(s) from COOLDOWN → ACTIVE.")
    return count


if __name__ == "__main__":
    init_db()
