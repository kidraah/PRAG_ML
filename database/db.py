"""
FeedLab — Database Schema & Helpers

Creates (or opens) `reels.db` with:
  accounts     the Instagram accounts the agent may use
  sessions     one row per agent run
  impressions  one row per reel shown — its metadata, what was extracted,
               the LLM's judgement, what the policy did, and how long the
               reel stayed on screen

The legacy `reels` table from earlier versions is left untouched.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reels.db")

IMPRESSION_COLUMNS = (
    "session_id", "seq", "code", "url", "creator", "caption", "hashtags",
    "audio_type", "audio_title", "duration", "like_count", "comment_count",
    "repost_count", "taken_at", "inventory_source", "is_ad",
    "transcript", "transcript_language", "cover_text",
    "score_basis", "value_score", "is_informational", "topic", "summary",
    "reasoning", "error", "gate_keyword", "gate_audit", "score_seconds", "scored_at",
    "decision", "decision_basis", "explored", "dwell_seconds", "planned",
    "liked", "saved", "followed",
)
SCORE_COLUMNS = (
    "transcript", "transcript_language", "cover_text", "score_basis",
    "value_score", "is_informational", "topic", "summary", "reasoning",
    "error", "gate_keyword", "gate_audit", "score_seconds",
)


def get_connection(path: str = DB_PATH) -> sqlite3.Connection:
    """Return a connection to the FeedLab SQLite database."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> None:
    """Create the accounts, sessions and impressions tables if they do not exist."""
    own = conn is None
    conn = conn or get_connection()
    conn.executescript(
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

        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            account     TEXT,
            started_at  TEXT DEFAULT (datetime('now')),
            ended_at    TEXT,
            dry_run     INTEGER NOT NULL DEFAULT 0,
            model       TEXT,
            scorer      TEXT,       -- INTERESTS + model fingerprint; scores compare only within one
            reels       INTEGER
        );

        CREATE TABLE IF NOT EXISTS impressions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          INTEGER NOT NULL REFERENCES sessions(id),
            seq                 INTEGER NOT NULL,        -- position within the session
            shown_at            TEXT DEFAULT (datetime('now')),
            code                TEXT,
            url                 TEXT,
            creator             TEXT,
            caption             TEXT,
            hashtags            TEXT,
            audio_type          TEXT,                    -- original | music
            audio_title         TEXT,
            duration            REAL,
            like_count          INTEGER,
            comment_count       INTEGER,
            repost_count        INTEGER,
            taken_at            INTEGER,
            inventory_source    TEXT,
            is_ad               INTEGER NOT NULL DEFAULT 0,
            transcript          TEXT,
            transcript_language TEXT,
            cover_text          TEXT,
            score_basis         TEXT,     -- score | keyword_gate | creator_prior | llm_error | ad; NULL = never scored
            value_score         REAL,
            is_informational    INTEGER,
            topic               TEXT,
            summary             TEXT,
            reasoning           TEXT,
            error               TEXT,
            gate_keyword        TEXT,
            gate_audit          INTEGER NOT NULL DEFAULT 0,   -- gated, but scored anyway to audit the gate
            score_seconds       REAL,     -- time spent scoring (download + Whisper/OCR + LLM)
            scored_at           TEXT,
            decision            TEXT,     -- SKIP | WATCH | FULL_WATCH | REWATCH
            decision_basis      TEXT,     -- what the decision rested on (timeout = score arrived too late)
            explored            INTEGER NOT NULL DEFAULT 0,
            dwell_seconds       REAL,
            planned             TEXT,     -- engagement the policy asked for, e.g. "like,save"
            liked               INTEGER NOT NULL DEFAULT 0,
            saved               INTEGER NOT NULL DEFAULT 0,
            followed            INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_impressions_session ON impressions(session_id, seq);
        CREATE INDEX IF NOT EXISTS idx_impressions_creator ON impressions(creator);
        """
    )
    conn.commit()
    if own:
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


# ── Sessions & Impressions ────────────────────────────────────────────

def start_session(account: str, *, dry_run: bool, model: str, scorer: str) -> int:
    """Open a session row and return its id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO sessions (account, dry_run, model, scorer) VALUES (?, ?, ?, ?)",
        (account, int(dry_run), model, scorer),
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def end_session(session_id: int) -> None:
    conn = get_connection()
    conn.execute(
        """
        UPDATE sessions
        SET ended_at = datetime('now'),
            reels = (SELECT COUNT(*) FROM impressions WHERE session_id = ?)
        WHERE id = ?
        """,
        (session_id, session_id),
    )
    conn.commit()
    conn.close()


def log_impression(row: dict, conn: sqlite3.Connection | None = None) -> int:
    """Insert one impression; keys must be IMPRESSION_COLUMNS. Returns the row id."""
    unknown = set(row) - set(IMPRESSION_COLUMNS)
    if unknown:
        raise ValueError(f"unknown impression columns: {sorted(unknown)}")
    own = conn is None
    conn = conn or get_connection()
    cols = list(row)
    cur = conn.execute(
        f"INSERT INTO impressions ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [row[c] for c in cols],
    )
    conn.commit()
    if own:
        conn.close()
    return cur.lastrowid


def update_impression_score(
    session_id: int, code: str, fields: dict, conn: sqlite3.Connection | None = None
) -> int:
    """Fill in a score that arrived after the reel was logged; returns rows updated."""
    fields = {k: v for k, v in fields.items() if k in SCORE_COLUMNS}
    if not fields:
        return 0
    own = conn is None
    conn = conn or get_connection()
    assignments = ", ".join(f"{k} = ?" for k in fields)
    cur = conn.execute(
        f"""
        UPDATE impressions SET {assignments}, scored_at = datetime('now')
        WHERE session_id = ? AND code = ? AND scored_at IS NULL
        """,
        [*fields.values(), session_id, code],
    )
    conn.commit()
    if own:
        conn.close()
    return cur.rowcount


def load_creator_history(
    scorer: str, conn: sqlite3.Connection | None = None
) -> tuple[list[tuple[str, str, float]], set[str]]:
    """(creator, code, best score) rows scored under `scorer`, and all creators ever followed."""
    own = conn is None
    conn = conn or get_connection()
    scores = conn.execute(
        """
        SELECT i.creator, i.code, MAX(i.value_score)
        FROM impressions i JOIN sessions s ON s.id = i.session_id
        WHERE s.scorer = ? AND i.score_basis = 'score'
          AND i.value_score IS NOT NULL AND COALESCE(i.creator, '') != ''
        GROUP BY i.creator, i.code
        """,
        (scorer,),
    ).fetchall()
    followed = {r[0] for r in conn.execute("SELECT DISTINCT creator FROM impressions WHERE followed = 1")}
    if own:
        conn.close()
    return scores, followed


if __name__ == "__main__":
    init_db()
