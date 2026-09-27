"""
PRAG — Database Schema & Helpers

Creates (or opens) `reels.db` with:
  accounts     the Instagram accounts the agent may use
  sessions     one row per agent run
  impressions  one row per reel shown — its metadata, what was extracted,
               the LLM's judgement, what the policy did, and how long the
               reel stayed on screen

The legacy `reels` table from earlier versions is left untouched.
"""

import datetime
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reels.db")

# Sessions run with --observe are scored afterwards by review (tools/review.py), not by the LLM.
REVIEW_SCORER = "review"

IMPRESSION_COLUMNS = (
    "session_id", "seq", "code", "url", "creator", "caption", "hashtags",
    "audio_type", "audio_title", "duration", "like_count", "comment_count",
    "repost_count", "taken_at", "inventory_source", "is_ad",
    "transcript", "transcript_language", "cover_text",
    "score_basis", "value_score", "is_informational", "topic", "summary",
    "reasoning", "error", "gate_keyword", "gate_audit", "score_seconds", "scored_at",
    "decision", "decision_basis", "explored", "dwell_seconds", "planned",
    "liked", "saved", "followed", "rule_score", "rule_reason", "ready_on_arrival",
)
# Columns added after the first release; init_db adds them to older databases.
LATER_COLUMNS = (("rule_score", "REAL"), ("rule_reason", "TEXT"), ("ready_on_arrival", "INTEGER"))
SCORE_COLUMNS = (
    "transcript", "transcript_language", "cover_text", "score_basis",
    "value_score", "is_informational", "topic", "summary", "reasoning",
    "error", "gate_keyword", "gate_audit", "score_seconds",
)


def get_connection(path: str = DB_PATH) -> sqlite3.Connection:
    """Return a connection to the PRAG SQLite database."""
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
            followed            INTEGER NOT NULL DEFAULT 0,
            rule_score          REAL,     -- the quick rules' score (--rules), kept next to the review label
            rule_reason         TEXT,
            ready_on_arrival    INTEGER   -- decision was ready the moment the reel appeared (preloaded feed data)
        );
        CREATE INDEX IF NOT EXISTS idx_impressions_session ON impressions(session_id, seq);
        CREATE INDEX IF NOT EXISTS idx_impressions_creator ON impressions(creator);
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(impressions)")}
    for column, kind in LATER_COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE impressions ADD COLUMN {column} {kind}")
    conn.commit()
    if own:
        conn.close()
        print("✅  Database initialised — reels.db is ready.")


def list_accounts(conn: sqlite3.Connection | None = None) -> list[dict]:
    """Every account: username, status and when it was last used."""
    own = conn is None
    conn = conn or get_connection()
    rows = conn.execute(
        "SELECT username, status, last_used_timestamp FROM accounts ORDER BY username"
    ).fetchall()
    if own:
        conn.close()
    return [{"username": u, "status": s, "last_used": t} for u, s, t in rows]


def choose_account(accounts: list[dict], requested: str | None) -> tuple[str | None, str | None]:
    """(username, None) for the one account a run should use, or (None, why not).

    Runs never pick between several accounts on their own: with more than one
    ACTIVE account, the account must be named explicitly.
    """
    add_hint = "Add it to accounts.json and run `python -m database.seed`."
    if requested:
        account = next((a for a in accounts if a["username"] == requested.lstrip("@")), None)
        if account is None:
            return None, f"Unknown account @{requested.lstrip('@')}. {add_hint}"
        if account["status"] != "ACTIVE":
            return None, f"@{account['username']} is {account['status']}, not ACTIVE."
        return account["username"], None
    active = [a["username"] for a in accounts if a["status"] == "ACTIVE"]
    if len(active) == 1:
        return active[0], None
    if not active:
        return None, f"No ACTIVE accounts. {add_hint}"
    return None, "Several accounts — pick one with --account: " + ", ".join(f"@{u}" for u in active)


def mark_account_used(username: str) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE accounts SET last_used_timestamp = datetime('now') WHERE username = ?", (username,)
    )
    conn.commit()
    conn.close()


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


def _today(now: datetime.datetime | None) -> tuple[str, str]:
    """(local midnight, now) as the database's UTC timestamps."""
    now = (now or datetime.datetime.now()).astimezone()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    utc = lambda ts: ts.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return utc(midnight), utc(now)


def minutes_today(account: str, now: datetime.datetime | None = None,
                  conn: sqlite3.Connection | None = None) -> float:
    """Minutes the agent has scrolled `account` since local midnight.

    A session that never closed (the process was killed) counts up to its last reel.
    """
    own = conn is None
    conn = conn or get_connection()
    seconds = conn.execute(
        """
        SELECT COALESCE(SUM(
            strftime('%s', COALESCE(s.ended_at,
                                    (SELECT MAX(i.shown_at) FROM impressions i WHERE i.session_id = s.id),
                                    s.started_at))
            - strftime('%s', s.started_at)), 0)
        FROM sessions s WHERE s.account = ? AND s.started_at >= ? AND s.started_at <= ?
        """,
        (account, *_today(now)),
    ).fetchone()[0]
    if own:
        conn.close()
    return seconds / 60


def engaged_today(account: str, now: datetime.datetime | None = None,
                  conn: sqlite3.Connection | None = None) -> bool:
    """Whether `account` already had a live session (likes, saves, follows allowed) that showed reels today."""
    own = conn is None
    conn = conn or get_connection()
    row = conn.execute(
        """
        SELECT 1 FROM sessions s
        WHERE s.account = ? AND s.dry_run = 0 AND s.started_at >= ? AND s.started_at <= ?
          AND EXISTS (SELECT 1 FROM impressions i WHERE i.session_id = s.id)
        LIMIT 1
        """,
        (account, *_today(now)),
    ).fetchone()
    if own:
        conn.close()
    return row is not None


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
    scorer: str, account: str, conn: sqlite3.Connection | None = None
) -> tuple[list[tuple[str, str, float]], set[str]]:
    """For one account: (creator, code, best score) rows scored under `scorer`, and creators it followed."""
    own = conn is None
    conn = conn or get_connection()
    scores = conn.execute(
        """
        SELECT i.creator, i.code, MAX(i.value_score)
        FROM impressions i JOIN sessions s ON s.id = i.session_id
        WHERE s.scorer = ? AND s.account = ? AND i.score_basis = 'score'
          AND i.value_score IS NOT NULL AND COALESCE(i.creator, '') != ''
        GROUP BY i.creator, i.code
        """,
        (scorer, account),
    ).fetchall()
    followed = {r[0] for r in conn.execute(
        """
        SELECT DISTINCT i.creator FROM impressions i JOIN sessions s ON s.id = i.session_id
        WHERE i.followed = 1 AND s.account = ?
        """,
        (account,),
    )}
    if own:
        conn.close()
    return scores, followed


if __name__ == "__main__":
    init_db()
