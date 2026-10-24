"""
PRAG — the Gradient's store: one SQLite file for the training corpus.

  items     one row per unique piece of content, whoever saw it, with where it
            came from (origin): an Instagram session, Wikipedia, an RSS
            headline, or written by an LLM for a feed (generate.py)
  feed      which items are in each account's feed: its knowledge, what chat
            and debate can cite. Instagram reels the account was shown, reels
            written for its feed, headlines the recommender picked for it
  labels    each persona's judgement of an item (would watch / would skip),
            by a teacher: Jev, the local LLM, the generator's intent, a person
  takes     the persona's own words about a reel in its feed: what the
            persona language model (lora.py) learns to sound like
  batches   every ingestion, in order: the corpus grows batch by batch
  runs      every training run, its data snapshot and every measured metric

Splits are assigned once per group_id (a story, or one generator brief), so
copies and retellings of one story never straddle train and test.
"""

import datetime
import hashlib
import json
import os
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "gradient")
DB_PATH = os.path.join(DATA_DIR, "corpus.db")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "gradient")
ACCOUNTS = ("mlproject01", "mlproject02")
ORIGINS = {  # origin: what the panel calls it
    "instagram": "Instagram sessions",
    "generated": "Written by an LLM (Qwen 3.5) for the feed",
    "rss": "News headlines (RSS)",
    "wikipedia": "Wikipedia",
}
SPLIT_SHARES = (("train", 0.7), ("val", 0.15), ("test", 0.15))

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id          TEXT PRIMARY KEY,
    origin      TEXT NOT NULL,          -- instagram | generated | rss | wikipedia
    source_name TEXT,
    source_url  TEXT,
    creator     TEXT,
    title       TEXT,
    caption     TEXT,
    on_screen   TEXT,
    transcript  TEXT,
    text        TEXT NOT NULL,          -- what the models read: caption | on-screen text | transcript
    language    TEXT,                   -- en | hi | hinglish
    topic       TEXT,                   -- taxonomy.TOPICS key
    category    TEXT,                   -- perception/categorize.py CATEGORIES key
    stance      TEXT,                   -- supportive | critical | neutral | not_political
    format      TEXT,
    group_id    TEXT NOT NULL,          -- the split unit
    split       TEXT NOT NULL,          -- train | val | test
    batch       TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS items_hash ON items(text_hash);
CREATE INDEX IF NOT EXISTS items_origin ON items(origin);
CREATE TABLE IF NOT EXISTS feed (
    account   TEXT NOT NULL,
    item_id   TEXT NOT NULL REFERENCES items(id),
    batch     TEXT NOT NULL,
    via       TEXT NOT NULL,            -- session | generator | recommender
    shown_at  TEXT,                     -- sessions only: when Instagram showed it
    PRIMARY KEY (account, item_id)
);
CREATE TABLE IF NOT EXISTS labels (
    item_id         TEXT NOT NULL REFERENCES items(id),
    account         TEXT NOT NULL,
    persona_version TEXT NOT NULL,      -- the persona's fingerprint when it was judged
    teacher         TEXT NOT NULL,      -- jev | local_llm | generator | human
    teacher_version TEXT NOT NULL,
    score           REAL,               -- graded, 0-1
    label           INTEGER NOT NULL,   -- 1 would watch to the end or engage, 0 would scroll past
    created_at      TEXT NOT NULL,
    PRIMARY KEY (item_id, account, teacher, teacher_version)
);
CREATE INDEX IF NOT EXISTS labels_account ON labels(account, teacher);
CREATE TABLE IF NOT EXISTS takes (
    item_id TEXT NOT NULL REFERENCES items(id),
    account TEXT NOT NULL,
    action  TEXT,                       -- skip | watch | like | save | share
    take    TEXT NOT NULL,
    PRIMARY KEY (item_id, account)
);
CREATE TABLE IF NOT EXISTS batches (
    id         TEXT PRIMARY KEY,
    step       INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    kind       TEXT NOT NULL,
    note       TEXT,
    counts     TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id         TEXT PRIMARY KEY,
    account    TEXT NOT NULL,
    kind       TEXT NOT NULL,           -- preference | feed | lora
    version    INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    status     TEXT NOT NULL,           -- done | failed
    items      INTEGER,
    batches    TEXT,                    -- JSON: the batches its data came from
    path       TEXT,
    report     TEXT,                    -- JSON: every metric, as measured
    error      TEXT
);
CREATE TABLE IF NOT EXISTS gen_briefs (
    id      TEXT PRIMARY KEY,           -- generate.py: a brief done is never written again
    account TEXT NOT NULL,
    done_at TEXT,
    reels   INTEGER,
    error   TEXT
);
"""


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or DB_PATH
    if path != ":memory:":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def text_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


def split_for(group_id: str) -> str:
    """A fixed split per group: the same group always lands in the same partition."""
    x = int(hashlib.sha1(f"split|{group_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    total = 0.0
    for name, share in SPLIT_SHARES:
        total += share
        if x < total:
            return name
    return SPLIT_SHARES[-1][0]


def model_text(caption: str | None, on_screen: str | None, transcript: str | None, limit: int = 1500) -> str:
    """The same three texts the session's judges read (ml/dataset.reel_text)."""
    parts = [p.strip() for p in (caption, on_screen, transcript) if p and p.strip()]
    return " | ".join(parts)[:limit]


def new_batch(conn: sqlite3.Connection, kind: str, note: str = "", batch_id: str | None = None) -> str:
    step = conn.execute("SELECT COALESCE(MAX(step), 0) + 1 FROM batches").fetchone()[0]
    batch_id = batch_id or f"b{step:03d}-{kind}"
    conn.execute("INSERT OR IGNORE INTO batches (id, step, created_at, kind, note) VALUES (?, ?, ?, ?, ?)",
                 (batch_id, step, now(), kind, note))
    return batch_id


def close_batch(conn: sqlite3.Connection, batch_id: str, counts: dict) -> None:
    conn.execute("UPDATE batches SET counts = ? WHERE id = ?", (json.dumps(counts), batch_id))
    conn.commit()


def add_item(conn: sqlite3.Connection, item: dict, batch: str) -> tuple[str, bool]:
    """(item id, whether it's new). An item whose text is already stored is the same item."""
    text = item.get("text") or model_text(item.get("caption"), item.get("on_screen"), item.get("transcript"))
    if not text.strip():
        raise ValueError("an item needs some text")
    digest = text_hash(text)
    found = conn.execute("SELECT id FROM items WHERE text_hash = ?", (digest,)).fetchone()
    if found:
        return found[0], False
    item_id = item.get("id") or digest[:16]
    group = item.get("group_id") or item_id
    conn.execute(
        "INSERT INTO items (id, origin, source_name, source_url, creator, title, caption, on_screen, transcript, text, "
        "language, topic, category, stance, format, group_id, split, batch, ingested_at, text_hash, meta) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item_id, item["origin"], item.get("source_name"), item.get("source_url"), item.get("creator"),
         item.get("title"), item.get("caption"), item.get("on_screen"), item.get("transcript"), text,
         item.get("language"), item.get("topic"), item.get("category"), item.get("stance"), item.get("format"),
         group, item.get("split") or split_for(group), batch, item.get("ingested_at") or now(), digest,
         json.dumps(item["meta"], ensure_ascii=False) if item.get("meta") else None))
    return item_id, True


def add_to_feed(conn: sqlite3.Connection, account: str, item_id: str, batch: str, via: str,
                shown_at: str | None = None) -> None:
    conn.execute("INSERT OR IGNORE INTO feed (account, item_id, batch, via, shown_at) VALUES (?, ?, ?, ?, ?)",
                 (account, item_id, batch, via, shown_at))


def add_label(conn: sqlite3.Connection, item_id: str, account: str, persona_version: str, teacher: str,
              teacher_version: str, score: float | None, label: int) -> None:
    conn.execute("INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (item_id, account, persona_version, teacher, teacher_version, score, int(label), now()))


def items_by_id(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    out = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        rows = conn.execute(f"SELECT * FROM items WHERE id IN ({','.join('?' * len(chunk))})", chunk)
        out.update({row["id"]: dict(row) for row in rows})
    return out
