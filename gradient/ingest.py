"""
PRAG — the Gradient's ingestion: everything that grows the training corpus.

  sessions     reels the Instagram sessions recorded (reels.db), into each
               account's feed, with when they were shown (dry runs too: the
               feed showed them all the same; only the likes were held back). Re-running only adds
               what's new, so daily.py runs it after every session.
  legacy       the earlier corpus (datasets/demo/gradient.db): Wikipedia
               articles, RSS headlines and written reels, with the Jev
               labels already paid for (both personas' texts are unchanged
               since they were judged, so the labels still hold)
  recommend    news headlines the recommender puts in each feed: drawn by
               that persona's Jev score, so what the persona likes it keeps
               getting, the way the real feed narrows

Generated reels come in through generate.py, labels through label.py.
"""

import json
import os
import random
import sqlite3

from agent import bias as biases
from gradient import store, taxonomy

LEGACY_DB = os.path.join(store.PROJECT_ROOT, "datasets", "demo", "gradient.db")
REELS_DB = os.path.join(store.PROJECT_ROOT, "reels.db")
LEGACY_ORIGINS = {"public_wikipedia": "wikipedia", "public_rss": "rss", "synthetic": "generated"}
LEGACY_PERSONAS = {"mlproject01": "p-e1c2205f0e", "mlproject02": "p-649d5e3289"}  # judged with today's texts
LEGACY_THRESHOLD = 0.5  # their labels: Jev's level "watches most of it" or more
CATEGORY_TOPIC = {"sports": "cricket", "food": "street_food", "comedy_memes": "hindi_comedy", "religion": "religion",
                  "film_tv": "film_tv", "music_dance": "indie_music", "travel_nature": "travel_nature",
                  "business_money": "economy", "education": "history_explainers", "tech_science": "tech_ai",
                  "lifestyle": "lifestyle", "promotion": "promotion", "news_politics": "party_politics"}

SESSION_SQL = """
SELECT i.code, i.creator, i.caption, i.cover_text, i.transcript, i.transcript_language, i.category, i.stance,
       i.topic, i.summary, MIN(i.shown_at) AS shown_at, s.account
FROM impressions i JOIN sessions s ON s.id = i.session_id
WHERE s.account = ? AND i.is_ad = 0 AND i.code IS NOT NULL
  AND (COALESCE(i.caption, '') <> '' OR COALESCE(i.transcript, '') <> '' OR COALESCE(i.cover_text, '') <> '')
GROUP BY i.code
ORDER BY MIN(i.id)
"""


def language_of(text: str, hint: str | None = None) -> str:
    if hint in ("hi", "en"):
        return hint
    devanagari = sum("ऀ" <= c <= "ॿ" for c in text)
    letters = sum(c.isalpha() for c in text) or 1
    if devanagari / letters > 0.3:
        return "hi"
    hinglish = sum(w in text.lower().split() for w in ("hai", "nahi", "kya", "bhai", "aur", "ke", "ki", "se", "mein"))
    return "hinglish" if hinglish >= 2 else "en"


def ingest_sessions(conn: sqlite3.Connection, accounts=store.ACCOUNTS, reels_db: str = REELS_DB) -> dict:
    """Every reel the accounts' sessions recorded and the corpus doesn't have yet."""
    source = sqlite3.connect(reels_db)
    source.row_factory = sqlite3.Row
    batch, counts = None, {}
    for account in accounts:
        new = 0
        for row in source.execute(SESSION_SQL, (account,)):
            if conn.execute("SELECT 1 FROM feed WHERE account = ? AND item_id = ?",
                            (account, f"ig-{row['code']}")).fetchone():
                continue
            batch = batch or store.new_batch(conn, "sessions", "reels recorded by the Instagram sessions")
            text = store.model_text(row["caption"], row["cover_text"], row["transcript"])
            item_id, _ = store.add_item(conn, {
                "id": f"ig-{row['code']}", "origin": "instagram", "source_name": "Instagram",
                "source_url": f"https://www.instagram.com/reel/{row['code']}/", "creator": row["creator"],
                "caption": row["caption"], "on_screen": row["cover_text"], "transcript": row["transcript"],
                "text": text, "language": language_of(text, row["transcript_language"]),
                "topic": CATEGORY_TOPIC.get(row["category"]), "category": row["category"], "stance": row["stance"],
                "group_id": f"ig-{row['code']}", "meta": {"summary": row["summary"], "topic_words": row["topic"]},
            }, batch)
            store.add_to_feed(conn, account, item_id, batch, "session", row["shown_at"])
            new += 1
        counts[account] = new
    source.close()
    if batch:
        store.close_batch(conn, batch, {"instagram": counts})
    conn.commit()
    return counts


def ingest_legacy(conn: sqlite3.Connection, path: str = LEGACY_DB) -> dict:
    """The earlier corpus and its Jev labels (once: items already here are skipped)."""
    if not os.path.exists(path) or conn.execute("SELECT 1 FROM batches WHERE kind = 'legacy'").fetchone():
        return {}
    old = sqlite3.connect(path)
    old.row_factory = sqlite3.Row
    batch = store.new_batch(conn, "legacy", "Wikipedia articles, RSS headlines and written reels, with Jev labels")
    ids, counts = {}, {}
    for row in old.execute("SELECT * FROM items WHERE origin IN ('public_wikipedia', 'public_rss', 'synthetic')"):
        origin = LEGACY_ORIGINS[row["origin"]]
        tags = json.loads(row["tags"] or "[]")
        topic = next((t for t in tags if t in taxonomy.TOPICS), None) or CATEGORY_TOPIC.get(row["category"])
        item_id, new = store.add_item(conn, {
            "id": row["id"], "origin": origin, "source_name": row["source_name"], "source_url": row["source_url"],
            "creator": row["creator"], "title": row["title"], "text": row["text"],
            "caption": row["title"] if origin != "wikipedia" else None, "language": row["language"],
            "topic": topic, "category": row["category"], "stance": row["stance"], "format": row["kind"],
            "group_id": row["group_id"] or row["story_group"], "ingested_at": row["ingested_at"],
            "meta": {"license": row["license"], "attribution": row["attribution"], "published_at": row["published_at"]},
        }, batch)
        ids[row["id"]] = item_id
        counts[origin] = counts.get(origin, 0) + new
    labelled = 0
    for account, version in LEGACY_PERSONAS.items():
        fingerprint = biases.load(account).fingerprint
        for row in old.execute("SELECT item_id, teacher_version, score FROM labels WHERE account = ? AND origin = 'jev' "
                               "AND persona_version = ?", (account, version)):
            if row["item_id"] in ids and row["score"] is not None:
                store.add_label(conn, ids[row["item_id"]], account, fingerprint, "jev", row["teacher_version"],
                                row["score"], int(row["score"] >= LEGACY_THRESHOLD))
                labelled += 1
    old.close()
    store.close_batch(conn, batch, {"legacy": counts, "jev_labels": labelled})
    return {**counts, "jev_labels": labelled}


def recommend(conn: sqlite3.Connection, account: str, n: int = 400, seed: int = 11) -> int:
    """Put n news headlines in the account's feed, drawn by its Jev score cubed (plus a little exploration)."""
    rows = conn.execute("SELECT i.id, l.score FROM items i JOIN labels l ON l.item_id = i.id AND l.account = ? "
                        "AND l.teacher = 'jev' WHERE i.origin = 'rss' AND NOT EXISTS "
                        "(SELECT 1 FROM feed f WHERE f.account = ? AND f.item_id = i.id)", (account, account)).fetchall()
    have = conn.execute("SELECT COUNT(*) FROM feed WHERE account = ? AND via = 'recommender'", (account,)).fetchone()[0]
    want = min(n - have, len(rows))
    if want <= 0:
        return 0
    rng = random.Random(f"{seed}|{account}|{have}")
    pool = list(rows)
    batch = store.new_batch(conn, "recommender", f"{account}: headlines the recommender served")
    picked = 0
    while picked < want and pool:
        weights = [0.02 + (r["score"] or 0) ** 3 for r in pool]
        choice = rng.choices(range(len(pool)), weights=weights)[0]
        store.add_to_feed(conn, account, pool.pop(choice)["id"], batch, "recommender")
        picked += 1
    store.close_batch(conn, batch, {"recommender": {account: picked}})
    return picked
