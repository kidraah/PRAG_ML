"""
PRAG — the Gradient's retrieval: evidence from one persona's own feed.

A persona can only cite what its feed showed it: the search runs over that
account's feed items and nothing else (not the other persona's feed, not
the rest of the corpus). Each item is ranked by

    score = similarity(question, reel) + PREF_WEIGHT × p(persona engages)

the first from the multilingual embeddings, the second from the persona's
trained preference model (prefs.py). So between two equally relevant reels,
the one this persona would have watched comes first, the way their own
memory would. Both parts are returned, so the panel can show what the model
contributed. Below MIN_SIMILARITY nothing counts as evidence, and the
persona is told so.
"""

import numpy as np

from gradient import prefs, store, vectors

PREF_WEIGHT = 0.15
MIN_SIMILARITY = 0.30
_index: dict[str, dict] = {}


def index(conn, account: str) -> dict:
    """The account's feed: items, their vectors and the preference model's scores (rebuilt when either changes)."""
    count = conn.execute("SELECT COUNT(*) FROM feed WHERE account = ?", (account,)).fetchone()[0]
    model = prefs.load(account)
    key = (count, model["run"] if model else None)
    cached = _index.get(account)
    if cached and cached["key"] == key:
        return cached
    rows = [dict(r) for r in conn.execute("SELECT i.*, f.via, f.shown_at FROM feed f JOIN items i ON i.id = f.item_id "
                                          "WHERE f.account = ? ORDER BY i.id", (account,))]
    matrix = vectors.for_items(rows)
    pref = prefs.predict(model, [r["text"] for r in rows]) if (model and rows) else np.zeros(len(rows))
    _index[account] = {"key": key, "rows": rows, "matrix": matrix, "pref": pref,
                       "model": model["run"] if model else None}
    return _index[account]


def search(conn, account: str, query: str, k: int = 5, exclude: set[str] | None = None) -> dict:
    idx = index(conn, account)
    if not idx["rows"]:
        return {"items": [], "model": idx["model"], "enough": False}
    q = vectors.embed_texts([query])[0]
    sim = idx["matrix"] @ q
    score = sim + PREF_WEIGHT * idx["pref"]
    order = [i for i in np.argsort(-score) if not exclude or idx["rows"][i]["id"] not in exclude]
    picked = [i for i in order[:k * 3] if sim[i] >= MIN_SIMILARITY][:k]
    by_sim = set(np.argsort(-sim)[:k].tolist())
    items = [{"id": idx["rows"][i]["id"], "similarity": round(float(sim[i]), 3),
              "preference": round(float(idx["pref"][i]), 3), "score": round(float(score[i]), 3),
              "moved_up_by_model": i not in by_sim, **public(idx["rows"][i])} for i in picked]
    return {"items": items, "model": idx["model"], "enough": bool(items)}


def public(row: dict) -> dict:
    """What the panel shows of an item."""
    return {k: row.get(k) for k in ("origin", "creator", "caption", "on_screen", "transcript", "title", "source_url",
                                    "language", "topic", "category", "stance", "via", "shown_at")} | {
        "text": row["text"][:600], "origin_name": store.ORIGINS.get(row.get("origin"), row.get("origin"))}
