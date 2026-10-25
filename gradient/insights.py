"""
PRAG — what the two personas' feeds and models show, side by side.

  divergence   day by day, how far apart the two accounts' feeds grow: the
               Jensen-Shannon distance between their topic mixes and between
               their political stance mixes, each feed's topic entropy (how
               varied it is) and top-3 share (how concentrated), its political
               share, and the creators both were shown. Read from the selected
               data source (Live: the recorded sessions; Simulated: the
               simulated ones), so on Simulated it describes the simulation's
               recommender, not Instagram's.
  explain      why a persona's preference model scores a text the way it does:
               the words that pushed its probability up and down (TF-IDF +
               logistic regression: each word's weight in the text times the
               model's coefficient for it)
  framing      how each persona's own feed covers one topic: the words that
               set one feed's reels on it apart from the other's (log-odds with
               an informative Dirichlet prior, Monroe et al. 2008), and the
               stance mix of those reels
  counter      reels on a topic the persona was never shown, from the side its
               feed leans away from: the material for blind-spot injection
               (chat.blindspot_job)

None of this changes a model; it reads them.
"""

import collections
import datetime
import math
import re
import sqlite3

import numpy as np

from gradient import prefs, retrieval, store, vectors

STANCES = ("supportive", "critical", "neutral", "not_political")
POLITICAL = ("supportive", "critical", "neutral")
WORD = re.compile(r"[\w\u0900-\u0963\u0966-\u097F]{3,}")  # Devanagari letters and signs, not the danda (।)
STOP = set("""the and for that this with are was were from have has had but not you your they their them its his her
our out all can will would could should about into than then there what when where which who why how also just more
most some such very been being over only other any each because these those here after before while hai hain kya
nahi aur bhi yeh woh mein par pe se ka ki ke ko ne jo tak liye ab toh lekin agar kuch sab apne unke iske uske
है हैं का की के को में से और भी यह वह पर ने तो कि लिए नहीं रहा रही रहे था थी थे गया गई एक इस उस""".split())


# ── Echo chambers: the feeds drifting apart ───────────────────────────

def _js(p: dict, q: dict) -> float | None:
    """Jensen-Shannon distance (base 2, 0 = same mix, 1 = nothing shared) between two count dicts."""
    keys = sorted(set(p) | set(q))
    a, b = np.array([p.get(k, 0) for k in keys], float), np.array([q.get(k, 0) for k in keys], float)
    if not a.sum() or not b.sum():
        return None
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2

    def kl(x):
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / m[mask])).sum())

    return round(math.sqrt(max(0.0, 0.5 * kl(a) + 0.5 * kl(b))), 3)


def _entropy(counts: dict, k: int) -> float | None:
    total = sum(counts.values())
    if not total or k < 2:
        return None
    p = np.array([v / total for v in counts.values() if v])
    return round(float(-(p * np.log(p)).sum() / math.log(k)), 3)


def local_day(utc_text: str) -> str:
    return datetime.datetime.fromisoformat(utc_text).replace(tzinfo=datetime.timezone.utc).astimezone().date().isoformat()


def divergence(conn: sqlite3.Connection, accounts: tuple[str, str] = tuple(store.ACCOUNTS)) -> dict:
    """Per day: each account's feed (entropy, top-3 share, political share, stance mix) and the distance between
    the two (topics, stance, shared creators). Ads and reels with no category are left out of the topic mixes."""
    rows = conn.execute("SELECT s.account, i.shown_at, i.category, i.stance, i.creator FROM impressions i JOIN sessions s "
                        "ON s.id = i.session_id WHERE i.is_ad = 0 AND i.shown_at IS NOT NULL AND s.account IN (?, ?)",
                        accounts).fetchall()
    days: dict[str, dict] = {}
    categories = set()
    for account, shown_at, category, stance, creator in rows:
        d = days.setdefault(local_day(shown_at), {a: {"cat": collections.Counter(), "stance": collections.Counter(),
                                                       "creators": set(), "n": 0} for a in accounts})[account]
        d["n"] += 1
        if category:
            d["cat"][category] += 1
            categories.add(category)
        if stance in STANCES:
            d["stance"][stance] += 1
        if creator:
            d["creators"].add(creator)
    out = []
    for day in sorted(days):
        a, b = (days[day][x] for x in accounts)
        if not a["n"] or not b["n"]:
            continue
        row = {"day": day, "topic_distance": _js(a["cat"], b["cat"]),
               "stance_distance": _js({k: a["stance"][k] for k in POLITICAL}, {k: b["stance"][k] for k in POLITICAL}),
               "shared_creators": round(len(a["creators"] & b["creators"]) / max(1, len(a["creators"] | b["creators"])), 3),
               "accounts": {}}
        for name, d in zip(accounts, (a, b)):
            cats = sum(d["cat"].values())
            political = sum(d["stance"][k] for k in POLITICAL)
            row["accounts"][name] = {
                "reels": d["n"], "entropy": _entropy(d["cat"], len(categories)),
                "top3": round(sum(v for _, v in d["cat"].most_common(3)) / cats, 3) if cats else None,
                "political": round(political / sum(d["stance"].values()), 3) if d["stance"] else None,
                "stance": {k: round(d["stance"][k] / political, 3) for k in POLITICAL} if political else None}
        out.append(row)

    def mean(key, part, account=None):
        vals = [(r["accounts"][account][key] if account else r[key]) for r in part]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    first, last = out[:3], out[-3:]
    summary = {"topic_distance": [mean("topic_distance", first), mean("topic_distance", last)],
               "stance_distance": [mean("stance_distance", first), mean("stance_distance", last)],
               "shared_creators": [mean("shared_creators", first), mean("shared_creators", last)],
               "entropy": {a: [mean("entropy", first, a), mean("entropy", last, a)] for a in accounts},
               "top3": {a: [mean("top3", first, a), mean("top3", last, a)] for a in accounts}}
    return {"accounts": list(accounts), "days": out, "summary": summary, "categories": len(categories)}


# ── Why the preference model scores a text so ─────────────────────────

def explain(saved: dict, text: str, top: int = 8) -> dict | None:
    """The words in `text` that pushed this TF-IDF model's probability up and down most; None for other model kinds.
    Character n-grams (the other half of its features) are summed into one number, since they don't read as words."""
    if not saved or saved["kind"] != "tfidf":
        return None
    pipeline = saved["model"]
    features, lr = pipeline.steps[0][1], pipeline.steps[-1][1]
    x = features.transform(np.array([prefs.scrub(text)], dtype=object)).tocsr()
    names = features.get_feature_names_out()
    contrib = x.multiply(lr.coef_[0]).tocsr()
    words, chars = [], 0.0
    for j, v in zip(contrib.indices, contrib.data):
        name = names[j]
        if name.startswith("words__"):
            words.append((name[len("words__"):], float(v)))
        else:
            chars += float(v)
    words.sort(key=lambda w: w[1])
    return {"intercept": round(float(lr.intercept_[0]), 3), "char_ngrams": round(chars, 3),
            "words_total": round(sum(v for _, v in words), 3),
            "up": [{"term": t, "weight": round(v, 3)} for t, v in reversed(words[-top:]) if v > 0],
            "down": [{"term": t, "weight": round(v, 3)} for t, v in words[:top] if v < 0]}


# ── How each feed frames a topic ──────────────────────────────────────

def topic_reels(conn, account: str, topic: str, k: int = 60) -> list[dict]:
    """The reels in this persona's feed closest to the topic (by meaning alone, no preference boost)."""
    idx = retrieval.index(conn, account)
    if not idx["rows"]:
        return []
    sim = idx["matrix"] @ vectors.embed_texts([topic])[0]
    order = [i for i in np.argsort(-sim)[:k] if sim[i] >= retrieval.MIN_SIMILARITY]
    return [{**idx["rows"][i], "similarity": float(sim[i])} for i in order]


def tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(prefs.scrub(text).lower()) if w not in STOP and not w.isdigit()]


def log_odds(a_texts: list[str], b_texts: list[str], prior_texts: list[str], top: int = 10) -> tuple[list, list]:
    """Words most distinctive of a vs b: log-odds ratio with an informative Dirichlet prior, z-scored (Monroe,
    Colaresi & Quinn 2008). Returns ([(word, z)] for a, [(word, z)] for b)."""
    ca = collections.Counter(w for t in a_texts for w in tokens(t))
    cb = collections.Counter(w for t in b_texts for w in tokens(t))
    prior = collections.Counter(w for t in prior_texts for w in tokens(t))
    prior.update(ca)
    prior.update(cb)
    na, nb = sum(ca.values()), sum(cb.values())
    strength = max(10.0, 0.1 * (na + nb))  # the prior's total pseudo-count: a tenth of the words compared
    scale = strength / (sum(prior.values()) or 1)
    z = {}
    for w in set(ca) | set(cb):
        if ca[w] + cb[w] < 3:
            continue
        aw = prior[w] * scale + 0.01
        la = math.log((ca[w] + aw) / (na + strength - ca[w] - aw))
        lb = math.log((cb[w] + aw) / (nb + strength - cb[w] - aw))
        z[w] = (la - lb) / math.sqrt(1 / (ca[w] + aw) + 1 / (cb[w] + aw))
    ranked = sorted(z.items(), key=lambda kv: kv[1])
    return ([(w, round(v, 2)) for w, v in reversed(ranked[-top:]) if v > 1.0],
            [(w, round(-v, 2)) for w, v in ranked[:top] if v < -1.0])


def framing(conn, topic: str) -> dict:
    """Each persona's own feed on a topic: how many reels, their stance mix, and the words that set it apart."""
    a, b = store.ACCOUNTS
    reels = {acc: topic_reels(conn, acc, topic) for acc in (a, b)}
    texts = {acc: [r["text"] for r in reels[acc]] for acc in (a, b)}
    words_a, words_b = log_odds(texts[a], texts[b], texts[a] + texts[b])
    out = {"topic": topic, "accounts": {}}
    for acc, words in ((a, words_a), (b, words_b)):
        mix = collections.Counter(r.get("stance") if r.get("stance") in STANCES else "not_political" for r in reels[acc])
        total = sum(mix.values()) or 1
        out["accounts"][acc] = {"reels": len(reels[acc]), "stance": {k: round(mix[k] / total, 3) for k in STANCES},
                                "words": [{"term": w, "z": z} for w, z in words]}
    shared = {r["id"] for r in reels[a]} & {r["id"] for r in reels[b]}
    out["shared_reels"] = len(shared)
    return out


# ── Blind spots: what the persona was never shown ─────────────────────

def leaning(conn, account: str, reels: list[dict]) -> str | None:
    """Which political side this persona's feed leans to on these reels (or overall, when they aren't political)."""
    mix = collections.Counter(r.get("stance") for r in reels if r.get("stance") in ("supportive", "critical"))
    if not mix:
        mix = collections.Counter(dict(conn.execute(
            "SELECT i.stance, COUNT(*) FROM feed f JOIN items i ON i.id = f.item_id WHERE f.account = ? AND "
            "i.stance IN ('supportive', 'critical') GROUP BY i.stance", (account,)).fetchall()))
    return mix.most_common(1)[0][0] if mix else None


def counter(conn, account: str, topic: str, k: int = 4) -> dict:
    """Reels on the topic from the corpus that this persona's feed never showed it, from the other side of how its
    feed leans: {"leans", "against", "items"}."""
    own = topic_reels(conn, account, topic, k=40)
    leans = leaning(conn, account, own)
    against = {"supportive": "critical", "critical": "supportive"}.get(leans)
    if against is None:
        return {"leans": leans, "against": None, "items": []}
    rows = [dict(r) for r in conn.execute(
        "SELECT i.* FROM items i WHERE i.stance = ? AND i.id NOT IN (SELECT item_id FROM feed WHERE account = ?)",
        (against, account))]
    if not rows:
        return {"leans": leans, "against": against, "items": []}
    sim = vectors.for_items(rows) @ vectors.embed_texts([topic])[0]
    picked = [i for i in np.argsort(-sim)[:k * 3] if sim[i] >= retrieval.MIN_SIMILARITY][:k]
    items = [{"id": rows[i]["id"], "similarity": round(float(sim[i]), 3), "injected": True,
              **retrieval.public(rows[i])} for i in picked]
    return {"leans": leans, "against": against, "items": items}
