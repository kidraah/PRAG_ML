"""
PRAG — The Feed Model: what the algorithm serves each persona

Learns the feed's topics from every account's reels together: what each
reel says (the scorer's topic and summary, the caption, the text on the
cover and the speech) is embedded with the multilingual sentence model
and clustered. Because the topics are shared, every persona's feed can be
described in the same terms and compared:

  mix          which topics the feed is made of, in the warm-up and once curated
  narrowness   the effective number of topics (e^entropy of the mix): it
               falls as the algorithm settles on what it thinks the persona wants
  day by day   how many reels, how narrow, and how much the mix moved from the
               day before (Jensen-Shannon distance, 0 = the same, 1 = nothing shared)
  rising       topics the latest days show more of than the days before
  apart        topics a persona is served more than the accounts overall
  alike        how similar the personas' feeds are (1 - JS distance)

One account's feed, or all of them. Retrained after each daily run, or:

    python -m ml.feed_model --train              learn the topics from every account
    python -m ml.feed_model [--account NAME]     the feed, overall or one persona's
"""

import argparse
import datetime
import math
import os
import re
import sqlite3
from collections import Counter

import numpy as np

from config import EMBEDDING_MODEL
from dashboard.feed import all_feeds
from database.db import get_connection, init_db
from perception.categorize import CATEGORIES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "feed-model.joblib")
VECTORS_PATH = os.path.join(MODEL_DIR, "feed-vectors.npz")  # embeddings already computed, by reel
MIN_REELS = 30        # fewer different reels and there are no topics to learn
MAX_TOPICS = 24
DAY_MIN_REELS = 5     # a day with fewer reels says little about the feed
RECENT_DAYS = 3       # "rising" compares the latest days with the ones before

WORD = re.compile(r"[a-z][a-z']{2,}|[ऀ-ॿ]{2,}")
NOISE = re.compile(r"https?://\S+|@[\w.]+")
STOPWORDS = set("""
the and for are but not you your yours with this that these those from have has had was were been being what when
where which who whom why how all any both each few more most other some such only own same than too very can will
just don should now into over under again further then once here there about above below after before between
during out off because until while its it's they them their theirs our ours she her hers him his himself herself
itself themselves ourselves yourself also even ever every get got gets like one two three new via per let lets make
made many much must need never next still well way ways yes yet would could may might shall upon within without
video videos reel reels instagram insta follow followers following like likes share shares comment comments
subscribe viral trending trend explore explorepage fyp foryou foryoupage page watch full part link bio today now
latest update updates day days time year years people world india indian news good best shows show
trendingreels viralreels reelsinstagram instareels reelsindia reelitfeelit reelkarofeelkaro feelkaroreelkaro
instagood instadaily viralvideo viralvideos viralpost trendingnow reelsvideo shorts ytshorts youtube youtubeshorts
dekhiye dekho dekhe suno har sabse wala wale wali
del esta que los las por para con una como pero
hai hain hei ka ki ke ko se me mein par aur ye yeh wo woh kya kaise kyun kyu nahi nhi bhi tha thi the ho hota hoti
kar karo karna karte kiya raha rahe rahi sab ek liye bhai log apne apna apni aap tum hum mera meri mere tera teri
jab tab ab abhi phir sirf bas bahut koi kuch agar lekin toh hi jo jis jise uska uski uske iska iski iske inke unke
""".split()) | set("""
है हैं का की के को से में पर और यह ये वह वो क्या कैसे क्यों नहीं भी था थी थे हो होता होती कर करो करना करते किया रहा रहे रही
सब एक लिए लोग अपने अपना अपनी आप तुम हम मेरा मेरी मेरे जब तब अब अभी फिर सिर्फ बहुत कोई कुछ अगर लेकिन तो ही जो जिस उसका उसकी
उसके इसका इसकी इसके इन उन इस उस साथ बाद पहले तक गया गई गए दिया दी दे ने जा रहे हुआ हुई हुए वाले वाली वाला
""".split())


def feed_text(row: dict) -> str:
    """What a reel says, as the model reads it."""
    parts = (row.get("topic"), row.get("summary"), (row.get("caption") or "")[:300], (row.get("cover_text") or "")[:200],
             (row.get("transcript") or "")[:300])
    return ". ".join(" ".join(str(p).split()) for p in parts if p and str(p).strip())[:900]


def tokens(row: dict) -> set[str]:
    """The distinctive words of a reel (for naming topics), each counted once."""
    text = " ".join(str(row.get(key) or "") for key in ("topic", "summary", "caption", "cover_text"))
    text = NOISE.sub(" ", text.lower()).replace("#", " ")
    return {w for w in WORD.findall(text) if w not in STOPWORDS}


def load_rows(conn: sqlite3.Connection) -> list[dict]:
    """Every organic reel every account was shown (with its account and phase) that says something."""
    rows = []
    for _account, feed_rows, _state in all_feeds(conn):
        rows += [r for r in feed_rows if not r["is_ad"] and feed_text(r)]
    return rows


# ── Training ──────────────────────────────────────────────────────────

def _cached_vectors(path: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["model"]) != EMBEDDING_MODEL:
                return {}
            return dict(zip(data["codes"].tolist(), data["vectors"]))
    except (OSError, KeyError, ValueError):
        return {}


def vectors(reels: list[dict], embed_fn=None, path: str = VECTORS_PATH) -> np.ndarray:
    """Unit-length embeddings of these reels, computing only the ones not cached yet."""
    cache = _cached_vectors(path)
    missing = [r for r in reels if r["code"] not in cache]
    if missing:
        if embed_fn is None:
            from stories.grouping import embed as embed_fn
        for reel, vector in zip(missing, embed_fn([feed_text(r) for r in missing])):
            cache[reel["code"]] = np.asarray(vector, dtype=np.float32)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        codes = list(cache)
        np.savez(path, model=np.array(EMBEDDING_MODEL), codes=np.array(codes),
                 vectors=np.stack([cache[c] for c in codes]))
    matrix = np.stack([cache[r["code"]] for r in reels]).astype(np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def topic_count(reels: int) -> int:
    return int(min(MAX_TOPICS, max(4, round(math.sqrt(reels / 2)))))


def name_topics(reels: list[dict], labels: list[int], k: int, top: int = 4) -> list[dict]:
    """Each topic's most distinctive words (class-based TF-IDF over reels), its usual category and its size."""
    members: dict[int, list[dict]] = {t: [] for t in range(k)}
    for reel, label in zip(reels, labels):
        members[label].append(reel)
    words = {t: Counter(w for r in rs for w in tokens(r)) for t, rs in members.items()}
    spread = Counter(w for counts in words.values() for w in counts)  # in how many topics each word appears
    topics = []
    for t in range(k):
        size = len(members[t]) or 1
        floor = 2 if size >= 4 else 1  # a word from a single reel doesn't name a topic
        ranked = sorted((w for w, n in words[t].items() if n >= floor),
                        key=lambda w: (-(words[t][w] / size) * math.log(1 + k / spread[w]), w))
        categories = Counter(r["category"] for r in members[t] if r.get("category"))
        topics.append({
            "id": t, "size": len(members[t]), "keywords": ranked[:top],
            "category": categories.most_common(1)[0][0] if categories else None,
            "examples": [r["topic"] for r in members[t] if r.get("topic")][:3],
        })
    return topics


def train(conn: sqlite3.Connection, *, embed_fn=None, path: str = MODEL_PATH, vectors_path: str = VECTORS_PATH,
          progress=print) -> dict | None:
    """Learn the topics from every account's reels and save the model (None: too few reels yet)."""
    import joblib
    from sklearn.cluster import KMeans

    rows = load_rows(conn)
    by_code: dict[str, dict] = {}
    for row in rows:  # one entry per reel, even when it was shown more than once
        by_code.setdefault(row["code"], row)
    reels = list(by_code.values())
    if len(reels) < MIN_REELS:
        progress(f"🧠  Only {len(reels)} different reels so far; the feed model needs {MIN_REELS}.")
        return None
    progress(f"🧠  Learning the feed's topics from {len(reels)} reels across "
             f"{len({r['account'] for r in rows})} accounts …")
    matrix = vectors(reels, embed_fn, vectors_path)
    k = min(topic_count(len(reels)), len(reels))
    fitted = KMeans(n_clusters=k, n_init=10, random_state=0).fit(matrix)
    by_size = sorted(range(k), key=lambda t: (-int(np.sum(fitted.labels_ == t)), t))  # topic 0 is the biggest
    renumber = {old: new for new, old in enumerate(by_size)}
    labels = [renumber[int(label)] for label in fitted.labels_]
    model = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": EMBEDDING_MODEL, "k": k, "reels": len(reels),
        "accounts": sorted({r["account"] for r in rows}),
        "centroids": fitted.cluster_centers_[by_size],
        "topics": name_topics(reels, labels, k),
        "assign": {reel["code"]: label for reel, label in zip(reels, labels)},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    progress(f"🧠  Saved {k} topics → {path}")
    return model


_loaded: dict[str, tuple[float, dict]] = {}


def load(path: str = MODEL_PATH) -> dict | None:
    """The saved model (re-read when the file changes), or None."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _loaded.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    import joblib

    model = joblib.load(path)
    _loaded[path] = (mtime, model)
    return model


# ── What the feed is made of ──────────────────────────────────────────

def _distribution(counts: Counter, k: int) -> np.ndarray:
    total = sum(counts.values())
    return np.array([counts[t] / total for t in range(k)]) if total else np.zeros(k)


def effective_topics(counts: Counter) -> float:
    """e^entropy: how many equally common topics would make a mix this varied."""
    total = sum(counts.values())
    if not total:
        return 0.0
    shares = [n / total for n in counts.values() if n]
    return round(math.exp(-sum(p * math.log(p) for p in shares)), 2)


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance in bits: 0 for the same mix, 1 for mixes with nothing in common."""
    m = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return round(math.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2)), 3)


def topic_label(topic: dict) -> str:
    return ", ".join(topic["keywords"][:3]) or f"Topic {topic['id'] + 1}"


def describe(model: dict, rows: list[dict], account: str | None = None) -> dict:
    """One persona's feed (account) or every account's, in the model's topics. `rows`: every account's reels."""
    k, assign = model["k"], model["assign"]
    organic = [r for r in rows if not r["is_ad"]]
    mine = [r for r in organic if account is None or r["account"] == account]
    known = [r for r in mine if r["code"] in assign]
    phase = {p: Counter(assign[r["code"]] for r in known if r["phase"] == p) for p in ("warm-up", "curated")}
    current_rows = [r for r in known if r["phase"] == "curated"] or known
    current = Counter(assign[r["code"]] for r in current_rows)

    days: dict[str, list[dict]] = {}
    for row in known:
        days.setdefault((row["shown_at"] or "")[:10], []).append(row)
    daily, before = [], None
    for day in sorted(d for d in days if d and len(days[d]) >= DAY_MIN_REELS):
        counts = Counter(assign[r["code"]] for r in days[day])
        mix = _distribution(counts, k)
        daily.append({"day": day, "reels": len(days[day]), "effective_topics": effective_topics(counts),
                      "moved": js_distance(before, mix) if before is not None else None})
        before = mix

    rising = []
    dated = sorted(d for d in days if d)
    if len(dated) > RECENT_DAYS:
        recent_days = set(dated[-RECENT_DAYS:])
        recent = Counter(assign[r["code"]] for r in known if (r["shown_at"] or "")[:10] in recent_days)
        earlier = Counter(assign[r["code"]] for r in known if (r["shown_at"] or "")[:10] not in recent_days)
        now, then = _distribution(recent, k), _distribution(earlier, k)
        rising = [{"topic": t, "from": round(float(then[t]), 3), "to": round(float(now[t]), 3)}
                  for t in np.argsort(then - now) if now[t] - then[t] >= 0.03 and recent[t] >= 3][:5]
        rising = [{**item, "topic": int(item["topic"])} for item in rising]

    out = {
        "trained": True, "created": model["created"], "k": k, "trained_reels": model["reels"],
        "trained_accounts": model["accounts"],
        "topics": [{**topic, "label": topic_label(topic)} for topic in model["topics"]],
        "reels": len(mine), "known": len(known), "newer": len(mine) - len(known),
        "mix": {p: {str(t): n for t, n in counts.items()} for p, counts in phase.items()},
        "effective_topics": effective_topics(current), "of": "curated" if phase["curated"] else "all",
        "daily": daily, "rising": rising,
    }
    everyone = Counter(assign[r["code"]] for r in organic if r["code"] in assign)
    if account is not None:
        shares, overall = _distribution(current, k), _distribution(everyone, k)
        apart = [{"topic": int(t), "share": round(float(shares[t]), 3), "overall": round(float(overall[t]), 3)}
                 for t in range(k) if current[t] >= 3 and overall[t] > 0 and shares[t] >= 1.3 * overall[t]]
        out["apart"] = sorted(apart, key=lambda a: a["overall"] - a["share"])[:5]  # most points above overall first
    else:
        accounts = sorted({r["account"] for r in organic})
        feeds = {}
        for name in accounts:
            theirs = [r for r in organic if r["account"] == name and r["code"] in assign]
            curated = [r for r in theirs if r["phase"] == "curated"] or theirs
            if curated:
                feeds[name] = _distribution(Counter(assign[r["code"]] for r in curated), k)
        names = list(feeds)
        out["alike"] = {"accounts": names,
                        "matrix": [[round(1 - js_distance(feeds[a], feeds[b]), 3) for b in names] for a in names]}
    return out


def analysis(conn: sqlite3.Connection, account: str | None = None, path: str = MODEL_PATH) -> dict:
    """What the panel shows: describe() when a model is saved, else how far there is to go."""
    model = load(path)
    rows = [r for _a, feed_rows, _s in all_feeds(conn) for r in feed_rows]
    if model is None:
        different = len({r["code"] for r in rows if not r["is_ad"] and feed_text(r)})
        return {"trained": False, "reels": different, "needed": MIN_REELS}
    return describe(model, rows, account)


# ── Command line ──────────────────────────────────────────────────────

def print_analysis(info: dict, account: str | None) -> None:
    if not info["trained"]:
        print(f"🧠  No feed model yet ({info['reels']} of the {info['needed']} reels it needs): run with --train.")
        return
    topics = {t["id"]: t for t in info["topics"]}
    who = f"@{account}'s feed" if account else "Every account's feed"
    print(f"🧠  {who}: {info['known']} reels in {info['k']} topics learned {info['created'][:16]} UTC"
          + (f" ({info['newer']} newer reels aren't in the model yet)" if info["newer"] else ""))
    print(f"    Spread over about {info['effective_topics']} topics ({info['of']} reels)")
    counts = Counter({int(t): n for p in ("warm-up", "curated") for t, n in info["mix"][p].items()})
    total = sum(counts.values())
    for t, n in counts.most_common(10):
        category = CATEGORIES.get(topics[t]["category"], ("", ""))[0]
        print(f"    {n / total:5.0%}  {topic_label(topics[t])[:40]:40} {category}")
    for day in info["daily"]:
        moved = "" if day["moved"] is None else f", moved {day['moved']:.2f}"
        print(f"    {day['day']}: {day['reels']} reels, about {day['effective_topics']} topics{moved}")
    for item in info["rising"]:
        print(f"    Rising: {topic_label(topics[item['topic']])} {item['from']:.0%} → {item['to']:.0%}")
    for item in info.get("apart", []):
        print(f"    More than overall: {topic_label(topics[item['topic']])} {item['share']:.0%} vs {item['overall']:.0%}")
    alike = info.get("alike")
    if alike and len(alike["accounts"]) > 1:
        print("    How alike the feeds are:")
        for name, row in zip(alike["accounts"], alike["matrix"]):
            print(f"      @{name:18} " + " ".join(f"{v:5.0%}" for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="The feed model: what the algorithm serves each persona")
    parser.add_argument("--train", action="store_true", help="Learn the topics from every account's reels")
    parser.add_argument("--account", help="One account's feed (default: every account's)")
    args = parser.parse_args()
    init_db()
    conn = get_connection()
    try:
        if args.train:
            train(conn)
        print_analysis(analysis(conn, args.account), args.account)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
