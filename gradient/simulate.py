"""
PRAG — 14 days of simulated sessions for both personas.

Every record this writes is a simulation, kept in its own database
(database.db.DEMO_DB_PATH) and shown in the panel only under the Simulated
data source, with a "Simulated sessions" banner. Nothing here is an Instagram
observation, and it never touches reels.db.

How a simulated session is made, for each account and each of the 14 complete
days before today (SESSIONS: six a day, three scheduled and three opened by
hand, about 3.7 hours of scrolling in all, a heavy user's day, so the two
accounts' 14 days come to 12,000+ impressions):

  serve     a simulated recommender picks each next reel from the training
            corpus's written reels (gradient/store.py). Reels about a real
            news story only appear from the day it was published, while it's
            fresh. Like a real recommender it never sees the persona's scores:
            it learns only from behaviour, an affinity per topic and per
            political stance that rises when the persona watches, likes or
            saves and falls when it skips. Each candidate is weighted by a
            cold-start prior (mass-appeal topics, the feed of an account with no
            history), fading over the 14 days, times its affinities raised to
            SHARPNESS/T, where T falls from the persona's start to end temperature, so
            the feed narrows as the recommender learns. A share of reels is
            served at random (exploration), and every so often an ad.
  judge     the score is Jev's judgement of the reel for this persona (the
            corpus labels): the same score a live session's judge gives
  act       the session's own engagement policy (agent/bandit.decide) turns
            the score into skip / watch / full watch / rewatch, dwell time,
            likes, saves and follows, with the per-session caps
  log       the session's console log, line for line in main.py's format,
            each line timestamped with its simulated time

Each session stores its simulated start and end (shown as the session's
time, labelled simulated) and, separately, when it was actually generated.
Daily digests are built from each simulated day's impressions with the
digest's own code; a news reel's source reference is the real article it was
written about. A feed-topic model is trained on the simulated impressions.

    python -m gradient.simulate [--days 14] [--sessions 6] [--seed 1] [--fresh]   a new simulation
    python -m gradient.simulate --append [--seed 2]     more sessions on the same days, existing ones kept
"""

import argparse
import datetime
import json
import math
import os
import random
import re
import shutil
import sqlite3

import numpy as np

from agent import bandit
from agent import bias as biases
from config import MAX_FOLLOWS_PER_SESSION, MAX_LIKES_PER_SESSION, MAX_SAVES_PER_SESSION, WATCH_THRESHOLD
from database import db
from gradient import store, taxonomy
from perception import jev

DEMO_DIR = os.path.dirname(db.DEMO_DB_PATH)
DEMO_DIGEST_DIR = os.path.join(DEMO_DIR, "digests")
DEMO_FEED_MODEL = os.path.join(DEMO_DIR, "feed-model.joblib")
DEMO_FEED_VECTORS = os.path.join(DEMO_DIR, "feed-vectors.npz")
EPSILON = 0.10
# What an account with no history is shown first: mass-appeal content, not the persona's (it fades as the
# recommender learns the persona's own engagement)
COLD_START = {"hindi_comedy": 3.0, "film_tv": 3.0, "fashion_beauty": 3.0, "lifestyle": 2.5, "astrology": 2.5,
              "promotion": 2.0, "street_food": 2.0, "religion": 2.0, "cricket": 2.0, "indie_music": 1.5, "gaming": 1.5,
              "satire": 1.5, "travel_nature": 1.5}
SHARPNESS = 0.3  # how hard the learned affinities steer: low enough that a feed narrows without collapsing
LEARN = {"SKIP": 0.965, "WATCH": 1.02, "FULL_WATCH": 1.05, "REWATCH": 1.08}
# (local start, how it was started, its length in minutes): the scheduled runs are longer than the ones opened by hand
SESSIONS = ((datetime.time(7, 15), "scheduled", (40, 55)), (datetime.time(10, 30), "manual", (20, 35)),
            (datetime.time(13, 0), "scheduled", (40, 55)), (datetime.time(16, 30), "manual", (20, 35)),
            (datetime.time(19, 45), "scheduled", (40, 55)), (datetime.time(22, 30), "manual", (20, 35)))
AD_EVERY = (9, 16)
FRESH_DAYS = 3
INFORMATIONAL = {"news_politics", "business_money", "tech_science", "education"}

TABLES = """
CREATE TABLE IF NOT EXISTS demo_sessions (
    session_id      INTEGER PRIMARY KEY,
    account         TEXT NOT NULL,
    simulated_date  TEXT NOT NULL,       -- the local calendar day it simulates
    simulated_start TEXT NOT NULL,       -- UTC, as sessions.started_at
    simulated_end   TEXT NOT NULL,
    kind            TEXT,                -- scheduled or manual
    batch           TEXT,                -- which simulation run made it: "base", or an appended batch
    generated_at    TEXT NOT NULL,       -- when this simulation was actually made (UTC)
    seed            TEXT NOT NULL,
    temperature     REAL,
    stats           TEXT
);
CREATE TABLE IF NOT EXISTS demo_log (
    session_id INTEGER NOT NULL,
    n          INTEGER NOT NULL,
    at         TEXT NOT NULL,            -- simulated time (UTC)
    line       TEXT NOT NULL,
    PRIMARY KEY (session_id, n)
);
CREATE TABLE IF NOT EXISTS demo_items (
    code        TEXT PRIMARY KEY,        -- the impression's code
    item_id     TEXT NOT NULL,           -- the corpus item (datasets/gradient/corpus.db)
    news_title  TEXT,                    -- for a reel about a real story: the article it was written about
    news_source TEXT,
    news_url    TEXT,
    news_date   TEXT
);
"""


def utc(ts: datetime.datetime) -> str:
    return ts.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def days_back(n: int, today: datetime.date | None = None) -> list[datetime.date]:
    """The n complete calendar days before today, oldest first."""
    today = today or datetime.date.today()
    return [today - datetime.timedelta(days=k) for k in range(n, 0, -1)]


def first_sentence(text: str, limit: int = 150) -> str:
    text = re.sub(r"#\S+", "", text or "").strip()
    text = " ".join(text.split())
    m = re.match(r"(.{15,}?[.!?।])(\s|$)", text)
    return (m.group(1) if m else text)[:limit]


# ── The content pool ──────────────────────────────────────────────────

def pool(corpus: sqlite3.Connection, account: str) -> dict:
    """Every written reel, with this persona's Jev score, topic and (for news reels) the story's date."""
    fingerprint = biases.load(account).fingerprint
    rows = corpus.execute(
        "SELECT i.*, l.score AS s FROM items i JOIN labels l ON l.item_id = i.id AND l.account = ? AND l.teacher = 'jev' "
        "AND l.persona_version = ? WHERE i.origin = 'generated' GROUP BY i.id", (account, fingerprint)).fetchall()
    items = []
    for r in rows:
        meta = json.loads(r["meta"] or "{}")
        published = meta.get("news_published_at")
        items.append({**dict(r), "meta": meta,
                      "news_date": datetime.datetime.fromisoformat(published).date() if published else None})
    return {"items": items, "score": np.array([float(i["s"] or 0.3) for i in items]),
            "topics": [i["topic"] or "other" for i in items],
            "news": np.array([i["news_date"] is not None for i in items])}


# ── One session ───────────────────────────────────────────────────────

def reel_line(seq: int, item: dict | None, score: float | None, decision, dwell: float, done: dict,
              duration: int, audio: str) -> str:
    """The console line main.py prints for a reel (⚡ flicked, ❤️ 🔖 ➕ engagement, 🧭 explored)."""
    clip = lambda t, w: t if len(t) <= w else t[:w - 1] + "…"  # noqa: E731
    if item is None:
        who, kind, verdict = "(ad)", "", "ad"
    else:
        who = "@" + (item["creator"] or "unknown").lstrip("@")
        kind = f"{audio} {duration}s"
        verdict = f"{score:.2f} {topic_of(item)}"
    marks = " ".join(m for m, on in (("❤️", done["liked"]), ("🔖", done["saved"]), ("➕", done["followed"])) if on)
    if decision.explored:
        marks = (marks + " 🧭").strip()
    if decision.action == "SKIP" and item is not None and done["flicked"]:
        marks = ("⚡ " + marks).strip()
    line = f"#{seq:<4} {clip(who, 26):26} {kind:12} {clip(verdict, 36):36} → {decision.action:10} {dwell:5.1f}s {marks}"
    if item is not None and decision.action != "SKIP":
        line += f"\n        “{clip(summary_of(item), 110)}”"
    return line


def summary_of(item: dict) -> str:
    """What the reel says: its first spoken sentence, else its caption's."""
    return first_sentence(item.get("transcript") or "") or first_sentence(item.get("caption") or item["text"])


def topic_of(item: dict) -> str:
    """A topic the way the live judge writes one: the story a news reel is about, else the reel's angle."""
    meta = item["meta"]
    title = meta.get("news_title") or meta.get("angle") or (item["topic"] or "other").replace("_", " ")
    return title if len(title) <= 80 else title[:79].rsplit(" ", 1)[0] + "…"


def simulate_session(conn, corpus_pool: dict, state: dict, account: str, day: datetime.date, slot: int, d_index: int,
                     n_days: int, rng: random.Random, scorer: str, generated_at: str, plan: tuple | None = None,
                     batch: str = "base") -> dict:
    """One simulated session. `plan`: (start, kind, minutes), else the day's slot in SESSIONS."""
    bias = biases.load(account)
    profile = bias.profile or {}
    explore = profile.get("exploration") or {}
    t0, t1 = float(explore.get("start_temperature", 0.8)), float(explore.get("end_temperature", 0.15))
    temperature = t0 + (t1 - t0) * (d_index / max(n_days - 1, 1))
    if plan:
        start, kind, minutes = plan
    else:
        at, kind, (shortest, longest) = SESSIONS[slot % len(SESSIONS)]
        local_start = datetime.datetime.combine(day, at).astimezone()
        start = local_start + datetime.timedelta(minutes=rng.randint(0, 60))
        minutes = rng.uniform(shortest, longest)
    cur = conn.execute("INSERT INTO sessions (account, started_at, dry_run, model, scorer) VALUES (?, ?, 0, ?, ?)",
                       (account, utc(start), "demo-simulation", scorer))
    session_id = cur.lastrowid
    log: list[tuple[datetime.datetime, str]] = [
        (start, f"🧪  Simulated {kind} session {session_id} for @{account}, simulated {start:%a %d %b %Y %H:%M} "
                f"(generated {generated_at} UTC)"),
        (start, f"🎭  Persona: {bias.name}"),
        (start, f"🧭  Recommender temperature {temperature:.2f} (day {d_index + 1} of {n_days}), exploration {EPSILON:.0%}"),
    ]
    items, scores, topics, news = corpus_pool["items"], corpus_pool["score"], corpus_pool["topics"], corpus_pool["news"]
    served = state["served"]
    fresh = np.array([(i["news_date"] is not None and 0 <= (day - i["news_date"]).days <= FRESH_DAYS) for i in items])
    eligible = (~news | fresh) & ~served
    clock, seq = start, 0
    likes = saves = follows = 0
    tally = {"SKIP": 0, "WATCH": 0, "FULL_WATCH": 0, "REWATCH": 0, "liked": 0, "saved": 0, "followed": 0, "flicked": 0}
    next_ad = rng.randint(*AD_EVERY)
    end_at = start + datetime.timedelta(minutes=minutes)
    while clock < end_at:
        seq += 1
        is_ad = seq == next_ad
        if is_ad:
            next_ad += rng.randint(*AD_EVERY)
            item, score = None, None
            decision = bandit.decide(value_score=None, basis="ad", duration=None, has_liked=False, has_saved=False,
                                     follow_ok=False, likes_left=0, saves_left=0, rng=rng)
            duration, audio = rng.randint(10, 30), "original"
        else:
            idx = np.flatnonzero(eligible)
            if not len(idx):
                break
            if rng.random() < EPSILON:
                k = int(idx[rng.randrange(len(idx))])
            else:
                aff = np.array([state["affinity"].get(topics[i], 1.0) * state["stance"].get(items[i]["stance"], 1.0)
                                for i in idx])
                fade = d_index / max(n_days - 1, 1)  # the cold-start prior's hold: 1 on day one, 0 on the last
                prior = np.array([COLD_START.get(topics[i], 0.6) for i in idx]) ** (1.0 - fade)
                w = prior * aff ** (SHARPNESS / temperature) * np.where(fresh[idx], 1.5, 1.0)
                k = int(idx[rng.choices(range(len(idx)), weights=w)[0]])
            eligible[k] = False
            served[k] = True
            item, score = items[k], float(scores[k])
            duration = rng.randint(14, 95)
            audio = "music" if item.get("format") in ("meme edit", "skit", "vlog") and rng.random() < 0.6 else "original"
            creator = item["creator"] or ""
            state["creator_hits"][creator] = state["creator_hits"].get(creator, 0) + (score >= 0.75)
            follow_ok = (state["creator_hits"][creator] >= 2 and follows < MAX_FOLLOWS_PER_SESSION
                         and creator not in state["followed"])
            decision = bandit.decide(value_score=score, basis="score", duration=duration, has_liked=False,
                                     has_saved=False, follow_ok=follow_ok, likes_left=MAX_LIKES_PER_SESSION - likes,
                                     saves_left=MAX_SAVES_PER_SESSION - saves, rng=rng, pre_analyzed=rng.random() < 0.7)
            # what the recommender learns from: only the behaviour, never the score
            signal = LEARN[decision.action] * (1.05 if decision.like else 1.0) * (1.05 if decision.save else 1.0)
            for table, key in ((state["affinity"], topics[k]), (state["stance"], item["stance"])):
                table[key] = min(8.0, max(0.1, table.get(key, 1.0) * signal))
        dwell = float(decision.dwell)
        done = {"liked": decision.like, "saved": decision.save, "followed": decision.follow,
                "flicked": decision.action == "SKIP" and not is_ad and dwell < 1.2}
        likes, saves, follows = likes + done["liked"], saves + done["saved"], follows + done["followed"]
        if done["followed"]:
            state["followed"].add(item["creator"])
        tally[decision.action] += 1
        for key in ("liked", "saved", "followed", "flicked"):
            tally[key] += bool(done[key])
        log.append((clock, reel_line(seq, item, score, decision, dwell, done, duration, audio)))
        code = f"demo{store.text_hash(item['id'])[:12]}" if item else f"demoad{session_id}x{seq}"
        meta = item["meta"] if item else {}
        db.log_impression({
            "session_id": session_id, "seq": seq, "code": code, "url": meta.get("news_url") or item.get("source_url")
            if item else None, "creator": (item["creator"] or "").lstrip("@") if item else None,
            "caption": item.get("caption") if item else None,
            "hashtags": " ".join(re.findall(r"#\w+", item.get("caption") or "")) if item else None,
            "audio_type": audio, "duration": duration, "like_count": int(rng.lognormvariate(7, 1.6)) if item else None,
            "comment_count": int(rng.lognormvariate(4, 1.5)) if item else None,
            "taken_at": int(datetime.datetime.combine(item["news_date"] or day, datetime.time(9)).timestamp())
            if item else None, "inventory_source": "demo-simulation", "is_ad": int(is_ad),
            "transcript": item.get("transcript") if item else None,
            "transcript_language": {"hi": "hi", "hinglish": "hi"}.get(item.get("language"), "en") if item else None,
            "cover_text": item.get("on_screen") if item else None, "score_basis": "ad" if is_ad else "score",
            "value_score": score, "is_informational": int(bool(item) and taxonomy.category(item["topic"]) in INFORMATIONAL),
            "topic": topic_of(item) if item else None,
            "summary": summary_of(item) if item else None, "reasoning": "Simulated session: Jev's score for this persona",
            "scored_at": utc(clock), "decision": decision.action, "decision_basis": decision.basis,
            "explored": int(decision.explored), "dwell_seconds": round(dwell, 2), "planned": decision.planned,
            "liked": int(done["liked"]), "saved": int(done["saved"]), "followed": int(done["followed"]),
            "ready_on_arrival": int(done["flicked"]), "stance": item.get("stance") if item else None,
        }, conn)
        conn.execute("UPDATE impressions SET shown_at = ?, category = ? WHERE session_id = ? AND seq = ?",
                     (utc(clock), taxonomy.category(item["topic"]) if item else None, session_id, seq))
        if item:
            conn.execute("INSERT OR IGNORE INTO demo_items VALUES (?, ?, ?, ?, ?, ?)",
                         (code, item["id"], meta.get("news_title"), meta.get("news_source"),
                          item.get("source_url") if meta.get("news_title") else None, meta.get("news_published_at")))
        clock += datetime.timedelta(seconds=dwell + rng.uniform(0.6, 2.2))
        if not is_ad:
            total = state["judged"]
            total.append(score >= WATCH_THRESHOLD)
            window = total[-100:]
            if not state["curated"] and len(window) == 100 and sum(window) >= 50:
                state["curated"] = True
                log.append((clock, f"🎯  The feed looks curated: {sum(window)} of the last 100 judged reels fit the persona"))
    minutes_done = (clock - start).total_seconds() / 60
    conn.execute("UPDATE sessions SET ended_at = ?, reels = ? WHERE id = ?", (utc(clock), seq, session_id))
    log += [
        (clock, f"🏁  Session {session_id} — {seq} reels in {minutes_done:.1f} min (simulated)"),
        (clock, f"    decisions: SKIP {tally['SKIP']} · WATCH {tally['WATCH']} · FULL_WATCH {tally['FULL_WATCH']}"
                f" · REWATCH {tally['REWATCH']}"),
        (clock, f"    engagement: ❤️ {tally['liked']} · 🔖 {tally['saved']} · ➕ {tally['followed']}"),
    ]
    rows, n = [], 0
    for at, text in log:
        for line in text.split("\n"):
            rows.append((session_id, n, utc(at), line))
            n += 1
    conn.executemany("INSERT INTO demo_log VALUES (?, ?, ?, ?)", rows)
    conn.execute("INSERT INTO demo_sessions (session_id, account, simulated_date, simulated_start, simulated_end, kind, "
                 "generated_at, seed, temperature, stats, batch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 (session_id, account, day.isoformat(), utc(start), utc(clock), kind, generated_at,
                  f"{state['seed']}|{batch}|{account}|{day}|{slot}", round(temperature, 3),
                  json.dumps({**tally, "reels": seq}), batch))
    return {"session_id": session_id, "reels": seq, "kind": kind, **tally}


# ── Digests ───────────────────────────────────────────────────────────

def demo_digest(conn, account: str, day: datetime.date, generated_at: str, **group_kwargs) -> str | None:
    """That simulated day's digest, built by stories/digest.py from its impressions; news reels cite their article."""
    from stories import digest as digests
    from stories.verify import Article, Verification

    end = datetime.datetime.combine(day, datetime.time(23, 59, 59)).astimezone()
    built = digests.build_digest(conn, account, hours=24, verify=False, now=end, **group_kwargs)
    if not built.reels_seen:
        return None
    sources = {r[0]: r[1:] for r in conn.execute("SELECT code, news_title, news_source, news_url, news_date FROM demo_items")}
    checks = []
    for story in built.stories:
        refs = [sources.get(r.code) for r in story.reels]
        refs = [x for x in refs if x and x[0]]
        if refs:
            outlets = sorted({x[1] for x in refs if x[1]})
            checks.append(Verification("reported", query=refs[0][0], outlets=outlets,
                                       headlines=[Article(title=x[0], source=x[1] or "", url=x[2] or "",
                                                          published=(x[3] or "")[:10]) for x in refs[:3]]))
        else:
            checks.append(Verification("not_checked"))
    built.checks = checks
    folder = os.path.join(DEMO_DIGEST_DIR, account)
    os.makedirs(folder, exist_ok=True)
    data = digests.to_json(built)
    data.update({"demo": True, "simulated_date": day.isoformat(), "generated_at": generated_at + "Z"})
    page = digests.render_html(built).replace(
        "<main>", '<main><div style="background:#f59e0b;color:#111;padding:8px 12px;font-weight:700;margin-bottom:14px">'
                  f"Simulated sessions. Simulated day {day:%d %B %Y}; generated {generated_at} UTC.</div>", 1)
    for reel in (r for s in data["stories"] for r in s["reels"]):
        if reel["url"].startswith("https://www.instagram.com/reels/demo"):
            reel["url"] = ""
    page = re.sub(r'href="https://www\.instagram\.com/reels/demo[^"]*"', 'href="#"', page)
    with open(os.path.join(folder, f"{day.isoformat()}.json"), "w") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(folder, f"{day.isoformat()}.html"), "w") as fh:
        fh.write(page)
    return day.isoformat()


# ── Everything ────────────────────────────────────────────────────────

def build(days: int = 14, sessions_per_day: int = len(SESSIONS), seed: int = 1, progress=print, today=None,
          fresh: bool = False, **group_kwargs) -> dict:
    """A fresh simulation database: accounts, 14 days of sessions with logs, digests, and its feed model. It replaces
    an existing one only with fresh=True (python -m gradient.simulate --fresh); to add to one, use append()."""
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(db.DEMO_DB_PATH) and not fresh:
        raise RuntimeError(f"{db.DEMO_DB_PATH} already holds simulated sessions: append to it (--append), or pass "
                           "--fresh to replace it")
    if os.path.exists(DEMO_DIR):
        for name in ("reels.db", "reels.db-wal", "reels.db-shm"):
            if os.path.exists(os.path.join(DEMO_DIR, name)):
                os.remove(os.path.join(DEMO_DIR, name))
        shutil.rmtree(DEMO_DIGEST_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR, exist_ok=True)
    conn = db.get_connection(db.DEMO_DB_PATH)
    db.init_db(conn)
    conn.executescript(TABLES)
    corpus = store.connect()
    out = {"generated_at": generated_at, "accounts": {}}
    span = days_back(days, today)
    for account in store.ACCOUNTS:
        conn.execute("INSERT OR IGNORE INTO accounts (username, password, status) VALUES (?, '', 'ACTIVE')", (account,))
        corpus_pool = pool(corpus, account)
        bias = biases.load(account)
        scorer = jev.jev_version(bias.interests, persona=True, vision=False)
        state = {"served": np.zeros(len(corpus_pool["items"]), dtype=bool), "affinity": {}, "stance": {}, "creator_hits": {},
                 "followed": set(), "judged": [], "curated": False, "seed": seed}
        made = []
        for d_index, day in enumerate(span):
            for slot in range(sessions_per_day):
                rng = random.Random(f"{seed}|{account}|{day}|{slot}")
                made.append(simulate_session(conn, corpus_pool, state, account, day, slot, d_index, days, rng, scorer,
                                             generated_at))
            conn.commit()
        progress(f"@{account}: {len(made)} simulated sessions, {sum(m['reels'] for m in made)} reels, "
                 f"{sum(m['liked'] for m in made)} likes; curated: {state['curated']}")
        out["accounts"][account] = {"sessions": len(made), "reels": sum(m["reels"] for m in made)}
    finish(conn, span, generated_at, out, progress, **group_kwargs)
    return out


def finish(conn, span: list[datetime.date], generated_at: str, out: dict, progress=print, **group_kwargs) -> None:
    """After sessions are added: every day's digest from all its sessions, the feed-topic model, the corpus feeds."""
    for account in store.ACCOUNTS:
        written = [d for d in (demo_digest(conn, account, day, generated_at, **group_kwargs) for day in span) if d]
        progress(f"@{account}: {len(written)} daily digests")
    conn.close()
    from ml import feed_model

    conn = db.get_connection(db.DEMO_DB_PATH)
    feed_model.train(conn, path=DEMO_FEED_MODEL, vectors_path=DEMO_FEED_VECTORS)
    conn.close()
    progress("Feed-topic model of the simulated sessions trained.")
    from gradient import ingest

    corpus = store.connect()
    out["corpus_feed"] = ingest.ingest_demo(corpus)
    corpus.close()
    progress(f"Served reels joined the personas' feeds in the training corpus: {out['corpus_feed']}")


# ── Adding to an existing simulation ──────────────────────────────────

# Sessions append() adds around the existing ones (07:30, 13:00, 20:00 plus up to 100 min, about 25 min each):
# (local start, how it was started, minutes); each starts up to APPEND_JITTER minutes late
APPEND_SESSIONS = ((datetime.time(6, 0), "manual", (30, 45)), (datetime.time(10, 0), "scheduled", (45, 60)),
                   (datetime.time(15, 20), "manual", (45, 60)), (datetime.time(17, 30), "scheduled", (40, 55)),
                   (datetime.time(22, 15), "manual", (35, 50)))
APPEND_JITTER = 25


def replay(conn, corpus_pool: dict, state: dict, session_id: int, index: dict) -> None:
    """What the recommender learned from an existing session, in the order it happened: the reels it served, the
    affinities its engagement taught, the creators it followed, and whether the feed counted as curated."""
    rows = conn.execute("SELECT i.decision, i.liked, i.saved, i.followed, i.value_score, d.item_id FROM impressions i "
                        "LEFT JOIN demo_items d ON d.code = i.code WHERE i.session_id = ? AND i.is_ad = 0 ORDER BY i.seq",
                        (session_id,)).fetchall()
    items, topics = corpus_pool["items"], corpus_pool["topics"]
    for decision, liked, saved, followed, score, item_id in rows:
        k = index.get(item_id)
        if k is None:  # a reel no longer in the corpus pool: nothing to learn about it
            continue
        state["served"][k] = True
        item = items[k]
        signal = LEARN.get(decision, 1.0) * (1.05 if liked else 1.0) * (1.05 if saved else 1.0)
        for table, key in ((state["affinity"], topics[k]), (state["stance"], item["stance"])):
            table[key] = min(8.0, max(0.1, table.get(key, 1.0) * signal))
        creator = item["creator"] or ""
        state["creator_hits"][creator] = state["creator_hits"].get(creator, 0) + ((score or 0) >= 0.75)
        if followed:
            state["followed"].add(item["creator"])
        state["judged"].append((score or 0) >= WATCH_THRESHOLD)
        window = state["judged"][-100:]
        if len(window) == 100 and sum(window) >= 50:
            state["curated"] = True


def free_start(start: datetime.datetime, minutes: float, busy: list[tuple], rng: random.Random) -> datetime.datetime:
    """The planned start, or, when that would overlap a session already on the account's day, just after it."""
    length = datetime.timedelta(minutes=minutes + 3)  # its last reel can run past the planned length
    moved = True
    while moved:
        moved = False
        for b_start, b_end in sorted(busy):
            if start < b_end and b_start < start + length:
                start, moved = b_end + datetime.timedelta(minutes=rng.randint(5, 15)), True
    return start


def append(seed: int = 2, progress=print, **group_kwargs) -> dict:
    """More simulated sessions on the same days as an existing simulation, which is kept as it is: every existing
    session, impression, log line and its generation time stay; the new ones record their own generation time and
    batch. The recommender carries on from what the existing sessions taught it, day by day in time order."""
    if not os.path.exists(db.DEMO_DB_PATH):
        raise RuntimeError("No simulation to add to yet: python -m gradient.simulate")
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    batch = f"append-{generated_at[:10]}-s{seed}"
    conn = db.get_connection(db.DEMO_DB_PATH)
    conn.row_factory = sqlite3.Row
    columns = {r[1] for r in conn.execute("PRAGMA table_info(demo_sessions)")}
    for column in ("kind", "batch"):  # older simulations: add the columns, leave their rows' values empty
        if column not in columns:
            conn.execute(f"ALTER TABLE demo_sessions ADD COLUMN {column} TEXT")
    span = [datetime.date.fromisoformat(d) for (d,) in
            conn.execute("SELECT DISTINCT simulated_date FROM demo_sessions ORDER BY 1")]
    if conn.execute("SELECT 1 FROM demo_sessions WHERE batch = ?", (batch,)).fetchone():
        raise RuntimeError(f"Batch {batch} was already added today: pick another --seed")
    before = conn.execute("SELECT COUNT(*) FROM impressions").fetchone()[0]
    corpus = store.connect()
    out = {"generated_at": generated_at, "batch": batch, "days": [span[0].isoformat(), span[-1].isoformat()],
           "accounts": {}}
    for account in store.ACCOUNTS:
        corpus_pool = pool(corpus, account)
        index = {item["id"]: k for k, item in enumerate(corpus_pool["items"])}
        bias = biases.load(account)
        scorer = jev.jev_version(bias.interests, persona=True, vision=False)
        state = {"served": np.zeros(len(corpus_pool["items"]), dtype=bool), "affinity": {}, "stance": {},
                 "creator_hits": {}, "followed": set(), "judged": [], "curated": False, "seed": seed}
        existing = conn.execute("SELECT session_id, simulated_date, simulated_start, simulated_end FROM demo_sessions "
                                "WHERE account = ? ORDER BY simulated_start", (account,)).fetchall()
        # every reel the existing sessions served, on any day, is theirs: a new session on an earlier day mustn't show
        # one of them first (the existing history was made without knowing about it)
        for (item_id,) in conn.execute("SELECT DISTINCT d.item_id FROM impressions i JOIN sessions s ON s.id = i.session_id "
                                       "JOIN demo_items d ON d.code = i.code WHERE s.account = ?", (account,)):
            if item_id in index:
                state["served"][index[item_id]] = True
        aware = lambda text: datetime.datetime.fromisoformat(text).replace(tzinfo=datetime.timezone.utc)  # noqa: E731
        made = []
        for d_index, day in enumerate(span):
            today = [r for r in existing if r["simulated_date"] == day.isoformat()]
            events = [(aware(r["simulated_start"]), "existing", r["session_id"]) for r in today]
            busy = [(aware(r["simulated_start"]), aware(r["simulated_end"])) for r in today]
            day_end = datetime.datetime.combine(day, datetime.time(23, 59)).astimezone()
            for slot, (at, kind, (shortest, longest)) in enumerate(APPEND_SESSIONS):
                rng = random.Random(f"{seed}|{batch}|{account}|{day}|{slot}")
                start = (datetime.datetime.combine(day, at).astimezone()
                         + datetime.timedelta(minutes=rng.randint(0, APPEND_JITTER)))
                minutes = rng.uniform(shortest, longest)
                start = free_start(start, minutes, busy, rng)
                if start + datetime.timedelta(minutes=minutes + 3) > day_end:
                    continue  # no room left in the day for it
                busy.append((start, start + datetime.timedelta(minutes=minutes + 3)))
                events.append((start, "new", (slot, rng, (start, kind, minutes))))
            for _at, what, payload in sorted(events, key=lambda e: e[0]):
                if what == "existing":
                    replay(conn, corpus_pool, state, payload, index)
                else:
                    slot, rng, plan = payload
                    made.append(simulate_session(conn, corpus_pool, state, account, day, slot, d_index, len(span), rng,
                                                 scorer, generated_at, plan=plan, batch=batch))
            conn.commit()
        progress(f"@{account}: {len(made)} more simulated sessions, {sum(m['reels'] for m in made)} reels, "
                 f"{sum(m['liked'] for m in made)} likes")
        out["accounts"][account] = {"sessions": len(made), "reels": sum(m["reels"] for m in made)}
    corpus.close()
    out["impressions"] = {"before": before, "added": conn.execute("SELECT COUNT(*) FROM impressions").fetchone()[0] - before}
    conn.row_factory = None
    finish(conn, span, generated_at, out, progress, **group_kwargs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="14 days of simulated sessions for both personas")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--sessions", type=int, default=len(SESSIONS), help="sessions per day per account")
    parser.add_argument("--seed", type=int, default=None, help="default: 1 for a new simulation, 2 for --append")
    parser.add_argument("--append", action="store_true",
                        help="add sessions around the existing ones on the same days, keeping every existing record")
    parser.add_argument("--fresh", action="store_true", help="replace an existing simulation")
    args = parser.parse_args()
    say = lambda m: print(m, flush=True)  # noqa: E731
    if args.append:
        out = append(2 if args.seed is None else args.seed, progress=say)
    else:
        out = build(args.days, args.sessions, 1 if args.seed is None else args.seed, progress=say, fresh=args.fresh)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
