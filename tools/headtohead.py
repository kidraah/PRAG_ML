"""
PRAG — Head-to-head: Jev or the local LLM, which judges a persona better?

Picks a fixed sample of stored reels (varied by category, with enough news
and politics for political personas), and scores each one as several
personas with both Jev and the local LLM. You label the same reels in the
panel's Label tab — would this person watch it to the end, or scroll past?
— without seeing either answer, and results() says which model agrees with
you more (the watch / skip call at WATCH_THRESHOLD), Hindi reels included.

The sample stays fixed once picked, so labels stay valid; running again
only fills in missing scores (--new picks a new sample, dropping labels).

    python -m tools.headtohead [--personas news-follower,right-leaning,left-leaning] [--reels 40] [--new]
"""

import argparse
import asyncio
import json
import os
import random
import re
import sqlite3
import time
from types import SimpleNamespace

import aiohttp
from dotenv import load_dotenv

from agent import bias as biases
from config import WATCH_THRESHOLD
from database.db import get_connection
from perception import jev
from perception.local_evaluator import ScoringError, build_user_message, evaluate_value_async, llm_serving

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H2H_DIR = os.path.join(PROJECT_ROOT, "logs", "headtohead")
RUN_PATH = os.path.join(H2H_DIR, "run.json")
LABELS_PATH = os.path.join(H2H_DIR, "labels.json")
DEFAULT_PERSONAS = ("news-follower", "right-leaning", "left-leaning")
LABELS = ("watch", "skip", "unsure")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
REELS_SQL = """
SELECT i.code, i.url, i.creator, i.caption, i.cover_text, i.transcript, i.transcript_language, i.audio_type,
       i.audio_title, i.duration, i.like_count, i.comment_count, i.taken_at, i.shown_at, i.category, i.topic, i.summary
FROM impressions i
WHERE i.is_ad = 0 AND (COALESCE(i.caption, '') != '' OR COALESCE(i.cover_text, '') != '' OR COALESCE(i.transcript, '') != '')
ORDER BY i.id
"""


# ── The sample ────────────────────────────────────────────────────────

def reel_text(row: dict) -> str:
    """What the LLM is shown for this reel, rebuilt from what was stored (Jev gets the same)."""
    meta = SimpleNamespace(creator=row["creator"], creator_full_name="", creator_verified=False,
                           taken_at=row["taken_at"], duration=row["duration"], like_count=row["like_count"],
                           comment_count=row["comment_count"], audio_type=row["audio_type"],
                           audio_title=row["audio_title"] or "", caption=row["caption"] or "")
    import datetime

    day = datetime.date.fromisoformat((row["shown_at"] or "")[:10] or datetime.date.today().isoformat())
    return build_user_message(meta, row["transcript"] or "", row["transcript_language"], row["cover_text"] or "", day)


def pick_sample(conn: sqlite3.Connection, size: int = 40, seed: int = 7) -> list[dict]:
    """Varied reels: up to a third news and politics, then round-robin across the other categories."""
    conn.row_factory = sqlite3.Row
    reels = list({row["code"]: dict(row) for row in conn.execute(REELS_SQL)}.values())
    rng = random.Random(seed)
    rng.shuffle(reels)
    by_category: dict[str, list[dict]] = {}
    for reel in reels:
        by_category.setdefault(reel["category"] or "other", []).append(reel)
    picked = by_category.pop("news_politics", [])[: size // 3]
    queues = [q for q in by_category.values() if q]
    while len(picked) < size and queues:
        for queue in list(queues):
            if len(picked) >= size:
                break
            picked.append(queue.pop())
            if not queue:
                queues.remove(queue)
    rng.shuffle(picked)
    return [{**r, "text": reel_text(r), "hindi": r["transcript_language"] == "hi" or bool(DEVANAGARI.search(
        " ".join(str(r[k] or "") for k in ("caption", "cover_text", "transcript"))))} for r in picked]


# ── Scoring ───────────────────────────────────────────────────────────

async def _score_all(run: dict, progress=print) -> None:
    llm_gate, jev_gate = asyncio.Semaphore(2), asyncio.Semaphore(6)
    scores = run.setdefault("scores", {})
    async with aiohttp.ClientSession() as session:
        async def one(model: str, persona: str, reel: dict):
            slot = scores.setdefault(persona, {}).setdefault(reel["code"], {})
            if model in slot:
                return
            interests = run["personas"][persona]["interests"]
            start = time.monotonic()
            try:
                if model == "jev":
                    async with jev_gate:
                        result = await jev.score_async(session, reel["text"], interests)
                    slot["jev"] = {"score": result.value_score, "seconds": result.seconds, "cost": result.cost}
                else:
                    async with llm_gate:
                        meta = SimpleNamespace(creator=reel["creator"], creator_full_name="", creator_verified=False,
                                               taken_at=reel["taken_at"], duration=reel["duration"],
                                               like_count=reel["like_count"], comment_count=reel["comment_count"],
                                               audio_type=reel["audio_type"], audio_title=reel["audio_title"] or "",
                                               caption=reel["caption"] or "")
                        start = time.monotonic()
                        judgement = await evaluate_value_async(
                            session, meta, reel["transcript"] or "", reel["transcript_language"],
                            reel["cover_text"] or "", interests=interests, persona=True, timeout=120)
                    slot["llm"] = {"score": judgement.value_score, "seconds": round(time.monotonic() - start, 2)}
            except (jev.JevError, ScoringError) as exc:
                run.setdefault("errors", []).append(f"{model} {persona} {reel['code']}: {exc}")

        tasks = [one(model, persona, reel) for persona in run["personas"] for reel in run["reels"]
                 for model in ("jev", "llm")]
        done = 0
        for task in asyncio.as_completed(tasks):
            await task
            done += 1
            if done % 20 == 0 or done == len(tasks):
                progress(f"  {done} of {len(tasks)} scored")
                save_run(run)


def load_run(path: str = RUN_PATH) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def save_run(run: dict, path: str = RUN_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(run, fh, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def load_labels(path: str = LABELS_PATH) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_label(persona: str, code: str, label: str | None, path: str = LABELS_PATH) -> dict:
    """Record (or with None, clear) your call on one reel for one persona."""
    run = load_run()
    if not run or persona not in run["personas"] or code not in {r["code"] for r in run["reels"]}:
        raise ValueError("That reel isn't in the head-to-head sample.")
    if label is not None and label not in LABELS:
        raise ValueError(f"A label is one of: {', '.join(LABELS)}")
    labels = load_labels(path)
    mine = labels.setdefault(persona, {})
    if label is None:
        mine.pop(code, None)
    else:
        mine[code] = label
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(labels, fh, indent=1)
    os.replace(path + ".tmp", path)
    return labels


# ── Who agrees with you ───────────────────────────────────────────────

def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """Share of calls that match, and Cohen's kappa (agreement beyond chance)."""
    n = len(pairs)
    if not n:
        return {"n": 0, "agree": None, "kappa": None}
    agree = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else 1.0
    return {"n": n, "agree": round(agree, 3), "kappa": round(kappa, 3)}


def results(run: dict, labels: dict) -> dict:
    """Per persona and overall: how often each model's watch / skip call matches yours."""
    reels = {r["code"]: r for r in run["reels"]}
    out, pooled = {}, {"jev": [], "llm": [], "jev_hindi": [], "llm_hindi": []}
    for persona in run["personas"]:
        mine = {c: l for c, l in labels.get(persona, {}).items() if l in ("watch", "skip") and c in reels}
        scores = run.get("scores", {}).get(persona, {})
        entry = {"labelled": len(mine), "of": len(reels), "misses": []}
        for model in ("jev", "llm"):
            pairs = [(mine[c] == "watch", scores[c][model]["score"] >= WATCH_THRESHOLD)
                     for c in mine if model in scores.get(c, {})]
            hindi = [(mine[c] == "watch", scores[c][model]["score"] >= WATCH_THRESHOLD)
                     for c in mine if model in scores.get(c, {}) and reels[c]["hindi"]]
            entry[model] = {**agreement(pairs), "hindi": agreement(hindi)}
            pooled[model] += pairs
            pooled[model + "_hindi"] += hindi
        for code, label in mine.items():
            s = scores.get(code, {})
            if "jev" in s and "llm" in s:
                calls = {m: s[m]["score"] >= WATCH_THRESHOLD for m in ("jev", "llm")}
                if any(calls[m] != (label == "watch") for m in calls):
                    entry["misses"].append({"code": code, "you": label, "jev": s["jev"]["score"], "llm": s["llm"]["score"],
                                            "creator": reels[code]["creator"]})
        out[persona] = entry
    timing = {m: [s[m]["seconds"] for p in run.get("scores", {}).values() for s in p.values() if m in s]
              for m in ("jev", "llm")}
    return {
        "personas": out,
        "overall": {m: {**agreement(pooled[m]), "hindi": agreement(pooled[m + "_hindi"])} for m in ("jev", "llm")},
        "seconds": {m: sorted(t)[len(t) // 2] if t else None for m, t in timing.items()},
        "jev_cost": round(sum(s["jev"].get("cost", 0) for p in run.get("scores", {}).values() for s in p.values()
                              if "jev" in s), 5),
        "scored": {m: len(t) for m, t in timing.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Jev or the local LLM: which judges a persona better?")
    parser.add_argument("--personas", default=",".join(DEFAULT_PERSONAS), help="Preset keys, comma-separated")
    parser.add_argument("--reels", type=int, default=40)
    parser.add_argument("--new", action="store_true", help="Pick a new sample (your labels are dropped)")
    args = parser.parse_args()
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    if not jev.api_key():
        print("⚠️  Set OPENROUTER_API_KEY in .env first.")
        return
    if not llm_serving():
        print("⚠️  LM Studio isn't serving the model: start it, then run this again.")
        return
    presets = biases.presets()
    keys = [k.strip() for k in args.personas.split(",") if k.strip()]
    unknown = [k for k in keys if k not in presets]
    if unknown:
        print(f"⚠️  Unknown presets: {', '.join(unknown)} (there are: {', '.join(presets)})")
        return
    run = None if args.new else load_run()
    if run is None:
        conn = get_connection()
        try:
            run = {"reels": pick_sample(conn, args.reels), "personas": {}}
        finally:
            conn.close()
        if os.path.exists(LABELS_PATH):
            os.remove(LABELS_PATH)
    for key in keys:
        run["personas"].setdefault(key, {"name": presets[key].name, "interests": presets[key].interests})
    run["errors"] = []
    save_run(run)
    print(f"⚖️  Scoring {len(run['reels'])} reels as {len(run['personas'])} personas with Jev and the local LLM …")
    asyncio.run(_score_all(run))
    save_run(run)
    if run["errors"]:
        print(f"⚠️  {len(run['errors'])} calls failed (run again to retry them), e.g. {run['errors'][0]}")
    r = results(run, load_labels())
    print(f"✅  Scored: Jev {r['scored']['jev']}, LLM {r['scored']['llm']} · median seconds: Jev {r['seconds']['jev']}, "
          f"LLM {r['seconds']['llm']} · Jev cost ${r['jev_cost']:.4f}")
    print("    Now label the reels in the panel's Label tab; the results appear there as you go.")


if __name__ == "__main__":
    main()
