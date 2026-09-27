"""
PRAG — The Feed Study: what an account is being shown

For studying what the algorithm feeds someone with the account's bias:
every reel the account was shown, with its category, split into the
warm-up (before the feed counted as curated; see dashboard/report.py) and
the curated feed, which is the part to study and to train on.

    python -m dashboard.feed --account NAME                  the mix, before and after curation
    python -m dashboard.feed --account NAME --categorize     categorise new reels (needs LM Studio)
    python -m dashboard.feed --account NAME --export [--curated-only] [--format jsonl|csv]
"""

import argparse
import csv
import datetime
import io
import json
import os
import sqlite3
from collections import Counter

from agent.bias import load as load_bias
from config import CURATION_WINDOW
from dashboard.report import curated_since, curation_verdict
from database.db import choose_account, get_connection, init_db, list_accounts
from perception.categorize import CATEGORIES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
WATCHED = ("WATCH", "FULL_WATCH", "REWATCH")

FEED_SQL = """
SELECT i.session_id, i.seq, i.shown_at, i.code, i.url, i.creator, i.caption, i.hashtags, i.audio_type,
       i.audio_title, i.duration, i.like_count, i.comment_count, i.taken_at, i.inventory_source, i.is_ad,
       i.transcript, i.transcript_language, i.cover_text, i.score_basis, i.value_score, i.is_informational,
       i.topic, i.summary, i.category, i.decision, i.dwell_seconds, i.liked, i.saved, i.followed, s.scorer
FROM impressions i JOIN sessions s ON s.id = i.session_id
WHERE s.account = ?
ORDER BY i.session_id, i.seq
"""
EXPORT_FIELDS = (
    "account", "bias", "persona", "phase", "session_id", "seq", "shown_at", "code", "url", "creator", "caption",
    "hashtags", "audio_type", "audio_title", "duration", "like_count", "comment_count", "taken_at",
    "inventory_source", "is_ad", "transcript", "transcript_language", "cover_text", "category", "topic",
    "summary", "is_informational", "value_score", "score_basis", "scorer", "decision", "dwell_seconds",
    "liked", "saved", "followed",
)


def load_feed(conn: sqlite3.Connection, account: str) -> tuple[list[dict], dict]:
    """(every reel shown to the account, oldest first, each with its "phase"; the curation state).

    Curation is judged under the latest session's scorer (its bias and model), like the report; every reel
    shown from the curation point on is "curated", whatever scored it.
    """
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(FEED_SQL, (account,))]
    scorer = rows[-1]["scorer"] if rows else None
    judged = [r for r in rows if r["scorer"] == scorer]
    since = curated_since(judged)
    boundary = (since["session_id"], since["seq"]) if since else None
    for row in rows:
        row["phase"] = "curated" if boundary and (row["session_id"], row["seq"]) >= boundary else "warm-up"
    return rows, {"since": since, "verdict": curation_verdict(judged), "scorer": scorer}


def mix(rows: list[dict]) -> dict[str, int]:
    """Categorised organic reels per category, in the categories' order."""
    counts = Counter(r["category"] for r in rows if not r["is_ad"] and r["category"])
    return {key: counts[key] for key in CATEGORIES if counts[key]}


def top_creators(rows: list[dict], limit: int = 12) -> list[dict]:
    """The creators the feed served most, with their usual category and how often the agent watched."""
    by_creator: dict[str, list[dict]] = {}
    for row in rows:
        if row["creator"] and not row["is_ad"]:
            by_creator.setdefault(row["creator"], []).append(row)
    ranked = sorted(by_creator.items(), key=lambda item: (-len(item[1]), item[0]))[:limit]
    out = []
    for creator, reels in ranked:
        categories = Counter(r["category"] for r in reels if r["category"])
        out.append({
            "creator": creator, "reels": len(reels),
            "category": categories.most_common(1)[0][0] if categories else None,
            "watched": round(sum(r["decision"] in WATCHED for r in reels) / len(reels), 2),
        })
    return out


def summary(conn: sqlite3.Connection, account: str, limit: int = 400) -> dict:
    """Everything the panel's Feed tab shows."""
    rows, state = load_feed(conn, account)
    bias = load_bias(account)
    curated = [r for r in rows if r["phase"] == "curated"]
    warmup = [r for r in rows if r["phase"] == "warm-up"]
    verdict = state["verdict"]
    return {
        "bias": {"name": bias.name, "persona": bias.persona},
        "curation": {
            "curated": bool(state["since"]), "since": state["since"], "judged": verdict["n"], "window": CURATION_WINDOW,
            "share": round(verdict["share"], 3), "target": verdict["target"], "enough": verdict["enough"],
        },
        "categories": {key: label for key, (label, _about) in CATEGORIES.items()},
        "counts": {
            "reels": len(rows), "ads": sum(bool(r["is_ad"]) for r in rows),
            "uncategorised": sum(1 for r in rows if not r["category"] and not r["is_ad"]),
            "warm-up": len(warmup), "curated": len(curated),
        },
        "mix": {"warm-up": mix(warmup), "curated": mix(curated)},
        "creators": top_creators(curated or rows),
        "reels": [{
            "session_id": r["session_id"], "seq": r["seq"], "shown_at": r["shown_at"], "creator": r["creator"],
            "url": r["url"], "category": r["category"], "topic": r["topic"],
            "summary": r["summary"] or " ".join((r["caption"] or "").split())[:160],
            "value_score": r["value_score"], "decision": r["decision"], "phase": r["phase"], "is_ad": bool(r["is_ad"]),
        } for r in reversed(rows[-limit:])],
    }


# ── The dataset ───────────────────────────────────────────────────────

def dataset(conn: sqlite3.Connection, account: str, curated_only: bool = False) -> list[dict]:
    """The reels as training examples: everything recorded about each, plus its bias, phase and category."""
    rows, _state = load_feed(conn, account)
    bias = load_bias(account)
    return [{**{key: r.get(key) for key in EXPORT_FIELDS}, "account": account, "bias": bias.name,
             "persona": bias.persona} for r in rows if not curated_only or r["phase"] == "curated"]


def to_jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def to_csv(records: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=EXPORT_FIELDS)
    writer.writeheader()
    writer.writerows(records)
    return out.getvalue()


def export(conn: sqlite3.Connection, account: str, *, curated_only: bool = False, fmt: str = "jsonl",
           out_dir: str = DATASET_DIR) -> tuple[str, int]:
    records = dataset(conn, account, curated_only)
    folder = os.path.join(out_dir, account)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{datetime.date.today().isoformat()}-{'curated' if curated_only else 'all'}.{fmt}")
    with open(path, "w", newline="") as fh:
        fh.write(to_jsonl(records) if fmt == "jsonl" else to_csv(records))
    return path, len(records)


# ── Command line ──────────────────────────────────────────────────────

def print_summary(info: dict, account: str) -> None:
    c, n = info["curation"], info["counts"]
    persona = " (persona)" if info["bias"]["persona"] else ""
    print(f"📊  @{account} · bias {info['bias']['name']}{persona} · {n['reels']} reels shown ({n['ads']} ads)")
    if c["curated"]:
        print(f"    Curated since {c['since']['shown_at']} UTC (judged reel #{c['since']['index']}): "
              f"{n['curated']} curated reels, {n['warm-up']} warm-up")
    else:
        print(f"    Not curated yet: {c['judged']} judged reels, {c['share']:.0%} on target "
              f"(needs {c['target']:.0%} over the latest 100)")
    if n["uncategorised"]:
        print(f"    {n['uncategorised']} reels not categorised yet → --categorize")
    for phase in ("warm-up", "curated"):
        counts = info["mix"][phase]
        total = sum(counts.values())
        if total:
            print(f"\n  {phase} ({total} categorised reels)")
            for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"    {info['categories'][key]:28} {count / total:5.0%}  {'█' * round(24 * count / total)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="What is this account's feed showing it?")
    parser.add_argument("--account", help="Account to study (required when several are ACTIVE)")
    parser.add_argument("--categorize", action="store_true", help="Categorise reels that have no category yet")
    parser.add_argument("--export", action="store_true", help="Write the dataset to datasets/<account>/")
    parser.add_argument("--curated-only", action="store_true", help="Export only the curated feed")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    args = parser.parse_args()

    init_db()  # adds the category column to an older database
    account, problem = choose_account(list_accounts(), args.account)
    if problem:
        print(f"⚠️  {problem}")
        return
    conn = get_connection()
    try:
        if args.categorize:
            from perception.categorize import categorize_pending
            from perception.local_evaluator import llm_serving

            if not llm_serving():
                print("⚠️  LM Studio isn't serving the model, so reels can't be categorised now.")
            else:
                done = categorize_pending(conn, account)
                print(f"✅  Categorised {done['done']} reels" + (f"; {done['failed']} failed" if done["failed"] else ""))
        if args.export:
            path, count = export(conn, account, curated_only=args.curated_only, fmt=args.format)
            print(f"💾  {count} reels → {path}")
        print_summary(summary(conn, account), account)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
