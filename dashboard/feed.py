"""
PRAG — The Feed Study: what an account is being shown

For studying what the algorithm feeds someone like the account's persona:
every reel the account was shown, with its category, split into the
warm-up (before the feed counted as curated; see dashboard/report.py) and
the curated feed, which is the part to study and to train on. Each
account has its own persona; summary_all() and dataset_all() put every
account's feed side by side, and together.

    python -m dashboard.feed --account NAME                  the mix, before and after curation
    python -m dashboard.feed --all                           every account, side by side
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

from agent.bias import has_own as has_own_persona
from agent.bias import load as load_bias
from config import CURATION_WINDOW, WATCH_THRESHOLD
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
       i.topic, i.summary, i.category, i.decision, i.dwell_seconds, i.liked, i.saved, i.followed, i.image_used, s.scorer
FROM impressions i JOIN sessions s ON s.id = i.session_id
WHERE s.account = ?
ORDER BY s.started_at, i.session_id, i.seq
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
    # rows are in the order they were shown (sessions by start time), so curated is everything from that reel on
    boundary = next((i for i, r in enumerate(rows) if (r["session_id"], r["seq"]) == (since["session_id"], since["seq"])),
                    None) if since else None
    for i, row in enumerate(rows):
        row["phase"] = "curated" if boundary is not None and i >= boundary else "warm-up"
    return rows, {"since": since, "verdict": curation_verdict(judged), "scorer": scorer}


def mix(rows: list[dict]) -> dict[str, int]:
    """Categorised organic reels per category, in the categories' order."""
    counts = Counter(r["category"] for r in rows if not r["is_ad"] and r["category"])
    return {key: counts[key] for key in CATEGORIES if counts[key]}


def top_creators(rows: list[dict], limit: int = 12) -> list[dict]:
    """The creators the feed served most, with their usual category, how often the agent watched, and (rows from
    several accounts) which accounts they reached."""
    by_creator: dict[str, list[dict]] = {}
    for row in rows:
        if row["creator"] and not row["is_ad"]:
            by_creator.setdefault(row["creator"], []).append(row)
    ranked = sorted(by_creator.items(), key=lambda item: (-len(item[1]), item[0]))[:limit]
    out = []
    for creator, reels in ranked:
        categories = Counter(r["category"] for r in reels if r["category"])
        item = {
            "creator": creator, "reels": len(reels),
            "category": categories.most_common(1)[0][0] if categories else None,
            "watched": round(sum(r["decision"] in WATCHED for r in reels) / len(reels), 2),
        }
        if any("account" in r for r in reels):
            item["accounts"] = sorted({r["account"] for r in reels})
        out.append(item)
    return out


def newest_first(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["shown_at"] or "", r["session_id"], r["seq"]), reverse=True)


def reel_item(r: dict) -> dict:
    """One reel as the panel lists it."""
    return {
        "account": r.get("account"), "session_id": r["session_id"], "seq": r["seq"], "shown_at": r["shown_at"],
        "creator": r["creator"], "url": r["url"], "category": r["category"], "topic": r["topic"],
        "summary": r["summary"] or " ".join((r["caption"] or "").split())[:160],
        "value_score": r["value_score"], "decision": r["decision"], "phase": r["phase"], "is_ad": bool(r["is_ad"]),
        "image_used": r.get("image_used"),
    }


def reel_items(rows: list[dict], limit: int) -> list[dict]:
    """The latest reels, newest first, as the panel lists them."""
    return [reel_item(r) for r in newest_first(rows)[:limit]]


def reel_page(rows: list[dict], *, day: str | None = None, phase: str | None = None, category: str | None = None,
              account: str | None = None, query: str | None = None, offset: int = 0, limit: int = 50) -> dict:
    """One page of the reels, newest first, after the filters: {"total", "offset", "limit", "items"}.
    `category` is a category key, "_none" (not categorised) or "_ad"; `query` matches creator, topic, caption or summary."""
    needle = (query or "").strip().lower()

    def keep(r: dict) -> bool:
        if phase and r["phase"] != phase or account and r.get("account") != account:
            return False
        if category == "_ad":
            if not r["is_ad"]:
                return False
        elif category == "_none":
            if r["category"] or r["is_ad"]:
                return False
        elif category and r["category"] != category:
            return False
        if day and local_day(r["shown_at"]) != day:
            return False
        return not needle or any(needle in (r.get(k) or "").lower() for k in ("creator", "topic", "caption", "summary"))

    shown = newest_first([r for r in rows if keep(r)])
    offset = max(0, offset)
    return {"total": len(shown), "offset": offset, "limit": limit,
            "items": [reel_item(r) for r in shown[offset:offset + limit]]}


def local_day(shown_at: str | None) -> str | None:
    """The Mac's calendar day a UTC "YYYY-MM-DD HH:MM:SS" falls on."""
    if not shown_at:
        return None
    return datetime.datetime.fromisoformat(shown_at).replace(tzinfo=datetime.timezone.utc).astimezone().date().isoformat()


def progression(rows: list[dict]) -> list[dict]:
    """Per day: reels shown (ads aside), how many fit the persona (score at or above the watch line), watched, liked."""
    days: dict[str, dict] = {}
    for r in rows:
        if r["is_ad"]:
            continue
        day = local_day(r["shown_at"])
        d = days.setdefault(day, {"day": day, "reels": 0, "fit": 0, "watched": 0, "liked": 0})
        d["reels"] += 1
        d["fit"] += r["value_score"] is not None and r["value_score"] >= WATCH_THRESHOLD
        d["watched"] += r["decision"] in WATCHED
        d["liked"] += bool(r["liked"])
    return [days[k] for k in sorted(k for k in days if k)]


def counts(rows: list[dict]) -> dict:
    return {
        "reels": len(rows), "ads": sum(bool(r["is_ad"]) for r in rows),
        "uncategorised": sum(1 for r in rows if not r["category"] and not r["is_ad"]),
        "warm-up": sum(r["phase"] == "warm-up" for r in rows), "curated": sum(r["phase"] == "curated" for r in rows),
    }


def summary(conn: sqlite3.Connection, account: str, limit: int = 400, day: str | None = None) -> dict:
    """Everything the Feed tab shows for one account; `day` narrows the reel list to one calendar day."""
    rows, state = load_feed(conn, account)
    for row in rows:
        row["account"] = account
    bias = load_bias(account)
    curated = [r for r in rows if r["phase"] == "curated"]
    warmup = [r for r in rows if r["phase"] == "warm-up"]
    verdict = state["verdict"]
    return {
        "scope": "account",
        "bias": {"name": bias.name, "persona": bias.persona, "own": has_own_persona(account)},
        "curation": {
            "curated": bool(state["since"]), "since": state["since"], "judged": verdict["n"], "window": CURATION_WINDOW,
            "share": round(verdict["share"], 3), "target": verdict["target"], "enough": verdict["enough"],
        },
        "categories": {key: label for key, (label, _about) in CATEGORIES.items()},
        "counts": counts(rows),
        "mix": {"warm-up": mix(warmup), "curated": mix(curated)},
        "creators": top_creators(curated or rows),
        "days": progression(rows), "day": day,
        "reels": reel_items([r for r in rows if local_day(r["shown_at"]) == day], 5000) if day else reel_items(rows, limit),
    }


def studied_accounts(conn: sqlite3.Connection) -> list[str]:
    """Every account that has been shown reels, most reels first."""
    return [account for (account,) in conn.execute(
        "SELECT s.account FROM sessions s JOIN impressions i ON i.session_id = s.id "
        "GROUP BY s.account ORDER BY COUNT(*) DESC, s.account")]


def all_feeds(conn: sqlite3.Connection) -> list[tuple[str, list[dict], dict]]:
    """(account, its reels with their phase and account, its curation state) for every studied account."""
    out = []
    for account in studied_accounts(conn):
        rows, state = load_feed(conn, account)
        for row in rows:
            row["account"] = account
        out.append((account, rows, state))
    return out


def summary_all(conn: sqlite3.Connection, limit: int = 400) -> dict:
    """Every account's feed side by side (each its own persona), and all of them together."""
    feeds = all_feeds(conn)
    accounts, everything = [], []
    for account, rows, state in feeds:
        everything += rows
        bias, verdict = load_bias(account), state["verdict"]
        curated = [r for r in rows if r["phase"] == "curated"]
        current = curated or rows
        accounts.append({
            "account": account, "persona": bias.name, "is_persona": bias.persona, "own": has_own_persona(account),
            "reels": len(rows), "curated": bool(state["since"]), "since": state["since"],
            "curated_reels": len(curated), "judged": verdict["n"], "share": round(verdict["share"], 3),
            "target": verdict["target"], "window": CURATION_WINDOW,
            "mix": mix(current), "mix_of": "curated" if curated else "all",
        })
    warmup = [r for r in everything if r["phase"] == "warm-up"]
    curated = [r for r in everything if r["phase"] == "curated"]
    return {
        "scope": "all",
        "accounts": accounts,
        "categories": {key: label for key, (label, _about) in CATEGORIES.items()},
        "counts": {**counts(everything), "accounts": len(accounts),
                   "curated_accounts": sum(a["curated"] for a in accounts)},
        "mix": {"warm-up": mix(warmup), "curated": mix(curated)},
        "creators": top_creators(curated or everything),
        "reels": reel_items(everything, limit),
    }


# ── The dataset ───────────────────────────────────────────────────────

def dataset(conn: sqlite3.Connection, account: str, curated_only: bool = False) -> list[dict]:
    """The reels as training examples: everything recorded about each, plus its bias, phase and category."""
    rows, _state = load_feed(conn, account)
    bias = load_bias(account)
    return [{**{key: r.get(key) for key in EXPORT_FIELDS}, "account": account, "bias": bias.name,
             "persona": bias.persona} for r in rows if not curated_only or r["phase"] == "curated"]


def dataset_all(conn: sqlite3.Connection, curated_only: bool = False) -> list[dict]:
    """Every account's reels in one dataset: each record says which account and persona it came from."""
    return [record for account in studied_accounts(conn) for record in dataset(conn, account, curated_only)]


def to_jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def to_csv(records: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=EXPORT_FIELDS)
    writer.writeheader()
    writer.writerows(records)
    return out.getvalue()


def export(conn: sqlite3.Connection, account: str | None, *, curated_only: bool = False, fmt: str = "jsonl",
           out_dir: str = DATASET_DIR) -> tuple[str, int]:
    """Write the dataset to datasets/<account>/ (account None: every account, to datasets/all/)."""
    records = dataset(conn, account, curated_only) if account else dataset_all(conn, curated_only)
    folder = os.path.join(out_dir, account or "all")
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


def print_all(info: dict) -> None:
    n = info["counts"]
    print(f"📊  {n['accounts']} accounts · {n['curated_accounts']} curated · {n['reels']} reels shown "
          f"({n['curated']} in curated feeds)")
    for a in info["accounts"]:
        state = "curated" if a["curated"] else f"{a['judged']}/{a['window']} judged, {a['share']:.0%} on target"
        total = sum(a["mix"].values())
        top = ", ".join(f"{info['categories'][k]} {v / total:.0%}"
                        for k, v in sorted(a["mix"].items(), key=lambda kv: -kv[1])[:3]) if total else "not categorised"
        print(f"    @{a['account']:18} {a['persona'][:28]:28} {a['reels']:5} reels  {state:30} {top}")


def main() -> None:
    parser = argparse.ArgumentParser(description="What is this account's feed showing it?")
    parser.add_argument("--account", help="Account to study (required when several are ACTIVE)")
    parser.add_argument("--all", action="store_true", help="Every account side by side (and --export: all of them)")
    parser.add_argument("--categorize", action="store_true", help="Categorise reels that have no category yet")
    parser.add_argument("--export", action="store_true", help="Write the dataset to datasets/<account>/")
    parser.add_argument("--curated-only", action="store_true", help="Export only the curated feed")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    args = parser.parse_args()

    init_db()  # adds the category column to an older database
    if args.all:
        conn = get_connection()
        try:
            if args.export:
                path, count = export(conn, None, curated_only=args.curated_only, fmt=args.format)
                print(f"💾  {count} reels from every account → {path}")
            print_all(summary_all(conn))
        finally:
            conn.close()
        return
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
