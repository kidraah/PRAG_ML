"""
PRAG — Curation Report

Answers "is the account curated yet?" from the impressions log:
  • the share of organic reels that are informative (value >= WATCH_THRESHOLD)
    and valuable (>= LIKE_THRESHOLD), per session and over time, with a
    verdict on the latest CURATION_WINDOW reels against the target
  • watch time by decision — skips must be short, or the feed learns nothing
  • what decided each reel, and an audit of the keyword / creator gates
  • top topics, best creators, and the latest valuable reels to eyeball

Each account has its own feed, so the report covers one account (by
default the one used most recently). Only reels scored with the current
prompt (the account's bias included) and model are compared; change either
and the trend starts over.

    python -m dashboard.report [--account NAME] [--window 50] [--recent 8]
"""

import argparse
import sqlite3
import statistics
from collections import Counter

from agent.bias import load as load_bias
from config import CURATION_TARGET_SHARE, CURATION_WINDOW, LIKE_THRESHOLD, WATCH_THRESHOLD
from database.db import DB_PATH, get_connection

SPARK = "▁▂▃▄▅▆▇█"

ROWS_SQL = """
SELECT i.session_id, i.seq, i.code, i.url, i.creator, i.is_ad, i.score_basis, i.value_score,
       i.topic, i.summary, i.decision, i.decision_basis, i.explored, i.dwell_seconds,
       i.gate_audit, i.score_seconds, i.liked, i.saved, i.followed, i.rule_score, i.ready_on_arrival,
       s.scorer
FROM impressions i JOIN sessions s ON s.id = i.session_id
WHERE s.account = ?
ORDER BY i.session_id, i.seq
"""

SESSIONS_SQL = """
SELECT s.id, strftime('%Y-%m-%d %H:%M', s.started_at, 'localtime') AS started,
       (julianday(COALESCE(s.ended_at, MAX(i.shown_at))) - julianday(s.started_at)) * 1440 AS minutes,
       s.dry_run, s.scorer, COUNT(i.id) AS reels, COALESCE(SUM(i.is_ad), 0) AS ads,
       COALESCE(SUM(i.liked), 0) AS liked, COALESCE(SUM(i.saved), 0) AS saved,
       COALESCE(SUM(i.followed), 0) AS followed
FROM sessions s LEFT JOIN impressions i ON i.session_id = s.id
WHERE s.account = ?
GROUP BY s.id ORDER BY s.id
"""


# ── Metrics (pure) ────────────────────────────────────────────────────

def on_target(row) -> bool | None:
    """Whether a reel counts as informative; None if not judged (ad, unscored, LLM error, awaiting review)."""
    if row["is_ad"] or row["score_basis"] in (None, "ad", "llm_error", "observed"):
        return None
    if row["score_basis"] in ("keyword_gate", "creator_prior"):
        return False  # skipped by a gate: presumed off-target (the gate audit checks this)
    return row["value_score"] is not None and row["value_score"] >= WATCH_THRESHOLD


def is_valuable(row) -> bool:
    return row["score_basis"] == "score" and (row["value_score"] or 0) >= LIKE_THRESHOLD


def window_stats(rows: list, size: int) -> list[dict]:
    """Informative / valuable shares over consecutive windows of judged reels."""
    judged = [r for r in rows if on_target(r) is not None]
    windows = []
    for start in range(0, len(judged), size):
        chunk = judged[start:start + size]
        windows.append({
            "first": start + 1,
            "last": start + len(chunk),
            "informative": sum(bool(on_target(r)) for r in chunk) / len(chunk),
            "valuable": sum(is_valuable(r) for r in chunk) / len(chunk),
        })
    return windows


def curation_verdict(rows: list, window: int = CURATION_WINDOW, target: float = CURATION_TARGET_SHARE) -> dict:
    judged = [r for r in rows if on_target(r) is not None][-window:]
    share = sum(bool(on_target(r)) for r in judged) / len(judged) if judged else 0.0
    enough = len(judged) >= window
    return {"n": len(judged), "share": share, "target": target, "enough": enough,
            "curated": enough and share >= target}


def sparkline(share: float) -> str:
    return SPARK[min(len(SPARK) - 1, int(share * len(SPARK)))]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


# ── Rendering ─────────────────────────────────────────────────────────

def render(conn: sqlite3.Connection, window: int = 50, recent: int = 8, account: str | None = None) -> str:
    conn.row_factory = sqlite3.Row
    accounts = conn.execute(
        "SELECT account, COUNT(*) AS n, MAX(id) AS last FROM sessions GROUP BY account ORDER BY last DESC"
    ).fetchall()
    if not accounts:
        return "No sessions yet — run `python main.py` first."
    account = (account or accounts[0]["account"]).lstrip("@")
    sessions = conn.execute(SESSIONS_SQL, (account,)).fetchall()
    if not sessions:
        return f"No sessions for @{account} yet."
    all_rows = conn.execute(ROWS_SQL, (account,)).fetchall()
    scorer = sessions[-1]["scorer"]
    rows = [r for r in all_rows if r["scorer"] == scorer]
    organic = [r for r in rows if not r["is_ad"]]
    judged = [r for r in rows if on_target(r) is not None]

    out = [f"PRAG — curation report for @{account}", "═" * 60]
    others = [f"@{a['account']} ({a['n']} sessions)" for a in accounts if a["account"] != account]
    if others:
        out.append("Other accounts (see --account): " + ", ".join(others))
    older = len(all_rows) - len(rows)
    out.append(
        f"Bias {load_bias(account).name} · scorer {scorer} · {len(sessions)} sessions · {len(rows)} reels shown · "
        f"{len(judged)} judged"
        + (f"  ({older} reels scored under another bias, prompt or model not compared)" if older else "")
    )

    awaiting = sum(r["score_basis"] == "observed" for r in rows)
    if awaiting:
        out.append(f"{awaiting} recorded reels await review → python -m tools.review export --account {account}")

    out += ["", "Sessions",
            f"  {'id':>3}  {'started (local)':16}  {'min':>5}  {'reels':>5}  {'ads':>4}  {'judged':>6}  "
            f"{'informative':>11}  {'valuable':>8}  {'❤️':>3} {'🔖':>3} {'➕':>3}  mode"]
    for s in sessions:
        srows = [r for r in all_rows if r["session_id"] == s["id"]]
        sj = [r for r in srows if on_target(r) is not None]
        inf = f"{sum(bool(on_target(r)) for r in sj) / len(sj):.0%}" if sj else "—"
        val = f"{sum(is_valuable(r) for r in sj) / len(sj):.0%}" if sj else "—"
        mode = "dry run" if s["dry_run"] else "live"
        if s["scorer"] != scorer:
            mode += " (old scorer)"
        out.append(
            f"  {s['id']:>3}  {s['started'] or '':16}  {(s['minutes'] or 0):5.1f}  {s['reels']:>5}  {s['ads']:>4}  "
            f"{len(sj):>6}  {inf:>11}  {val:>8}  {s['liked']:>3} {s['saved']:>3} {s['followed']:>3}  {mode}"
        )

    out += ["", f"Feed quality over time (judged organic reels, windows of {window})"]
    for w in window_stats(rows, window):
        out.append(
            f"  reels {w['first']:>5}–{w['last']:<5}  informative {sparkline(w['informative'])} "
            f"{w['informative']:>4.0%}   valuable {w['valuable']:>4.0%}"
        )
    v = curation_verdict(rows)
    if not v["n"]:
        out.append("  (no judged reels yet)")
    elif not v["enough"]:
        out.append(f"Verdict: only {v['n']} judged reels so far — need {CURATION_WINDOW} for a verdict "
                   f"(currently {v['share']:.0%} informative, target {v['target']:.0%}).")
    else:
        state = "CURATED ✅" if v["curated"] else "not curated yet"
        out.append(f"Verdict: latest {v['n']} judged reels — {v['share']:.0%} informative "
                   f"(target {v['target']:.0%}) → {state}")

    out += ["", "Watch time by decision (median seconds on screen)"]
    parts = []
    for action in ("SKIP", "WATCH", "FULL_WATCH", "REWATCH", "OBSERVE"):
        dwell = [r["dwell_seconds"] for r in organic if r["decision"] == action and r["dwell_seconds"] is not None]
        if dwell:
            parts.append(f"{action} {_median(dwell):.1f}s (n={len(dwell)})")
    out.append("  " + (" · ".join(parts) or "—"))
    known = [r["ready_on_arrival"] for r in organic if r["ready_on_arrival"] is not None]
    if known:
        out.append(f"  known before the reel appeared (preloaded feed data): {sum(known) / len(known):.0%}"
                   " — skips of those are flicks")
    late = sum(r["decision_basis"] == "timeout" for r in organic)
    timing = _median([r["score_seconds"] for r in organic if r["score_seconds"] is not None])
    if organic:
        out.append(f"  decided before the score arrived: {late / len(organic):.0%} of organic reels"
                   + (f" · median scoring time {timing:.1f}s" if timing is not None else ""))

    bases = Counter(r["decision_basis"] for r in rows)
    if rows:
        out += ["", "What decided each reel",
                "  " + " · ".join(f"{b} {n / len(rows):.0%}" for b, n in bases.most_common())]

    audits = [r for r in rows if r["gate_audit"] and r["score_basis"] == "score"]
    if audits:
        missed = sum(bool(on_target(r)) for r in audits)
        out.append(f"Gate audit: {len(audits)} gated reels scored anyway — {missed} were informative "
                   f"({missed / len(audits):.0%}); if this is high, remove some of the bias's skip words.")

    ruled = [r for r in rows if r["rule_score"] is not None and on_target(r) is not None]
    if ruled:
        watched = [r for r in ruled if r["rule_score"] >= WATCH_THRESHOLD]
        informative = [r for r in ruled if on_target(r)]
        hits = sum(bool(on_target(r)) for r in watched)
        out += ["", f"Quick rules vs scores ({len(ruled)} scored reels the rules also judged)",
                f"  rules watched {len(watched)}: {hits} informative, {len(watched) - hits} not"
                f" · informative reels the rules skipped: {len(informative) - hits} of {len(informative)}"]

    scored = [r for r in rows if r["score_basis"] == "score"]
    topics = Counter((r["topic"] or "").strip().lower() for r in scored[-200:] if r["topic"])
    if topics:
        out += ["", "Top topics (latest 200 scored)",
                "  " + " · ".join(f"{t} {n}" for t, n in topics.most_common(12))]

    by_creator: dict[str, dict[str, float]] = {}
    for r in scored:
        if r["creator"]:
            by_creator.setdefault(r["creator"], {})[r["code"]] = r["value_score"]
    best = sorted(
        ((c, sum(s.values()) / len(s), len(s)) for c, s in by_creator.items() if len(s) >= 3),
        key=lambda x: -x[1],
    )[:8]
    if best:
        out += ["", "Best creators (3+ reels scored)",
                "  " + " · ".join(f"@{c} {avg:.2f} ({n})" for c, avg, n in best)]

    valuable = [r for r in scored if is_valuable(r)]
    if valuable:
        out += ["", "Latest valuable reels"]
        seen = set()
        for r in reversed(valuable):
            if r["code"] in seen:
                continue
            seen.add(r["code"])
            out.append(f"  {r['value_score']:.2f}  @{r['creator']}  “{(r['summary'] or r['topic'] or '')[:100]}”")
            out.append(f"        {r['url']}")
            if len(seen) >= recent:
                break
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Is the PRAG account curated yet?")
    parser.add_argument("--account", help="Account to report on (default: the one used most recently)")
    parser.add_argument("--window", type=int, default=50, help="Reels per trend window")
    parser.add_argument("--recent", type=int, default=8, help="How many recent valuable reels to list")
    parser.add_argument("--db", default=DB_PATH, help="Path to reels.db")
    args = parser.parse_args()
    conn = get_connection(args.db)
    try:
        print(render(conn, window=args.window, recent=args.recent, account=args.account))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
