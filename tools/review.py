"""
PRAG — Review Recorded Reels

Sessions run with `main.py --observe` record every reel (caption, cover
text, speech transcript) without judging it. This tool exports those
reels for review and imports the judgements, after which the curation
report counts them like LLM-scored reels. The judgements also form a
labelled set for checking the local LLM scorer later.

    python -m tools.review export --account NAME [--session N] [--out FILE]
    python -m tools.review import FILE

Export writes one JSON object per reel. Import reads one per line with
session, code, value_score (0-1), topic, summary, is_informational and
reasoning — the same fields the LLM scorer produces.
"""

import argparse
import datetime
import json
import os
import sqlite3

from database.db import get_connection

EXPORT_SQL = """
SELECT i.session_id AS session, i.seq, i.code, i.url, i.creator, i.caption, i.hashtags,
       i.audio_type, i.audio_title, i.duration, i.like_count, i.taken_at,
       i.cover_text, i.transcript, i.transcript_language, i.error
FROM impressions i JOIN sessions s ON s.id = i.session_id
WHERE s.account = ? AND i.score_basis = 'observed' AND (? IS NULL OR i.session_id = ?)
ORDER BY i.session_id, i.seq
"""


def export_reels(conn: sqlite3.Connection, account: str, session: int | None = None) -> list[dict]:
    """Recorded reels still awaiting review, one dict per impression."""
    conn.row_factory = sqlite3.Row
    reels = []
    for row in conn.execute(EXPORT_SQL, (account, session, session)):
        reel = dict(row)
        if reel["taken_at"]:
            reel["posted"] = datetime.datetime.fromtimestamp(reel.pop("taken_at"), datetime.timezone.utc).date().isoformat()
        else:
            reel.pop("taken_at")
        reels.append({k: v for k, v in reel.items() if v not in (None, "")})
    return reels


def import_judgements(conn: sqlite3.Connection, judgements: list[dict]) -> tuple[int, list[str]]:
    """Write judgements onto recorded reels; returns (rows updated, problems)."""
    updated, problems = 0, []
    for j in judgements:
        try:
            score = float(j["value_score"])
            session, code = int(j["session"]), str(j["code"])
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"skipped {j!r:.80}: {exc}")
            continue
        if not 0.0 <= score <= 1.0:
            problems.append(f"skipped {code}: value_score {score} outside 0-1")
            continue
        cur = conn.execute(
            """
            UPDATE impressions
            SET score_basis = 'score', value_score = ?, is_informational = ?, topic = ?,
                summary = ?, reasoning = ?, scored_at = datetime('now')
            WHERE session_id = ? AND code = ? AND score_basis = 'observed'
            """,
            (score, int(bool(j.get("is_informational"))), str(j.get("topic") or "")[:60],
             str(j.get("summary") or "")[:400], str(j.get("reasoning") or "")[:300], session, code),
        )
        if cur.rowcount:
            updated += cur.rowcount
        else:
            problems.append(f"no recorded reel {code} awaiting review in session {session}")
    conn.commit()
    return updated, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="Write recorded reels awaiting review to a JSONL file")
    exp.add_argument("--account", required=True)
    exp.add_argument("--session", type=int)
    exp.add_argument("--out", help="Default: debug/review/<account>[-s<session>].jsonl")
    imp = sub.add_parser("import", help="Read judgements from a JSONL file")
    imp.add_argument("file")
    args = parser.parse_args()

    conn = get_connection()
    try:
        if args.command == "export":
            account = args.account.lstrip("@")
            reels = export_reels(conn, account, args.session)
            suffix = f"-s{args.session}" if args.session else ""
            out = args.out or os.path.join("debug", "review", f"{account}{suffix}.jsonl")
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as fh:
                for reel in reels:
                    fh.write(json.dumps(reel, ensure_ascii=False) + "\n")
            print(f"📝  {len(reels)} reels awaiting review → {out}")
        else:
            with open(args.file) as fh:
                judgements = [json.loads(line) for line in fh if line.strip()]
            updated, problems = import_judgements(conn, judgements)
            print(f"✅  {updated} reels scored from {len(judgements)} judgements")
            for problem in problems:
                print(f"  ⚠️  {problem}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
