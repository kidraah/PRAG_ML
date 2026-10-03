"""
PRAG — The Writer: the local LLM's words for reels Jev judged

Jev decides, in time, whether the persona watches a reel; it can't write.
The local LLM (LM Studio) writes each reel's topic and summary, says
whether it's informational, and gives its own score as a second opinion
(llm_score), so every reel still has its text for the digest, the Feed
tab and the dataset.

The pipeline writes during a session, watched reels first, as far as the
LLM keeps up; write_pending() finishes the rest afterwards (daily.py runs
it after each session, and the panel's "Categorise now" does too).

    python -m perception.writer --account NAME
"""

import argparse
import asyncio
import datetime
import sqlite3
from types import SimpleNamespace

import aiohttp

from agent.bias import load as load_bias
from database.db import choose_account, get_connection, init_db, list_accounts
from perception.local_evaluator import ScoringError, ValueJudgement, evaluate_value_async, llm_serving

PENDING_SQL = """
SELECT i.id, i.code, i.creator, i.caption, i.cover_text, i.transcript, i.transcript_language, i.audio_type,
       i.audio_title, i.duration, i.like_count, i.comment_count, i.taken_at, i.shown_at
FROM impressions i JOIN sessions s ON s.id = i.session_id
WHERE s.account = ? AND s.scorer LIKE 'jev-%' AND i.is_ad = 0 AND i.score_basis = 'score' AND i.llm_score IS NULL
ORDER BY i.value_score DESC, i.id
"""


def text_fields(judgement: ValueJudgement) -> dict:
    """The columns the writer fills in."""
    return {"topic": judgement.topic, "summary": judgement.summary,
            "is_informational": int(judgement.is_informational), "reasoning": judgement.reasoning,
            "llm_score": judgement.value_score}


def meta_from_row(row: dict) -> SimpleNamespace:
    """What local_evaluator.build_user_message needs, from a stored impression."""
    return SimpleNamespace(creator=row["creator"], creator_full_name="", creator_verified=False,
                           taken_at=row["taken_at"], duration=row["duration"], like_count=row["like_count"],
                           comment_count=row["comment_count"], audio_type=row["audio_type"],
                           audio_title=row["audio_title"] or "", caption=row["caption"] or "")


def pending(conn: sqlite3.Connection, account: str) -> dict[str, list[dict]]:
    """Reels Jev judged that have no text yet: {code: [its impressions]}, the highest scored first."""
    conn.row_factory = sqlite3.Row
    out: dict[str, list[dict]] = {}
    for row in conn.execute(PENDING_SQL, (account,)):
        out.setdefault(row["code"], []).append(dict(row))
    return out


def save(conn: sqlite3.Connection, ids: list[int], fields: dict) -> None:
    conn.executemany(f"UPDATE impressions SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                     [[*fields.values(), i] for i in ids])


async def _write_all(reels: dict[str, list[dict]], bias, parallel: int) -> tuple[dict, list[str]]:
    written, errors = {}, []
    gate = asyncio.Semaphore(parallel)
    async with aiohttp.ClientSession() as session:
        async def one(code: str, rows: list[dict]):
            row = rows[0]
            day = datetime.date.fromisoformat((row["shown_at"] or "")[:10] or datetime.date.today().isoformat())
            async with gate:
                try:
                    judgement = await evaluate_value_async(
                        session, meta_from_row(row), row["transcript"] or "", row["transcript_language"],
                        row["cover_text"] or "", today=day, interests=bias.interests, persona=bias.persona, timeout=120)
                    written[code] = text_fields(judgement)
                except ScoringError as exc:
                    errors.append(f"{code}: {exc}")
        await asyncio.gather(*(one(code, rows) for code, rows in reels.items()))
    return written, errors


def write_pending(conn: sqlite3.Connection, account: str, *, parallel: int = 2, progress=print) -> dict:
    """Write the text of every reel Jev judged for this account that has none yet: {"done", "failed", "left"}."""
    reels = pending(conn, account)
    if not reels:
        return {"done": 0, "failed": 0, "left": 0}
    if not llm_serving():
        progress(f"✍️  {len(reels)} reels are waiting for their summaries: LM Studio isn't serving the model.")
        return {"done": 0, "failed": 0, "left": len(reels)}
    progress(f"✍️  Writing topics and summaries for {len(reels)} reels with the local LLM …")
    written, errors = asyncio.run(_write_all(reels, load_bias(account), parallel))
    for code, fields in written.items():
        save(conn, [r["id"] for r in reels[code]], fields)
    conn.commit()
    if errors:
        progress(f"  ⚠️  {len(errors)} failed (they're tried again next time), e.g. {errors[0]}")
    return {"done": len(written), "failed": len(errors), "left": len(reels) - len(written)}


def main() -> None:
    parser = argparse.ArgumentParser(description="The local LLM's topics and summaries for reels Jev judged")
    parser.add_argument("--account", help="Account (required when several are ACTIVE)")
    args = parser.parse_args()
    init_db()
    account, problem = choose_account(list_accounts(), args.account)
    if problem:
        print(f"⚠️  {problem}")
        return
    conn = get_connection()
    try:
        done = write_pending(conn, account)
    finally:
        conn.close()
    print(f"✅  Wrote {done['done']} reels' text" + (f"; {done['failed']} failed" if done["failed"] else "")
          + (f"; {done['left']} still waiting" if done["left"] - done["failed"] > 0 else ""))


if __name__ == "__main__":
    main()
