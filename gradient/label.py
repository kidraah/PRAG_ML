"""
PRAG — the Gradient's teacher: Jev judges every item for every persona.

Each persona's preference model learns from labels, and a feed alone has no
negatives worth the name, so Jev (perception/jev.py, the persona question
the head-to-head measured) judges every item in the corpus for each
persona: would they watch it to the end and engage? Its graded answer is
kept (score, 0-1) apart from the binary label the classifiers learn
(score >= THRESHOLD: "watches most of it" or more).

Labels are kept per persona fingerprint: change a persona's text and its
items are judged again. A run stops before it spends more than
config.JEV_MAX_SPEND_USD; what's left is labelled next run.

    python -m gradient.label [--account mlproject01] [--parallel 16]
"""

import argparse
import asyncio
import os
import time

import aiohttp
from dotenv import load_dotenv

from agent import bias as biases
from config import JEV_MAX_SPEND_USD
from gradient import store
from perception import jev

load_dotenv(os.path.join(store.PROJECT_ROOT, ".env"))
THRESHOLD = 0.5

PENDING_SQL = """
SELECT i.* FROM items i
WHERE NOT EXISTS (SELECT 1 FROM labels l WHERE l.item_id = i.id AND l.account = ? AND l.teacher = 'jev'
                  AND l.persona_version = ?)
ORDER BY CASE i.origin WHEN 'instagram' THEN 0 WHEN 'generated' THEN 1 ELSE 2 END, i.id
"""


def reel_text(item: dict) -> str:
    """The item as Jev reads it: the parts a reel has, in the order the session's judges get them."""
    lines = []
    if item.get("creator"):
        lines.append(f"Creator: {item['creator']}")
    if item.get("title") and item.get("origin") in ("wikipedia", "rss"):
        lines.append(f"Title: {item['title']}")
    if item.get("caption"):
        lines.append(f"Caption: {item['caption']}")
    if item.get("on_screen"):
        lines.append(f"Text on screen: {item['on_screen']}")
    if item.get("transcript"):
        lines.append(f"Speech transcript: {item['transcript']}")
    if len(lines) <= 2 and not item.get("transcript"):
        lines.append(f"Text: {item['text']}")
    return "\n".join(lines)[:2400]


def pending(conn, account: str, fingerprint: str) -> list[dict]:
    return [dict(row) for row in conn.execute(PENDING_SQL, (account, fingerprint))]


async def label_account(conn, account: str, parallel: int = 16, max_spend: float = JEV_MAX_SPEND_USD,
                        progress=print, should_stop=lambda: False) -> dict:
    bias = biases.load(account)
    items = pending(conn, account, bias.fingerprint)
    version = jev.jev_version(bias.interests, persona=True, vision=False)
    stats = {"done": 0, "failed": 0, "usd": 0.0, "left": len(items)}
    if not items:
        return stats
    progress(f"{account}: Jev is judging {len(items)} items")
    gate = asyncio.Semaphore(parallel)
    started = time.monotonic()
    errors: list[str] = []

    async with aiohttp.ClientSession() as session:
        async def one(item):
            if should_stop() or stats["usd"] >= max_spend:
                return
            async with gate:
                if should_stop() or stats["usd"] >= max_spend:
                    return
                try:
                    score = await jev.judge_async(session, reel_text(item), bias.interests, persona=True,
                                                  labels=False, vision=False)
                except jev.JevError as exc:
                    stats["failed"] += 1
                    errors.append(str(exc))
                    return
                stats["usd"] += score.cost
                store.add_label(conn, item["id"], account, bias.fingerprint, "jev", version, score.value_score,
                                int(score.value_score >= THRESHOLD))
                stats["done"] += 1
                if stats["done"] % 500 == 0:
                    conn.commit()
                    rate = stats["done"] / (time.monotonic() - started) * 60
                    progress(f"{account}: {stats['done']}/{len(items)} judged ({rate:.0f}/min, ${stats['usd']:.3f})")

        await asyncio.gather(*(one(item) for item in items))
    conn.commit()
    stats["left"] = len(items) - stats["done"]
    if stats["usd"] >= max_spend:
        progress(f"{account}: stopped at the ${max_spend:.2f} cap (config.JEV_MAX_SPEND_USD); the rest wait")
    if errors:
        progress(f"{account}: {len(errors)} failed (tried again next run), e.g. {errors[0][:160]}")
    progress(f"{account}: {stats['done']} judged, ${stats['usd']:.3f}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Jev labels every corpus item for each persona")
    parser.add_argument("--account", choices=store.ACCOUNTS)
    parser.add_argument("--parallel", type=int, default=16)
    args = parser.parse_args()
    conn = store.connect()
    for account in [args.account] if args.account else store.ACCOUNTS:
        asyncio.run(label_account(conn, account, args.parallel, progress=lambda m: print(m, flush=True)))
    conn.close()


if __name__ == "__main__":
    main()
