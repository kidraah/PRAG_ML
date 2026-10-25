"""
PRAG — written answers: what the persona language model learns to answer with.

A grounded answer can't be assembled from a template (the first adapters
learned to paste captions that way). Here the writer LLM (generate.py's
backend) is given the persona and a few reels from its feed on one story
or topic, and writes what a friend would ask about them and how this person
would answer: three to five sentences, first person, saying what the reels
report and what the persona makes of it, each claim cited [n], no pasted
captions or hashtags. A share of the tasks are debate turns: the other
persona's real take on the same subject, which this persona answers from
its own reels.

Every task draws its reels from one split, so answers stay on their side of
train / validation / test. lora.build_examples trains on them.

    python -m gradient.answers --account mlproject01 --n 3000
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp

from agent import bias as biases
from gradient import generate, store

PER_CALL = 4
DEBATE_SHARE = 0.2
WRITER = f"{generate.GENERATOR}|answers-v1"

SYSTEM = """You write training examples for a chatbot that speaks as the person described below. For each numbered \
task you get a few Instagram reels from this person's own feed, numbered [1], [2], …

Write:
- question: something a friend might naturally ask this person about the subject of those reels (vary it: opinions, \
what's going on, should I…, kya scene hai…; sometimes Hinglish). For a debate task, the question is given.
- answer: this person's reply, 3-5 sentences, first person, in their voice and language (English or Hinglish). Say \
what the reels actually report or argue, in your own words, and what this person makes of it. Put the reel's number \
right after each claim it supports, like [2]. Never paste captions, hashtags or emojis. They can disagree with a \
reel, and they say so when the reels are thin. Don't add facts that aren't in the reels.

Answer in JSON: {"examples": [{"task": 1, "question": "...", "answer": "..."}, ...]}"""

SCHEMA = {"name": "answers", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["examples"],
    "properties": {"examples": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["task", "question", "answer"],
        "properties": {"task": {"type": "integer"}, "question": {"type": "string"}, "answer": {"type": "string"}}}}}}}

TABLE = """
CREATE TABLE IF NOT EXISTS answers (
    id         TEXT PRIMARY KEY,
    account    TEXT NOT NULL,
    kind       TEXT NOT NULL,           -- answer | debate
    items      TEXT NOT NULL,           -- JSON list of item ids, in the order they're numbered
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    split      TEXT NOT NULL,
    writer     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def reel(n: int, item: dict) -> str:
    return f"[{n}] {item.get('creator') or item.get('source_name') or 'a page'}: {item['text'][:450]}"


def tasks(conn, account: str, n: int, seed: int = 3) -> list[dict]:
    """n tasks: reels from this feed sharing a story (or else a topic) and a split; some are debate turns."""
    rng = random.Random(f"{seed}|{account}")
    rows = [dict(r) for r in conn.execute("SELECT i.* FROM feed f JOIN items i ON i.id = f.item_id WHERE f.account = ?",
                                          (account,))]
    by_group, by_topic = {}, {}
    for r in rows:
        by_group.setdefault((r["group_id"], r["split"]), []).append(r)
        by_topic.setdefault((r["topic"] or "other", r["split"]), []).append(r)
    rivals: dict[str, list[str]] = {}
    for row in conn.execute("SELECT i.topic, t.take FROM takes t JOIN items i ON i.id = t.item_id WHERE t.account <> ? "
                            "AND t.action <> 'skip'", (account,)):
        rivals.setdefault(row["topic"] or "other", []).append(row["take"])
    done = {r[0] for r in conn.execute("SELECT id FROM answers WHERE account = ?", (account,))}
    anchors = rows[:]
    rng.shuffle(anchors)
    out = []
    for anchor in anchors:
        if len(out) >= n:
            break
        group = [r for r in by_group[(anchor["group_id"], anchor["split"])] if r["id"] != anchor["id"]]
        topic = [r for r in by_topic[(anchor["topic"] or "other", anchor["split"])] if r["id"] != anchor["id"]]
        pool = group if len(group) >= 2 else topic
        if len(pool) < 2:
            continue
        items = [anchor] + rng.sample(pool, min(len(pool), rng.randint(2, 3)))
        rng.shuffle(items)
        task_id = f"{account}-{anchor['id']}"
        if task_id in done:
            continue
        rival = rivals.get(anchor["topic"] or "other")
        debate = bool(rival) and rng.random() < DEBATE_SHARE
        subject = (anchor["topic"] or "this").replace("_", " ")
        out.append({"id": task_id, "account": account, "items": items, "split": anchor["split"],
                    "kind": "debate" if debate else "answer",
                    "question": f"Debate on {subject}. The other side said: “{rng.choice(rival)}” Answer them."
                    if debate else None})
    return out


def prompt(batch: list[dict]) -> str:
    parts = []
    for i, t in enumerate(batch, 1):
        head = f"Task {i}" + (f" (debate; the question is fixed: {t['question']})" if t["question"] else "")
        parts.append(head + "\n" + "\n".join(reel(n, it) for n, it in enumerate(t["items"], 1)))
    return "\n\n".join(parts)


async def run(account: str, n: int, parallel: int = 12, progress=print, max_spend: float = 1.5) -> dict:
    bias = biases.load(account)
    conn = store.connect()
    conn.executescript(TABLE)
    todo = tasks(conn, account, n)
    stats = {"done": 0, "failed": 0, "usd": 0.0}
    if not todo:
        conn.close()
        return stats
    progress(f"{account}: writing {len(todo)} answers")
    system = SYSTEM + "\n\n" + generate.viewer_text(bias)
    gate, spend, started = asyncio.Semaphore(parallel), {"usd": 0.0}, time.monotonic()
    async with aiohttp.ClientSession() as session:
        async def one(batch):
            async with gate:
                if spend["usd"] >= max_spend:
                    return
                url, headers, body = generate.chat_request([{"role": "system", "content": system},
                                                            {"role": "user", "content": prompt(batch)}], SCHEMA)
                try:
                    examples = (await generate._ask(session, url, headers, body, 180, spend))["examples"]
                except Exception:  # noqa: BLE001 - left for the next run
                    stats["failed"] += len(batch)
                    return
                for ex in examples if isinstance(examples, list) else []:
                    if not isinstance(ex, dict):
                        continue
                    k = ex.get("task")
                    if not (isinstance(k, int) and 1 <= k <= len(batch)) or len(str(ex.get("answer", "")).split()) < 12:
                        continue
                    t = batch[k - 1]
                    conn.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (t["id"], account, t["kind"], json.dumps([it["id"] for it in t["items"]]),
                                  t["question"] or str(ex["question"]).strip(), str(ex["answer"]).strip(), t["split"],
                                  WRITER, store.now()))
                    stats["done"] += 1
                conn.commit()
                if stats["done"] and stats["done"] % 400 < PER_CALL:
                    progress(f"{account}: {stats['done']} answers ({stats['done'] / (time.monotonic() - started) * 60:.0f}/min, "
                             f"${spend['usd']:.3f})")

        await asyncio.gather(*(one(todo[i:i + PER_CALL]) for i in range(0, len(todo), PER_CALL)))
    conn.close()
    stats["usd"] = spend["usd"]
    progress(f"{account}: {stats['done']} answers written (${spend['usd']:.3f})")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the persona language model's grounded answers")
    parser.add_argument("--account", choices=store.ACCOUNTS)
    parser.add_argument("--n", type=int, default=3000)
    args = parser.parse_args()
    for account in [args.account] if args.account else store.ACCOUNTS:
        asyncio.run(run(account, args.n, progress=lambda m: print(m, flush=True)))


if __name__ == "__main__":
    main()
