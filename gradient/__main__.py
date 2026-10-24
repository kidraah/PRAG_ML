"""
PRAG — the Gradient from the terminal.

    python -m gradient status                     what's in the corpus, per persona, and the latest models
    python -m gradient build                      first build: earlier corpus, sessions, headlines, labels, models
    python -m gradient update [--lora]            what daily runs need: new session reels → labels → retrain
    python -m gradient chat --account mlproject02 "What's going on with air pollution?"
    python -m gradient debate --topic "fuel prices" [--rounds 2]
    python -m gradient compare "Is the new expressway worth the tolls?"
"""

import argparse
import asyncio
import json
import os
import time

from gradient import generate, ingest, label, prefs, store


def status(conn) -> dict:
    out = {"items": conn.execute("SELECT COUNT(*) FROM items").fetchone()[0],
           "origins": dict(conn.execute("SELECT origin, COUNT(*) FROM items GROUP BY origin").fetchall()),
           "accounts": {}}
    for account in store.ACCOUNTS:
        out["accounts"][account] = {
            "feed": dict(conn.execute("SELECT via, COUNT(*) FROM feed WHERE account = ? GROUP BY via",
                                      (account,)).fetchall()),
            "jev_labels": conn.execute("SELECT COUNT(DISTINCT item_id) FROM labels WHERE account = ? AND teacher = "
                                       "'jev'", (account,)).fetchone()[0],
            "takes": conn.execute("SELECT COUNT(*) FROM takes WHERE account = ?", (account,)).fetchone()[0],
            "runs": [dict(r) for r in conn.execute("SELECT id, kind, version, created_at FROM runs WHERE account = ? "
                                                   "ORDER BY created_at DESC LIMIT 4", (account,))]}
    return out


def update(conn, *, lora_too: bool = False, progress=print, first: bool = False) -> dict:
    """New session reels in, everything labelled, headlines topped up, both preference models retrained."""
    done = {}
    if first:
        done["legacy"] = ingest.ingest_legacy(conn)
        progress(f"Earlier corpus: {done['legacy']}")
    done["sessions"] = ingest.ingest_sessions(conn)
    progress(f"New Instagram reels: {done['sessions']}")
    for account in store.ACCOUNTS:
        done[f"label_{account}"] = asyncio.run(label.label_account(conn, account, progress=progress))
    for account in store.ACCOUNTS:
        done[f"recommended_{account}"] = ingest.recommend(conn, account)
    if os.environ.get("OPENROUTER_API_KEY") or generate.GEN_BACKEND == "local":
        for account in store.ACCOUNTS:  # the persona's reaction to what it was served, for the language model
            done[f"takes_{account}"] = asyncio.run(generate.write_takes(account, progress=progress))["done"]
    for account in store.ACCOUNTS:
        done[f"model_{account}"] = prefs.train(conn, account, progress=progress)["run"]
    done["model_feed"] = prefs.train(conn, store.ACCOUNTS[0], kind="feed", progress=progress)["run"]
    if lora_too:
        from gradient import lora

        lora.write_shared_tests(conn)
        for account in store.ACCOUNTS:
            done[f"lora_{account}"] = lora.train(conn, account, resume=lora.latest_adapter(account) is not None,
                                                 progress=progress)["version"]
    return done


def talk(kind: str, params: dict) -> None:
    from gradient import chat

    job = chat.submit(kind, params)
    shown = 0
    while True:
        job = chat.job(job["id"])
        for event in job["events"][shown:]:
            if event["type"] in ("answer", "turn"):
                head = f"@{event['account']}" + (f" (round {event['round']})" if event.get("round") else "")
                print(f"\n{head} [{event['adapter'] or 'base model'}]:\n{event['text']}")
                for n in event["cited"]:
                    e = event["evidence"][n - 1]
                    print(f"   [{n}] {e['origin_name']}: {e.get('creator') or ''} {e['text'][:110]}")
            elif event["type"] == "moderator":
                print(f"\nModerator:\n{event['text']}")
            elif event["type"] == "gap":
                print("\n" + json.dumps({k: v for k, v in event.items() if k != "type"}, indent=1))
        shown = len(job["events"])
        if job["status"] in ("done", "failed", "stopped"):
            if job.get("error"):
                print(f"\n⚠️  {job['error']}")
            return
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gradient", description="The Gradient: per-persona models")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("build")
    up = sub.add_parser("update")
    up.add_argument("--lora", action="store_true", help="also continue each persona's LoRA adapter")
    c = sub.add_parser("chat")
    c.add_argument("--account", required=True, choices=store.ACCOUNTS)
    c.add_argument("question")
    d = sub.add_parser("debate")
    d.add_argument("--topic", required=True)
    d.add_argument("--rounds", type=int, default=2)
    q = sub.add_parser("compare")
    q.add_argument("question")
    args = parser.parse_args()
    if args.cmd in ("chat", "debate", "compare"):
        params = {"chat": lambda: {"account": args.account, "question": args.question},
                  "debate": lambda: {"topic": args.topic, "rounds": args.rounds},
                  "compare": lambda: {"question": args.question}}[args.cmd]()
        return talk(args.cmd, params)
    conn = store.connect()
    try:
        if args.cmd == "status":
            print(json.dumps(status(conn), indent=1))
        else:
            print(json.dumps(update(conn, lora_too=getattr(args, "lora", False), first=args.cmd == "build",
                                    progress=lambda m: print(m, flush=True)), indent=1))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
