"""
PRAG — Daily Run: scroll, then write the digest (the one command)

    python daily.py --account NAME [--engagement live|dry|mixed] [--minutes 40] [--headless] [--start-llm] [--open]
    python daily.py --account NAME --digest-only

--engagement (default: config.DAILY_ENGAGEMENT, "mixed"):
  live    likes, saves and follows (within the per-session caps)
  dry     watch time only (--dry-run is short for this)
  mixed   live for the day's first session, dry runs after — at most one
          session's worth of likes, saves and follows a day

1. Doesn't scroll (the digest is still refreshed) when the account already
   had DAILY_MAX_MINUTES today or the Mac is on battery below MIN_BATTERY.
   Two daily runs never overlap.
2. Scrolls one session with the LLM when LM Studio is serving the model.
   Otherwise, or when the LLM fails to warm up, the quick rules steer and
   the trained model scores (python -m ml.train --save), so the digest
   still fills. LM Studio is started and the model loaded only with
   --start-llm.
3. Categorises the reels it hasn't yet (for the panel's Feed tab), when
   LM Studio is up.
4. Rebuilds the digest for the last DIGEST_HOURS — digests/<account>/
   latest.html is always the newest — and posts a notification. A login
   that needs a security code notifies you while it waits for you.

Everything printed also goes to logs/<account>/<date>_<time>.log (and only
there when FEEDLAB_LOG_ONLY=1, as in scheduled runs). logs/daily.lock names
the run in progress, so launch.py can follow and stop it.
"""

import argparse
import asyncio
import contextlib
import datetime
import fcntl
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

from dotenv import load_dotenv
load_dotenv()

import main as agent
from agent.bias import load as load_bias
from automation.notify import notify
from config import DAILY_ENGAGEMENT, DAILY_MAX_MINUTES, DIGEST_HOURS, ENGAGEMENT_MODES, MIN_BATTERY, SESSION_MINUTES
from database.db import (
    choose_account, engaged_today, get_connection, init_db, list_accounts, minutes_today, reset_cooldown_accounts,
)
from ml import feed_model
from ml.scorer import ModelScorer
from perception.categorize import categorize_pending
from perception.local_evaluator import llm_serving
from stories.digest import build_digest, story_title, write_digest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOCK_PATH = os.path.join(LOG_DIR, "daily.lock")
KEEP_LOGS = 60  # per account
MIN_SESSION_MINUTES = 5  # less than this left in the day's budget isn't worth opening the browser for

# What the notification says when a session didn't happen or ended early.
SESSION_PROBLEMS = {
    "no_account": "No account to use.",
    "busy": "Another PRAG run is using this account.",
    "login_needed": "Instagram needs you to log in: start a session with the browser window (not headless) and log in there.",
    "llm_unavailable": "Neither the LLM nor the fallback could score.",
    "no_reels": "Instagram showed no reels.",
    "stuck": "The feed stopped advancing.",
    "interrupted": "The session was interrupted.",
    "error": "The session crashed — see the log.",
}
FINE = {"ok", "stuck"}  # a feed that stops advancing still leaves a full session behind


# ── Guards ────────────────────────────────────────────────────────────

def battery() -> tuple[bool, int | None]:
    """(running on battery, charge %) — (False, None) on a Mac without one or if pmset fails."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False, None
    charge = re.search(r"(\d+)%", out)
    return "'Battery Power'" in out, int(charge.group(1)) if charge else None


def skip_reason(account: str, used: float, on_battery: bool, charge: int | None) -> str | None:
    """Why not to scroll now, or None."""
    if DAILY_MAX_MINUTES - used < MIN_SESSION_MINUTES:
        return f"@{account} already scrolled {used:.0f} min today (daily limit {DAILY_MAX_MINUTES})"
    if on_battery and charge is not None and charge < MIN_BATTERY:
        return f"the Mac is on battery at {charge}% (below {MIN_BATTERY}%)"
    return None


# ── Session ───────────────────────────────────────────────────────────

def choose_dry_run(engagement: str, account: str) -> tuple[bool, str]:
    """(dry run?, how it was decided) for this run's session."""
    if engagement == "mixed":
        if engaged_today(account):
            return True, "mixed → dry run (today's live session already ran)"
        return False, "mixed → live (first session today)"
    return engagement == "dry", {"live": "live", "dry": "dry run"}[engagement]


def session_args(account: str, minutes: float, dry_run: bool, mode: str, headless: bool = False) -> argparse.Namespace:
    return argparse.Namespace(account=account, minutes=minutes, max_reels=None, dry_run=dry_run,
                              rules=mode == "rules", observe=False, headless=headless)


async def scroll(account: str, minutes: float, dry_run: bool, start_llm: bool, headless: bool = False) -> dict:
    """One session: with the LLM when it's there, else (or if it fails to warm up) rules + trained model."""
    has_model = ModelScorer.available(bias=load_bias(account))
    fallback = ("the trained model scores" if has_model
                else "no trained model for this bias yet, so reels are only recorded for review")
    mode = "llm" if start_llm or llm_serving() else "rules"
    if mode == "rules":
        print(f"🧠  LM Studio isn't serving the model: the quick rules steer, {fallback}.")
    outcome = await agent.run(session_args(account, minutes, dry_run, mode, headless))
    if outcome["status"] == "llm_unavailable" and mode == "llm":
        print(f"🧠  The LLM didn't respond: scrolling with the quick rules instead, {fallback}.")
        mode = "rules"
        outcome = await agent.run(session_args(account, minutes, dry_run, mode, headless))
    outcome["mode"] = mode if mode == "llm" or has_model else "record"
    outcome["dry_run"] = dry_run
    return outcome


def categorize_new(account: str) -> None:
    """Label the reels this account was shown that have no category yet (never fails the run)."""
    if not llm_serving():
        return
    conn = get_connection()
    try:
        done = categorize_pending(conn, account, progress=lambda message: None)
        if done["done"]:
            print(f"\n🏷  Categorised {done['done']} new reels for the Feed tab"
                  + (f" ({done['failed']} left for next time)" if done["failed"] else "") + ".")
    except Exception as exc:  # noqa: BLE001
        print(f"\n⚠️  Couldn't categorise the new reels: {exc}")
    finally:
        conn.close()


def update_feed_model() -> None:
    """Relearn the feed's topics with the new reels, every account's (never fails the run)."""
    conn = get_connection()
    try:
        model = feed_model.train(conn, progress=lambda message: None)
        if model:
            print(f"\n🧠  Feed model: {model['k']} topics from {model['reels']} reels across "
                  f"{len(model['accounts'])} accounts.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n⚠️  Couldn't update the feed model: {exc}")
    finally:
        conn.close()


# ── Output ────────────────────────────────────────────────────────────

class Tee:
    """A stream that writes to the terminal and the run's log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):  # encoding, isatty, fileno … come from the terminal stream
        return getattr(self.streams[0], name)


@contextlib.contextmanager
def logging_to(path: str):
    """Print to the log file too — only to it with FEEDLAB_LOG_ONLY=1 (a scheduled run, where nobody watches)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", buffering=1) as log:
        out, err = sys.stdout, sys.stderr
        log_only = os.environ.get("FEEDLAB_LOG_ONLY") == "1"
        sys.stdout, sys.stderr = (log, log) if log_only else (Tee(out, log), Tee(err, log))
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = out, err


def current_run() -> dict | None:
    """The daily run in progress — {"pid", "account", "log", "started"} — or None."""
    try:
        with open(LOCK_PATH) as fh:
            info = json.load(fh)
        os.kill(info["pid"], 0)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    command = subprocess.run(["ps", "-p", str(info["pid"]), "-o", "command="], capture_output=True, text=True).stdout
    return info if "daily.py" in command else None  # not a reused pid


def prune_logs(folder: str, keep: int = KEEP_LOGS) -> None:
    for old in sorted(glob.glob(os.path.join(folder, "*.log")))[:-keep]:
        os.remove(old)


def session_line(outcome: dict | None, skipped: str | None) -> str:
    if skipped:
        return f"Didn't scroll: {skipped}."
    if outcome is None:
        return "Digest only."
    if outcome["status"] not in FINE:
        return SESSION_PROBLEMS.get(outcome["status"], outcome["status"]).format(account=outcome.get("account", ""))
    by = {"llm": "LLM", "rules": "rules + model"}.get(outcome.get("mode"), "rules; recorded for review")
    how = "dry run" if outcome.get("dry_run") else "live"
    scored = outcome.get("scored") or 0
    found = f", {outcome.get('informative', 0)} of {scored} informative" if scored else ""
    return f"Scrolled {outcome.get('minutes', 0):.0f} min ({by}, {how}): {outcome.get('reels', 0)} reels{found}."


# ── Run ───────────────────────────────────────────────────────────────

def daily(args: argparse.Namespace, account: str) -> int:
    print(f"📅  PRAG daily run — @{account}, {datetime.datetime.now():%a %d %b %H:%M}")
    outcome, skipped = None, None
    if not args.digest_only:
        used = minutes_today(account)
        skipped = skip_reason(account, used, *battery())
        if skipped:
            print(f"⏭  Not scrolling: {skipped}.")
        else:
            dry_run, how = choose_dry_run(args.engagement, account)
            print(f"🎚  Engagement: {how}  (choose with --engagement {'|'.join(ENGAGEMENT_MODES)})")
            minutes = min(args.minutes, DAILY_MAX_MINUTES - used)
            outcome = asyncio.run(scroll(account, minutes, dry_run, args.start_llm, args.headless))
            if outcome["status"] == "interrupted":
                return 130

    categorize_new(account)
    update_feed_model()
    print(f"\n📰  Building the digest for the last {DIGEST_HOURS} h (checking stories against the news) …")
    conn = get_connection()
    try:
        digest = build_digest(conn, account, DIGEST_HOURS)
    finally:
        conn.close()
    page, _data = write_digest(digest)
    for story, check in list(zip(digest.stories, digest.checks))[:8]:
        print(f"  {story.score:.2f}  {story_title(story, check)[:48]:48}  · {check.label}")
    latest = os.path.join(os.path.dirname(page), "latest.html")
    print(f"\n  {len(digest.stories)} stories · {latest}")

    line = session_line(outcome, skipped)
    top = story_title(digest.stories[0], digest.checks[0]) if digest.stories else ""
    count = f"{len(digest.stories)} {'story' if len(digest.stories) == 1 else 'stories'}"
    message = f"{count}. Top: {top}" if top else "Nothing informative in the last day yet."
    notify(f"PRAG · @{account}", message, subtitle=line)
    if args.open:
        subprocess.run(["open", latest], check=False)
    return 0 if outcome is None or outcome["status"] in FINE else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRAG daily run: scroll one session, then write the digest")
    parser.add_argument("--account", help="Instagram account (required when several are ACTIVE)")
    parser.add_argument("--minutes", type=float, default=SESSION_MINUTES, help="Session length")
    parser.add_argument("--engagement", choices=list(ENGAGEMENT_MODES), default=DAILY_ENGAGEMENT,
                        help="; ".join(f"{mode}: {what}" for mode, what in ENGAGEMENT_MODES.items())
                        + f" (default: {DAILY_ENGAGEMENT})")
    parser.add_argument("--dry-run", dest="engagement", action="store_const", const="dry",
                        help="Same as --engagement dry")
    parser.add_argument("--headless", action="store_true",
                        help="No browser window (Instagram can tell; a login that needs a code needs the window)")
    parser.add_argument("--start-llm", action="store_true",
                        help="Start LM Studio and load the model if it isn't serving (otherwise: rules + model)")
    parser.add_argument("--digest-only", action="store_true", help="Don't scroll; just rebuild the digest")
    parser.add_argument("--open", action="store_true", help="Open the digest in the browser when done")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    # Ctrl+C — or the panel's Stop — ends the run cleanly even when whatever started it ignores SIGINT
    # (a shell's background jobs do, and children inherit that).
    signal.signal(signal.SIGINT, signal.default_int_handler)

    init_db()
    reset_cooldown_accounts()
    account, problem = choose_account(list_accounts(), args.account)
    if problem:
        print(f"⚠️  {problem}")
        notify("PRAG", problem)
        return 2

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOCK_PATH, "a+") as lock:  # not "w": that would wipe the running holder's details
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("⏭  Another daily run is in progress.")
            return 0
        folder = os.path.join(LOG_DIR, account)
        log_path = os.path.join(folder, f"{datetime.datetime.now():%Y-%m-%d_%H%M}.log")
        lock.seek(0)
        lock.truncate()
        json.dump({"pid": os.getpid(), "account": account, "log": log_path, "started": time.time()}, lock)
        lock.flush()
        with logging_to(log_path) as path:
            try:
                return daily(args, account)
            except KeyboardInterrupt:
                print("\n⏹  Stopped.")
                return 130
            finally:
                print(f"\n📝  Log: {path}")
                prune_logs(folder)


if __name__ == "__main__":
    sys.exit(main())
