"""
FeedLab — Pre-seed an Account by Following Creators

Gives a fresh account a head start: follows a list of accounts (by default
seeds/news_india.txt), one profile at a time at a person's pace. Only
verified accounts are followed, so a mistyped handle or an impostor page
is skipped instead of followed. If the account isn't logged in yet, it
logs in with the password from accounts.json, or waits for you to log in
by hand in the browser window.

    python -m tools.follow_seed --account NAME [--check] [--file PATH] [handle ...]

--check only visits the profiles and reports what it would do.
"""

import argparse
import asyncio
import os
import random
from collections import Counter

from automation.browser_driver import (
    PROJECT_ROOT,
    close_browser,
    follow_open_profile,
    init_browser,
    open_profile,
    profile_dir,
)
from automation.login import ensure_logged_in
from database.db import choose_account, list_accounts

DEFAULT_SEEDS = os.path.join(PROJECT_ROOT, "seeds", "news_india.txt")


def read_handles(path: str) -> list[str]:
    """Handles from a seed file: one per line, '#' starts a comment, '@' is optional."""
    handles = []
    with open(path) as fh:
        for line in fh:
            handle = line.split("#", 1)[0].strip().lstrip("@")
            if handle and handle not in handles:
                handles.append(handle)
    return handles


async def seed(account: str, handles: list[str], check_only: bool) -> Counter:
    outcomes: Counter = Counter()
    _context, page = await init_browser(profile_dir(account))
    try:
        await page.goto("https://www.instagram.com/", wait_until="domcontentloaded")
        if not await ensure_logged_in(page, account):
            print("❌  Not logged in.")
            return outcomes
        verb = "Checking" if check_only else "Following"
        print(f"  {verb} {len(handles)} accounts for @{account} …")
        for i, handle in enumerate(handles):
            state = await open_profile(page, handle)
            if state.get("unavailable") or not state.get("found"):
                outcome = "not found"
            elif not state.get("verified"):
                outcome = "skipped: not verified"
            elif (state.get("follow") or {}).get("text") in ("Following", "Requested"):
                outcome = "already following"
            elif check_only:
                outcome = "would follow"
            else:
                outcome = "followed" if await follow_open_profile(page) else "follow failed"
            outcomes[outcome] += 1
            print(f"    @{handle:<20} {state.get('followers') or '?':>7} followers  → {outcome}")
            if i < len(handles) - 1:
                await asyncio.sleep(random.uniform(2, 4) if check_only else random.uniform(6, 15))
    finally:
        await close_browser()
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("handles", nargs="*", help="Accounts to follow (default: the seed file)")
    parser.add_argument("--account", help="Which of your accounts does the following")
    parser.add_argument("--file", default=DEFAULT_SEEDS, help="Seed file, one handle per line")
    parser.add_argument("--check", action="store_true", help="Only check the handles; follow nothing")
    args = parser.parse_args()

    account, problem = choose_account(list_accounts(), args.account)
    if problem:
        print(f"⚠️  {problem}")
        return
    handles = [h.lstrip("@") for h in args.handles] or read_handles(args.file)
    outcomes = asyncio.run(seed(account, handles, args.check))
    if outcomes:
        print("\n  " + " · ".join(f"{name}: {n}" for name, n in outcomes.most_common()))


if __name__ == "__main__":
    main()
