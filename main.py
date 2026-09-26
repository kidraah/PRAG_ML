"""
FeedLab — Curation Agent (P.R.A.G.: Predictive Reel Accumulation & Grouping)

Phase 1: curate the account. The agent scrolls Instagram Reels on a
dedicated account and engages the way a person with config.INTERESTS
would — skipping fast, watching what's informative, watching valuable
reels through (the best of them twice) and liking / saving some of them,
following creators who keep delivering — so the account's feed converges
on valuable content. Every reel shown is logged; `python -m
dashboard.report` says whether the feed is curated yet.

    python main.py [--account NAME] [--minutes 40] [--max-reels N] [--dry-run] [--rules | --observe]

Without LM Studio (while the project is being built):
  --rules    steers with quick rules — watch reels from followed accounts and
             news-like captions, flick past the rest — and records every reel
  --observe  records every reel with the same watch time on each (no steering)
Recorded reels (caption, cover text, speech) are scored afterwards with
tools/review.py.

1. Starts LM Studio if needed and warms up Whisper, OCR and the LLM, then
   opens instagram.com/reels in the persistent browser profile.
2. FeedCatalog captures each reel's JSON as it arrives; ScoringPipeline
   scores it in the background, usually before it reaches the screen.
3. For the reel on screen: wait briefly for its score, decide, act, log,
   scroll.
"""

import argparse
import asyncio
import datetime
import random
import subprocess
import time
import traceback
from collections import Counter
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from agent.bandit import Decision, decide
from agent.creators import CreatorMemory
from agent.pipeline import ScoreResult, ScoringPipeline
from agent.rules import RuleVerdict, rule_score
from automation.browser_driver import (
    close_browser,
    follow_active_creator,
    init_browser,
    is_logged_in,
    like_active_reel,
    open_reels,
    profile_dir,
    save_active_reel,
    scroll_next,
    wait_for_first_reel,
)
from automation.feed_capture import FeedCatalog, ReelMeta
from automation.login import ensure_logged_in
from config import (
    AD_DWELL,
    LIKE_THRESHOLD,
    LLM_MODEL_NAME,
    MAX_DECISION_WAIT,
    MAX_FOLLOWS_PER_SESSION,
    MAX_LIKES_PER_SESSION,
    MAX_SAVES_PER_SESSION,
    OBSERVE_DWELL,
    SESSION_MINUTES,
    WATCH_THRESHOLD,
)
from database.db import (
    REVIEW_SCORER,
    choose_account,
    end_session,
    init_db,
    list_accounts,
    load_creator_history,
    log_impression,
    mark_account_used,
    reset_cooldown_accounts,
    start_session,
    update_impression_score,
)
from perception.local_evaluator import ScoringError, ensure_llm_ready, scorer_version

MAX_STUCK_SCROLLS = 3


class FeedStuck(Exception):
    """The feed stopped advancing."""


@dataclass
class Performed:
    liked: bool = False
    saved: bool = False
    followed: bool = False


# ── Action Executor ───────────────────────────────────────────────────

async def _sleep_until(deadline: float) -> None:
    delay = deadline - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)


async def execute_decision(
    page, decision: Decision, meta: ReelMeta | None, shown_at: float, *, dry_run: bool, rng: random.Random
) -> Performed:
    """Carry out a decision on the reel on screen, keeping total dwell on target.

    Engagement happens part-way through the watch, like a person reacting
    to what they're seeing. In a dry run nothing is clicked (watch time
    still happens, so the feed still learns from it).
    """
    done = Performed()
    if (decision.like or decision.save or decision.follow) and not dry_run:
        await _sleep_until(shown_at + decision.dwell * rng.uniform(0.35, 0.75))
        if decision.like:
            done.liked = await like_active_reel(page)
        if decision.save:
            await asyncio.sleep(rng.uniform(0.6, 1.6))
            done.saved = await save_active_reel(page)
        if decision.follow and meta is not None:
            await asyncio.sleep(rng.uniform(0.6, 1.6))
            done.followed = await follow_active_creator(page, meta.creator)
    await _sleep_until(shown_at + decision.dwell)
    return done


# ── Logging ───────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def impression_row(
    session_id: int, seq: int, code: str | None, meta: ReelMeta | None, state: dict, is_ad: bool,
    result: ScoreResult | None, decision: Decision, dwell: float, done: Performed,
) -> dict:
    row = {
        "session_id": session_id,
        "seq": seq,
        "code": code,
        "url": meta.url if meta else None,
        "creator": meta.creator if meta else state.get("creator"),
        "is_ad": int(is_ad),
        "decision": decision.action,
        "decision_basis": decision.basis,
        "explored": int(decision.explored),
        "dwell_seconds": round(dwell, 2),
        "planned": decision.planned,
        "liked": int(done.liked),
        "saved": int(done.saved),
        "followed": int(done.followed),
    }
    if meta is not None:
        row.update(
            caption=meta.caption,
            hashtags=" ".join(meta.hashtags),
            audio_type=meta.audio_type,
            audio_title=meta.audio_title,
            duration=meta.duration,
            like_count=meta.like_count,
            comment_count=meta.comment_count,
            repost_count=meta.repost_count,
            taken_at=meta.taken_at,
            inventory_source=meta.inventory_source,
        )
    if is_ad:
        row.update(score_basis="ad", scored_at=_utc_now())
    elif result is not None:
        row.update(result.score_fields(), scored_at=_utc_now())
    return row


def describe(
    seq: int, meta: ReelMeta | None, state: dict, is_ad: bool, result: ScoreResult | None,
    decision: Decision, dwell: float, done: Performed, dry_run: bool,
    rule: RuleVerdict | None = None, ready: bool = False,
) -> str:
    """One console line per reel (⚡ = flicked: known to be skippable before it appeared)."""
    if is_ad:
        who = "(ad)"
    elif meta and meta.creator:
        who = f"@{meta.creator}"
    else:
        who = f"@{state['creator']}" if state.get("creator") else "(unknown)"
    kind = f"{meta.audio_type or '?'} {meta.duration:.0f}s" if meta and meta.duration else ""

    if rule is not None:
        verdict = f"rule {rule.score:.1f} · {rule.reason}"
    elif result and result.basis == "score":
        verdict = f"{result.value_score:.2f} {result.topic}"
    elif result and result.basis == "keyword_gate":
        verdict = f"gate: {result.gate_keyword}"
    elif result and result.basis == "creator_prior":
        verdict = "creator: never valuable"
    elif result and result.basis == "llm_error":
        verdict = f"error: {result.error}"
    elif result and result.basis == "observed":
        found = [label for label, text in (("speech", result.transcript), ("cover text", result.cover_text)) if text]
        verdict = "recorded" + (f": {', '.join(found)}" if found else "")
    else:
        verdict = {"ad": "ad", "timeout": "not scored in time", "no_metadata": "no metadata",
                   "observe": "recording …"}.get(
            decision.basis, decision.basis
        )

    marks = " ".join(m for m, on in (("❤️", done.liked), ("🔖", done.saved), ("➕", done.followed)) if on)
    if dry_run and decision.planned:
        marks = f"(would {decision.planned})"
    if decision.explored:
        marks = (marks + " 🧭").strip()
    if decision.action == "SKIP" and ready and decision.basis != "ad":
        marks = ("⚡ " + marks).strip()
    line = f"#{seq:<4} {who[:26]:26} {kind:12} {verdict[:36]:36} → {decision.action:10} {dwell:5.1f}s {marks}"
    if result and result.summary and decision.action != "SKIP":
        line += f"\n        “{result.summary[:110]}”"
    return line


# ── Main Loop ─────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    """One curation session."""
    init_db()
    reset_cooldown_accounts()
    username, problem = choose_account(list_accounts(), args.account)
    if problem:
        print(f"⚠️  {problem}")
        return
    mark_account_used(username)
    observe, rules = args.observe, args.rules
    record = observe or rules  # the LLM is off: every reel is recorded for review

    if not record:
        try:
            await asyncio.to_thread(ensure_llm_ready)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            print(f"❌  {exc}")
            return

    scorer = REVIEW_SCORER if record else scorer_version()
    creators = CreatorMemory() if record else CreatorMemory.from_history(*load_creator_history(scorer, username))
    catalog = FeedCatalog()
    rng = random.Random()
    model = "none" if observe else "rules" if rules else LLM_MODEL_NAME
    session_id = start_session(username, dry_run=args.dry_run or observe, model=model, scorer=scorer)
    if observe:
        mode = "observe — recording every reel, no scoring or engagement"
    elif rules:
        mode = "rules — quick rules steer, every reel recorded" + (", no likes/saves/follows" if args.dry_run else "")
    else:
        mode = "dry run — no likes, saves or follows" if args.dry_run else "live"
    print(f"\n🚀  FeedLab curation — @{username}, session {session_id} ({mode}, {args.minutes} min)\n")

    pipeline: ScoringPipeline | None = None
    worker: asyncio.Task | None = None
    started = time.monotonic()
    seq = likes = saves = follows = 0
    tally: Counter = Counter()

    try:
        _context, page = await init_browser(profile_dir(username))

        def on_result(meta: ReelMeta, result: ScoreResult) -> None:
            # Scores that arrive after their reel was logged (decided on timeout).
            update_impression_score(session_id, meta.code, result.score_fields())

        pipeline = ScoringPipeline(
            catalog, creators, rng=rng,
            user_agent=await page.evaluate("navigator.userAgent"),
            on_result=on_result,
            use_llm=not record,
        )
        print("  🔥  Warming up Whisper and OCR …" if record else "  🔥  Warming up Whisper, OCR and the LLM …")
        warm_start = time.monotonic()
        try:
            await pipeline.warm_up()
        except ScoringError as exc:
            print(f"❌  LLM warm-up failed: {exc}")
            return
        print(f"  🔥  Ready in {time.monotonic() - warm_start:.0f}s")

        catalog.on_new_feed_reel = pipeline.enqueue
        catalog.attach(page)
        worker = asyncio.create_task(pipeline.run())

        await open_reels(page)
        if not await is_logged_in(page):
            if not await ensure_logged_in(page, username):
                print("❌  Not logged in — ending the session.")
                return
            await open_reels(page)
        await catalog.ingest_embedded(page)

        state = await wait_for_first_reel(page)
        if not state.get("src"):
            print("❌  No reel started playing.")
            return

        deadline = started + args.minutes * 60
        while time.monotonic() < deadline and (args.max_reels is None or seq < args.max_reels):
            seq += 1
            shown_at = time.monotonic()
            code = state.get("code")
            meta_ready = catalog.get(code) is not None  # its feed data arrived before it appeared
            meta = await catalog.wait_for(code, timeout=1.5)
            is_ad = meta.is_ad if meta is not None else bool(state.get("sponsored"))
            pipeline.set_current(code, None if is_ad else meta)
            # Could the verdict be known before the reel reached the screen (preloaded feed data)?
            ready_on_arrival = meta_ready and (record or pipeline.result(code) is not None)
            rule: RuleVerdict | None = None

            if observe:
                # Same neutral behaviour on every reel: the recording is judged afterwards.
                decision = (Decision("SKIP", rng.uniform(*AD_DWELL), "ad") if is_ad
                            else Decision("OBSERVE", rng.uniform(*OBSERVE_DWELL), "observe"))
            elif rules:
                rule = rule_score(meta) if meta is not None and not is_ad else None
                decision = decide(
                    value_score=rule.score if rule else None,
                    basis="ad" if is_ad else "no_metadata" if rule is None else "score",
                    duration=meta.duration if meta else state.get("duration"),
                    has_liked=bool(meta and meta.has_liked),
                    has_saved=bool(meta and meta.has_saved),
                    follow_ok=False,
                    likes_left=MAX_LIKES_PER_SESSION - likes,
                    saves_left=MAX_SAVES_PER_SESSION - saves,
                    rng=rng,
                    pre_analyzed=ready_on_arrival,
                )
                if rule is not None:
                    decision.basis = "rules"
            else:
                result = None
                if meta is not None and not is_ad:
                    result = await pipeline.wait_for(code, MAX_DECISION_WAIT - (time.monotonic() - shown_at))

                if is_ad:
                    basis = "ad"
                elif meta is None:
                    basis = "no_metadata"
                elif result is None:
                    basis = "timeout"
                else:
                    basis = result.basis
                value = result.value_score if result is not None else None
                follow_ok = (
                    meta is not None and value is not None and value >= LIKE_THRESHOLD
                    and follows < MAX_FOLLOWS_PER_SESSION and not meta.creator_following
                    and creators.qualifies_for_follow(meta.creator)
                )
                decision = decide(
                    value_score=value,
                    basis=basis,
                    duration=meta.duration if meta else state.get("duration"),
                    has_liked=bool(meta and meta.has_liked),
                    has_saved=bool(meta and meta.has_saved),
                    follow_ok=follow_ok,
                    likes_left=MAX_LIKES_PER_SESSION - likes,
                    saves_left=MAX_SAVES_PER_SESSION - saves,
                    rng=rng,
                    pre_analyzed=ready_on_arrival,
                )

            done = await execute_decision(page, decision, meta, shown_at, dry_run=args.dry_run, rng=rng)
            # Session caps count what was done — in a dry run, what would have been.
            likes += decision.like if args.dry_run else done.liked
            saves += decision.save if args.dry_run else done.saved
            follows += decision.follow if args.dry_run else done.followed
            if meta is not None:
                meta.has_liked = meta.has_liked or done.liked
                meta.has_saved = meta.has_saved or done.saved
                if done.followed:
                    meta.creator_following = True
                    creators.mark_following(meta.creator)
            dwell = time.monotonic() - shown_at

            latest = pipeline.result(code)  # may have landed during the dwell
            row = impression_row(session_id, seq, code, meta, state, is_ad, latest, decision, dwell, done)
            row["ready_on_arrival"] = int(ready_on_arrival)
            if rule is not None:
                row.update(rule_score=rule.score, rule_reason=rule.reason)
            log_impression(row)
            print(describe(seq, meta, state, is_ad, latest, decision, dwell, done, args.dry_run, rule, ready_on_arrival))

            tally[decision.action] += 1
            tally["flicked"] += decision.action == "SKIP" and ready_on_arrival and decision.basis != "ad"
            tally["timeout"] += decision.basis == "timeout"
            tally["liked"] += done.liked
            tally["saved"] += done.saved
            tally["followed"] += done.followed
            if latest is not None and latest.basis == "score":
                tally["scored"] += 1
                tally["informative"] += latest.value_score >= WATCH_THRESHOLD
                tally["valuable"] += latest.value_score >= LIKE_THRESHOLD

            prev_src = state.get("src")
            for _attempt in range(MAX_STUCK_SCROLLS):
                state = await scroll_next(page, prev_src)
                if state.get("src") and state.get("src") != prev_src:
                    break
            else:
                raise FeedStuck

    except FeedStuck:
        print("  🛑  The feed stopped advancing — ending the session.")
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n⏹  Session interrupted.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌  Fatal error: {exc}")
        traceback.print_exc()
    finally:
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        if pipeline is not None:
            await pipeline.close()
        end_session(session_id)
        await close_browser()

    minutes = (time.monotonic() - started) / 60
    if record:
        watched = tally["WATCH"] + tally["FULL_WATCH"] + tally["REWATCH"]
        steer = f" · rules watched {watched}, skipped {tally['SKIP']} ({tally['flicked']} flicked)" if rules else ""
        print(f"\n🏁  Session {session_id} — {seq} reels in {minutes:.1f} min{steer} · recorded for review")
        print(f"    Next: python -m tools.review export --account {username} --session {session_id}\n")
        return
    scored = tally["scored"]
    share = f"{tally['informative'] / scored:.0%}" if scored else "—"
    print(f"\n🏁  Session {session_id} — {seq} reels in {minutes:.1f} min")
    print(f"    informative: {tally['informative']} of {scored} scored ({share}) · valuable: {tally['valuable']}")
    print(
        f"    decisions: SKIP {tally['SKIP']} · WATCH {tally['WATCH']} · FULL_WATCH {tally['FULL_WATCH']}"
        f" · REWATCH {tally['REWATCH']} · decided before a score arrived: {tally['timeout']}"
    )
    if not args.dry_run:
        print(f"    engagement: ❤️ {tally['liked']} · 🔖 {tally['saved']} · ➕ {tally['followed']}")
    print("    Is the feed curated yet? →  python -m dashboard.report\n")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="FeedLab curation agent")
    parser.add_argument("--account", help="Instagram account to use (required when several are ACTIVE)")
    parser.add_argument("--minutes", type=float, default=SESSION_MINUTES, help="Session length")
    parser.add_argument("--max-reels", type=int, default=None, help="Stop after this many reels")
    parser.add_argument("--dry-run", action="store_true", help="Never like, save or follow (watch time still steers)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--rules", action="store_true",
                       help="Steer with quick rules instead of the LLM (no LM Studio); record every reel for review")
    modes.add_argument("--observe", action="store_true",
                       help="Record every reel for review with the same watch time on each; no LM Studio, no steering")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
