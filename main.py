"""
PRAG — Curation Agent (Parametric Reel Aggregation and Gradient)

Phase 1: curate the account. The agent scrolls Instagram Reels on a
dedicated account and engages the way a person with config.INTERESTS
would — skipping fast, watching what's informative, watching valuable
reels through (the best of them twice) and liking / saving some of them,
following creators who keep delivering — so the account's feed converges
on valuable content. Every reel shown is logged; `python -m
dashboard.report` says whether the feed is curated yet.

    python main.py [--account NAME] [--minutes 40] [--max-reels N] [--dry-run] [--rules | --observe]

Without LM Studio:
  --rules    steers with quick rules — watch reels from followed accounts and
             news-like captions, flick past the rest. The trained model
             (python -m ml.train --save) scores every reel if one is saved;
             otherwise the reels are recorded for review.
  --observe  records every reel with the same watch time on each (no steering)
Recorded reels (caption, cover text, speech) are scored afterwards with
tools/review.py. For the everyday run — scroll, then the digest — use daily.py.

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

from agent.bandit import Decision, decide, watch_dwell
from agent.bias import load as load_bias
from agent.creators import CreatorMemory
from agent.pipeline import ScoreResult, ScoringPipeline
from agent.rules import RuleVerdict, rule_score
from ml.scorer import ModelScorer
from automation.browser_driver import (
    active_reel,
    close_browser,
    follow_active_creator,
    init_browser,
    is_logged_in,
    like_active_reel,
    open_reels,
    profile_dir,
    profile_lock,
    save_active_reel,
    scroll_next,
    set_input_lock,
    wait_for_first_reel,
)
from automation.feed_capture import FeedCatalog, ReelMeta
from automation.login import ensure_logged_in
from config import (
    AD_DWELL,
    LATE_SCORE_WAIT,
    LIKE_THRESHOLD,
    LLM_MODEL_NAME,
    MAX_DECISION_WAIT,
    MAX_FOLLOWS_PER_SESSION,
    MAX_LIKES_PER_SESSION,
    MAX_SAVES_PER_SESSION,
    OBSERVE_DWELL,
    SCORER,
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
from perception import jev, visual
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


def interference(before: dict, after: dict, done: Performed, tried: frozenset = frozenset()) -> list[str]:
    """What changed on screen during a reel that the agent didn't do (someone at the Mac, or the browser).

    `tried`: the buttons the agent pressed on this reel ("like", "save", "follow"), whatever came of it, so a
    like of its own that registered late is never taken for someone else's.
    """
    if not after.get("src") or not after.get("code"):
        return ["left the Reels feed"]
    if (after["code"] != before["code"]) if before.get("code") else (after["src"] != before.get("src")):
        return ["moved to another reel"]  # (the same reel's video can reload, so the code decides)
    reasons = []
    for key, off in (("like", "Like"), ("save", "Save")):
        was, now = (before.get(key) or {}).get("label"), (after.get(key) or {}).get("label")
        if key not in tried and was and now and was != now:
            reasons.append(f"{'un' if now == off else ''}{'liked' if key == 'like' else 'saved'} it")
    was, now = (before.get("follow") or {}).get("text"), (after.get("follow") or {}).get("text")
    if "follow" not in tried and was and now and was != now:
        reasons.append("followed the creator" if was == "Follow" else "unfollowed the creator")
    return reasons


async def recover(page, reasons: list[str]) -> dict:
    """Get back to a reel playing in the feed, locked; returns its state."""
    if "left the Reels feed" in reasons:
        await open_reels(page)
        state = await wait_for_first_reel(page)
    else:
        state = await active_reel(page)
    await set_input_lock(page, True)
    return state


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

async def await_score(pipeline: ScoringPipeline, code: str, meta: ReelMeta,
                      shown_at: float) -> tuple[ScoreResult | None, RuleVerdict | None]:
    """The reel's score, waited for MAX_DECISION_WAIT after it appeared — longer (LATE_SCORE_WAIT) when
    the quick rules call it news-like. Long reels take 10-20 s to score; skipping a news reel for being
    slow would teach Instagram the opposite of the goal and leave the scorer behind on every next reel.
    Returns the rule verdict too when it was consulted.
    """
    result = await pipeline.wait_for(code, MAX_DECISION_WAIT - (time.monotonic() - shown_at))
    if result is not None:
        return result, None
    rule = rule_score(meta, getattr(pipeline, "bias", None))
    if rule.score >= WATCH_THRESHOLD:
        result = await pipeline.wait_for(code, LATE_SCORE_WAIT - (time.monotonic() - shown_at))
    return result, rule


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


def _clip(text: str, width: int) -> str:
    """At most `width` characters, ending in … where it was cut (the panel shows these lines too)."""
    return text if len(text) <= width else text[:width - 1] + "…"


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

    scored = result is not None and result.basis == "score"
    if rule is not None and (decision.basis == "rules" or not scored):  # the rules decided (or stood in)
        verdict = f"rule {rule.score:.1f} · {rule.reason}"
    elif scored:
        labels = result.labels or {}  # Jev: no topic until the LLM has written it, but its labels
        what = result.topic or ", ".join(v.replace("_", " ") for v in (labels.get("source_type"), labels.get("stance")) if v)
        verdict = f"{result.value_score:.2f} {what}"
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
    if result is not None and result.image_used:
        marks = (marks + " 🖼").strip()  # Jev saw the cover frame with the text
    if decision.action == "SKIP" and ready and decision.basis != "ad":
        marks = ("⚡ " + marks).strip()
    line = f"#{seq:<4} {_clip(who, 26):26} {kind:12} {_clip(verdict, 36):36} → {decision.action:10} {dwell:5.1f}s {marks}"
    if result and result.summary and decision.action != "SKIP":
        line += f"\n        “{_clip(result.summary, 110)}”"
    if result and result.visual and decision.action != "SKIP":
        v = result.visual
        line += (f"\n        🖼 Cover frame: {visual.short(v)} · brightness {v['brightness']:.2f} · colour "
                 f"{v['colourfulness']:.2f} · edges {v['edge_density']:.2f}")
    return line


# ── Main Loop ─────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> dict:
    """One curation session. Returns its outcome, e.g. {"status": "ok", "session_id": 7, "reels": 60}.

    status: ok | no_account | busy | llm_unavailable | login_needed | no_reels | stuck | interrupted | error
    """
    init_db()
    reset_cooldown_accounts()
    username, problem = choose_account(list_accounts(), args.account)
    if problem:
        print(f"⚠️  {problem}")
        return {"status": "no_account", "detail": problem}
    with profile_lock(username) as held:  # one browser per profile: a scheduled and a manual run can't overlap
        if not held:
            print(f"⚠️  @{username} is already being scrolled by another PRAG run.")
            return {"status": "busy", "account": username}
        return await _session(args, username)


async def jev_problem(bias) -> str | None:
    """Why Jev can't judge reels right now (no key, no credit, down), or None: one small call."""
    import aiohttp

    if not jev.api_key():
        return "OPENROUTER_API_KEY isn't set in .env"
    try:
        async with aiohttp.ClientSession() as session:
            await jev.judge_async(session, "Caption: Warm-up request.", bias.interests, persona=bias.persona,
                                  labels=False)
        return None
    except jev.JevError as exc:
        return str(exc)


async def _session(args: argparse.Namespace, username: str) -> dict:
    mark_account_used(username)
    observe, rules = args.observe, args.rules
    record = observe or rules  # the LLM is off: every reel is recorded
    # With the LLM off, --rules scores the recordings with the trained model when there is one;
    # otherwise (and always with --observe) they wait for review.
    bias = load_bias(username)  # what this account looks for (the panel's Bias tab)
    fallback = ModelScorer(bias=bias) if rules and ModelScorer.available(bias=bias) else None

    # Who judges reels. Jev (config.SCORER): the session only runs when Jev answers, so an account's persona is
    # always judged the same way; the local LLM writes the text as far as it can. Otherwise the LLM does both.
    use_jev = not record and SCORER == "jev"
    if use_jev:
        problem = await jev_problem(bias)
        if problem:
            print(f"❌  Jev can't be reached ({problem}): no session now, so this persona is never judged by "
                  "anything else.")
            return {"status": "scorer_unavailable", "account": username, "detail": problem}
    elif not record:
        try:
            await asyncio.to_thread(ensure_llm_ready)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            print(f"❌  {exc}")
            return {"status": "llm_unavailable", "account": username, "detail": str(exc)}

    scorer = ((fallback.version if fallback else REVIEW_SCORER) if record
              else jev.jev_version(bias.interests, bias.persona) if use_jev
              else scorer_version(bias.interests, bias.persona))
    creators = CreatorMemory() if record else CreatorMemory.from_history(*load_creator_history(scorer, username))
    catalog = FeedCatalog()
    rng = random.Random()
    model = "none" if observe else "rules" if rules else jev.JEV_MODEL if use_jev else LLM_MODEL_NAME
    session_id = start_session(username, dry_run=args.dry_run or observe, model=model, scorer=scorer)
    if observe:
        mode = "observe — recording every reel, no scoring or engagement"
    elif rules:
        scored_by = "the trained model scores them" if fallback else "every reel recorded for review"
        mode = f"rules — quick rules steer, {scored_by}" + (", no likes/saves/follows" if args.dry_run else "")
    else:
        mode = "dry run — no likes, saves or follows" if args.dry_run else "live"
    headless = getattr(args, "headless", False)
    print(f"\n🚀  PRAG curation — @{username}, session {session_id} ({mode}, {args.minutes} min"
          + (", headless)" if headless else ")") + f"\n🎯  Bias: {bias.name}" + (" (persona)" * bias.persona) + "\n")

    pipeline: ScoringPipeline | None = None
    worker: asyncio.Task | None = None
    started = time.monotonic()
    seq = likes = saves = follows = 0
    tally: Counter = Counter()
    lock: dict = {}  # the window's input lock: {"locked", "blocked"}
    outcome = {"status": "ok", "account": username, "session_id": session_id}

    try:
        _context, page = await init_browser(profile_dir(username), headless=headless)

        def on_result(meta: ReelMeta, result: ScoreResult) -> None:
            # Scores that arrive after their reel was logged (decided on timeout).
            update_impression_score(session_id, meta.code, result.score_fields())

        def on_text(meta: ReelMeta, result: ScoreResult) -> None:
            # The LLM's words for a reel Jev judged (a reel not logged yet takes them from its result).
            update_impression_score(session_id, meta.code, {
                "topic": result.topic, "summary": result.summary, "reasoning": result.reasoning,
                "is_informational": int(bool(result.is_informational)), "llm_score": result.llm_score})
            tally["written"] += 1
            if result.summary:
                print(f"        ✍  {result.topic or 'Written'} (@{meta.creator or '?'}): “{result.summary[:110]}”")

        pipeline = ScoringPipeline(
            catalog, creators, rng=rng,
            user_agent=await page.evaluate("navigator.userAgent"),
            on_result=on_result,
            use_llm=not record,
            fallback_scorer=fallback,
            bias=bias,
            scorer="jev" if use_jev else "llm",
            on_text=on_text,
        )
        print("  🔥  Warming up Whisper and OCR …" if record else
              "  🔥  Warming up Whisper, OCR, Jev and the writer …" if use_jev else
              "  🔥  Warming up Whisper, OCR and the LLM …")
        warm_start = time.monotonic()
        try:
            await pipeline.warm_up()
        except ScoringError as exc:
            print(f"❌  {'Jev and writer' if use_jev else 'LLM'} warm-up failed: {exc}")
            outcome.update(status="llm_unavailable", detail=str(exc))
            return outcome
        print(f"  🔥  Ready in {time.monotonic() - warm_start:.0f}s")

        catalog.on_new_feed_reel = pipeline.enqueue
        catalog.attach(page)
        worker = asyncio.create_task(pipeline.run())

        await open_reels(page)
        if not await is_logged_in(page):
            if not await ensure_logged_in(page, username, headless=headless):
                print("❌  Not logged in — ending the session.")
                outcome["status"] = "login_needed"
                return outcome
            await open_reels(page)
        await catalog.ingest_embedded(page)

        state = await wait_for_first_reel(page)
        if not state.get("src"):
            print("❌  No reel started playing.")
            outcome["status"] = "no_reels"
            return outcome
        lock = await set_input_lock(page, True)  # from here on, only the agent steers this window
        print("  🔒  The window ignores clicks, scrolls and keys while the session runs.")

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
                rule = rule_score(meta, bias) if meta is not None and not is_ad else None
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
                    result, rule = await await_score(pipeline, code, meta, shown_at)

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
                if basis == "timeout" and rule is not None and rule.score >= WATCH_THRESHOLD:
                    decision = Decision("WATCH", watch_dwell(meta.duration, rng), "timeout")  # waited; rules said watch

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

            # Did anything the agent didn't do reach the reel? Then its watch isn't the persona's.
            now = await active_reel(page)
            tried = frozenset() if args.dry_run else frozenset(
                key for key, on in (("like", decision.like), ("save", decision.save), ("follow", decision.follow)) if on)
            reasons = interference(state, now, done, tried)
            lock = await set_input_lock(page, True) or lock  # re-armed if the page reloaded
            latest = pipeline.result(code)  # may have landed during the dwell
            row = impression_row(session_id, seq, code, meta, state, is_ad, latest, decision, dwell, done)
            if reasons:
                row["decision_basis"] = "interrupted"
            row["ready_on_arrival"] = int(ready_on_arrival)
            if rule is not None:
                row.update(rule_score=rule.score, rule_reason=rule.reason)
            log_impression(row)
            print(describe(seq, meta, state, is_ad, latest, decision, dwell, done, args.dry_run, rule, ready_on_arrival))
            if reasons:
                print(f"        ✋  Interrupted ({', '.join(reasons)}): not counted as the persona's; carrying on.")
                tally["interrupted"] += 1

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

            if reasons:  # carry on from whatever is on screen now, without scrolling past it
                state = await recover(page, reasons)
                if state.get("src"):
                    continue
            prev_src = state.get("src")
            for _attempt in range(MAX_STUCK_SCROLLS):
                state = await scroll_next(page, prev_src)
                if state.get("src") and state.get("src") != prev_src:
                    break
            else:
                raise FeedStuck

    except FeedStuck:
        print("  🛑  The feed stopped advancing — ending the session.")
        outcome["status"] = "stuck"
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n⏹  Session interrupted.")
        outcome["status"] = "interrupted"
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌  Fatal error: {exc}")
        traceback.print_exc()
        outcome.update(status="error", detail=str(exc))
    finally:
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        if pipeline is not None:
            await pipeline.close()
        end_session(session_id)
        await close_browser()

    minutes = (time.monotonic() - started) / 60
    outcome.update(reels=seq, minutes=round(minutes, 1), informative=tally["informative"], scored=tally["scored"])
    if record and fallback is None:
        watched = tally["WATCH"] + tally["FULL_WATCH"] + tally["REWATCH"]
        steer = f" · rules watched {watched}, skipped {tally['SKIP']} ({tally['flicked']} flicked)" if rules else ""
        print(f"\n🏁  Session {session_id} — {seq} reels in {minutes:.1f} min{steer} · recorded for review")
        print(f"    Next: python -m tools.review export --account {username} --session {session_id}\n")
        return outcome
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
    if lock.get("blocked") or tally["interrupted"]:
        print(f"    🔒 ignored {lock.get('blocked', 0)} clicks, scrolls or keys from outside the agent; "
              f"{tally['interrupted']} reels interrupted anyway (not counted as the persona's)")
    print("    Is the feed curated yet? →  python -m dashboard.report\n")
    return outcome


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="PRAG curation agent")
    parser.add_argument("--account", help="Instagram account to use (required when several are ACTIVE)")
    parser.add_argument("--minutes", type=float, default=SESSION_MINUTES, help="Session length")
    parser.add_argument("--max-reels", type=int, default=None, help="Stop after this many reels")
    parser.add_argument("--dry-run", action="store_true", help="Never like, save or follow (watch time still steers)")
    parser.add_argument("--headless", action="store_true",
                        help="No browser window (Instagram can tell; a login that needs a code needs the window)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--rules", action="store_true",
                       help="Steer with quick rules instead of the LLM (no LM Studio); the trained model scores "
                            "the reels when one is saved (python -m ml.train --save), else they're recorded for review")
    modes.add_argument("--observe", action="store_true",
                       help="Record every reel for review with the same watch time on each; no LM Studio, no steering")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
