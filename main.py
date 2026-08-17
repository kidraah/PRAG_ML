"""
FeedLab — P.R.A.G. Local Fast Async Orchestration Loop
(Predictive Reel Accumulation & Grouping)

1. Initializes SQLite database & launches persistent async Chromium browser.
2. Navigates to Instagram Reels.
3. Continuous async event loop:
   a. Intercepts/records active reel video and extracts DOM metadata.
   b. Spawns background perception task (ffmpeg audio -> mlx-whisper -> local LM Studio).
   c. Plays video naturally in foreground (mimicking human viewing).
   d. Awaits background evaluation result and computes bandit policy.
   e. Executes action (Like/Watch/Skip) with elapsed time compensation.
   f. Logs to SQLite, cleans up temporary files, and scrolls to next reel.
"""

import asyncio
import os
import random
import time
import traceback

from dotenv import load_dotenv
load_dotenv()

from config import TARGET_BIAS
from database.db import (
    get_next_available_account,
    init_db,
    log_reel_event_async,
    reset_cooldown_accounts,
)
from perception.heuristics import fast_heuristic_check
from perception.local_evaluator import (
    extract_audio,
    transcribe_audio,
    evaluate_alignment_async,
    cleanup_audio_file,
)
from agent.bandit import decide_timing_and_action
from automation.browser_driver import (
    init_browser,
    close_browser,
    open_reels,
    get_current_reel_url,
    scroll_next,
    fast_skip,
    like_reel,
    capture_active_reel_and_metadata,
    cleanup_temp_file,
    is_logged_in,
    TEMP_VIDEO,
)

TEMP_AUDIO = "temp_audio.mp3"
MAX_CONSECUTIVE_ERRORS = 5


# ── Background Perception Pipeline ────────────────────────────────────

async def run_perception_pipeline(video_path: str, caption: str) -> tuple[dict, str]:
    """Execute local audio extraction, transcription, and scoring in background.

    Runs CPU/GPU-bound tasks (ffmpeg, mlx-whisper) in worker threads and
    network I/O (LM Studio) natively asynchronously.
    """
    # 1. Extract audio track via ffmpeg (~50ms)
    audio_path = await asyncio.to_thread(extract_audio, video_path, TEMP_AUDIO)

    # 2. Transcribe speech using mlx-whisper on Apple Silicon
    transcript = ""
    if audio_path:
        transcript = await asyncio.to_thread(transcribe_audio, audio_path)

    # 3. Score alignment with local text LLM via aiohttp
    eval_result = await evaluate_alignment_async(
        transcript=transcript,
        caption=caption,
        target_bias=TARGET_BIAS,
    )

    return eval_result, transcript


# ── Action Executor ───────────────────────────────────────────────────

async def execute_action_async(
    page,
    action: str,
    target_watch_seconds: float,
    reel_start_time: float,
) -> None:
    """Execute the bandit's chosen action with elapsed time compensation.

    Factors in the time already spent on this reel during in-browser recording
    and async evaluation, so total dwell time reflects the policy target
    plus natural human variability.
    """
    try:
        elapsed = time.time() - reel_start_time

        # If policy dictates liking, trigger like now while the video is playing
        if action == "LIKE_AND_WATCH":
            await like_reel(page)

        # Calculate remaining watch time needed to reach the target duration
        remaining_watch = max(0.0, target_watch_seconds - elapsed)
        extra_human_delay = random.uniform(0.5, 1.5)
        total_wait = remaining_watch + extra_human_delay

        print(f"  ⏱  Watch time: {elapsed + total_wait:.1f}s total (analyzed in {elapsed:.1f}s, waiting remaining {total_wait:.1f}s) …")
        await asyncio.sleep(total_wait)

    except Exception as exc:
        print(f"  ⚠️  Action '{action}' failed: {exc}")


# ── Main Async Loop ───────────────────────────────────────────────────

async def main_async() -> None:
    """Run the FeedLab local async P.R.A.G. agent."""
    init_db()
    reset_cooldown_accounts()

    # ── Step 1: Get account for DB tracking ──
    account = get_next_available_account()
    if account is None:
        print("⚠️  No ACTIVE accounts available. Exiting.")
        return

    username = account["username"]
    print(f"\n🚀  FeedLab Local Async Agent — Starting (tracking @{username})\n")

    # ── Step 2: Launch browser ──
    try:
        context, page = await init_browser()
    except Exception as exc:
        print(f"❌  Browser launch failed: {exc}")
        return

    reel_count = 0

    try:
        # ── Step 3: Navigate to Reels ──
        await open_reels(page)

        # ── Step 3b: Login check ──
        if not await is_logged_in(page):
            print("\n" + "=" * 60)
            print("  🔐  Not logged in!")
            print("  Please log into Instagram in the browser window.")
            print("  Press ENTER here once you're done …")
            print("=" * 60 + "\n")
            await asyncio.to_thread(input)
            await open_reels(page)

        # ── Step 4: P.R.A.G. Loop ──
        consecutive_errors = 0

        print(f"\n{'═' * 60}")
        print(f"  P.R.A.G. local async loop started for @{username}")
        print(f"{'═' * 60}\n")

        while True:
            reel_count += 1
            reel_start_time = time.time()
            print(f"── Reel #{reel_count} ──")

            try:
                # ── Stage 1: Fast-Path Heuristic Gate ──
                # Read DOM metadata quickly before heavy video recording
                reel_data = await capture_active_reel_and_metadata(page)
                
                reel_url = reel_data["reel_url"]
                caption = reel_data["caption"]
                like_count = reel_data["like_count"]
                comment_count = reel_data["comment_count"]

                print(f"  URL: {reel_url}")
                print(f"  📊  Likes: {like_count:,}  Comments: {comment_count:,}")
                if caption:
                    print(f"  📝  Caption: {caption[:80]}…")

                heuristic_pass = fast_heuristic_check(caption)
                if not heuristic_pass:
                    print("  🛑  Heuristic Filter: Excluded keyword detected.")
                    action, target_watch_seconds = decide_timing_and_action(0.0, heuristic_pass=False)
                    await execute_action_async(page, action, target_watch_seconds, reel_start_time)
                    await log_reel_event_async(
                        reel_url=reel_url, caption=caption[:500], transcript="",
                        alignment_score=0.0, like_count=like_count, comment_count=comment_count, action_taken=action
                    )
                    await fast_skip(page)
                    continue

                # ── Stage 2: Capture & Perception ──
                if not reel_data["file_path"] or not os.path.exists(TEMP_VIDEO):
                    print("  ⚠️  No video captured — skipping.")
                    await scroll_next(page)
                    continue

                # b. Spawn Background Perception Task
                perception_task = asyncio.create_task(
                    run_perception_pipeline(TEMP_VIDEO, caption)
                )

                # c. Foreground Baseline Watch (plays naturally on screen)
                await asyncio.sleep(2.0)

                # d. Await Background Perception Resolution
                eval_result, transcript = await perception_task
                score = eval_result.get("alignment_score", 0.0)
                reasoning = eval_result.get("reasoning", "No reasoning.")

                if transcript:
                    print(f"  🎙  Transcript: \"{transcript[:90]}…\"")
                print(f"  📈  Score: {score:.2f} — {reasoning}")

                # ── Stage 3: Bandit Policy & Action ──
                action, target_watch_seconds = decide_timing_and_action(score, heuristic_pass=True)
                print(f"  🎯  Policy → {action} ({target_watch_seconds:.1f}s target)")

                # f. Execute action with timing compensation
                await execute_action_async(page, action, target_watch_seconds, reel_start_time)

                # g. Log to Database asynchronously
                await log_reel_event_async(
                    reel_url=reel_url,
                    caption=caption[:500],
                    transcript=transcript[:2000],
                    alignment_score=score,
                    like_count=like_count,
                    comment_count=comment_count,
                    action_taken=action,
                )

                # h. Cleanup temporary files
                cleanup_temp_file(TEMP_VIDEO)
                cleanup_audio_file(TEMP_AUDIO)

                # i. Scroll to next reel
                await scroll_next(page)

                consecutive_errors = 0

            except Exception as exc:
                consecutive_errors += 1
                print(f"  ❌  Error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {exc}")

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print("  🛑  Too many consecutive errors — stopping.")
                    break

                try:
                    await scroll_next(page)
                except Exception:
                    pass

                cleanup_temp_file(TEMP_VIDEO)
                cleanup_audio_file(TEMP_AUDIO)

            print()

    except asyncio.CancelledError:
        print("\n⏹  Session cancelled.")
    except KeyboardInterrupt:
        print("\n⏹  Session interrupted by user.")
    except Exception as exc:
        print(f"\n❌  Fatal error: {exc}")
        traceback.print_exc()
    finally:
        cleanup_temp_file(TEMP_VIDEO)
        cleanup_audio_file(TEMP_AUDIO)
        await close_browser()

    print(f"\n🏁  FeedLab Local Async Agent — Done ({reel_count} reels processed)\n")


def main() -> None:
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
