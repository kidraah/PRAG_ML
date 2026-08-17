---
description: Builds the Appium automation controllers, the ADB wiper for account cycling, and the Contextual Bandit RL loop that steers the feed.
---

## Steps

1. **Appium Driver Setup**
   - Create `automation/appium_driver.py`. Write `init_driver()` to connect to Android Emulator at `http://localhost:4723`.
   - Write swipe functions using screen coordinates (not UI IDs): `swipe_up()`, `double_tap_like()`, and `tap_save()`. Use `random.uniform(0.5, 1.5)` for human-like delays.

2. **ADB Device State Isolator**
   - Create `automation/adb_manager.py`. Write `wipe_device_state()`.
   - Use `subprocess` to run `adb shell pm clear com.instagram.android` and randomize `android_id` to prevent account bans during cycling.

3. **The RL Policy / Bandit**
   - Create `agent/bandit.py`. Write `decide_action(alignment_score)`.
   - Logic: score >= 0.75 -> `["WATCH_FULL", "LIKE"]`; 0.4 <= score < 0.75 -> `["WATCH_FULL"]`; score < 0.4 -> `["SWIPE"]`.

4. **The Main Orchestration Loop**
   - Create `main.py` in the root.
   - Import database modules, appium drivers, and perception evaluator.
   - Write a `main()` loop that:
     1. Queries the DB for an `ACTIVE` account (using `get_next_available_account()`).
     2. Simulates taking screenshots/audio (placeholder for now).
     3. Calls `evaluate_alignment()`.
     4. Passes the score to `decide_action()`.
     5. Executes the Appium gestures based on the policy.
     6. Logs the score and action to the `reels` table via `log_reel_event()`.
     7. Tracks session time, marks the account as `COOLDOWN` in the DB after 45 minutes, calls `wipe_device_state()`, and cycles to the next account.