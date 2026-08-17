---
description: Refactors FeedLab from Android Appium to Playwright browser automation with persistent session storage, and upgrades the perception pipeline to a multimodal scoring engine using yt-dlp, mlx-whisper, OpenCV keyframe extraction, and local LM Studio vis
---

## Steps

1. **Update Dependencies (`requirements.txt`)**
   - Update `requirements.txt` to include:
     ```text
     playwright
     playwright-stealth
     yt-dlp
     opencv-python
     mlx-whisper
     requests
     pandas
     streamlit
     ```
   - Run `pip install -r requirements.txt` and execute `playwright install chromium` via the terminal tool.

2. **Update Database Schema (`database/db.py`)**
   - Update `reels.db` schema initialization to ensure the `reels` table contains:
     - `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
     - `timestamp` (DATETIME DEFAULT CURRENT_TIMESTAMP)
     - `account_username` (TEXT)
     - `reel_url` (TEXT)
     - `transcript` (TEXT)
     - `caption` (TEXT)
     - `alignment_score` (REAL)
     - `like_count` (INTEGER)
     - `view_count` (INTEGER)
     - `comment_count` (INTEGER)
     - `action_taken` (TEXT)
   - Update `log_reel_event()` to accept and persist these metadata parameters.

3. **Build the yt-dlp Media Extractor (`perception/scraper.py`)**
   - Create `perception/scraper.py`.
   - Implement `download_reel_data(url: str, output_path: str = "temp_reel.mp4") -> dict`:
     - Configure `yt-dlp` to download video at <= 480p (`'bestvideo[height<=480]+bestaudio/best[height<=480]'`).
     - Suppress output logs (`'quiet': True`).
     - Save the downloaded media to `output_path`.
     - Extract and return a dictionary:
       ```python
       {
           "file_path": output_path,
           "caption": info_dict.get("description", "") or "",
           "view_count": int(info_dict.get("view_count") or 0),
           "like_count": int(info_dict.get("like_count") or 0),
           "comment_count": int(info_dict.get("comment_count") or 0)
       }
       ```
     - Wrap download calls in exception handling returning fallback empty values on failure.

4. **Build the Multimodal Vision Evaluator (`perception/evaluator.py`)**
   - Rewrite `perception/evaluator.py`:
     - Implement `extract_keyframes(video_path: str, num_frames: int = 4) -> list[str]`:
       - Open video with `cv2.VideoCapture`.
       - Sample `num_frames` evenly spaced frames across duration.
       - Encode frames as JPEG and convert to base64 strings.
     - Implement `evaluate_alignment(transcript: str, frames_base64: list[str], caption: str, target_bias: str) -> dict`:
       - Construct an OpenAI-compatible Vision payload for `http://127.0.0.1:1234/v1/chat/completions`.
       - System message: Specify strict JSON output matching `{"alignment_score": <float 0.0 to 1.0>, "reasoning": "<brief summary>"}`.
       - User message content: Combine target bias, caption, speech transcript text, and the base64 keyframe image objects (`{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}`).
       - Send POST request with a 30s timeout.
       - **Robust Error Handling**: If JSON parsing fails, print raw `response.text` to the console for debugging before returning a fallback `{ "alignment_score": 0.0, "reasoning": "Parse Error" }`.

5. **Build Playwright Browser Controller (`automation/browser_driver.py`)**
   - Create `automation/browser_driver.py` using `playwright.sync_api` and `playwright_stealth`:
     - Implement `init_browser(user_data_dir: str = "./browser_profile")`:
       - Launch a persistent browser context (`chromium.launch_persistent_context`) with viewport `1280x800` and `headless=False`.
       - Apply `stealth_sync(page)`.
       - Return `(browser_context, page)`.
     - Implement browser action helpers with randomized timing (`random.uniform`):
       - `open_reels(page)`: Navigates to `https://www.instagram.com/reels/`.
       - `get_current_reel_url(page) -> str`: Returns `page.url`.
       - `scroll_next(page)`: Simulates pressing `ArrowDown` with human-like pre- and post-delay.
       - `like_reel(page)`: Double-clicks video viewport or clicks like button.
       - `save_reel(page)`: Clicks the bookmark button if available.

6. **Update Decision Bandit (`agent/bandit.py`)**
   - Update `decide_action(alignment_score: float) -> tuple[str, float]`:
     - High alignment (>= 0.75): Return `("LIKE_AND_WATCH", random.uniform(8.0, 14.0))`
     - Moderate alignment (0.40 - 0.74): Return `("WATCH", random.uniform(4.0, 7.0))`
     - Low alignment (< 0.40): Return `("SKIP", random.uniform(1.0, 2.5))`

7. **Assemble Orchestration Pipeline (`main.py`)**
   - Refactor `main.py` into a unified execution loop:
     1. Initialize database and launch persistent browser controller.
     2. Navigate to `https://www.instagram.com/reels/`.
     3. If not logged in, prompt user in terminal to complete login in the open browser window once.
     4. Loop continuously:
        a. Read `current_url = get_current_reel_url(page)`.
        b. Call `download_reel_data(current_url)`.
        c. Call `transcribe_audio("temp_reel.mp4")` via `perception/transcriber.py`.
        d. Call `extract_keyframes("temp_reel.mp4", num_frames=4)`.
        e. Call `evaluate_alignment(transcript, frames, caption, TARGET_BIAS)`.
        f. Pass score to `decide_action()` and sleep for the assigned watch duration.
        g. Execute like/save action if designated by policy.
        h. Persist all metrics and actions to `reels.db`.
        i. Advance to next reel with `scroll_next(page)`.
        j. Remove `temp_reel.mp4` safely to conserve disk space.