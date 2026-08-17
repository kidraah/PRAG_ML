"""
FeedLab — Asynchronous Playwright Browser Controller & Video Capture

Persistent Chromium browser with stealth patches for Instagram Reels.
Uses `playwright.async_api` for non-blocking browser automation, live video
recording directly from the active HTML5 <video> element, and natural human gestures.
"""

import base64
import os
import random
import asyncio

from playwright.async_api import async_playwright, BrowserContext, Page
from playwright_stealth import Stealth


# ── Module State ──────────────────────────────────────────────────────

_playwright = None
_context: BrowserContext | None = None
_page: Page | None = None

TEMP_VIDEO = "temp_reel.webm"


async def _human_delay(lo: float = 0.5, hi: float = 1.5) -> None:
    """Non-blocking randomised sleep interval."""
    await asyncio.sleep(random.uniform(lo, hi))


# ── Browser Lifecycle ─────────────────────────────────────────────────

async def init_browser(user_data_dir: str = "./browser_profile") -> tuple[BrowserContext, Page]:
    """Launch a persistent Chromium context with stealth in async mode.

    Parameters
    ----------
    user_data_dir : str
        Directory for persistent cookies / local-storage.

    Returns
    -------
    tuple[BrowserContext, Page]
    """
    global _playwright, _context, _page

    _playwright = await async_playwright().start()
    _context = await _playwright.chromium.launch_persistent_context(
        user_data_dir,
        headless=False,
        viewport={"width": 1280, "height": 800},
        locale="en-US",
        args=["--disable-blink-features=AutomationControlled"],
    )

    _page = _context.pages[0] if _context.pages else await _context.new_page()
    await Stealth().apply_stealth_async(_page)

    print("🌐  Playwright async browser launched (persistent profile).")
    return _context, _page


def get_page() -> Page:
    """Return the active page, raising if browser not initialised."""
    if _page is None:
        raise RuntimeError("Browser not initialised. Call init_browser() first.")
    return _page


async def close_browser() -> None:
    """Cleanly shut down browser context and Playwright."""
    global _playwright, _context, _page
    try:
        if _context is not None:
            await _context.close()
        if _playwright is not None:
            await _playwright.stop()
    except Exception:
        pass
    _context = None
    _page = None
    _playwright = None
    print("🌐  Browser closed.")


# ── Navigation ────────────────────────────────────────────────────────

async def open_reels(page: Page) -> None:
    """Navigate to the Instagram Reels feed."""
    await page.goto("https://www.instagram.com/reels/", wait_until="domcontentloaded")
    await _human_delay(2.0, 3.5)
    print("  🎬  Opened Instagram Reels.")


def get_current_reel_url(page: Page) -> str:
    """Return the current page URL."""
    return page.url


async def is_logged_in(page: Page) -> bool:
    """Check if the user appears to be logged into Instagram."""
    await page.wait_for_load_state("domcontentloaded")
    await asyncio.sleep(2)

    url = page.url.lower()
    if "accounts/login" in url or "accounts/emailsignup" in url:
        return False
    if "instagram.com/reels" in url or "instagram.com/p/" in url:
        return True
    if "instagram.com" in url:
        return True
    return False


# ── In-Browser Video Capture & Metadata ───────────────────────────────

async def capture_active_reel_and_metadata(page: Page) -> dict:
    """Record active reel video directly from the browser <video> element asynchronously.

    Captures ~3.5 seconds of live video + audio from the playing HTML5 <video>
    element, saves to disk, and extracts deduplicated likes, comments, and caption.

    Returns
    -------
    dict
        ``{"file_path", "reel_url", "caption", "like_count", "comment_count"}``
    """
    await asyncio.sleep(0.8)

    record_js = """
    async () => {
        const videos = Array.from(document.querySelectorAll('video'));
        if (!videos.length) return { error: 'No video element found in DOM' };

        const activeVideo = videos.find(v => {
            const r = v.getBoundingClientRect();
            return r.top >= -100 && r.bottom <= window.innerHeight + 100 && r.width > 100;
        }) || videos[0];

        try {
            const stream = activeVideo.captureStream ? activeVideo.captureStream() : activeVideo.mozCaptureStream();
            const recorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp8,opus' });
            const chunks = [];

            recorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) chunks.push(e.data);
            };

            const recordPromise = new Promise((resolve) => {
                recorder.onstop = () => {
                    const blob = new Blob(chunks, { type: 'video/webm' });
                    const reader = new FileReader();
                    reader.onloadend = () => resolve({ size: blob.size, dataUrl: reader.result });
                    reader.readAsDataURL(blob);
                };
            });

            recorder.start();
            await new Promise(r => setTimeout(r, 3500));
            recorder.stop();

            return await recordPromise;
        } catch (err) {
            return { error: err.toString() };
        }
    }
    """

    res = await page.evaluate(record_js)
    file_path = ""

    if res and "dataUrl" in res and res.get("size", 0) > 10000:
        b64_data = res["dataUrl"].split(",")[1]
        raw_bytes = base64.b64decode(b64_data)
        with open(TEMP_VIDEO, "wb") as f:
            f.write(raw_bytes)
        file_path = TEMP_VIDEO
        print(f"  📹  Captured live reel ({len(raw_bytes) / 1024:.0f} KB)")
    else:
        err = res.get("error", "Unknown error") if res else "No response"
        print(f"  ⚠️  Video capture note: {err}")

    # Read DOM metadata
    reel_url = page.url
    caption, like_count, comment_count = await _extract_dom_metadata(page)

    return {
        "file_path": file_path,
        "reel_url": reel_url,
        "caption": caption,
        "like_count": like_count,
        "comment_count": comment_count,
    }


async def _extract_dom_metadata(page: Page) -> tuple[str, int, int]:
    """Extract caption, like count, and comment count from the DOM asynchronously."""
    meta_js = """
    () => {
        const parseNum = (str) => {
            if (!str) return 0;
            let mult = 1;
            const s = str.toUpperCase().trim();
            if (s.endsWith('K')) mult = 1000;
            else if (s.endsWith('M')) mult = 1000000;
            else if (s.endsWith('B')) mult = 1000000000;
            const num = parseFloat(s.replace(/[^0-9.]/g, ''));
            return isNaN(num) ? 0 : Math.round(num * mult);
        };

        const likeSvgs = Array.from(document.querySelectorAll('svg[aria-label="Like"]'));
        const activeLikeSvg = likeSvgs.find(s => {
            const r = s.getBoundingClientRect();
            return r.top >= -100 && r.bottom <= window.innerHeight + 100;
        }) || likeSvgs[0];

        let likeCount = 0;
        let commentCount = 0;
        let caption = '';

        if (activeLikeSvg) {
            let p = activeLikeSvg.parentElement;
            for (let i = 0; i < 10 && p; i++) {
                const buttons = Array.from(p.querySelectorAll('[role="button"], button'));
                if (buttons.length >= 3) {
                    const uniqueCounts = [];
                    for (const b of buttons) {
                        const txt = b.innerText.trim();
                        // Match pure numbers like 144K, 1.2M, 500
                        if (txt && /^[0-9,.]+[KMBkmb]?$/.test(txt)) {
                            if (!uniqueCounts.includes(txt)) {
                                uniqueCounts.push(txt);
                            }
                        } 
                        // Match hidden likes like "Liked by user and others" or "Liked by user and 12,304 others"
                        else if (txt && txt.toLowerCase().includes('liked by')) {
                            // Try to extract the number if present, otherwise default to 0
                            const match = txt.match(/and ([0-9,.]+[KMBkmb]?) others/i);
                            if (match && match[1]) {
                                uniqueCounts.push(match[1]);
                            } else {
                                uniqueCounts.push("0");
                            }
                        }
                    }
                    if (uniqueCounts.length >= 1) likeCount = parseNum(uniqueCounts[0]);
                    if (uniqueCounts.length >= 2) commentCount = parseNum(uniqueCounts[1]);
                    break;
                }
                p = p.parentElement;
            }
        }

        // Caption extraction
        const main = document.querySelector('main') || document.body;
        const spans = Array.from(main.querySelectorAll('h1, span[dir="auto"], span.x1lliihq'));
        for (const s of spans) {
            const t = s.innerText ? s.innerText.trim() : '';
            if (t.length > 5 && !t.includes('Follow') && !t.includes('Original audio') && !t.includes('Audio is muted') && !t.includes('likes') && !t.includes('comments')) {
                caption = t;
                break;
            }
        }

        return { caption, likeCount, commentCount };
    }
    """
    try:
        data = await page.evaluate(meta_js)
        return (
            data.get("caption", "")[:500],
            int(data.get("likeCount", 0)),
            int(data.get("commentCount", 0)),
        )
    except Exception:
        return "", 0, 0


def cleanup_temp_file(path: str = TEMP_VIDEO) -> None:
    """Safely remove the temporary video file."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ── Gesture Actions ───────────────────────────────────────────────────

async def fast_skip(page: Page) -> None:
    """Execute an immediate, sub-second skip to the next reel."""
    old_url = page.url
    await _human_delay(0.2, 0.4)
    await page.keyboard.press("ArrowDown")
    
    deadline = asyncio.get_event_loop().time() + 2.0
    while asyncio.get_event_loop().time() < deadline:
        if page.url != old_url:
            break
        await asyncio.sleep(0.1)
        
    if page.url == old_url:
        await page.keyboard.press("PageDown")
    
    print("  ⏭️  Fast-skipped irrelevant reel.")

async def scroll_next(page: Page) -> None:
    """Advance to the next reel using ArrowDown (native Instagram trigger).

    Waits for the page URL to change (up to 5s) so the next loop
    iteration reads the new reel.
    """
    old_url = page.url
    await _human_delay(0.3, 0.6)

    # Use ArrowDown which cleanly triggers the desktop reels transition
    await page.keyboard.press("ArrowDown")

    # Wait for URL to update
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        if page.url != old_url:
            break
        await asyncio.sleep(0.3)

    if page.url == old_url:
        await page.keyboard.press("PageDown")
        await asyncio.sleep(1.0)

    await _human_delay(1.0, 1.8)
    print("  ↓  Scrolled to next reel.")


async def like_reel(page: Page) -> None:
    """Like the current reel via double-click on the video viewport or Like button."""
    try:
        video = await page.query_selector("video")
        if video:
            box = await video.bounding_box()
            if box:
                cx = box["x"] + box["width"] / 2 + random.randint(-30, 30)
                cy = box["y"] + box["height"] / 2 + random.randint(-30, 30)
                await page.mouse.dblclick(cx, cy)
                await _human_delay(0.5, 1.0)
                print("  ❤️  Liked reel (double-click).")
                return
    except Exception:
        pass

    try:
        like_btn = await page.query_selector('[aria-label="Like"]')
        if like_btn:
            await like_btn.click()
            await _human_delay(0.5, 1.0)
            print("  ❤️  Liked reel (button).")
            return
    except Exception:
        pass

    print("  ⚠️  Could not like reel.")
