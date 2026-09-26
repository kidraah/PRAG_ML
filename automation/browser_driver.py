"""
FeedLab — Asynchronous Playwright Browser Controller

Persistent Chromium profile with stealth patches, Reels navigation, and
actions on whichever reel is on screen. The on-screen reel is found from
the DOM — the <video> with the most visible area, and the action bar in
the same container — so a like, save or follow can never land on one of
the preloaded neighbours stacked above and below it.
"""

import asyncio
import os
import random
import re

from playwright.async_api import BrowserContext, Page, async_playwright
from playwright_stealth import Stealth


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(PROJECT_ROOT, "browser_profile")
REELS_URL = "https://www.instagram.com/reels/"
REEL_CODE_RE = re.compile(r"instagram\.com/reels?/([A-Za-z0-9_-]{6,})")


# ── Module State ──────────────────────────────────────────────────────

_playwright = None
_context: BrowserContext | None = None
_page: Page | None = None


async def _human_delay(lo: float = 0.5, hi: float = 1.5) -> None:
    """Non-blocking randomised sleep interval."""
    await asyncio.sleep(random.uniform(lo, hi))


# ── Browser Lifecycle ─────────────────────────────────────────────────

async def init_browser(user_data_dir: str = PROFILE_DIR) -> tuple[BrowserContext, Page]:
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
    await page.goto(REELS_URL, wait_until="domcontentloaded")
    await _human_delay(2.0, 3.5)
    print("  🎬  Opened Instagram Reels.")


async def is_logged_in(page: Page) -> bool:
    """Check whether Instagram is showing a login wall instead of the feed."""
    await page.wait_for_load_state("domcontentloaded")
    await asyncio.sleep(2)
    url = page.url.lower()
    if "accounts/login" in url or "accounts/emailsignup" in url:
        return False
    return await page.query_selector('input[name="username"]') is None


def reel_code_from_url(url: str) -> str | None:
    """Shortcode from a /reels/<code>/ URL, else None."""
    match = REEL_CODE_RE.search(url or "")
    return match.group(1) if match else None


# ── On-Screen Reel ────────────────────────────────────────────────────

ACTIVE_REEL_JS = """
() => {
    const vh = window.innerHeight;
    let video = null, bestVis = 0;
    for (const v of document.querySelectorAll('video')) {
        const r = v.getBoundingClientRect();
        const vis = Math.max(0, Math.min(vh, r.bottom) - Math.max(0, r.top));
        if (vis > bestVis) { bestVis = vis; video = v; }
    }
    if (!video) return null;

    let container = null, node = video;
    for (let i = 0; i < 30 && node; i++, node = node.parentElement) {
        if (node.querySelector && node.querySelector('svg[aria-label="Like"], svg[aria-label="Unlike"]')) {
            container = node;
            break;
        }
    }
    const box = el => {
        const r = el.getBoundingClientRect();
        return { x: r.left + r.width / 2, y: r.top + r.height / 2, visible: r.bottom > 0 && r.top < vh };
    };
    const out = {
        src: video.currentSrc || '', duration: video.duration || null,
        time: video.currentTime, paused: video.paused,
    };
    if (!container) return out;

    const like = container.querySelector('svg[aria-label="Like"], svg[aria-label="Unlike"]');
    const save = container.querySelector('svg[aria-label="Save"], svg[aria-label="Remove"]');
    const follow = Array.from(container.querySelectorAll('div, button, [role="button"]'))
        .filter(e => ['Follow', 'Following'].includes((e.innerText || '').trim()))
        .pop();  // deepest match
    const creatorLink = container.querySelector('a[href$="/reels/"]');
    out.like = like ? { label: like.getAttribute('aria-label'), ...box(like) } : null;
    out.save = save ? { label: save.getAttribute('aria-label'), ...box(save) } : null;
    out.follow = follow ? { text: follow.innerText.trim(), ...box(follow) } : null;
    out.creator = creatorLink ? creatorLink.getAttribute('href').split('/').filter(Boolean)[0] : null;
    out.sponsored = /(^|\\n)Sponsored(\\n|$)/.test(container.innerText || '');
    return out;
}
"""

ACTIVE_SRC_JS = """
() => {
    const vh = window.innerHeight;
    let src = '', bestVis = 0;
    for (const v of document.querySelectorAll('video')) {
        const r = v.getBoundingClientRect();
        const vis = Math.max(0, Math.min(vh, r.bottom) - Math.max(0, r.top));
        if (vis > bestVis) { bestVis = vis; src = v.currentSrc || ''; }
    }
    return src;
}
"""


async def active_reel(page: Page) -> dict:
    """Describe the reel on screen: its video, action buttons, creator and shortcode."""
    try:
        state = await page.evaluate(ACTIVE_REEL_JS) or {}
    except Exception:  # noqa: BLE001 — page mid-navigation
        state = {}
    state["code"] = reel_code_from_url(page.url)
    return state


async def wait_for_first_reel(page: Page, timeout: float = 15.0) -> dict:
    """Wait until a reel is playing and the URL carries its shortcode."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    state = await active_reel(page)
    while not (state.get("src") and state.get("code")) and loop.time() < deadline:
        await asyncio.sleep(0.3)
        state = await active_reel(page)
    return state


async def _wait_for_new_video(page: Page, prev_src: str, timeout: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            src = await page.evaluate(ACTIVE_SRC_JS)
        except Exception:  # noqa: BLE001
            src = ""
        if src and src != prev_src:
            return True
        await asyncio.sleep(0.12)
    return False


async def scroll_next(page: Page, prev_src: str) -> dict:
    """Advance to the next reel and return its state (see active_reel).

    ArrowDown is the native desktop trigger; PageDown and a mouse-wheel
    scroll are fallbacks. If nothing moves, the returned state still has
    prev_src, which the caller treats as a stuck feed.
    """
    await _human_delay(0.15, 0.45)
    for method in ("ArrowDown", "PageDown", "wheel"):
        if method == "wheel":
            await page.mouse.move(640 + random.uniform(-40, 40), 400 + random.uniform(-60, 60))
            await page.mouse.wheel(0, 800)
        else:
            await page.keyboard.press(method)
        if await _wait_for_new_video(page, prev_src, timeout=4.0 if method == "ArrowDown" else 3.0):
            break
    await asyncio.sleep(0.3)  # the URL is updated just after the swap
    return await active_reel(page)


# ── Actions on the On-Screen Reel ─────────────────────────────────────

async def _click(page: Page, target: dict) -> None:
    x = target["x"] + random.uniform(-3, 3)
    y = target["y"] + random.uniform(-3, 3)
    await page.mouse.move(x, y, steps=random.randint(8, 16))
    await asyncio.sleep(random.uniform(0.08, 0.25))
    await page.mouse.click(x, y)


async def _toggle(page: Page, key: str, off_label: str) -> bool:
    """Click the on-screen reel's `key` button if it's in its off state; True once it's on."""
    button = (await active_reel(page)).get(key)
    if not button or not button.get("visible"):
        return False
    if button["label"] != off_label:
        return True  # already liked / saved — clicking again would undo it
    await _click(page, button)
    await asyncio.sleep(0.9)
    after = (await active_reel(page)).get(key) or {}
    return after.get("label") not in (None, off_label)


async def like_active_reel(page: Page) -> bool:
    """Like the reel on screen; True if it ends up liked."""
    return await _toggle(page, "like", "Like")


async def save_active_reel(page: Page) -> bool:
    """Save the reel on screen; True if it ends up saved."""
    return await _toggle(page, "save", "Save")


async def follow_active_creator(page: Page, username: str) -> bool:
    """Follow the on-screen reel's creator, only if it is `username`; True if followed."""
    state = await active_reel(page)
    follow = state.get("follow")
    if not follow or not follow.get("visible") or state.get("creator") != username:
        return False
    if follow["text"] != "Follow":
        return True
    await _click(page, follow)
    await asyncio.sleep(1.2)
    after = (await active_reel(page)).get("follow") or {}
    return after.get("text") != "Follow"
