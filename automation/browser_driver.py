"""
PRAG — Asynchronous Playwright Browser Controller

Persistent Chromium profile with stealth patches, Reels navigation, and
actions on whichever reel is on screen. The on-screen reel is found from
the DOM — the <video> with the most visible area, and the action bar in
the same container — so a like, save or follow can never land on one of
the preloaded neighbours stacked above and below it.

While a session scrolls, the window ignores people: clicks, double-clicks,
scrolls, keys and touches are swallowed (INPUT_LOCK_JS) unless the agent
is acting (agent_input), so nobody at the Mac can steer the persona's
feed by accident. Anything that still gets through (the browser's own
buttons, say) is caught by the session loop, which checks the reel after
each one (main.py).
"""

import asyncio
import contextlib
import fcntl
import os
import random
import re

from playwright.async_api import BrowserContext, Page, async_playwright
from playwright_stealth import Stealth


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(PROJECT_ROOT, "browser_profiles")
REELS_URL = "https://www.instagram.com/reels/"
REEL_CODE_RE = re.compile(r"instagram\.com/reels?/([A-Za-z0-9_-]{6,})")


# ── Input Lock ────────────────────────────────────────────────────────

INPUT_LOCK_JS = r"""
(() => {
    if (window.__pragLockInstalled) return;
    window.__pragLockInstalled = true;
    window.__pragLock = false;       // on once the session is scrolling (set_input_lock)
    window.__pragAgentUntil = 0;     // the agent is acting until then: its own input passes
    window.__pragBlocked = 0;        // how many events were swallowed
    const block = (event) => {
        if (!window.__pragLock || Date.now() < window.__pragAgentUntil) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        window.__pragBlocked++;
    };
    for (const type of ["wheel", "mousewheel", "touchstart", "touchmove", "touchend", "keydown", "keyup", "keypress",
                        "mousedown", "mouseup", "click", "dblclick", "auxclick", "contextmenu", "pointerdown",
                        "pointerup", "dragstart", "selectstart"]) {
        window.addEventListener(type, block, {capture: true, passive: false});
    }
})();
"""

SET_LOCK_JS = r"""
(on) => {
    window.__pragLock = on;
    let banner = document.getElementById("prag-lock-banner");
    if (on && !banner) {
        banner = document.createElement("div");
        banner.id = "prag-lock-banner";
        banner.textContent = "PRAG is scrolling this window: clicks, scrolls and keys are ignored. Stop the session from the panel.";
        banner.style.cssText = "position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:2147483647;" +
            "pointer-events:none;background:rgba(14,14,17,.88);color:#fff;font:600 12px -apple-system,sans-serif;" +
            "padding:6px 12px;border:1px solid rgba(255,255,255,.55)";
        document.documentElement.appendChild(banner);
    }
    if (banner) banner.hidden = !on;
    return {locked: window.__pragLock, blocked: window.__pragBlocked || 0};
}
"""


async def set_input_lock(page: Page, on: bool) -> dict:
    """Make the window ignore people (on) or not; returns {"locked", "blocked"}. Safe mid-navigation."""
    try:
        await page.evaluate(INPUT_LOCK_JS)  # a page loaded before the init script: install it now
        return await page.evaluate(SET_LOCK_JS, on) or {}
    except Exception:  # noqa: BLE001 — page mid-navigation: the next check turns it back on
        return {}


@contextlib.asynccontextmanager
async def agent_input(page: Page, seconds: float = 6.0):
    """Let the agent's own clicks, keys and scrolls through the input lock while inside."""
    with contextlib.suppress(Exception):
        await page.evaluate(f"window.__pragAgentUntil = Date.now() + {int(seconds * 1000)}")
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            await page.evaluate("window.__pragAgentUntil = 0")


# ── Module State ──────────────────────────────────────────────────────

_playwright = None
_context: BrowserContext | None = None
_page: Page | None = None


async def _human_delay(lo: float = 0.5, hi: float = 1.5) -> None:
    """Non-blocking randomised sleep interval."""
    await asyncio.sleep(random.uniform(lo, hi))


# ── Browser Lifecycle ─────────────────────────────────────────────────

def profile_dir(account: str) -> str:
    """Each Instagram account gets its own browser profile, which holds its login."""
    return os.path.join(PROFILES_DIR, account)


@contextlib.contextmanager
def profile_lock(account: str):
    """Hold the account's browser profile for this run; yields False if another run already holds it."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(os.path.join(PROFILES_DIR, f"{account}.lock"), "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


async def init_browser(user_data_dir: str, headless: bool = False) -> tuple[BrowserContext, Page]:
    """Launch a persistent Chromium context with stealth in async mode.

    Parameters
    ----------
    user_data_dir : str
        Directory for persistent cookies / local-storage.
    headless : bool
        Run without a window, in Chromium's full headless mode: the same browser build, so reels play as they
        do in a window. It reports itself as HeadlessChrome, and a login that needs a code can't be finished.

    Returns
    -------
    tuple[BrowserContext, Page]
    """
    global _playwright, _context, _page

    _playwright = await async_playwright().start()
    _context = await _playwright.chromium.launch_persistent_context(
        user_data_dir,
        headless=headless,
        channel="chromium" if headless else None,  # the full browser, not the stripped-down headless shell
        viewport={"width": 1280, "height": 800},
        locale="en-US",
        args=["--disable-blink-features=AutomationControlled"],
    )

    await _context.add_init_script(INPUT_LOCK_JS)  # every page gets the lock (off until a session turns it on)
    _page = _context.pages[0] if _context.pages else await _context.new_page()
    await Stealth().apply_stealth_async(_page)

    print("🌐  Playwright async browser launched (persistent profile" + (", headless)." if headless else ")."))
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


NOT_LOGGED_IN_URLS = ("accounts/login", "accounts/emailsignup", "/challenge", "two_factor", "auth_platform",
                      "accounts/suspended")
CHECK_URLS = ("/challenge", "two_factor", "auth_platform")  # Instagram asking for a security code


async def is_logged_in(page: Page) -> bool:
    """Whether Instagram shows the logged-in app (not a login, sign-up or security-check page)."""
    await page.wait_for_load_state("domcontentloaded")
    await asyncio.sleep(2)
    url = page.url.lower()
    if any(part in url for part in NOT_LOGGED_IN_URLS):
        return False
    if await page.query_selector('input[type="password"]'):
        return False
    # Only the logged-in app has the navigation bar.
    return await page.query_selector('a[href="/explore/"], a[href^="/direct/inbox"]') is not None


async def wait_for_login(page: Page, timeout: float = 300.0) -> bool:
    """Wait up to `timeout` seconds for someone to log in by hand in the browser window."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            if "accounts/onetap" in page.url:  # "Save your login info?" — the profile keeps it anyway
                await page.goto("https://www.instagram.com/", wait_until="domcontentloaded")
            if await is_logged_in(page):
                return True
        except Exception:  # noqa: BLE001 — the page navigates while the login form submits
            pass
        await asyncio.sleep(3)
    return False


LOGIN_URL = "https://www.instagram.com/accounts/login/"
# The form's field names change between versions (username/password, email/pass), so match by role.
USERNAME_FIELD = 'input[name="email"], input[name="username"], input[autocomplete^="username"]'
PASSWORD_FIELD = 'input[type="password"]'
LOGIN_ERRORS = (
    "password was incorrect", "password you've entered is incorrect", "incorrect password",
    "doesn't belong to an account", "isn't connected to an account", "couldn't find your account",
)


async def log_in(page: Page, username: str, password: str, timeout: float = 60.0) -> str:
    """Log in through Instagram's login form.

    Returns "ok", "wrong_credentials", "challenge" (Instagram wants a
    security code sent by email or SMS) or "failed".
    """
    await page.goto(LOGIN_URL, wait_until="domcontentloaded")
    try:
        await page.wait_for_selector(USERNAME_FIELD, timeout=15000)
    except Exception:  # noqa: BLE001 — no form: maybe already logged in
        return "ok" if await is_logged_in(page) else "failed"
    for label in ("Allow all cookies", "Decline optional cookies"):
        button = page.get_by_role("button", name=label)
        if await button.count():
            await button.first.click()
            break
    await page.fill(USERNAME_FIELD, username)
    await _human_delay(0.3, 0.8)
    await page.fill(PASSWORD_FIELD, password)
    await _human_delay(0.3, 0.8)
    await page.press(PASSWORD_FIELD, "Enter")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await asyncio.sleep(2)
        url = page.url.lower()
        if any(part in url for part in CHECK_URLS):
            return "challenge"
        try:
            body = (await page.inner_text("body")).lower()
        except Exception:  # noqa: BLE001 — page mid-navigation
            continue
        if any(err in body for err in LOGIN_ERRORS):
            return "wrong_credentials"
        if await wait_for_login(page, timeout=1):
            return "ok"
    return "failed"


# The security-code box on Instagram's check pages (challenge, two-factor, the newer code entry).
CODE_FIELD = ", ".join(f'input{match}:not([type="hidden"]):not([type="password"])' for match in (
    '[autocomplete="one-time-code"]', '[name="verificationCode"]', '[name="security_code"]',
    '[name="approvals_code"]', '[name="code"]', '[aria-label*="code" i]', '[placeholder*="code" i]',
    '[inputmode="numeric"]',
))
SUBMIT_BUTTON = re.compile(r"^(confirm|submit|continue|next|verify|log in)$", re.I)


async def find_code_field(page: Page, timeout: float = 15.0):
    """The security-code box on the page (a Playwright locator), or None if there isn't one."""
    field = page.locator(CODE_FIELD).first
    try:
        await field.wait_for(state="visible", timeout=timeout * 1000)
    except Exception:  # noqa: BLE001 — no such box (or the page moved on)
        return None
    return field


async def enter_code(page: Page, field, code: str) -> None:
    """Type the code and submit it: Enter, then a Confirm-like button if the box is still showing."""
    await field.fill(code)
    await _human_delay(0.3, 0.8)
    await field.press("Enter")
    await asyncio.sleep(3)
    try:
        if await field.is_visible():
            button = page.get_by_role("button", name=SUBMIT_BUTTON)
            if await button.count():
                await button.first.click()
    except Exception:  # noqa: BLE001 — the page moved on
        pass


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
        async with agent_input(page):
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
    async with agent_input(page):
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


# ── Profile Pages ─────────────────────────────────────────────────────

PROFILE_JS = """
() => {
    const unavailable = /Sorry, this page isn't available|Profile isn't available/i.test(document.body.innerText || '');
    const header = document.querySelector('header');
    if (!header) return { unavailable, found: false };
    const follow = Array.from(header.querySelectorAll('button, [role="button"], div'))
        .filter(e => ['Follow', 'Follow Back', 'Following', 'Requested'].includes((e.innerText || '').trim()))
        .pop();  // deepest match
    const box = follow ? follow.getBoundingClientRect() : null;
    const followers = ((header.innerText || '').match(/([\\d.,]+[KMB]?) followers/) || [])[1] || null;
    return {
        unavailable, found: true, followers,
        verified: !!header.querySelector('svg[aria-label="Verified"]'),
        follow: follow ? {
            text: follow.innerText.trim(), x: box.left + box.width / 2, y: box.top + box.height / 2,
            visible: box.bottom > 0 && box.top < window.innerHeight,
        } : null,
    };
}
"""


async def open_profile(page: Page, handle: str) -> dict:
    """Open instagram.com/<handle>/ and describe it: found, verified, followers, follow button."""
    await page.goto(f"https://www.instagram.com/{handle}/", wait_until="domcontentloaded")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    state: dict = {}
    while loop.time() < deadline:
        state = await page.evaluate(PROFILE_JS)
        if state.get("unavailable") or (state.get("found") and state.get("follow")):
            break
        await asyncio.sleep(0.5)
    return state


async def follow_open_profile(page: Page) -> bool:
    """Follow the account whose profile is open; True once it shows Following / Requested."""
    follow = (await page.evaluate(PROFILE_JS)).get("follow")
    if not follow or not follow.get("visible"):
        return False
    if follow["text"] in ("Following", "Requested"):
        return True
    await _click(page, follow)
    await asyncio.sleep(1.5)
    after = (await page.evaluate(PROFILE_JS)).get("follow") or {}
    return after.get("text") in ("Following", "Requested")
