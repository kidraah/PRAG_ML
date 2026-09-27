"""
PRAG — Appium Driver & Gesture Controller

Connects to an Android Emulator via Appium and provides:
  • Direct navigation to the Reels tab.
  • Human-like swipe / tap gestures (Appium + ADB fallback).
  • Popup dismissal during scrolling to prevent crashes.

Assumes the user has already logged into Instagram manually.
"""

import random
import subprocess
import time

from appium import webdriver
from appium.options.android import UiAutomator2Options
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)


# ── Default Capabilities ──────────────────────────────────────────────

APPIUM_SERVER = "http://localhost:4723"

DEFAULT_CAPS = {
    "platformName": "Android",
    "automationName": "UiAutomator2",
    "appPackage": "com.instagram.android",
    "appActivity": "com.instagram.android.activity.MainTabActivity",
    "noReset": True,
}

ELEMENT_WAIT = 5  # seconds for element searches


def _human_delay(lo: float = 0.5, hi: float = 1.5) -> None:
    """Sleep for a randomised, human-like interval."""
    time.sleep(random.uniform(lo, hi))


# ── Driver Lifecycle ──────────────────────────────────────────────────

_driver: webdriver.Remote | None = None


def init_driver() -> webdriver.Remote:
    """Initialise and return an Appium WebDriver session."""
    global _driver

    options = UiAutomator2Options()
    for key, value in DEFAULT_CAPS.items():
        options.set_capability(key, value)

    _driver = webdriver.Remote(APPIUM_SERVER, options=options)
    _driver.implicitly_wait(ELEMENT_WAIT)
    print("📱  Appium driver connected to Android Emulator.")
    return _driver


def get_driver() -> webdriver.Remote:
    """Return the current driver, raising if not initialised."""
    if _driver is None:
        raise RuntimeError("Appium driver not initialised. Call init_driver() first.")
    return _driver


def quit_driver() -> None:
    """Cleanly shut down the Appium session."""
    global _driver
    if _driver is not None:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None
        print("📱  Appium driver disconnected.")


# ── Element Helpers ───────────────────────────────────────────────────

def _find(driver, strategy: str, locator: str, timeout: float = ELEMENT_WAIT):
    """Try to find a single element, return it or None."""
    try:
        driver.implicitly_wait(timeout)
        return driver.find_element(strategy, locator)
    except (NoSuchElementException, WebDriverException):
        return None
    finally:
        driver.implicitly_wait(ELEMENT_WAIT)


def _find_any(driver, candidates: list[tuple[str, str]], timeout: float = 3):
    """Try multiple (strategy, locator) pairs, return the first match or None."""
    for strategy, locator in candidates:
        el = _find(driver, strategy, locator, timeout=timeout)
        if el is not None:
            return el
    return None


# ── ADB Gesture Helpers ──────────────────────────────────────────────

def _adb_swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 350) -> None:
    """Perform a swipe gesture via ADB (bypasses W3C actions issues)."""
    subprocess.run(
        ["adb", "shell", "input", "swipe",
         str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
        capture_output=True,
        timeout=10,
    )


def _adb_tap(x: int, y: int) -> None:
    """Perform a tap via ADB."""
    subprocess.run(
        ["adb", "shell", "input", "tap", str(x), str(y)],
        capture_output=True,
        timeout=10,
    )


# ── Reels Navigation ─────────────────────────────────────────────────

def navigate_to_reels(driver) -> bool:
    """Navigate to the Instagram Reels tab.

    Tries the ADB deep-link intent first, then falls back to tapping
    the Reels icon in the bottom nav bar.

    Returns True if Reels feed is confirmed active.
    """
    print("  🎬  Navigating to Reels …")

    # Method 1: ADB deep link
    try:
        subprocess.run(
            ["adb", "shell", "am", "start",
             "-a", "android.intent.action.VIEW",
             "-d", "instagram://reels"],
            capture_output=True,
            timeout=10,
        )
        _human_delay(2.0, 3.0)

        if _verify_reels_feed(driver):
            print("  ✅  Reels feed active (via deep link).")
            return True
    except Exception as exc:
        print(f"  ⚠️  Deep link failed: {exc}")

    # Method 2: Tap the Reels icon
    reels_icon = _find_any(driver, [
        ("xpath", "//*[contains(@content-desc,'Reels')]"),
        ("xpath", "//*[contains(@content-desc,'reels')]"),
        ("id", "com.instagram.android:id/clips_tab"),
    ], timeout=5)

    if reels_icon is not None:
        reels_icon.click()
        _human_delay(2.0, 3.0)
        if _verify_reels_feed(driver):
            print("  ✅  Reels feed active (via tab icon).")
            return True

    # Method 3: Maybe already on Reels
    if _verify_reels_feed(driver):
        print("  ✅  Already on Reels feed.")
        return True

    print("  ⚠️  Could not confirm Reels feed — proceeding anyway.")
    return False


def _verify_reels_feed(driver) -> bool:
    """Heuristic: check if the current screen looks like the Reels feed."""
    indicators = ["reel_viewer", "clips_viewer", "clips_tab", "reel"]
    try:
        source = driver.page_source.lower()
        return any(ind in source for ind in indicators)
    except WebDriverException:
        return False


# ── Popup Dismisser ──────────────────────────────────────────────────

def dismiss_unexpected_popup(driver) -> bool:
    """Try to dismiss any unexpected popup/dialog during reel scrolling.

    Returns True if a popup was found and dismissed.
    """
    popup_dismiss = [
        ("xpath", "//*[@text='Not Now']"),
        ("xpath", "//*[@text='Not now']"),
        ("xpath", "//*[@text='OK']"),
        ("xpath", "//*[@text='Dismiss']"),
        ("xpath", "//*[@text='Cancel']"),
        ("xpath", "//*[@text='Close']"),
        ("xpath", "//*[@text='Got It']"),
        ("xpath", "//*[@text='Allow']"),
        ("id", "com.android.permissioncontroller:id/permission_deny_button"),
        ("xpath", "//*[contains(@content-desc,'Back')]"),
    ]
    el = _find_any(driver, popup_dismiss, timeout=1)
    if el is not None:
        try:
            el.click()
            print("  🔕  Dismissed unexpected popup.")
            _human_delay(0.5, 1.0)
            return True
        except (StaleElementReferenceException, WebDriverException):
            pass
    return False


# ── Gesture Helpers ───────────────────────────────────────────────────

def _screen_size() -> tuple[int, int]:
    """Return (width, height) of the current device screen."""
    size = get_driver().get_window_size()
    return size["width"], size["height"]


def swipe_up() -> None:
    """Swipe up to scroll to the next reel.

    Tries Appium W3C swipe first; falls back to ADB input swipe if
    the driver action fails.
    """
    driver = get_driver()
    width, height = _screen_size()

    start_x = width // 2 + random.randint(-20, 20)
    start_y = int(height * 0.75)
    end_y = int(height * 0.25)
    duration_ms = random.randint(250, 450)

    try:
        driver.swipe(start_x, start_y, start_x, end_y, duration_ms)
    except WebDriverException:
        # Fallback: use ADB swipe (bypasses W3C action chain issues)
        _adb_swipe(start_x, start_y, start_x, end_y, duration_ms)

    _human_delay()
    print("  ↑  Swiped up (next reel)")


def double_tap_like() -> None:
    """Double-tap the centre of the screen to like the current reel."""
    driver = get_driver()
    width, height = _screen_size()

    tap_x = width // 2 + random.randint(-15, 15)
    tap_y = height // 2 + random.randint(-15, 15)

    try:
        driver.tap([(tap_x, tap_y)], duration=80)
        time.sleep(random.uniform(0.08, 0.15))
        driver.tap([(tap_x, tap_y)], duration=80)
    except WebDriverException:
        # Fallback: ADB double tap
        _adb_tap(tap_x, tap_y)
        time.sleep(random.uniform(0.08, 0.15))
        _adb_tap(tap_x, tap_y)

    _human_delay()
    print("  ❤️  Double-tapped to like")


def tap_save() -> None:
    """Tap the save/bookmark icon (bottom-right of a reel)."""
    driver = get_driver()
    width, height = _screen_size()

    save_x = int(width * 0.93) + random.randint(-5, 5)
    save_y = int(height * 0.88) + random.randint(-5, 5)

    try:
        driver.tap([(save_x, save_y)], duration=100)
    except WebDriverException:
        _adb_tap(save_x, save_y)

    _human_delay()
    print("  🔖  Tapped save")
