"""
PRAG — Account Login

Makes sure the browser profile is logged into its Instagram account, with
the password from accounts.json when there is one. When Instagram needs
you — it sent a security code, or there's no working password — PRAG
asks in a dialog box (and in the panel, if it's open) and types in what
you enter; nothing typed is saved. A check it can't fill in is left for
you to finish in the browser window. The account's browser profile keeps
the login afterwards.
"""

import os

from automation.browser_driver import enter_code, find_code_field, is_logged_in, log_in, wait_for_login
from automation.notify import notify
from automation.prompt import ask
from database.seed import account_password

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGIN_PROBLEMS = {
    "wrong_credentials": "Instagram rejected the username or password.",
    "challenge": "Instagram wants a security check (a code sent by email or SMS).",
    "failed": "the login didn't go through.",
}
NO_PASSWORD = "No password for it in accounts.json."
ATTEMPTS = 3  # passwords, and codes, asked for at most this many times


async def ensure_logged_in(page, username: str, manual_timeout: float = 600.0, headless: bool = False) -> bool:
    """True once the page shows Instagram logged in as `username`'s profile.

    A question waits up to manual_timeout seconds for an answer. Headless there's no window to finish a
    check by hand, so it gives up when the questions can't get it through.
    """
    if await is_logged_in(page):
        return True
    password = account_password(username)
    problem = NO_PASSWORD
    for _ in range(ATTEMPTS):
        if not password:
            password = await ask(password_question(username, problem), account=username, kind="password",
                                 secret=True, timeout=manual_timeout)
            if not password:
                break
        print(f"  🔐  Logging in as @{username} …")
        outcome = await log_in(page, username, password)
        if outcome == "ok":
            print(f"  🔓  Logged in as @{username}.")
            return True
        problem = LOGIN_PROBLEMS[outcome]
        print(f"  ⚠️  Couldn't log in automatically: {problem}")
        if outcome == "challenge" and await enter_security_code(page, username, manual_timeout):
            print(f"  🔓  Logged in as @{username}.")
            return True
        if outcome != "wrong_credentials":
            break
        password = None  # ask for the right one

    if await is_logged_in(page):  # finished by hand in the window meanwhile
        return True
    if headless:
        shot = await _screenshot(page, username)
        print(f"  🔐  Logging in as @{username} needs the browser window: start a session with headless off."
              + (f" What Instagram showed: {shot}" if shot else ""))
        notify("PRAG needs you", f"Log in as @{username}: start a session with headless off and log in in "
               "the window.", subtitle=problem)
        return False
    print(f"  🔐  Log in as @{username} in the browser window (waiting up to {manual_timeout / 60:.0f} minutes) …")
    notify("PRAG needs you", f"Log in as @{username} in the PRAG browser window "
           f"(it waits {manual_timeout / 60:.0f} min).", subtitle=problem)
    return await wait_for_login(page, timeout=manual_timeout)


def password_question(username: str, problem: str) -> str:
    if problem == NO_PASSWORD:
        return (f"PRAG needs the Instagram password for @{username}: there's none in accounts.json. "
                "It's only used to log in, and isn't saved.")
    return f"Instagram rejected the password for @{username}. Enter the right one (only used to log in, not saved):"


async def enter_security_code(page, username: str, timeout: float) -> bool:
    """Ask for the code Instagram sent and type it in.

    False when the page has no code box PRAG recognises, or no working code arrives.
    """
    for attempt in range(ATTEMPTS):
        field = await find_code_field(page)
        if field is None:
            print("  🔐  Instagram's check page has no code box PRAG recognises.")
            return False
        print(f"  🔐  Asking for @{username}'s security code (a dialog box, and the PRAG panel if it's open) …")
        question = (f"Instagram sent a security code for @{username} (by email or text message, or it's in your "
                    "authenticator app). Enter it here:") if attempt == 0 else (
                    f"That code didn't work for @{username}. Check it, or enter the newest code:")
        code = await ask(question, account=username, kind="code", timeout=timeout,
                         resolved=lambda: is_logged_in(page))
        if code is None:
            return await is_logged_in(page)  # finished in the window instead, or nobody answered
        print("  🔐  Entering the code …")
        await enter_code(page, field, code)
        if await wait_for_login(page, timeout=30):
            return True
    return False


async def _screenshot(page, username: str) -> str | None:
    """Save what Instagram shows (headless, nobody saw it) to work out what it wanted."""
    path = os.path.join(PROJECT_ROOT, "logs", f"login-{username}.png")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        await page.screenshot(path=path)
        return path
    except Exception:  # noqa: BLE001
        return None
