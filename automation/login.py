"""
FeedLab — Account Login

Makes sure the browser profile is logged into its Instagram account: with
the password from accounts.json if there is one, otherwise (or if that
fails) by waiting for someone to log in by hand in the window. The
account's browser profile keeps the login afterwards.
"""

from automation.browser_driver import is_logged_in, log_in, wait_for_login
from database.seed import account_password

LOGIN_PROBLEMS = {
    "wrong_credentials": "Instagram rejected the username or password in accounts.json.",
    "challenge": "Instagram wants a security check (a code sent by email or SMS).",
    "failed": "the login didn't go through.",
}


async def ensure_logged_in(page, username: str, manual_timeout: float = 600.0) -> bool:
    """True once the page shows Instagram logged in as `username`'s profile."""
    if await is_logged_in(page):
        return True
    password = account_password(username)
    if password:
        print(f"  🔐  Logging in as @{username} …")
        outcome = await log_in(page, username, password)
        if outcome == "ok":
            print(f"  🔓  Logged in as @{username}.")
            return True
        print(f"  ⚠️  Couldn't log in automatically: {LOGIN_PROBLEMS[outcome]}")
    print(f"  🔐  Log in as @{username} in the browser window (waiting up to {manual_timeout / 60:.0f} minutes) …")
    return await wait_for_login(page, timeout=manual_timeout)
