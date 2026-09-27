"""
PRAG — Account Seeder

Reads `accounts.json` from the project root and inserts each entry into
the `accounts` table using INSERT OR IGNORE to avoid duplicates.

Entries only need a username, e.g. [{"username": "my_test_account"}];
entries with a blank username are skipped. Passwords are never copied
into the database. If an entry has one, the agent uses it (read from
accounts.json at that moment, never printed) to log the account's browser
profile in; otherwise you log in by hand in the window on its first run.
Either way, that account's browser profile keeps the login afterwards.
"""

import json
import os
import sqlite3

from database.db import get_connection, init_db

ACCOUNTS_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "accounts.json"
)


def seed_accounts(json_path: str = ACCOUNTS_JSON, conn: sqlite3.Connection | None = None) -> list[str]:
    """Read accounts from a JSON file and insert them into the database.

    Returns the usernames that were processed (blank slots are skipped).
    """
    with open(json_path, "r") as fh:
        accounts = json.load(fh)

    own = conn is None
    conn = conn or get_connection()
    usernames = []
    for acct in accounts:
        username = (acct.get("username") or "").strip().lstrip("@")
        if not username:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO accounts (username, password, status) VALUES (?, '', ?)",
            (username, acct.get("status") or "ACTIVE"),
        )
        usernames.append(username)

    conn.commit()
    if own:
        conn.close()
    return usernames


def account_password(username: str, json_path: str = ACCOUNTS_JSON) -> str | None:
    """The account's password from accounts.json, if one is filled in (read only to log in)."""
    try:
        with open(json_path) as fh:
            accounts = json.load(fh)
    except (OSError, ValueError):
        return None
    for acct in accounts:
        if (acct.get("username") or "").strip().lstrip("@") == username:
            return acct.get("password") or None
    return None


def print_accounts_table() -> None:
    """Print a formatted table of all accounts and their statuses."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT username, status, daily_scroll_minutes, last_used_timestamp "
        "FROM accounts ORDER BY username"
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("⚠️  No accounts found in the database.")
        return

    # Header
    print()
    print(f"{'Username':<20} {'Status':<14} {'Daily Min':<12} {'Last Used':<22}")
    print("─" * 68)

    for username, status, minutes, last_used in rows:
        last_used = last_used or "—"
        print(f"{username:<20} {status:<14} {minutes or 0:<12} {last_used:<22}")

    print()
    print(f"Total accounts: {len(rows)}")


if __name__ == "__main__":
    # Ensure the schema exists before seeding
    init_db()

    usernames = seed_accounts()
    print(f"\n🌱  Seeded {len(usernames)} account(s) from accounts.json")

    print_accounts_table()
