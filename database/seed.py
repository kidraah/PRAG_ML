"""
FeedLab — Account Seeder

Reads `accounts.json` from the project root and inserts each entry into
the `accounts` table using INSERT OR IGNORE to avoid duplicates.
"""

import json
import os
import sqlite3

from database.db import get_connection, init_db

ACCOUNTS_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "accounts.json"
)


def seed_accounts(json_path: str = ACCOUNTS_JSON) -> list[dict]:
    """Read accounts from a JSON file and insert them into the database.

    Returns the list of account dicts that were processed.
    """
    with open(json_path, "r") as fh:
        accounts = json.load(fh)

    conn = get_connection()
    cursor = conn.cursor()

    for acct in accounts:
        cursor.execute(
            """
            INSERT OR IGNORE INTO accounts (username, password, status)
            VALUES (?, ?, ?)
            """,
            (acct["username"], acct["password"], acct.get("status", "WARMING_UP")),
        )

    conn.commit()
    conn.close()
    return accounts


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

    accounts = seed_accounts()
    print(f"\n🌱  Seeded {len(accounts)} account(s) from accounts.json")

    print_accounts_table()
