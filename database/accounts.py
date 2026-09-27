"""
PRAG — Instagram Accounts: add, change and remove them

The accounts the agent may use live in two places: accounts.json (username,
password and status; the only place a password is kept) and the
database's accounts table (username and status, never the password). These
functions keep the two in step for the panel's Accounts tab.

Passwords are written to accounts.json (readable only by you) and read
back only to log in (database/seed.py). Nothing here returns one.
Removing an account keeps what it collected (sessions, reels, digests and
its bias), so adding it again reconnects its history.
"""

import json
import os
import re
import shutil
import sqlite3

from agent.bias import has_own as has_own_bias
from agent.bias import load as load_bias
from automation.browser_driver import profile_dir, profile_lock
from database.db import get_connection
from database.seed import ACCOUNTS_JSON

USERNAME = re.compile(r"[A-Za-z0-9._]{1,30}")
MAX_PASSWORD = 128


def valid_username(name) -> str:
    name = str(name or "").strip().lstrip("@")
    if not USERNAME.fullmatch(name):
        raise ValueError("An Instagram username is 1-30 letters, numbers, dots or underscores.")
    return name


def _checked_password(password) -> str:
    password = "" if password is None else str(password).strip("\r\n")
    if len(password) > MAX_PASSWORD:
        raise ValueError(f"That password is longer than {MAX_PASSWORD} characters.")
    return password


def _read(path: str) -> list:
    try:
        with open(path) as fh:
            entries = json.load(fh)
    except FileNotFoundError:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"{os.path.basename(path)} should be a list of accounts.")
    return entries


def _write(path: str, entries: list) -> None:
    """Replace the file in one step, readable only by this user (it holds passwords)."""
    temp = path + ".tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(temp, path)
    os.chmod(path, 0o600)


def _name(entry) -> str:
    return (entry.get("username") or "").strip().lstrip("@") if isinstance(entry, dict) else ""


def _entry(entries: list, username: str) -> dict | None:
    return next((e for e in entries if _name(e) == username), None)


def _in_database(conn: sqlite3.Connection, username: str) -> bool:
    return conn.execute("SELECT 1 FROM accounts WHERE username = ?", (username,)).fetchone() is not None


def overview(conn: sqlite3.Connection | None = None, path: str = ACCOUNTS_JSON) -> list[dict]:
    """Every account, as the panel lists it (never with its password)."""
    entries = _read(path)
    own = conn is None
    conn = conn or get_connection()
    try:
        rows = {u: (s, t) for u, s, t in conn.execute("SELECT username, status, last_used_timestamp FROM accounts")}
        reels = dict(conn.execute("SELECT s.account, COUNT(i.id) FROM sessions s "
                                  "JOIN impressions i ON i.session_id = s.id GROUP BY s.account"))
    finally:
        if own:
            conn.close()
    names = sorted({name for name in [_name(e) for e in entries] + list(rows) if name})
    out = []
    for name in names:
        entry = _entry(entries, name)
        status, last_used = rows.get(name, (None, None))
        out.append({
            "username": name,
            "status": status or (entry or {}).get("status") or "ACTIVE",
            "in_database": name in rows,
            "has_password": bool(entry and entry.get("password")),
            "last_used": last_used,
            "reels": reels.get(name, 0),
            "bias": load_bias(name).name,
            "persona_set": has_own_bias(name),  # False: it scrolls with the default until given a persona
            "login_saved": os.path.isdir(profile_dir(name)),
        })
    return out


def add(username, password=None, *, conn: sqlite3.Connection | None = None, path: str = ACCOUNTS_JSON) -> str:
    """Add an account (password optional: the login prompt asks for it otherwise)."""
    username = valid_username(username)
    password = _checked_password(password)
    entries = _read(path)
    if _entry(entries, username) is not None:
        raise ValueError(f"@{username} is already added.")
    own = conn is None
    conn = conn or get_connection()
    try:
        conn.execute("INSERT OR IGNORE INTO accounts (username, password, status) VALUES (?, '', 'ACTIVE')", (username,))
        conn.commit()
    finally:
        if own:
            conn.close()
    blank = next((e for e in entries if isinstance(e, dict) and not _name(e)), None)  # fill an empty slot first
    if blank is not None:
        blank.update(username=username, password=password, status=blank.get("status") or "ACTIVE")
    else:
        entries.append({"username": username, "password": password, "status": "ACTIVE"})
    _write(path, entries)
    return username


def set_password(username, password, *, conn: sqlite3.Connection | None = None, path: str = ACCOUNTS_JSON) -> None:
    """Save a new password for the account, or forget it ("")."""
    username = valid_username(username)
    password = _checked_password(password)
    entries = _read(path)
    entry = _entry(entries, username)
    if entry is None:
        own = conn is None
        conn = conn or get_connection()
        try:
            known = _in_database(conn, username)
        finally:
            if own:
                conn.close()
        if not known:
            raise ValueError(f"@{username} isn't added.")
        entries.append({"username": username, "password": password, "status": "ACTIVE"})
    else:
        entry["password"] = password
    _write(path, entries)


def remove(username, *, delete_login: bool = False, conn: sqlite3.Connection | None = None,
           path: str = ACCOUNTS_JSON) -> dict:
    """Stop using an account: forget its password and database entry, and (if asked) its saved Instagram login.

    What it collected stays. Refused while a run has the account's browser open.
    """
    username = valid_username(username)
    with profile_lock(username) as free:
        if not free:
            raise ValueError(f"@{username} is being used by a run: stop it first.")
        entries = _read(path)
        entry = _entry(entries, username)
        own = conn is None
        conn = conn or get_connection()
        try:
            if entry is None and not _in_database(conn, username):
                raise ValueError(f"@{username} isn't added.")
            if entry is not None:
                entries.remove(entry)
                _write(path, entries)
            conn.execute("DELETE FROM accounts WHERE username = ?", (username,))
            conn.commit()
        finally:
            if own:
                conn.close()
        login = profile_dir(username)
        login_deleted = delete_login and os.path.isdir(login)
        if login_deleted:
            shutil.rmtree(login)
    return {"username": username, "login_deleted": login_deleted}
