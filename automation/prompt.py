"""
PRAG — Ask the Person at the Mac

When Instagram needs something only you have — the security code it just
sent, or the password when accounts.json has none or a wrong one — the
agent asks in a dialog box and, if it's open, in the PRAG panel
(launch.py). The first answer is used; the other closes.

Answers are never printed or logged. One typed in the panel reaches the
run through a file only you can read (logs/prompts/), deleted as soon as
it's read.
"""

import asyncio
import glob
import json
import os
import re
import sys
import time
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(PROJECT_ROOT, "logs", "prompts")
RESOLVED_EVERY = 3.0  # seconds between checks whether the question still matters
REQUEST_ID = re.compile(r"[0-9a-f]{32}")
NO_ANSWER = "__FEEDLAB_NO_ANSWER__"
# The question and its settings go in as arguments, so nothing in them can break the script.
DIALOG = """
on run argv
    set question to item 1 of argv
    set waitSeconds to (item 2 of argv) as integer
    activate
    if item 3 of argv is "1" then
        set reply to display dialog question default answer "" with hidden answer with title "PRAG" buttons {"Cancel", "OK"} default button "OK" cancel button "Cancel" giving up after waitSeconds
    else
        set reply to display dialog question default answer "" with title "PRAG" buttons {"Cancel", "OK"} default button "OK" cancel button "Cancel" giving up after waitSeconds
    end if
    if gave up of reply then return "__FEEDLAB_NO_ANSWER__"
    return text returned of reply
end run
"""


def _path(kind: str, request_id: str) -> str:
    return os.path.join(PROMPT_DIR, f"{kind}-{request_id}.json")


def _write(path: str, data: dict) -> None:
    """Write JSON only this user can read, all at once."""
    os.makedirs(PROMPT_DIR, mode=0o700, exist_ok=True)
    temp = path + ".tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh)
    os.replace(temp, path)


def _remove(*paths: str) -> None:
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def pending(now: float | None = None) -> list[dict]:
    """The questions waiting for an answer, oldest first (the panel shows these)."""
    now = time.time() if now is None else now
    waiting = []
    for path in glob.glob(os.path.join(PROMPT_DIR, "request-*.json")):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if data.get("expires", 0) <= now:
            _remove(path)  # left behind by a run that ended
            continue
        waiting.append({key: data.get(key) for key in ("id", "account", "kind", "message", "secret", "asked", "expires")})
    return sorted(waiting, key=lambda item: item["asked"] or 0)


def answer(request_id: str, text: str | None) -> bool:
    """Answer a waiting question (None cancels it); False if it isn't waiting any more."""
    if not REQUEST_ID.fullmatch(str(request_id)) or not os.path.exists(_path("request", request_id)):
        return False
    _write(_path("response", request_id), {"id": request_id, "answer": text})
    return True


async def _open_dialog(message: str, secret: bool, timeout: float):
    """The dialog box as a running osascript process, or None where there's no such thing."""
    if sys.platform != "darwin":
        return None
    try:
        return await asyncio.create_subprocess_exec(
            "osascript", "-e", DIALOG, message, str(max(1, int(timeout))), "1" if secret else "0",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return None


async def ask(message: str, *, account: str, kind: str, secret: bool = False, timeout: float = 600.0,
              resolved=None) -> str | None:
    """Ask `message` and wait up to `timeout` seconds for the answer.

    kind is "code" or "password" (the panel words its box by it); secret hides what's typed. Returns the
    answer, or None if it was cancelled, nobody answered in time, or `resolved()` — an async check made
    every RESOLVED_EVERY seconds — says the question no longer matters (say, you logged in in the window).
    """
    request_id = uuid.uuid4().hex
    request, response = _path("request", request_id), _path("response", request_id)
    now = time.time()
    _write(request, {"id": request_id, "account": account, "kind": kind, "message": message, "secret": secret,
                     "asked": now, "expires": now + timeout})
    dialog = await _open_dialog(message, secret, timeout)
    reply = asyncio.ensure_future(dialog.communicate()) if dialog is not None else None
    loop = asyncio.get_running_loop()
    deadline, next_check = loop.time() + timeout, loop.time() + RESOLVED_EVERY
    try:
        while loop.time() < deadline:
            if os.path.exists(response):  # answered in the panel
                with open(response) as fh:
                    text = json.load(fh).get("answer")
                return (text.strip() or None) if isinstance(text, str) else None
            if reply is not None and reply.done():
                stdout, stderr = reply.result()
                if dialog.returncode == 0:
                    text = stdout.decode().rstrip("\n")
                    return None if text == NO_ANSWER else (text.strip() or None)
                if b"-128" in stderr:  # Cancel
                    return None
                reply = None  # the dialog couldn't open: the panel can still answer
            if resolved is not None and loop.time() >= next_check:
                next_check = loop.time() + RESOLVED_EVERY
                if await resolved():
                    return None
            await asyncio.sleep(0.25)
        return None
    finally:
        if dialog is not None and dialog.returncode is None:
            dialog.kill()  # closes the dialog box
            await dialog.wait()
        if reply is not None and not reply.done():
            reply.cancel()
        _remove(request, response)
