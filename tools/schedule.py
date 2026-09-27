"""
PRAG — Schedule the Daily Run (macOS launchd)

Runs daily.py for one account at the times given, every day, as a
LaunchAgent of the logged-in user (a browser window opens while it runs).

    python -m tools.schedule install --account NAME --at 09:00 --at 14:00 [--engagement live|dry|mixed]
                                     [--minutes 40] [--headless] [--start-llm]
    python -m tools.schedule show
    python -m tools.schedule remove --account NAME

If the Mac is asleep at a scheduled time, launchd runs the job once when
it wakes; if it's shut down, that run is skipped. Each run keeps its own
log in logs/<account>/; logs/launchd-<account>.log only catches output
from before a run's log opens (such as a crash on start).
"""

import argparse
import glob
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time

from config import DAILY_ENGAGEMENT, ENGAGEMENT_MODES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS_DIR = os.path.expanduser("~/Library/LaunchAgents")
LABEL_PREFIX = "com.feedlab.daily."
BASE_PATH = ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]


def label(account: str) -> str:
    return LABEL_PREFIX + account


def plist_path(account: str, agents_dir: str = AGENTS_DIR) -> str:
    return os.path.join(agents_dir, label(account) + ".plist")


def parse_time(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a time like 09:00 or 21:30")
    return int(match.group(1)), int(match.group(2))


def search_path() -> str:
    """PATH for the job: launchd's default lacks Homebrew, where ffmpeg (needed by Whisper) lives."""
    dirs = [os.path.dirname(found) for tool in ("ffmpeg",) if (found := shutil.which(tool))]
    dirs += ["/opt/homebrew/bin", "/usr/local/bin"] + BASE_PATH
    return ":".join(dict.fromkeys(dirs))


def build_plist(account: str, times: list[tuple[int, int]], *, minutes: float | None, engagement: str,
                start_llm: bool, headless: bool = False, python: str = sys.executable, root: str = PROJECT_ROOT) -> dict:
    command = [python, os.path.join(root, "daily.py"), "--account", account, "--engagement", engagement]
    if minutes is not None:
        command += ["--minutes", f"{minutes:g}"]
    command += ["--headless"] * headless + ["--start-llm"] * start_llm
    log = os.path.join(root, "logs", f"launchd-{account}.log")
    return {
        "Label": label(account),
        "ProgramArguments": command,
        "WorkingDirectory": root,
        "StartCalendarInterval": [{"Hour": h, "Minute": m} for h, m in sorted(set(times))],
        "EnvironmentVariables": {"PATH": search_path(), "PYTHONUNBUFFERED": "1", "FEEDLAB_LOG_ONLY": "1"},
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "RunAtLoad": False,
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(account: str, plist: dict, agents_dir: str = AGENTS_DIR) -> str:
    path = plist_path(account, agents_dir)
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(os.path.dirname(plist["StandardOutPath"]), exist_ok=True)
    _launchctl("bootout", f"{_domain()}/{plist['Label']}")  # replace an earlier schedule, if any
    with open(path, "wb") as fh:
        plistlib.dump(plist, fh)
    result = _launchctl("bootstrap", _domain(), path)
    if result.returncode != 0:  # right after a bootout launchd can need a moment
        time.sleep(1)
        result = _launchctl("bootstrap", _domain(), path)
    if result.returncode != 0:
        raise RuntimeError(f"launchctl couldn't load {path}: {(result.stderr or result.stdout).strip()}")
    return path


def remove(account: str, agents_dir: str = AGENTS_DIR) -> bool:
    path = plist_path(account, agents_dir)
    _launchctl("bootout", f"{_domain()}/{label(account)}")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def read(plist: dict) -> dict:
    """What a schedule runs: account, times, minutes, engagement, start_llm."""
    args = plist["ProgramArguments"][2:]
    value = lambda flag, default: args[args.index(flag) + 1] if flag in args else default
    return {
        "account": value("--account", "?"),
        "times": [f"{t['Hour']:02d}:{t['Minute']:02d}" for t in plist["StartCalendarInterval"]],
        "minutes": value("--minutes", None),
        "engagement": value("--engagement", DAILY_ENGAGEMENT),
        "headless": "--headless" in args,
        "start_llm": "--start-llm" in args,
    }


def describe(plist: dict) -> str:
    info = read(plist)
    extras = " · headless" * info["headless"] + " · starts LM Studio if needed" * info["start_llm"]
    return (f"@{info['account']} at {', '.join(info['times'])} · {info['minutes'] or 'default'} min · "
            f"{info['engagement']}: {ENGAGEMENT_MODES[info['engagement']]}{extras}")


def installed(agents_dir: str = AGENTS_DIR) -> list[dict]:
    """The installed schedules, each with its description, whether launchd has it loaded and its last exit."""
    items = []
    for path in sorted(glob.glob(os.path.join(agents_dir, LABEL_PREFIX + "*.plist"))):
        with open(path, "rb") as fh:
            plist = plistlib.load(fh)
        state = _launchctl("print", f"{_domain()}/{plist['Label']}")
        last = re.search(r"last exit code = (.+)", state.stdout or "")
        items.append({**read(plist), "description": describe(plist), "loaded": state.returncode == 0,
                      "last_exit": last.group(1).strip() if last else None})
    return items


def show(agents_dir: str = AGENTS_DIR) -> list[str]:
    return [f"{item['description']} · {'loaded' if item['loaded'] else 'NOT loaded (run install again)'}"
            + (f" · last exit: {item['last_exit']}" if item["last_exit"] else "")
            for item in installed(agents_dir)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Schedule the PRAG daily run (macOS launchd)")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("install", help="Run daily.py for an account at set times every day")
    add.add_argument("--account", required=True)
    add.add_argument("--at", dest="times", type=parse_time, action="append", required=True,
                     help="A time of day, 24-hour HH:MM (repeat for several runs)")
    add.add_argument("--minutes", type=float, help="Session length (default: config.SESSION_MINUTES)")
    add.add_argument("--engagement", choices=list(ENGAGEMENT_MODES), default=DAILY_ENGAGEMENT,
                     help="; ".join(f"{mode}: {what}" for mode, what in ENGAGEMENT_MODES.items())
                     + f" (default: {DAILY_ENGAGEMENT})")
    add.add_argument("--dry-run", dest="engagement", action="store_const", const="dry",
                     help="Same as --engagement dry")
    add.add_argument("--headless", action="store_true", help="No browser window (a login that needs a code needs one)")
    add.add_argument("--start-llm", action="store_true", help="Start LM Studio and load the model when it's off")
    sub.add_parser("show", help="List the installed schedules")
    rm = sub.add_parser("remove", help="Remove an account's schedule")
    rm.add_argument("--account", required=True)
    args = parser.parse_args()

    if args.command == "install":
        account = args.account.lstrip("@")
        plist = build_plist(account, args.times, minutes=args.minutes, engagement=args.engagement,
                            start_llm=args.start_llm, headless=args.headless)
        path = install(account, plist)
        print(f"🗓  Scheduled {describe(plist)}\n   {path}")
        if not shutil.which("ffmpeg"):
            print("   ⚠️  ffmpeg wasn't found: speech won't be transcribed (brew install ffmpeg).")
    elif args.command == "show":
        print("\n".join(show()) or "No PRAG schedules installed.")
    else:
        account = args.account.lstrip("@")
        print(f"🗑  Removed the schedule for @{account}." if remove(account) else f"No schedule for @{account}.")


if __name__ == "__main__":
    main()
