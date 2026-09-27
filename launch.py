"""
PRAG — Control Panel

    python launch.py [--port 8765] [--no-browser]

Opens a page in your browser, served by this Mac to itself only, that does
what the commands do: start a session with the engagement mode you pick,
follow its output and stop it, rebuild and read the digest, check the
curation report, set each account's bias (what it curates towards, and
which accounts to follow first), and add or remove the daily schedule.

Runs go through daily.py, so they're guarded and logged the same way, and
they keep going if the page or this terminal is closed — reopen the panel
to follow or stop one, including a run the schedule started. When a run
needs a login code or password, the panel asks for it too (as does a
dialog box on the Mac).
"""

import argparse
import glob
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import daily
from agent import bias as biases
from automation import prompt
from config import DAILY_ENGAGEMENT, DAILY_MAX_MINUTES, ENGAGEMENT_MODES, MIN_BATTERY, SESSION_MINUTES
from database.db import engaged_today, init_db, list_accounts, minutes_today
from ml.scorer import ModelScorer
from perception.local_evaluator import llm_serving
from stories.digest import DIGEST_DIR
from tools import schedule

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_OUTPUT = os.path.join(daily.LOG_DIR, "panel-run.out")  # what the latest run started here printed
RUN_INFO = os.path.join(daily.LOG_DIR, "panel-run.json")   # which run that was
TOKEN = secrets.token_urlsafe(24)  # every API call carries it; other websites can't read the page to learn it
ALLOWED_HOSTS: set[str] = set()    # 127.0.0.1:<port> and localhost:<port> — refuses DNS-rebinding tricks
NOISE = re.compile(r"pin_memory|super\(\)\.__init__\(loader\)|HF_TOKEN|HF Hub|Loading weights|resource_tracker|"
                   r"^\s+warnings\.warn\($")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
DIGEST_FILE = re.compile(r"(\d{4}-\d{2}-\d{2}|latest)\.html")
USERNAME = re.compile(r"[A-Za-z0-9._]{1,30}")
PAGE_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
DIGEST_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src https: data:; frame-ancestors 'none'"


# ── What the page shows ───────────────────────────────────────────────

def active_accounts() -> list[str]:
    """ACTIVE accounts, most recently used first."""
    accounts = sorted(list_accounts(), key=lambda a: a.get("last_used") or "", reverse=True)
    return [a["username"] for a in accounts if a["status"] == "ACTIVE"]


def checked_account(name) -> str:
    if name not in active_accounts():
        raise ValueError(f"Unknown or inactive account: {name!r}")
    return name


_cache: dict[str, tuple[float, object]] = {}


def cached(key: str, seconds: float, compute):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < seconds:
        return hit[1]
    value = compute()
    _cache[key] = (time.time(), value)
    return value


def account_status(account: str) -> dict:
    on_battery, charge = cached("battery", 30, daily.battery)
    used = minutes_today(account)
    engaged = engaged_today(account)
    return {
        "llm": cached("llm", 10, llm_serving),
        "minutes_used": round(used),
        "minutes_cap": DAILY_MAX_MINUTES,
        "engaged_today": engaged,
        "mixed_next": "dry run (today's live session already ran)" if engaged else "live (first session today)",
        "battery": {"on_battery": on_battery, "charge": charge},
        "skip": daily.skip_reason(account, used, on_battery, charge),
    }


def read_output(path: str, offset: int, limit: int = 400_000) -> tuple[str, int, bool]:
    """(whole new lines after `offset`, the next offset, whether the file started over), minus library noise."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "", 0, offset > 0
    restarted = offset > size
    if restarted:
        offset = 0
    with open(path, "rb") as fh:
        fh.seek(offset)
        data = fh.read(limit)
    end = data.rfind(b"\n") + 1 or (len(data) if len(data) == limit else 0)  # a partial last line waits
    text = data[:end].decode("utf-8", "replace")
    lines = text.split("\n")[:-1] if text.endswith("\n") else text.split("\n")
    kept = [ANSI.sub("", line.rsplit("\r", 1)[-1]) for line in lines if line]  # a progress bar keeps its last state
    return "".join(line + "\n" for line in kept if not NOISE.search(line)), offset + end, restarted


def digest_data(account: str, date: str | None) -> dict:
    folder = os.path.join(DIGEST_DIR, account)
    dates = sorted((os.path.basename(p)[:-5] for p in glob.glob(os.path.join(folder, "*.json"))), reverse=True)
    if not dates:
        return {"dates": [], "date": None, "digest": None, "page": None}
    date = date if date in dates else dates[0]
    with open(os.path.join(folder, f"{date}.json")) as fh:
        return {"dates": dates, "date": date, "digest": json.load(fh), "page": f"/digest/{account}/{date}.html"}


def bias_info(account: str) -> dict:
    current = biases.load(account)
    trained = ModelScorer.trained_for()
    model_bias = next((b for b in [current, *biases.PRESETS.values()] if b.fingerprint == trained), None)
    return {
        "bias": current.to_dict(),
        "own": biases.has_own(account),
        "presets": {key: preset.to_dict() for key, preset in biases.PRESETS.items()},
        "model_interests": model_bias.interests if model_bias else None,  # the fallback model scores only for these
        "model_name": model_bias.name if model_bias else None,
        "default_name": biases.PRESETS[biases.DEFAULT].name,
    }


def report_text(account: str) -> str:
    proc = subprocess.run([sys.executable, "-m", "dashboard.report", "--account", account], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, timeout=120)
    return ANSI.sub("", proc.stdout + proc.stderr).strip()


# ── Runs ──────────────────────────────────────────────────────────────

class Runner:
    """The daily.py run the page follows: one started here, or one running without the panel (schedule, terminal)."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.info: dict | None = None  # the run self.proc is
        self.seen: dict | None = None  # the latest run noticed running outside the panel
        self.lock = threading.Lock()

    @staticmethod
    def _saved() -> dict | None:
        try:
            with open(RUN_INFO) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _save(info: dict) -> None:
        try:
            with open(RUN_INFO, "w") as fh:
                json.dump(info, fh)
        except OSError:
            pass

    def current(self) -> tuple[dict, str | None]:
        """(what the page shows about the run in progress or the latest one, the file its output goes to)."""
        mine = self.info or self._saved()
        if self.proc is not None and self.proc.poll() is None:
            return {**self.info, "running": True, "source": "panel"}, RUN_OUTPUT
        other = daily.current_run()
        if other is not None:
            if mine and mine.get("pid") == other["pid"]:  # started here before the panel was reopened
                return {**mine, "running": True, "source": "panel"}, RUN_OUTPUT
            self.seen = {"pid": other["pid"], "label": f"Run for @{other['account']} started outside the panel",
                         "started": other.get("started") or 0, "key": f"{other['pid']}-{other.get('started')}",
                         "log": other["log"]}
            return {**self._public(self.seen), "running": True, "source": "elsewhere"}, other["log"]
        finished = []
        if mine and mine.get("key"):
            if self.proc is not None and self.proc.pid == mine.get("pid") and "exit" not in mine:
                mine = self.info = {**mine, "exit": self.proc.returncode}  # remembered for a reopened panel
                self._save(mine)
            finished.append(({**mine, "running": False, "source": "panel", "exit": mine.get("exit")}, RUN_OUTPUT))
        if self.seen:
            finished.append(({**self._public(self.seen), "running": False, "source": "elsewhere", "exit": None},
                             self.seen["log"]))
        if not finished:
            return {"key": None, "running": False}, None
        return max(finished, key=lambda item: item[0].get("started") or 0)

    @staticmethod
    def _public(info: dict) -> dict:
        return {k: v for k, v in info.items() if k != "log"}

    def start(self, account: str, engagement: str, minutes: float, start_llm: bool, digest_only: bool,
              headless: bool = False) -> dict:
        command = [sys.executable, os.path.join(PROJECT_ROOT, "daily.py"), "--account", account]
        if digest_only:
            command.append("--digest-only")
            label = f"Digest for @{account}"
        else:
            command += ["--engagement", engagement, "--minutes", f"{minutes:g}"]
            command += ["--headless"] * headless + ["--start-llm"] * start_llm
            label = f"Session for @{account} ({engagement}, {minutes:g} min{', headless' * headless}), then the digest"
        return self._launch(command, label)

    def start_seed(self, account: str, handles: list[str], check_only: bool) -> dict:
        """Follow (or only check) the bias's accounts-to-follow-first with tools/follow_seed.py."""
        command = [sys.executable, "-m", "tools.follow_seed", "--account", account] + ["--check"] * check_only
        verb = "Checking" if check_only else "Following"
        return self._launch(command + handles, f"{verb} {len(handles)} accounts for @{account}")

    def _launch(self, command: list[str], label: str) -> dict:
        with self.lock:
            if self.current()[0]["running"]:
                raise ValueError("A run is already in progress.")
            os.makedirs(daily.LOG_DIR, exist_ok=True)
            with open(RUN_OUTPUT, "wb") as out:
                self.proc = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    start_new_session=True,  # its own process group: closing the panel's terminal doesn't stop it
                )
            started = time.time()
            self.info = {"pid": self.proc.pid, "label": label, "started": started, "key": f"{self.proc.pid}-{started}"}
            self._save(self.info)
            return self.info

    def stop(self) -> None:
        with self.lock:
            info, _ = self.current()
            if not info["running"]:
                raise ValueError("Nothing is running.")
            pid = info["pid"]
            os.kill(pid, signal.SIGINT)  # like Ctrl+C: the session ends cleanly and closes its browser
        threading.Thread(target=self._escalate, args=(pid,), daemon=True).start()

    def _alive(self, pid: int) -> bool:
        if self.proc is not None and self.proc.pid == pid:
            return self.proc.poll() is None
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _escalate(self, pid: int) -> None:
        """If Ctrl+C isn't enough, terminate — and at last kill — the run and the browser it started."""
        for sig, grace in ((signal.SIGTERM, 45), (signal.SIGKILL, 15)):
            deadline = time.time() + grace
            while time.time() < deadline:
                if not self._alive(pid):
                    return
                time.sleep(0.5)
            try:
                if os.getpgid(pid) == pid:  # it leads its own group (runs started here do)
                    os.killpg(pid, sig)
                else:
                    os.kill(pid, sig)
            except OSError:
                return


# ── HTTP ──────────────────────────────────────────────────────────────

class Panel(BaseHTTPRequestHandler):
    runner = Runner()

    def log_message(self, format, *args):  # noqa: A002 — keep the terminal quiet
        pass

    def _send(self, status: int, body, content_type: str, headers: dict | None = None) -> None:
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        for name, value in {"Content-Type": content_type, "Content-Length": str(len(data)),
                            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                            "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer", **(headers or {})}.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj), "application/json; charset=utf-8")

    def _allowed(self, api: bool) -> bool:
        if self.headers.get("Host") not in ALLOWED_HOSTS:
            self._send(403, "Forbidden", "text/plain; charset=utf-8")
            return False
        if api and not secrets.compare_digest(self.headers.get("X-PRAG-Token", ""), TOKEN):
            self._json({"error": "The panel was restarted: reload this page."}, 403)
            return False
        return True

    def _handle(self, routes: dict, path: str, arg) -> None:
        handler = routes.get(path)
        if handler is None:
            return self._json({"error": "Not found"}, 404)
        try:
            self._json(handler(arg))
        except (ValueError, TypeError, RuntimeError, argparse.ArgumentTypeError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001 — show it on the page rather than drop the request
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if not self._allowed(api=url.path.startswith("/api/")):
            return
        if url.path == "/":
            return self._send(200, page(), "text/html; charset=utf-8", {"Content-Security-Policy": PAGE_CSP})
        if url.path.startswith("/digest/"):
            return self._digest_file(url.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        self._handle({"/api/state": self.state, "/api/output": self.output, "/api/digest": self.digest,
                      "/api/report": self.report, "/api/schedules": self.schedules, "/api/bias": self.bias},
                     url.path, query)

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if not self._allowed(api=True):
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._json({"error": "Expected a JSON object"}, 400)
        self._handle({"/api/run": self.run, "/api/stop": self.stop, "/api/open": self.open_folder,
                      "/api/prompt": self.answer_prompt, "/api/bias": self.save_bias, "/api/seed": self.seed,
                      "/api/schedule/install": self.install_schedule,
                      "/api/schedule/remove": self.remove_schedule}, url.path, body)

    def _digest_file(self, path: str) -> None:
        parts = path.split("/")  # "", "digest", account, file
        if len(parts) == 4 and parts[2] in active_accounts() and DIGEST_FILE.fullmatch(parts[3]):
            file = os.path.join(DIGEST_DIR, parts[2], parts[3])
            if os.path.isfile(file):
                with open(file, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8", {"Content-Security-Policy": DIGEST_CSP})
        self._send(404, "No such digest", "text/plain; charset=utf-8")

    # GET

    def state(self, query: dict) -> dict:
        accounts = active_accounts()
        account = query.get("account") if query.get("account") in accounts else (accounts[0] if accounts else None)
        info, _ = self.runner.current()
        return {"accounts": accounts, "account": account, "run": info, "prompts": prompt.pending(),
                "status": account_status(account) if account else None}

    def output(self, query: dict) -> dict:
        info, path = self.runner.current()
        if path is None:
            return {"key": None, "text": "", "offset": 0, "reset": True}
        same = query.get("key") == info["key"]
        text, offset, restarted = read_output(path, int(query.get("offset") or 0) if same else 0)
        return {"key": info["key"], "text": text, "offset": offset, "reset": restarted or not same}

    def digest(self, query: dict) -> dict:
        return digest_data(checked_account(query.get("account")), query.get("date"))

    def report(self, query: dict) -> dict:
        return {"text": report_text(checked_account(query.get("account")))}

    def schedules(self, _query: dict) -> dict:
        return {"items": schedule.installed()}

    def bias(self, query: dict) -> dict:
        return bias_info(checked_account(query.get("account")))

    # POST

    def run(self, body: dict) -> dict:
        account = checked_account(body.get("account"))
        engagement = body.get("engagement", DAILY_ENGAGEMENT)
        if engagement not in ENGAGEMENT_MODES:
            raise ValueError(f"Engagement must be one of: {', '.join(ENGAGEMENT_MODES)}")
        minutes = float(body.get("minutes") or SESSION_MINUTES)
        if not 1 <= minutes <= DAILY_MAX_MINUTES:
            raise ValueError(f"Minutes must be between 1 and {DAILY_MAX_MINUTES}")
        return self.runner.start(account, engagement, minutes, bool(body.get("start_llm")),
                                 bool(body.get("digest_only")), headless=bool(body.get("headless")))

    def stop(self, _body: dict) -> dict:
        self.runner.stop()
        return {"ok": True}

    def save_bias(self, body: dict) -> dict:
        account = checked_account(body.get("account"))
        biases.save(account, biases.validated(body))
        return bias_info(account)

    def seed(self, body: dict) -> dict:
        account = checked_account(body.get("account"))
        handles = biases.load(account).seeds
        if not handles:
            raise ValueError("Add accounts to follow first, and save the bias.")
        return self.runner.start_seed(account, handles, bool(body.get("check_only")))

    def answer_prompt(self, body: dict) -> dict:
        text = None if body.get("cancel") else str(body.get("answer") or "").strip()[:200]
        if text == "":
            raise ValueError("Type the answer first.")
        if not prompt.answer(str(body.get("id", "")), text):
            raise ValueError("That question isn't waiting any more.")
        return {"ok": True}

    def open_folder(self, body: dict) -> dict:
        folders = {"logs": daily.LOG_DIR, "digests": DIGEST_DIR}
        folder = folders.get(body.get("what"))
        if folder is None:
            raise ValueError("Unknown folder")
        os.makedirs(folder, exist_ok=True)
        subprocess.run(["open", folder], check=False)
        return {"ok": True}

    def install_schedule(self, body: dict) -> dict:
        account = checked_account(body.get("account"))
        engagement = body.get("engagement", DAILY_ENGAGEMENT)
        if engagement not in ENGAGEMENT_MODES:
            raise ValueError(f"Engagement must be one of: {', '.join(ENGAGEMENT_MODES)}")
        times = [schedule.parse_time(t) for t in re.split(r"[,\s]+", str(body.get("times", "")).strip()) if t]
        if not times:
            raise ValueError("Add at least one time, like 09:00")
        minutes = float(body.get("minutes") or SESSION_MINUTES)
        if not 1 <= minutes <= DAILY_MAX_MINUTES:
            raise ValueError(f"Minutes must be between 1 and {DAILY_MAX_MINUTES}")
        plist = schedule.build_plist(account, times, minutes=minutes, engagement=engagement,
                                     start_llm=bool(body.get("start_llm")), headless=bool(body.get("headless")))
        schedule.install(account, plist)
        return {"description": schedule.describe(plist)}

    def remove_schedule(self, body: dict) -> dict:
        account = str(body.get("account", ""))
        if not USERNAME.fullmatch(account):
            raise ValueError("Unknown account")
        return {"removed": schedule.remove(account)}


# ── The page ──────────────────────────────────────────────────────────

def page() -> str:
    script_json = lambda value: json.dumps(value).replace("<", "\\u003c")
    defaults = {"engagement": DAILY_ENGAGEMENT, "minutes": SESSION_MINUTES, "cap": DAILY_MAX_MINUTES,
                "min_battery": MIN_BATTERY}
    return (PAGE.replace("__TOKEN__", script_json(TOKEN)).replace("__MODES__", script_json(ENGAGEMENT_MODES))
            .replace("__DEFAULTS__", script_json(defaults)))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PRAG</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%232f6fde'/%3E%3Cpath d='M10 23V9h12M10 16h9' stroke='white' stroke-width='3.2' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<style>
:root {
  --bg: #f5f4f0; --panel: #ffffff; --text: #1c1c1e; --muted: #6c6c72; --line: #e4e2db;
  --accent: #2f6fde; --button: #2f6fde; --ok: #0b7a53; --ok-soft: #e3f4ec; --warn: #9a5800; --warn-soft: #fcf1e1;
  --bad: #c0392b; --console-bg: #15161a; --console-text: #e6e6ea;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111113; --panel: #1b1b1f; --text: #ececf0; --muted: #9d9da6; --line: #2d2d33;
    --accent: #6d9cf2; --ok: #4cc79a; --ok-soft: #173229; --warn: #e6a955; --warn-soft: #362a17;
    --bad: #f07466; --console-bg: #0b0b0d;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans Devanagari", sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 20px 16px 40px; }
header { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; }
.brand { font-weight: 700; font-size: 1.3rem; letter-spacing: -0.01em; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-left: auto; }
.chip { font-size: .8rem; padding: 3px 10px; border-radius: 999px; background: var(--panel);
        border: 1px solid var(--line); color: var(--muted); white-space: nowrap; }
.chip.ok { color: var(--ok); background: var(--ok-soft); border-color: transparent; }
.chip.warn { color: var(--warn); background: var(--warn-soft); border-color: transparent; }
select, input[type=number], input[type=text] { font: inherit; color: var(--text); background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; max-width: 100%; }
input[type=number] { width: 90px; }
nav { display: flex; gap: 2px; margin: 18px 0 14px; border-bottom: 1px solid var(--line); overflow-x: auto; }
nav button { font: inherit; background: none; border: 0; border-bottom: 2px solid transparent; padding: 8px 14px;
             color: var(--muted); cursor: pointer; margin-bottom: -1px; }
nav button.active { color: var(--text); border-bottom-color: var(--accent); font-weight: 600; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; }
.card h3 { margin: 0 0 8px; font-size: 1rem; }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin: 10px 0; }
.label { width: 110px; color: var(--muted); font-size: .9rem; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 9px; overflow: hidden; }
.seg label { padding: 6px 16px; cursor: pointer; border-right: 1px solid var(--line); font-size: .92rem; user-select: none; }
.seg label:last-child { border-right: 0; }
.seg label.on { background: var(--button); color: #fff; }
.seg input { position: absolute; opacity: 0; pointer-events: none; }
.check { display: inline-flex; gap: 8px; align-items: center; cursor: pointer; }
.hint { color: var(--muted); font-size: .88rem; }
.notice { padding: 8px 12px; border-radius: 8px; background: var(--warn-soft); color: var(--warn); font-size: .9rem; margin: 10px 0 0; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.btn { font: inherit; font-weight: 600; padding: 8px 16px; border-radius: 9px; border: 1px solid var(--line);
       background: var(--panel); color: var(--text); cursor: pointer; }
.btn.primary { background: var(--button); border-color: var(--button); color: #fff; }
.btn.danger { color: var(--bad); }
.btn:disabled { opacity: .45; cursor: default; }
.linkish { font: inherit; background: none; border: 0; color: var(--accent); cursor: pointer; padding: 0; font-size: .88rem; }
.status { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--line); flex: none; }
.dot.live { background: var(--ok); box-shadow: 0 0 0 4px var(--ok-soft); }
.spacer { flex: 1; }
pre { margin: 10px 0 0; border-radius: 10px; padding: 12px 14px; overflow: auto; white-space: pre-wrap; word-break: break-word;
      font: 12.5px/1.45 ui-monospace, "SF Mono", Menlo, monospace; }
pre.console { background: var(--console-bg); color: var(--console-text); height: 440px; }
pre.console:empty::before { content: "A run's output appears here."; color: #85858d; }
pre.report { background: var(--bg); border: 1px solid var(--line); }
.story { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; margin: 10px 0; }
.story-head { display: flex; gap: 10px; align-items: baseline; }
.story h3 { margin: 0; font-size: 1.02rem; }
.story p { margin: 6px 0 0; }
.score { font: 600 .78rem ui-monospace, Menlo, monospace; color: var(--muted); }
.meta { margin-top: 8px; font-size: .86rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; }
.badge { font-size: .75rem; font-weight: 600; padding: 1px 8px; border-radius: 999px; border: 1px solid currentColor; }
.badge.ok { color: var(--ok); } .badge.warn { color: var(--warn); }
ul.news { margin: 6px 0 0; padding-left: 18px; font-size: .88rem; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
textarea { font: inherit; color: var(--text); background: var(--panel); border: 1px solid var(--line);
           border-radius: 8px; padding: 8px 10px; width: 100%; resize: vertical; }
textarea.words { font-size: .9rem; }
.field { margin: 16px 0 0; }
.field > label { font-weight: 600; display: block; }
.field > .hint { margin: 2px 0 6px; }
.field input[type=text] { width: 100%; }
.presets { display: flex; flex-wrap: wrap; gap: 6px; }
.presets button { font: inherit; font-size: .9rem; padding: 5px 12px; border-radius: 999px; border: 1px solid var(--line);
                  background: var(--panel); color: var(--text); cursor: pointer; }
.presets button.on { background: var(--button); border-color: var(--button); color: #fff; }
.dirty { color: var(--warn); font-weight: 600; }
.sched { display: flex; gap: 12px; align-items: center; justify-content: space-between; padding: 10px 0; border-top: 1px solid var(--line); }
.sched:first-child { border-top: 0; }
footer { color: var(--muted); font-size: .8rem; margin-top: 24px; }
.overlay { position: fixed; inset: 0; background: rgba(0, 0, 0, .45); display: flex; align-items: center;
           justify-content: center; padding: 16px; z-index: 10; }
.overlay[hidden] { display: none; }
.modal { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 14px;
         padding: 20px 22px; width: min(460px, 100%); box-shadow: 0 20px 60px rgba(0, 0, 0, .35); }
.modal h3 { margin: 0 0 6px; font-size: 1.1rem; }
.modal p { margin: 0 0 10px; }
.modal input { width: 100%; font-size: 1.25rem; letter-spacing: .06em; padding: 10px 12px; margin: 4px 0 2px; }
.modal .actions { align-items: center; }
#toast { position: fixed; left: 50%; bottom: 20px; transform: translateX(-50%); background: var(--text); color: var(--bg);
         padding: 10px 16px; border-radius: 10px; font-size: .9rem; max-width: calc(100vw - 32px); box-shadow: 0 6px 24px rgba(0,0,0,.25); }
#toast.bad { background: var(--bad); color: #fff; }
@media (max-width: 600px) { .label { width: 100%; } .chips { margin-left: 0; } pre.console { height: 360px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">PRAG</div>
    <select id="account" aria-label="Account"></select>
    <div class="chips">
      <span class="chip" id="chip-llm" hidden></span>
      <span class="chip" id="chip-today" hidden></span>
      <span class="chip" id="chip-live" hidden></span>
      <span class="chip" id="chip-power" hidden></span>
    </div>
  </header>
  <p class="notice" id="no-account" hidden>No ACTIVE account. Add one to accounts.json and run <code>python -m database.seed</code>.</p>

  <nav>
    <button data-tab="run">Run</button>
    <button data-tab="digest">Digest</button>
    <button data-tab="report">Report</button>
    <button data-tab="bias">Bias</button>
    <button data-tab="schedule">Schedule</button>
  </nav>

  <main>
    <section id="tab-run">
      <div class="card">
        <div class="row"><span class="label">Engagement</span><div class="seg" id="engagement"></div></div>
        <div class="row"><span class="label"></span><span class="hint" id="eng-hint"></span></div>
        <div class="row"><span class="label">Minutes</span><input id="minutes" type="number" min="1" step="1"><span class="hint" id="minutes-hint"></span></div>
        <div class="row"><span class="label">Browser</span><label class="check"><input type="checkbox" id="headless"> Headless: no browser window</label></div>
        <div class="row"><span class="label"></span><span class="hint" id="headless-hint"></span></div>
        <div class="row"><span class="label"></span><label class="check"><input type="checkbox" id="start-llm"> Start LM Studio if it's off</label></div>
        <p class="notice" id="skip" hidden></p>
        <div class="actions">
          <button class="btn primary" id="btn-run">Start session</button>
          <button class="btn" id="btn-digest">Rebuild digest only</button>
          <button class="btn danger" id="btn-stop" disabled>Stop</button>
        </div>
      </div>
      <div class="card">
        <div class="status"><span class="dot" id="run-dot"></span><span id="run-status">Loading…</span><span class="spacer"></span>
          <button class="linkish" id="btn-logs">Open logs folder</button></div>
        <pre class="console" id="console"></pre>
      </div>
    </section>

    <section id="tab-digest" hidden>
      <div class="card">
        <div class="row" style="margin:0"><select id="digest-date" aria-label="Digest date"></select>
          <span class="hint" id="digest-stats"></span><span class="spacer"></span>
          <a id="digest-open" target="_blank" rel="noopener" hidden>Open the full page ↗</a></div>
      </div>
      <div id="stories"></div>
    </section>

    <section id="tab-report" hidden>
      <div class="card">
        <div class="row" style="margin:0"><button class="btn" id="btn-report">Refresh</button>
          <span class="hint">Is the feed curated yet? The verdict needs 100 judged reels.</span></div>
        <pre class="report" id="report"></pre>
      </div>
    </section>

    <section id="tab-bias" hidden>
      <div class="card">
        <div class="status"><span id="bias-state">Loading…</span><span class="dirty" id="bias-dirty" hidden>· unsaved changes</span></div>
        <div class="row"><span class="label">Start from</span><div class="presets" id="presets"></div></div>
        <p class="hint">A preset only fills in the form below; nothing changes until you save.</p>

        <div class="field"><label for="bias-name">Name</label>
          <input type="text" id="bias-name" maxlength="40"></div>
        <div class="field"><label for="bias-interests">What to look for</label>
          <p class="hint">The LLM scores every reel against this: say what's high, medium and low value to you.</p>
          <textarea id="bias-interests" rows="14"></textarea></div>
        <div class="field"><label for="bias-watch">Watch words</label>
          <p class="hint">When LM Studio is off, the quick rules watch reels whose caption or hashtags contain one of these. Separate with commas.</p>
          <textarea class="words" id="bias-watch" rows="3"></textarea></div>
        <div class="field"><label for="bias-skip">Skip words</label>
          <p class="hint">Reels whose caption or hashtags contain one of these are skipped at once, without the LLM. Single words.</p>
          <textarea class="words" id="bias-skip" rows="3"></textarea></div>
        <div class="field"><label for="bias-hints">Account-name hints</label>
          <p class="hint">Parts of account names that suggest a relevant page, like "news". The quick rules watch those accounts.</p>
          <input type="text" id="bias-hints"></div>
        <div class="field"><label for="bias-seeds">Accounts to follow first</label>
          <p class="hint">Optional. Following these gives a fresh account a head start. Only verified accounts get followed.</p>
          <textarea class="words" id="bias-seeds" rows="2"></textarea>
          <div class="actions"><button class="btn" id="btn-seed-check">Check them</button>
            <button class="btn" id="btn-seed-follow">Follow them</button></div></div>

        <div class="actions"><button class="btn primary" id="btn-bias-save">Save bias</button></div>
        <p class="hint" id="bias-note"></p>
      </div>
    </section>

    <section id="tab-schedule" hidden>
      <div class="card"><h3>Installed</h3><div id="schedules"></div></div>
      <div class="card">
        <h3>Schedule this account</h3>
        <div class="row"><span class="label">Times</span><input type="text" id="sch-times" placeholder="09:00, 14:00, 20:00"></div>
        <div class="row"><span class="label">Engagement</span><select id="sch-eng"></select></div>
        <div class="row"><span class="label">Minutes</span><input type="number" id="sch-minutes" min="1" step="1"></div>
        <div class="row"><span class="label">Browser</span><label class="check"><input type="checkbox" id="sch-headless"> Headless: no browser window</label></div>
        <div class="row"><span class="label"></span><label class="check"><input type="checkbox" id="sch-llm"> Start LM Studio if it's off</label></div>
        <div class="actions"><button class="btn primary" id="btn-schedule">Install schedule</button></div>
        <p class="hint">Runs every day at these times while you're logged in to the Mac, replacing any earlier schedule for this
          account. If the Mac is asleep, the run starts when it wakes; if it's shut down, that run is skipped. A browser window
          opens while it runs, unless it's headless.</p>
      </div>
    </section>
  </main>
  <footer>Only this Mac can open this panel. Closing it (Ctrl+C in its terminal) doesn't stop a run in progress.</footer>
</div>
<div class="overlay" id="prompt" hidden>
  <form class="modal" id="prompt-form" autocomplete="off">
    <h3 id="prompt-title">Instagram needs you</h3>
    <p id="prompt-message"></p>
    <input id="prompt-input" type="text" spellcheck="false" aria-label="Answer">
    <div class="actions"><button type="submit" class="btn primary">Send</button>
      <button type="button" class="btn" id="prompt-cancel">Cancel</button><span class="spacer"></span>
      <span class="hint" id="prompt-wait"></span></div>
    <p class="hint">A dialog box on the Mac asks the same thing: answer either one.</p>
  </form>
</div>
<div id="toast" hidden></div>

<script>
"use strict";
const TOKEN = __TOKEN__, MODES = __MODES__, DEFAULTS = __DEFAULTS__;
const $ = (id) => document.getElementById(id);
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem("feedlab." + key); return v === null ? fallback : JSON.parse(v); }
    catch { return fallback; }
  },
  set(key, value) { try { localStorage.setItem("feedlab." + key, JSON.stringify(value)); } catch {} },
};
let state = null, tab = "run", wasRunning = false, shownPrompt = null;
const out = {key: null, offset: 0, busy: false};

async function api(path, body) {
  const init = {headers: {"X-PRAG-Token": TOKEN}, cache: "no-store"};
  if (body !== undefined) {
    init.method = "POST";
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  let data = {};
  try { data = await res.json(); } catch {}
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children) if (child !== null && child !== undefined && child !== false) node.append(child);
  return node;
}

const safeUrl = (url) => (/^https?:\/\//i.test(url || "") ? url : null);
const link = (url, text) => (safeUrl(url) ? h("a", {href: url, target: "_blank", rel: "noopener", text}) : h("span", {text}));

function toast(message, bad) {
  const el = $("toast");
  el.textContent = message;
  el.className = bad ? "bad" : "";
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, bad ? 7000 : 4000);
}

function chip(id, spec) {
  const el = $(id);
  el.hidden = !spec;
  if (spec) { el.textContent = spec[0]; el.className = "chip " + (spec[1] || ""); }
}

const account = () => $("account").value || store.get("account", "");
const mode = () => (document.querySelector('input[name="eng"]:checked') || {}).value || DEFAULTS.engagement;
const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

// ── State ──

async function refresh() {
  try {
    state = await api("/api/state?account=" + encodeURIComponent(account()));
    render();
  } catch (e) {
    $("run-status").textContent = e.message;
  }
}

function render() {
  const select = $("account");
  if ([...select.options].map((o) => o.value).join() !== state.accounts.join()) {
    select.replaceChildren(...state.accounts.map((a) => h("option", {value: a, text: "@" + a})));
  }
  if (state.account) select.value = state.account;
  $("no-account").hidden = state.accounts.length > 0;

  const s = state.status;
  chip("chip-llm", s && (s.llm ? ["● LLM ready", "ok"] : ["○ LLM off · rules + model score", "warn"]));
  chip("chip-today", s && [`${s.minutes_used} / ${s.minutes_cap} min today`, s.minutes_used >= s.minutes_cap ? "warn" : ""]);
  chip("chip-live", s && [s.engaged_today ? "Live session done today" : "No live session yet today", ""]);
  chip("chip-power", s && s.battery.charge !== null &&
       [s.battery.on_battery ? `On battery · ${s.battery.charge}%` : "On power",
        s.battery.on_battery && s.battery.charge < DEFAULTS.min_battery ? "warn" : ""]);
  renderForm();
  renderStatus();
  renderPrompt();

  const running = state.run.running;
  if (wasRunning && !running) {
    const code = state.run.exit;
    toast(code === 130 || code < 0 ? "The run stopped." : code > 0 ? "The run ended with a problem: see its output."
          : "The run finished.", code > 0 && code !== 130);
    if (tab === "digest") loadDigest();
  }
  wasRunning = running;
}

function effectiveMode() {
  const m = mode(), s = state && state.status;
  if (m === "mixed") return s && s.engaged_today ? "dry run" : "live";
  return m === "dry" ? "dry run" : "live";
}

function renderForm() {
  document.querySelectorAll("#engagement label").forEach((l) => l.classList.toggle("on", l.querySelector("input").checked));
  const s = state && state.status, m = mode();
  $("eng-hint").textContent = MODES[m] + (m === "mixed" && s ? ` · next run: ${s.mixed_next}` : "");
  if (!state) return;
  if (s) {
    const left = Math.max(0, s.minutes_cap - s.minutes_used);
    $("minutes-hint").textContent = `${left} of ${s.minutes_cap} min left today`;
    $("skip").hidden = !s.skip;
    $("skip").textContent = s.skip ? `Won't scroll now: ${s.skip}. Starting only rebuilds the digest.` : "";
  }
  const running = state.run.running, none = !state.account;
  $("btn-run").textContent = `Start session · ${effectiveMode()}` + ($("headless").checked ? " · headless" : "");
  $("btn-run").disabled = running || none;
  $("btn-digest").disabled = running || none;
  $("btn-stop").disabled = !running;
}

function renderStatus() {
  const r = state.run;
  let text = "No run yet. Pick a mode and start a session.";
  if (r.key && r.running) {
    text = `Running · ${r.label}` + (r.started ? ` · ${clock(Date.now() / 1000 - r.started)}` : "");
  } else if (r.key) {
    const ended = r.exit === null || r.exit === undefined || r.exit === 0 ? "finished"
      : r.exit === 130 || r.exit < 0 ? "stopped" : `ended with a problem (exit ${r.exit}), see the output`;
    const at = r.started ? " · started " + new Date(r.started * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"}) : "";
    text = `${r.label} · ${ended}${at}`;
  }
  $("run-status").textContent = text;
  $("run-dot").className = "dot" + (r.running ? " live" : "");
}

// ── Questions from a run (a login code, a password) ──

function renderPrompt() {
  const current = (state.prompts || [])[0], box = $("prompt");
  document.title = current ? "🔐 PRAG needs you" : "PRAG";
  if (!current) {
    box.hidden = true;
    return;
  }
  $("prompt-wait").textContent = `Waits ${clock(Math.max(0, current.expires - Date.now() / 1000))}`;
  if (shownPrompt === current.id) return;  // already shown (or just answered)
  shownPrompt = current.id;
  const code = current.kind === "code", input = $("prompt-input");
  $("prompt-title").textContent = `${code ? "Security code" : "Password"} for @${current.account}`;
  $("prompt-message").textContent = current.message;
  input.value = "";
  input.type = current.secret ? "password" : "text";
  input.inputMode = code ? "numeric" : "text";
  input.autocomplete = code ? "one-time-code" : "off";
  box.hidden = false;
  input.focus();
}

async function answerPrompt(cancel) {
  const input = $("prompt-input"), value = input.value.trim();
  if (!cancel && !value) { input.focus(); return; }
  $("prompt").hidden = true;
  input.value = "";
  try {
    await api("/api/prompt", cancel ? {id: shownPrompt, cancel: true} : {id: shownPrompt, answer: value});
    toast(cancel ? "Cancelled." : "Sent. PRAG is entering it.");
  } catch (e) { toast(e.message, true); }
  refresh();
}

// ── Output ──

async function pollOutput() {
  if (out.busy || !state || !state.run.key) return;
  out.busy = true;
  try {
    const d = await api(`/api/output?key=${encodeURIComponent(out.key || "")}&offset=${out.offset}`);
    const box = $("console");
    if (d.reset) box.textContent = "";
    out.key = d.key;
    out.offset = d.offset;
    if (d.text) {
      const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
      box.append(d.text);
      if (box.textContent.length > 400000) box.textContent = box.textContent.slice(-300000);
      if (atBottom) box.scrollTop = box.scrollHeight;
    }
  } catch {} finally { out.busy = false; }
}

async function start(digestOnly) {
  const body = {account: account(), engagement: mode(), minutes: Number($("minutes").value) || DEFAULTS.minutes,
                start_llm: $("start-llm").checked, digest_only: digestOnly, headless: $("headless").checked};
  try {
    await api("/api/run", body);
    toast(digestOnly ? "Rebuilding the digest…" : body.headless ? "Starting, headless: follow it below."
          : "Starting. A browser window opens for the session.");
    await refresh();
    pollOutput();
  } catch (e) { toast(e.message, true); }
}

async function stop() {
  if (!confirm("Stop the run? The session ends and its browser closes.")) return;
  try { await api("/api/stop", {}); toast("Stopping…"); } catch (e) { toast(e.message, true); }
}

// ── Digest, report, schedule ──

function storyCard(s) {
  const v = s.verification || {};
  const badge = {widely_reported: "ok", reported: "ok", reels_only: "warn"}[v.status] || "";
  const meta = h("div", {class: "meta"}, h("span", {class: "badge " + badge, text: v.label || "Not checked"}));
  const byCreator = new Map();
  for (const r of s.reels) byCreator.set(r.creator, [...(byCreator.get(r.creator) || []), r.url]);
  for (const [creator, urls] of byCreator) {
    meta.append(link(urls[0], "@" + creator + (urls.length > 1 ? ` ×${urls.length}` : "")));
  }
  const news = (v.headlines || []).slice(0, 2).map((a) => h("li", {}, link(a.url, a.title), ` — ${a.source}`));
  return h("article", {class: "story"},
    h("div", {class: "story-head"}, h("span", {class: "score", text: Number(s.score).toFixed(2)}), h("h3", {text: s.title})),
    s.summary && s.summary !== s.title ? h("p", {text: s.summary}) : null,
    meta,
    news.length ? h("ul", {class: "news"}, ...news) : null);
}

async function loadDigest(date) {
  try {
    const d = await api(`/api/digest?account=${encodeURIComponent(account())}` + (date ? `&date=${encodeURIComponent(date)}` : ""));
    $("digest-date").replaceChildren(...d.dates.map((x) => h("option", {value: x, text: x})));
    $("digest-date").hidden = !d.dates.length;
    if (d.date) $("digest-date").value = d.date;
    if (!d.digest) {
      $("digest-stats").textContent = "No digest yet: run a session or rebuild the digest.";
      $("digest-open").hidden = true;
      $("stories").replaceChildren();
      return;
    }
    const st = d.digest.stats;
    $("digest-stats").textContent = `${st.stories} stories from ${st.informative_reels} informative reels · ` +
      `${st.reels_seen} scrolled in ${Math.round(st.agent_minutes)} min`;
    $("digest-open").href = d.page;
    $("digest-open").hidden = false;
    $("stories").replaceChildren(...d.digest.stories.map(storyCard));
  } catch (e) { toast(e.message, true); }
}

async function loadReport() {
  $("report").textContent = "Loading…";
  try { $("report").textContent = (await api("/api/report?account=" + encodeURIComponent(account()))).text; }
  catch (e) { $("report").textContent = e.message; }
}

async function loadSchedules() {
  try {
    const items = (await api("/api/schedules")).items;
    $("schedules").replaceChildren(...(items.length ? items.map((item) => {
      const remove = h("button", {class: "btn danger", text: "Remove"});
      remove.onclick = () => removeSchedule(item.account);
      const note = item.loaded ? "Active" : "Not loaded: install it again";
      return h("div", {class: "sched"},
        h("div", {}, h("div", {text: item.description}),
          h("div", {class: "hint", text: note + (item.last_exit ? ` · last run's exit code ${item.last_exit}` : "")})),
        remove);
    }) : [h("p", {class: "hint", text: "Nothing scheduled: runs happen only when you start them."})]));
  } catch (e) { toast(e.message, true); }
}

async function installSchedule() {
  const body = {account: account(), times: $("sch-times").value, engagement: $("sch-eng").value,
                minutes: Number($("sch-minutes").value) || DEFAULTS.minutes, start_llm: $("sch-llm").checked,
                headless: $("sch-headless").checked};
  if (!body.times.trim()) { toast("Add at least one time, like 09:00.", true); return; }
  if (!confirm(`Scroll @${body.account} every day at ${body.times} (${body.engagement}, ${body.minutes} min each)?`)) return;
  try { toast("Scheduled: " + (await api("/api/schedule/install", body)).description); loadSchedules(); }
  catch (e) { toast(e.message, true); }
}

async function removeSchedule(name) {
  if (!confirm(`Remove the daily schedule for @${name}?`)) return;
  try { await api("/api/schedule/remove", {account: name}); toast("Schedule removed."); loadSchedules(); }
  catch (e) { toast(e.message, true); }
}

// ── Bias ──

let biasData = null;
const BIAS_FIELDS = {name: "bias-name", interests: "bias-interests", watch_words: "bias-watch",
                     skip_words: "bias-skip", creator_hints: "bias-hints", seeds: "bias-seeds"};
const LIST_FIELDS = ["watch_words", "skip_words", "creator_hints", "seeds"];

function biasForm() {
  return Object.fromEntries(Object.entries(BIAS_FIELDS).map(([key, id]) => [key, $(id).value]));
}

function canonical(bias) {  // how the server will store it, to spot unsaved changes
  const list = (v) => (Array.isArray(v) ? v : String(v).split(/[,\n]/))
    .map((w) => w.trim().replace(/^[#@]+/, "")).filter(Boolean)
    .map((w) => (/^[\x00-\x7f]*$/.test(w) ? w.toLowerCase() : w));
  return JSON.stringify([String(bias.name).trim(), String(bias.interests).trim(), ...LIST_FIELDS.map((k) => list(bias[k]))]);
}

function fillBias(bias) {
  $("bias-name").value = bias.name;
  $("bias-interests").value = bias.interests.trim();
  for (const key of LIST_FIELDS) $(BIAS_FIELDS[key]).value = bias[key].join(", ");
  biasChanged();
}

function biasChanged() {
  if (!biasData) return;
  const form = biasForm(), interests = form.interests.trim();
  document.querySelectorAll("#presets button").forEach((b) =>
    b.classList.toggle("on", biasData.presets[b.dataset.key].interests.trim() === interests));
  $("bias-dirty").hidden = canonical(form) === canonical(biasData.bias);
  const fits = biasData.model_interests !== null && biasData.model_interests.trim() === interests;
  $("bias-note").textContent = "Changing \"what to look for\" starts a fresh comparison in the report; earlier reels " +
    "stay but aren't compared with the new ones." + (fits ? "" : " When LM Studio is off, reels are only recorded " +
    "for review with this bias: the fallback model " + (biasData.model_name ? `was trained for ${biasData.model_name}.`
    : "hasn't been trained yet."));
}

async function loadBias() {
  try {
    biasData = await api("/api/bias?account=" + encodeURIComponent(account()));
  } catch (e) { toast(e.message, true); return; }
  $("presets").replaceChildren(...Object.entries(biasData.presets).map(([key, preset]) => {
    const b = h("button", {type: "button", text: preset.name, "data-key": key});
    b.onclick = () => fillBias(preset);
    return b;
  }));
  $("bias-state").textContent = biasData.own ? `@${account()} has its own bias: ${biasData.bias.name}`
    : `@${account()} uses the default bias: ${biasData.default_name}`;
  fillBias(biasData.bias);
}

async function saveBias() {
  try {
    biasData = {...biasData, ...(await api("/api/bias", {account: account(), ...biasForm()}))};
    await loadBias();
    toast("Saved. The next session uses this bias.");
  } catch (e) { toast(e.message, true); }
}

async function followSeeds(checkOnly) {
  if (!biasData) return;
  if (!$("bias-dirty").hidden) { toast("Save the bias first.", true); return; }
  const n = biasData.bias.seeds.length;
  if (!n) { toast("Add accounts to follow first, and save.", true); return; }
  if (!checkOnly && !confirm(`Follow ${n} accounts from @${account()}? Only verified ones get followed, at a person's ` +
                             "pace, in a browser window.")) return;
  try {
    await api("/api/seed", {account: account(), check_only: checkOnly});
    toast(checkOnly ? "Checking the accounts: see the output." : "Following: see the output.");
    showTab("run");
    await refresh();
    pollOutput();
  } catch (e) { toast(e.message, true); }
}

function showTab(name) {
  tab = name;
  store.set("tab", name);
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll("main > section").forEach((s) => { s.hidden = s.id !== "tab-" + name; });
  if (name === "digest") loadDigest();
  if (name === "report") loadReport();
  if (name === "schedule") loadSchedules();
  if (name === "bias") loadBias();
}

// ── Start ──

function init() {
  const seg = $("engagement");
  for (const [value, label] of [["mixed", "Mixed"], ["live", "Live"], ["dry", "Dry run"]]) {
    seg.append(h("label", {title: MODES[value]}, h("input", {type: "radio", name: "eng", value}), label));
  }
  const saved = store.get("engagement", DEFAULTS.engagement);
  (seg.querySelector(`input[value="${MODES[saved] ? saved : DEFAULTS.engagement}"]`)).checked = true;
  seg.addEventListener("change", () => { store.set("engagement", mode()); renderForm(); });

  $("minutes").max = DEFAULTS.cap;
  $("minutes").value = store.get("minutes", DEFAULTS.minutes);
  $("minutes").addEventListener("change", () => store.set("minutes", Number($("minutes").value)));
  $("start-llm").checked = store.get("start_llm", false);
  $("start-llm").addEventListener("change", () => store.set("start_llm", $("start-llm").checked));
  const headlessHint = () => {
    $("headless-hint").textContent = $("headless").checked
      ? "Instagram can tell the browser is headless. If it asks for a login code, run once with the window."
      : "A browser window opens while the session runs.";
  };
  $("headless").checked = store.get("headless", false);
  $("headless").addEventListener("change", () => { store.set("headless", $("headless").checked); headlessHint(); renderForm(); });
  headlessHint();
  $("account").addEventListener("change", () => { store.set("account", $("account").value); refresh().then(() => showTab(tab)); });

  $("btn-run").onclick = () => start(false);
  $("btn-digest").onclick = () => start(true);
  $("btn-stop").onclick = stop;
  $("btn-logs").onclick = () => api("/api/open", {what: "logs"}).catch((e) => toast(e.message, true));
  $("btn-report").onclick = loadReport;
  $("digest-date").onchange = () => loadDigest($("digest-date").value);
  $("sch-eng").replaceChildren(...Object.entries(MODES).map(([m, what]) => h("option", {value: m, text: `${m}: ${what}`})));
  $("sch-eng").value = DEFAULTS.engagement;
  $("sch-minutes").max = DEFAULTS.cap;
  $("sch-minutes").value = DEFAULTS.minutes;
  $("btn-schedule").onclick = installSchedule;
  $("prompt-form").addEventListener("submit", (e) => { e.preventDefault(); answerPrompt(false); });
  $("prompt-cancel").onclick = () => answerPrompt(true);
  for (const id of Object.values(BIAS_FIELDS)) $(id).addEventListener("input", biasChanged);
  $("btn-bias-save").onclick = saveBias;
  $("btn-seed-check").onclick = () => followSeeds(true);
  $("btn-seed-follow").onclick = () => followSeeds(false);
  document.querySelectorAll("nav button").forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });

  renderForm();
  refresh().then(() => { showTab(store.get("tab", "run")); pollOutput(); });
  setInterval(refresh, 2000);
  setInterval(pollOutput, 1000);
}
document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="PRAG control panel (opens in your browser)")
    parser.add_argument("--port", type=int, default=8765, help="Port on 127.0.0.1 (another one if it's taken)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser")
    args = parser.parse_args()

    init_db()
    server = None
    for port in (args.port, 0):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Panel)
            break
        except OSError:
            continue
    port = server.server_address[1]
    ALLOWED_HOSTS.update({f"127.0.0.1:{port}", f"localhost:{port}"})
    url = f"http://127.0.0.1:{port}/"
    print(f"🎛  PRAG panel: {url}\n    Ctrl+C closes the panel; a run in progress keeps going.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋  Panel closed.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
