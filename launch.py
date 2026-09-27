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
from dashboard import feed as feed_study
from database import accounts as account_store
from config import DAILY_ENGAGEMENT, DAILY_MAX_MINUTES, ENGAGEMENT_MODES, MIN_BATTERY, SESSION_MINUTES
from database.db import engaged_today, get_connection, init_db, list_accounts, minutes_today
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
FONT_FILE = re.compile(r"[a-z0-9-]+\.woff2")
FONTS_DIR = os.path.join(PROJECT_ROOT, "assets", "fonts")  # Orbitron, Rajdhani, JetBrains Mono (SIL OFL), served locally
USERNAME = re.compile(r"[A-Za-z0-9._]{1,30}")
PAGE_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
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
    bias = biases.load(account)
    return {
        "llm": cached("llm", 10, llm_serving),
        "minutes_used": round(used),
        "minutes_cap": DAILY_MAX_MINUTES,
        "engaged_today": engaged,
        "mixed_next": "dry run (today's live session already ran)" if engaged else "live (first session today)",
        "battery": {"on_battery": on_battery, "charge": charge},
        "skip": daily.skip_reason(account, used, on_battery, charge),
        "bias": {"name": bias.name, "persona": bias.persona},
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
        "model_persona": model_bias.persona if model_bias else False,
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
        return self._launch(command, label, minutes=None if digest_only else minutes)

    def start_seed(self, account: str, handles: list[str], check_only: bool) -> dict:
        """Follow (or only check) the bias's accounts-to-follow-first with tools/follow_seed.py."""
        command = [sys.executable, "-m", "tools.follow_seed", "--account", account] + ["--check"] * check_only
        verb = "Checking" if check_only else "Following"
        return self._launch(command + handles, f"{verb} {len(handles)} accounts for @{account}")

    def _launch(self, command: list[str], label: str, minutes: float | None = None) -> dict:
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
            if minutes:
                self.info["minutes"] = minutes  # how long the session scrolls, for the page's progress line
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
        if url.path.startswith("/fonts/"):
            return self._font_file(url.path[len("/fonts/"):])
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path == "/api/feed/export":
            return self._feed_export(query)
        self._handle({"/api/state": self.state, "/api/output": self.output, "/api/digest": self.digest,
                      "/api/report": self.report, "/api/schedules": self.schedules, "/api/bias": self.bias,
                      "/api/feed": self.feed, "/api/accounts": self.account_list},
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
                      "/api/feed/categorize": self.categorize, "/api/accounts/add": self.add_account,
                      "/api/accounts/password": self.account_password, "/api/accounts/delete": self.delete_account,
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

    def _font_file(self, name: str) -> None:
        file = os.path.join(FONTS_DIR, name)
        if FONT_FILE.fullmatch(name) and os.path.isfile(file):
            with open(file, "rb") as fh:
                return self._send(200, fh.read(), "font/woff2", {"Cache-Control": "max-age=604800"})
        self._send(404, "No such font", "text/plain; charset=utf-8")

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

    def feed(self, query: dict) -> dict:
        account = checked_account(query.get("account"))
        conn = get_connection()
        try:
            return feed_study.summary(conn, account)
        finally:
            conn.close()

    def _feed_export(self, query: dict) -> None:
        """The account's dataset as a download: the curated feed or everything, as JSONL or CSV."""
        try:
            account = checked_account(query.get("account"))
            fmt = query.get("format", "jsonl")
            if fmt not in ("jsonl", "csv"):
                raise ValueError("format must be jsonl or csv")
            curated = query.get("phase", "curated") == "curated"
            conn = get_connection()
            try:
                records = feed_study.dataset(conn, account, curated_only=curated)
            finally:
                conn.close()
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        body = feed_study.to_jsonl(records) if fmt == "jsonl" else feed_study.to_csv(records)
        name = f"{account}-{'curated' if curated else 'all'}.{fmt}"
        self._send(200, body, "application/x-ndjson; charset=utf-8" if fmt == "jsonl" else "text/csv; charset=utf-8",
                   {"Content-Disposition": f'attachment; filename="{name}"'})

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

    def account_list(self, _query: dict) -> dict:
        return {"items": account_store.overview()}

    def add_account(self, body: dict) -> dict:
        return {"username": account_store.add(body.get("username"), body.get("password"))}

    def account_password(self, body: dict) -> dict:
        if not body.get("remove") and not body.get("password"):
            raise ValueError("Type the new password, or forget the saved one.")
        account_store.set_password(body.get("username"), "" if body.get("remove") else body.get("password"))
        return {"ok": True}

    def delete_account(self, body: dict) -> dict:
        username = account_store.valid_username(body.get("username"))
        if self.runner.current()[0]["running"]:
            raise ValueError("A run is in progress: stop it first, then delete the account.")
        done = account_store.remove(username, delete_login=bool(body.get("delete_login")))
        done["schedule_removed"] = schedule.remove(username)
        return done

    def categorize(self, body: dict) -> dict:
        account = checked_account(body.get("account"))
        if not cached("llm", 10, llm_serving):
            raise ValueError("LM Studio isn't serving the model, so reels can't be categorised now.")
        return self.runner._launch([sys.executable, "-m", "dashboard.feed", "--account", account, "--categorize"],
                                   f"Categorising reels for @{account}")

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
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' fill='%230e0e11'/%3E%3Crect x='3' y='3' width='26' height='26' fill='none' stroke='%23fff' stroke-width='2.5'/%3E%3Cpath d='M12 10v12l10-6z' fill='%23fff'/%3E%3C/svg%3E">
<script>try { if (JSON.parse(localStorage.getItem("feedlab.theme")) === "brutal") document.documentElement.setAttribute("data-theme", "brutal"); } catch (e) {}</script>
<style>
@font-face { font-family: 'JetBrains Mono'; font-style: normal; font-weight: 400 500; font-display: swap;
  src: url(/fonts/jetbrainsmono-latin-ext.woff2) format('woff2'); unicode-range: U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329, U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C0, U+2113, U+2C60-2C7F, U+A720-A7FF; }
@font-face { font-family: 'JetBrains Mono'; font-style: normal; font-weight: 400 500; font-display: swap;
  src: url(/fonts/jetbrainsmono-latin.woff2) format('woff2'); unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD; }
@font-face { font-family: 'Orbitron'; font-style: normal; font-weight: 400 900; font-display: swap;
  src: url(/fonts/orbitron-latin.woff2) format('woff2'); unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD; }
@font-face { font-family: 'Rajdhani'; font-style: normal; font-weight: 500; font-display: swap;
  src: url(/fonts/rajdhani-devanagari-500.woff2) format('woff2'); unicode-range: U+0900-097F, U+1CD0-1CF9, U+200C-200D, U+20A8, U+20B9, U+20F0, U+25CC, U+A830-A839, U+A8E0-A8FF, U+11B00-11B09; }
@font-face { font-family: 'Rajdhani'; font-style: normal; font-weight: 500; font-display: swap;
  src: url(/fonts/rajdhani-latin-ext-500.woff2) format('woff2'); unicode-range: U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329, U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C0, U+2113, U+2C60-2C7F, U+A720-A7FF; }
@font-face { font-family: 'Rajdhani'; font-style: normal; font-weight: 500; font-display: swap;
  src: url(/fonts/rajdhani-latin-500.woff2) format('woff2'); unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD; }
@font-face { font-family: 'Rajdhani'; font-style: normal; font-weight: 600; font-display: swap;
  src: url(/fonts/rajdhani-devanagari-600.woff2) format('woff2'); unicode-range: U+0900-097F, U+1CD0-1CF9, U+200C-200D, U+20A8, U+20B9, U+20F0, U+25CC, U+A830-A839, U+A8E0-A8FF, U+11B00-11B09; }
@font-face { font-family: 'Rajdhani'; font-style: normal; font-weight: 600; font-display: swap;
  src: url(/fonts/rajdhani-latin-ext-600.woff2) format('woff2'); unicode-range: U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329, U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C0, U+2113, U+2C60-2C7F, U+A720-A7FF; }
@font-face { font-family: 'Rajdhani'; font-style: normal; font-weight: 600; font-display: swap;
  src: url(/fonts/rajdhani-latin-600.woff2) format('woff2'); unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD; }
@font-face { font-family: 'Rajdhani'; font-style: normal; font-weight: 700; font-display: swap;
  src: url(/fonts/rajdhani-devanagari-700.woff2) format('woff2'); unicode-range: U+0900-097F, U+1CD0-1CF9, U+200C-200D, U+20A8, U+20B9, U+20F0, U+25CC, U+A830-A839, U+A8E0-A8FF, U+11B00-11B09; }
@font-face { font-family: 'Rajdhani'; font-style: normal; font-weight: 700; font-display: swap;
  src: url(/fonts/rajdhani-latin-ext-700.woff2) format('woff2'); unicode-range: U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329, U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C0, U+2113, U+2C60-2C7F, U+A720-A7FF; }
@font-face { font-family: 'Rajdhani'; font-style: normal; font-weight: 700; font-display: swap;
  src: url(/fonts/rajdhani-latin-700.woff2) format('woff2'); unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD; }

/* The look of the KIDRAAH_ workspace: square corners, 2px rules, hard offset shadows, Orbitron for names,
   Rajdhani for reading. Two themes share that shape and differ only in palette. */
:root {
  color-scheme: dark;
  --bg: #0e0e11; --rail: #050507; --bar: #050507; --panel: #14141a; --glass: rgba(255, 255, 255, .05);
  --text: #e2e2e2; --muted: #8b8b92; --bar-text: #e2e2e2; --bar-muted: #8b8b92;
  --line: rgba(255, 255, 255, .55); --rule: rgba(255, 255, 255, .16); --hover: rgba(255, 255, 255, .14);
  --accent: #ffffff; --on-accent: #0e0e11; --focus: #ffffff; --link: #ffffff;
  --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444;
  --shadow-sm: 3px 3px 0 rgba(255, 255, 255, .55); --shadow-lg: 5px 5px 0 rgba(255, 255, 255, .55);
  --nav-bg: #14141a; --nav-text: #e2e2e2; --nav-line: rgba(255, 255, 255, .55);
  --nav-on-bg: #ffffff; --nav-on-text: #0e0e11;
  --contour: 255, 255, 255; --contour-alpha: .18;
  --console-bg: #050507; --console-text: #e2e2e2;
  --chevron: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M7 10l5 5 5-5z' fill='%23e2e2e2'/%3E%3C/svg%3E");
  /* Feed chart: one grey ramp (warm-up dimmer, curated brighter), validated against #14141a */
  --viz-before: #6e6e6e; --viz-after: #e2e2e2; --viz-bar: #a8a8a8; --viz-grid: #26262d; --viz-axis: #4a4a52; --viz-link: #4a4a52;
  --display: "Orbitron", "Avenir Next", sans-serif;
  --body: "Rajdhani", "Avenir Next", "Kohinoor Devanagari", sans-serif;
  --mono: "JetBrains Mono", Menlo, monospace;
}
[data-theme="brutal"] {
  color-scheme: light;
  --bg: #fbf1da; --rail: #d6791c; --bar: #edb03c; --panel: #fbf1da; --glass: rgba(18, 33, 58, .05);
  --text: #12213a; --muted: #4f5868; --bar-text: #12213a; --bar-muted: #12213a;
  --line: #12213a; --rule: rgba(18, 33, 58, .2); --hover: rgba(214, 121, 28, .18);
  --accent: #d6791c; --on-accent: #12213a; --focus: #12213a; --link: #12213a;
  --ok: #166534; --warn: #92400e; --bad: #b91c1c;
  --shadow-sm: 3px 3px 0 #12213a; --shadow-lg: 5px 5px 0 #12213a;
  --nav-bg: #ffffff; --nav-text: #12213a; --nav-line: #12213a;
  --nav-on-bg: #12213a; --nav-on-text: #ffffff;
  --contour: 251, 241, 218; --contour-alpha: .5;
  --console-bg: #12213a; --console-text: #fbf1da;
  --chevron: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M7 10l5 5 5-5z' fill='%2312213a'/%3E%3C/svg%3E");
  /* One orange ramp, validated against #fbf1da */
  --viz-before: #d98a3a; --viz-after: #8a4a12; --viz-bar: #b0621a; --viz-grid: #eadcbc; --viz-axis: #b8ab8c; --viz-link: #c9b994;
}

/* Base */
*, *::before, *::after { box-sizing: border-box; }
[hidden] { display: none !important; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--text); font: 500 16px/1.5 var(--body); -webkit-font-smoothing: antialiased; }
a { color: var(--link); text-decoration: underline; text-decoration-color: var(--line); text-underline-offset: 3px; }
a:hover { text-decoration-color: currentColor; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
code { font: 13px var(--mono); }
.hint { color: var(--muted); font-size: 14.5px; }
p.hint { max-width: 80ch; }
.spacer { flex: 1; }
.caps, .label, .island-label, .chip, table.data th, .control .label, .box-title, .card h3 {
  font-family: var(--display); font-weight: 500; text-transform: uppercase; }

/* The frame: the rail on the left, the page beside it */
.app { display: grid; grid-template-columns: auto minmax(0, 1fr); min-height: 100vh; }
.sidebar { position: sticky; top: 0; height: 100vh; width: 248px; background: var(--rail); border-right: 1px solid var(--line);
           z-index: 20; transition: width .35s cubic-bezier(.2, 0, 0, 1); }
.sidebar.collapsed { width: 68px; }
.canvas-clip { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
#contours { position: absolute; top: 0; left: 0; width: 248px; height: 100%; }
.sidebar-content { position: relative; z-index: 1; display: flex; flex-direction: column; height: 100%; padding: 16px; }
.sidebar.collapsed .sidebar-content { padding: 16px 0; align-items: center; }
.sidebar-header { display: flex; align-items: center; justify-content: space-between; height: 42px; padding: 0 6px 0 12px;
                  margin-bottom: 20px; background: var(--nav-bg); color: var(--nav-text); border: 2px solid var(--nav-line);
                  box-shadow: var(--shadow-sm); }
.brand { font: 500 14px var(--display); letter-spacing: 2px; white-space: nowrap; }
.toggle { width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; padding: 0; flex: none;
          background: transparent; color: inherit; border: 1px solid transparent; cursor: pointer; }
.toggle:hover { background: var(--hover); border-color: var(--nav-line); }
.toggle svg { width: 16px; height: 16px; fill: currentColor; }
.sidebar.collapsed .sidebar-header { background: transparent; border-color: transparent; box-shadow: none; padding: 0; justify-content: center; }
.sidebar.collapsed .brand { display: none; }
.sidebar.collapsed .toggle { width: 36px; height: 36px; background: var(--nav-bg); color: var(--nav-text); border: 2px solid var(--nav-line);
                             box-shadow: var(--shadow-sm); }
.sidebar nav { display: flex; flex-direction: column; gap: 8px; }
.nav-btn { position: relative; display: flex; align-items: center; gap: 11px; width: 100%; padding: 8px 12px; cursor: pointer;
           background: var(--nav-bg); color: var(--nav-text); border: 2px solid var(--nav-line); box-shadow: var(--shadow-sm);
           font: 600 15.5px/1.2 var(--body); letter-spacing: .3px; text-align: left; }
.nav-btn svg { width: 17px; height: 17px; fill: currentColor; flex: none; }
.nav-btn:hover { border-color: var(--accent); }
[data-theme="brutal"] .nav-btn:hover { border-color: var(--nav-line); background: #fff4df; }
.nav-btn.active { background: var(--nav-on-bg); color: var(--nav-on-text); border-color: var(--nav-on-bg); }
.nav-btn .btn-text { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.nav-meta { margin-left: auto; font: 500 12px var(--mono); display: inline-flex; align-items: center; gap: 6px; }
.nav-meta:not(:empty)::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor;
                                animation: blink 1.2s steps(1) infinite; }
.sidebar.collapsed .nav-btn { width: 36px; height: 36px; padding: 0; justify-content: center; }
.sidebar.collapsed .nav-btn .btn-text, .sidebar.collapsed .nav-meta:empty { display: none; }
.sidebar.collapsed .nav-meta { position: absolute; top: -5px; right: -5px; font-size: 0; gap: 0; }
.sidebar.collapsed .nav-meta::before { width: 9px; height: 9px; box-shadow: 0 0 0 2px var(--rail); }
.sidebar-bottom { margin-top: auto; padding-top: 14px; border-top: 1px solid var(--nav-line); position: relative; width: 100%; }
.sidebar.collapsed .sidebar-bottom { width: auto; border-top-color: transparent; }
.theme-menu { position: absolute; bottom: 0; left: calc(100% + 16px); z-index: 60; min-width: 150px; padding: 4px;
              display: flex; flex-direction: column; gap: 2px; background: var(--panel); border: 2px solid var(--line);
              box-shadow: var(--shadow-lg); }
.menu-item { background: transparent; border: 0; padding: 8px 12px; text-align: left; cursor: pointer; color: var(--text);
             font: 600 15px var(--body); display: flex; justify-content: space-between; gap: 12px; }
.menu-item:hover { background: var(--hover); }
.menu-item.active::after { content: "✓"; }
.column { min-width: 0; width: 100%; max-width: 1320px; padding: 22px clamp(18px, 3vw, 40px) 56px; }
footer { color: var(--muted); font: 12px var(--mono); margin-top: 40px; }

/* The top bars: which account, and what the machine can do right now */
header.top { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 16px 24px; margin-bottom: 30px; }
.island { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; padding: 7px 10px; background: var(--bar);
          color: var(--bar-text); border: 1px solid var(--line); box-shadow: var(--shadow-lg); }
.island-label { align-self: stretch; display: flex; align-items: center; padding: 0 10px 0 2px; margin-right: 2px;
                border-right: 1px solid var(--line); color: var(--bar-muted); font-size: 10px; letter-spacing: 1.6px; }
#account { font: 700 18px/1.2 var(--body); color: var(--bar-text); background: transparent var(--chevron) no-repeat right 6px center / 18px;
           border: 2px solid var(--line); padding: 4px 32px 4px 10px; appearance: none; -webkit-appearance: none; cursor: pointer; }
#account:hover { background-color: var(--hover); }
.text-btn { font: 500 10.5px/1 var(--display); letter-spacing: 1px; text-transform: uppercase; color: var(--bar-text); cursor: pointer;
            background: transparent; border: 2px solid var(--line); padding: 9px 12px; min-height: 34px; white-space: nowrap; }
.text-btn:hover { background: var(--hover); border-color: var(--accent); }
[data-theme="brutal"] .text-btn:hover { border-color: var(--line); }
.chip { display: inline-flex; align-items: center; gap: 8px; padding: 4px 8px; font-size: 10.5px; letter-spacing: 1px;
        color: var(--bar-text); white-space: nowrap; }
.chip::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--bar-muted); opacity: .6; }
.chip.ok::before { background: var(--ok); opacity: 1; }
.chip.warn::before { background: var(--warn); opacity: 1; }
[data-theme="brutal"] .chip.ok::before, [data-theme="brutal"] .chip.warn::before { box-shadow: 0 0 0 1.5px var(--line); }

/* Boxes: the border is the structure, the shadow is the depth */
.card { background: var(--panel); border: 2px solid var(--line); box-shadow: var(--shadow-sm); padding: 20px 22px 22px; margin: 0 0 30px; }
.card h3, .box-title { font-size: 13px; letter-spacing: 2px; margin: 0 0 12px; }
.headline { font: 600 26px/1.2 var(--display); letter-spacing: .5px; margin: 0 0 10px; }
.lede { font-size: 17.5px; max-width: 70ch; margin: 0; }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin: 12px 0; }
.status { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; }
.label { width: 120px; color: var(--muted); font-size: 10.5px; letter-spacing: 1.2px; }
select, input[type=number], input[type=text], input[type=password], textarea {
  font: 500 16px/1.4 var(--body); color: var(--text); background: var(--glass); border: 2px solid var(--line); border-radius: 0;
  padding: 8px 12px; max-width: 100%; }
select { cursor: pointer; }
select option { background: var(--panel); color: var(--text); }
input[type=number] { width: 100px; font: 500 15px var(--mono); }
textarea { width: 100%; resize: vertical; line-height: 1.5; }
textarea.words { font-size: 15px; }
input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
input:focus-visible, select:focus-visible, textarea:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
input[type=checkbox] { appearance: none; -webkit-appearance: none; position: relative; flex: none; width: 36px; height: 20px; margin: 0;
                       background: var(--glass); border: 2px solid var(--line); cursor: pointer; }
input[type=checkbox]::before { content: ""; position: absolute; top: 2px; left: 2px; width: 12px; height: 12px; background: var(--muted);
                               transition: transform .2s; }
input[type=checkbox]:checked { background: var(--accent); border-color: var(--accent); }
input[type=checkbox]:checked::before { transform: translateX(16px); background: var(--on-accent); }
[data-theme="brutal"] input[type=checkbox]:checked { border-color: var(--line); }
.check { display: inline-flex; gap: 12px; align-items: center; cursor: pointer; font-weight: 600; }
.field { margin: 24px 0 0; }
.field > label, .field > .name { display: block; font-weight: 700; font-size: 16.5px; }
.field > .hint { margin: 2px 0 8px; }
.field input[type=text] { width: 100%; }
#acc-user, #acc-pass { width: 320px; }
.seg { display: inline-flex; flex-wrap: wrap; background: var(--glass); border: 2px solid var(--line); }
.seg label { position: relative; padding: 9px 14px; cursor: pointer; user-select: none; color: var(--muted);
             font: 500 10.5px/1.2 var(--display); letter-spacing: 1.2px; text-transform: uppercase; }
.seg label + label { border-left: 1px solid var(--line); }
.seg label:hover { color: var(--text); background: var(--hover); }
.seg label.on { background: var(--accent); color: var(--on-accent); }
.seg input { position: absolute; opacity: 0; pointer-events: none; }
.seg label:focus-within { outline: 2px solid var(--focus); outline-offset: 2px; }
.seg.small label { padding: 7px 11px; font-size: 10px; }
.monitor-head .seg label:not(.on) { color: var(--bar-muted); }  /* on the bar's own ground */
.presets { display: flex; flex-wrap: wrap; gap: 10px; }
.presets button { font: 600 15px var(--body); padding: 6px 14px; border: 2px solid var(--line); background: var(--glass); color: var(--text);
                  cursor: pointer; }
.presets button:hover { background: var(--hover); }
.presets button.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
[data-theme="brutal"] .presets button.on { border-color: var(--line); }
.notice { padding: 10px 14px; background: var(--glass); border: 1px solid var(--rule); border-left: 4px solid var(--warn); color: var(--text);
          font-size: 15px; margin: 14px 0 0; max-width: 80ch; }
.notice button { font: inherit; color: inherit; text-decoration: underline; background: none; border: 0; padding: 0; cursor: pointer; }
.badge { display: inline-block; font: 500 9.5px/1.4 var(--display); letter-spacing: 1px; text-transform: uppercase; padding: 3px 8px;
         border: 1px solid var(--line); color: var(--muted); }
.badge.ok { border-color: var(--ok); color: var(--ok); } .badge.warn { border-color: var(--warn); color: var(--warn); }

/* Buttons: a box on its shadow, which it drops onto when pressed */
.actions { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 20px; }
.btn { font: 700 15px/1 var(--body); letter-spacing: .6px; padding: 11px 18px; cursor: pointer; color: var(--text);
       background: var(--panel); border: 2px solid var(--line); box-shadow: var(--shadow-sm); }
.btn:hover:not(:disabled) { background: var(--hover); border-color: var(--accent); }
.btn:active:not(:disabled) { transform: translate(3px, 3px); box-shadow: none; }
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
.btn.primary:hover:not(:disabled) { background: var(--accent); box-shadow: var(--shadow-lg); }
[data-theme="brutal"] .btn:hover:not(:disabled) { border-color: var(--line); }
[data-theme="brutal"] .btn.primary { border-color: var(--line); }
.btn.danger { color: var(--bad); }
.btn.danger:hover:not(:disabled) { border-color: var(--bad); }
.btn:disabled { opacity: .35; cursor: not-allowed; box-shadow: none; }
.linkish { font: 700 15px var(--body); background: none; border: 0; padding: 0; cursor: pointer; color: var(--link);
           text-decoration: underline; text-decoration-color: var(--line); text-underline-offset: 3px; }
.linkish:hover { text-decoration-color: currentColor; }
.linkish:disabled { opacity: .35; cursor: not-allowed; }
.linkish.danger { color: var(--bad); }
.row-actions { display: inline-flex; gap: 18px; white-space: nowrap; }

/* Run: the session's state, its controls, and every reel it sees */
#tab-run { display: grid; grid-template-columns: minmax(290px, 370px) minmax(0, 1fr); grid-template-rows: auto 1fr; gap: 0 36px;
           align-items: start; }
.run-head { grid-column: 1; grid-row: 1; }
.deck { grid-column: 1; grid-row: 2; }
.monitor { grid-column: 2; grid-row: 1 / 3; }
.run-head .state { font: 600 clamp(26px, 2.4vw, 34px)/1.1 var(--display); letter-spacing: 4px; text-transform: uppercase; margin: 0 0 10px;
                   white-space: nowrap; }
.run-head .state.live::after { content: "_"; animation: blink 1.1s steps(1) infinite; }
.run-head .state.bad { color: var(--bad); }
.run-head .detail { color: var(--muted); font-size: 15px; margin: 0; }
.run-head .clock { display: block; margin-top: 8px; font: 500 13px var(--mono); color: var(--text); }
.track { height: 3px; background: var(--rule); margin: 14px 0 0; overflow: hidden; }
.track i { display: block; width: 0; height: 100%; background: var(--accent); transition: width 1s linear; }
.track.indeterminate i { width: 30%; animation: sweep 1.8s ease-in-out infinite; }
@keyframes sweep { from { transform: translateX(-100%); } to { transform: translateX(340%); } }
@keyframes blink { 50% { opacity: 0; } }
.control { margin: 0 0 20px; }
.control .label { display: block; width: auto; margin: 0 0 8px; }
.control .hint { margin: 6px 0 0; }
.control .check + .hint { margin: 6px 0 14px 48px; }
.inline { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }
.deck .actions { margin-top: 24px; }
.deck .quiet { margin-top: 16px; gap: 22px; }
.monitor { padding: 0; box-shadow: var(--shadow-lg); overflow: hidden; }
.monitor-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; padding: 12px 14px 12px 18px;
                background: var(--bar); color: var(--bar-text); border-bottom: 2px solid var(--line); }
.monitor-head .island-label { align-self: auto; padding: 4px 12px 4px 0; }
.tally { font: 500 12.5px var(--mono); color: var(--bar-text); }
.stream { height: 660px; overflow-y: auto; overscroll-behavior: contain; scrollbar-width: thin; scrollbar-color: var(--line) transparent; }
.stream:empty::before { content: "// READY. AWAITING_REELS..."; display: block; padding: 150px 24px 10px; text-align: center;
                        font: 500 12px var(--display); letter-spacing: 2px; color: var(--muted); }
.stream:empty::after { content: "Start a session: every reel the account is shown lands here as it happens."; display: block;
                       text-align: center; padding: 0 24px; color: var(--muted); }
.v-row { display: grid; grid-template-columns: 50px minmax(0, 1fr) auto; gap: 2px 16px; padding: 13px 16px 13px 12px;
         border-left: 3px solid var(--rule); border-bottom: 1px solid var(--rule); }
.v-row.watched { border-left-color: var(--ok); }
.v-row.skipped { border-left-color: var(--warn); }
.fresh { animation: arrive .35s ease-out; }
@keyframes arrive { from { transform: translateY(-8px); opacity: 0; } }
.v-seq { font: 500 12px var(--mono); color: var(--muted); padding-top: 3px; }
.v-main { min-width: 0; }
.v-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 12px; }
.v-head .who { font-weight: 700; font-size: 16.5px; color: var(--text); text-decoration: none; overflow-wrap: anywhere; }
.v-head a.who:hover { text-decoration: underline; }
.v-head .kind { font: 12px var(--mono); color: var(--muted); }
.v-main .topic { margin-top: 1px; font-size: 16px; }
.v-main .summary { margin-top: 3px; color: var(--muted); font-size: 15px; }
.v-main .extras { margin-top: 4px; font: 500 10px var(--display); letter-spacing: 1px; text-transform: uppercase; }
.v-verdict { text-align: right; min-width: 132px; }
.stamp { font: 600 11.5px/1.3 var(--display); letter-spacing: 1.4px; text-transform: uppercase; }
.v-row.watched .stamp, .v-row.watched .meter { color: var(--ok); }
.v-row.skipped .stamp, .v-row.skipped .meter { color: var(--warn); }
.v-row.other .stamp, .v-row.other .meter, .v-row.other .v-head .who { color: var(--muted); }
.dwell { margin-top: 6px; font: 12px var(--mono); color: var(--muted); }
.meter { display: flex; align-items: center; justify-content: flex-end; gap: 8px; margin-top: 6px; font: 500 12px var(--mono); }
.meter b { display: block; width: 56px; height: 3px; background: var(--rule); overflow: hidden; }
.meter b i { display: block; height: 100%; background: currentColor; }
.v-note { padding: 12px 16px 12px 65px; border-left: 3px solid transparent; border-bottom: 1px solid var(--rule);
          font: 12.5px/1.6 var(--mono); color: var(--muted); }
.v-note div::before { content: "// "; }
pre { margin: 0; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; font: 12.5px/1.55 var(--mono); }
pre.console { display: none; height: 660px; padding: 16px 20px; background: var(--console-bg); color: var(--console-text); }
pre.console:empty::before { content: "// NO OUTPUT YET"; opacity: .6; }
.monitor.show-raw .stream { display: none; }
.monitor.show-raw pre.console { display: block; }

/* Feed */
.legend { display: flex; flex-wrap: wrap; gap: 18px; font-size: 15px; color: var(--muted); margin: 10px 0 4px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 7px; }
.chart { position: relative; margin-top: 8px; max-width: 980px; }
.chart svg { display: block; width: 100%; height: auto; }
.chart text { font: 600 14px var(--body); }
.chart .cat { fill: var(--text); }
.chart .val, .chart .tick { fill: var(--muted); font: 500 12px var(--mono); }
.chart .hit { fill: transparent; outline: none; }
.chart .hit:hover, .chart .hit:focus { fill: var(--hover); }
.tip { position: absolute; z-index: 5; pointer-events: none; background: var(--panel); border: 2px solid var(--line); padding: 10px 14px;
       font-size: 14.5px; box-shadow: var(--shadow-sm); min-width: 200px; }
.tip .title { font-weight: 700; margin-bottom: 4px; }
.tip .line { display: flex; align-items: center; gap: 8px; }
.tip .line b { font: 500 13px var(--mono); min-width: 44px; }
.tip .line .name { color: var(--muted); }
.tip .key { display: inline-block; width: 12px; height: 3px; }
details summary { cursor: pointer; margin-top: 14px; }
table.data { width: 100%; border-collapse: collapse; font-size: 15.5px; margin-top: 10px; }
table.data th, table.data td { text-align: left; padding: 10px 10px; vertical-align: baseline; }
table.data th { font-size: 10px; letter-spacing: 1.2px; color: var(--muted); border-bottom: 2px solid var(--line); }
table.data td { border-bottom: 1px solid var(--rule); }
table.data .num { text-align: right; font: 500 13px var(--mono); }
table.data th.num { font: 500 10px var(--display); letter-spacing: 1.2px; }
.table-scroll { overflow-x: auto; }
.reel-row { display: grid; grid-template-columns: 150px 170px 170px minmax(0, 1fr) 132px; gap: 16px; padding: 12px 0;
            border-top: 1px solid var(--rule); font-size: 15.5px; align-items: baseline; }
.reel-row .when { color: var(--muted); font: 12px var(--mono); }
.reel-row .num { display: flex; gap: 12px; justify-content: flex-end; color: var(--muted); font: 12px var(--mono); }
.reel-row a { overflow-wrap: anywhere; }
.tag-cat { font: 500 9.5px/1.4 var(--display); letter-spacing: .8px; text-transform: uppercase; padding: 3px 8px;
           border: 1px solid var(--line); white-space: nowrap; }

/* Digest, report, bias, schedule */
.story { background: var(--panel); border: 2px solid var(--line); box-shadow: var(--shadow-sm); padding: 18px 22px; margin: 0 0 18px;
         max-width: 920px; }
.story-head { display: flex; gap: 14px; align-items: baseline; }
.story h3 { margin: 0; font: 600 17px/1.35 var(--display); letter-spacing: .3px; }
.story p { margin: 8px 0 0; max-width: 72ch; }
.score { font: 500 12.5px var(--mono); color: var(--muted); }
.meta { margin-top: 12px; font-size: 15px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; }
ul.news { margin: 10px 0 0; padding-left: 18px; font-size: 15.5px; }
ul.news .src { color: var(--muted); margin-left: 8px; }
pre.report { padding: 18px 20px; background: var(--glass); border: 1px solid var(--rule); margin-top: 18px; }
.two { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 0 44px; align-items: start; }
.sched { display: flex; gap: 14px; align-items: center; justify-content: space-between; padding: 14px 0; border-top: 1px solid var(--rule); }
.sched:first-child { border-top: 0; }

/* Dialogs and toasts */
.overlay { position: fixed; inset: 0; background: rgba(0, 0, 0, .6); backdrop-filter: blur(6px); display: flex;
           align-items: center; justify-content: center; padding: 16px; z-index: 40; }
.modal { background: var(--panel); color: var(--text); border: 2px solid var(--line); box-shadow: var(--shadow-lg); padding: 24px 24px 22px;
         width: min(480px, 100%); }
.modal h3 { margin: 0 0 10px; font: 600 17px/1.3 var(--display); letter-spacing: .5px; }
.modal p { margin: 0 0 12px; }
.modal input[type=text], .modal input[type=password] { width: 100%; font: 500 20px var(--mono); letter-spacing: .08em; padding: 11px 14px; margin: 4px 0 2px; }
#toast { position: fixed; right: 20px; bottom: 20px; z-index: 50; max-width: min(420px, calc(100vw - 40px)); padding: 11px 16px;
         background: var(--panel); color: var(--text); border: 2px solid var(--line); border-left: 5px solid var(--ok);
         box-shadow: var(--shadow-lg); font-size: 15.5px; font-weight: 600; }
#toast.bad { border-left-color: var(--bad); }

/* Narrower windows: the stream under the controls; then the rail becomes a bar along the bottom */
@media (max-width: 1100px) {
  #tab-run, .two { grid-template-columns: minmax(0, 1fr); }
  #tab-run { grid-template-rows: none; }
  .run-head, .deck, .monitor { grid-column: 1; grid-row: auto; }
  .deck { order: 1; } .monitor { order: 2; }
  #tab-run.running .monitor { order: 1; margin-bottom: 30px; } #tab-run.running .deck { order: 2; }
  .stream, pre.console { height: 540px; }
}
@media (max-width: 760px) {
  .app { grid-template-columns: minmax(0, 1fr); }
  .sidebar, .sidebar.collapsed { position: fixed; top: auto; bottom: 0; left: 0; right: 0; width: auto; height: auto;
                                 border-right: 0; border-top: 1px solid var(--line); background: var(--bar); }
  .canvas-clip, .sidebar-header, .sidebar-bottom { display: none; }
  .sidebar-content, .sidebar.collapsed .sidebar-content { padding: 4px 4px calc(4px + env(safe-area-inset-bottom)); }
  .sidebar nav { flex-direction: row; justify-content: space-around; gap: 0; width: 100%; }
  .nav-btn, .sidebar.collapsed .nav-btn { flex-direction: column; gap: 3px; width: auto; height: auto; min-width: 44px; padding: 6px 2px;
                                          background: transparent; color: var(--bar-text); border: 0; border-top: 2px solid transparent;
                                          box-shadow: none; font-size: 11px; }
  .sidebar.collapsed .nav-btn .btn-text { display: inline; }
  .nav-btn.active { background: transparent; color: var(--bar-text); border-top-color: var(--bar-text); }
  .nav-meta { position: absolute; top: 2px; right: 2px; font-size: 0; }
  .column { padding: 16px 16px 100px; }
  header.top { margin-bottom: 22px; }
  .label { width: 100%; }
  .reel-row { grid-template-columns: minmax(0, 1fr); gap: 3px; }
  .reel-row .num { justify-content: flex-start; }
  .v-row { grid-template-columns: minmax(0, 1fr) auto; }
  .v-seq { display: none; }
  .v-note { padding-left: 16px; }
  #acc-user, #acc-pass { width: 100%; }
  #toast { left: 16px; right: 16px; bottom: 84px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  .track.indeterminate i { width: 100%; opacity: .45; }
}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar" id="sidebar">
    <div class="canvas-clip"><canvas id="contours" aria-hidden="true"></canvas></div>
    <div class="sidebar-content">
      <div class="sidebar-header">
        <span class="brand">PRAG_</span>
        <button class="toggle" id="collapse" title="Collapse the menu" aria-label="Collapse the menu" aria-expanded="true">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/></svg></button>
      </div>
      <nav aria-label="Sections">
        <button class="nav-btn" data-tab="run" title="Run"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg><span class="btn-text">Run</span><span class="nav-meta" id="nav-run-meta"></span></button>
        <button class="nav-btn" data-tab="feed" title="Feed"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg><span class="btn-text">Feed</span></button>
        <button class="nav-btn" data-tab="digest" title="Digest"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg><span class="btn-text">Digest</span></button>
        <button class="nav-btn" data-tab="report" title="Report"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4zm8.94 3A8.99 8.99 0 0 0 13 3.06V1h-2v2.06A8.99 8.99 0 0 0 3.06 11H1v2h2.06A8.99 8.99 0 0 0 11 20.94V23h2v-2.06A8.99 8.99 0 0 0 20.94 13H23v-2h-2.06zM12 19c-3.87 0-7-3.13-7-7s3.13-7 7-7 7 3.13 7 7-3.13 7-7 7z"/></svg><span class="btn-text">Report</span></button>
        <button class="nav-btn" data-tab="bias" title="Bias"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z"/></svg><span class="btn-text">Bias</span></button>
        <button class="nav-btn" data-tab="schedule" title="Schedule"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z"/></svg><span class="btn-text">Schedule</span></button>
        <button class="nav-btn" data-tab="accounts" title="Accounts"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg><span class="btn-text">Accounts</span></button>
      </nav>
      <div class="sidebar-bottom">
        <button class="nav-btn" id="theme-btn" title="Theme" aria-haspopup="true" aria-expanded="false"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9c.83 0 1.5-.67 1.5-1.5 0-.39-.15-.74-.39-1.01-.23-.26-.38-.61-.38-.99 0-.83.67-1.5 1.5-1.5H16c2.76 0 5-2.24 5-5 0-4.42-4.03-8-9-8zm-5.5 9c-.83 0-1.5-.67-1.5-1.5S5.67 9 6.5 9 8 9.67 8 10.5 7.33 12 6.5 12zm3-4C8.67 8 8 7.33 8 6.5S8.67 5 9.5 5s1.5.67 1.5 1.5S10.33 8 9.5 8zm5 0c-.83 0-1.5-.67-1.5-1.5S13.67 5 14.5 5s1.5.67 1.5 1.5S15.33 8 14.5 8zm3 4c-.83 0-1.5-.67-1.5-1.5S18.67 9 19.5 9s1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/></svg><span class="btn-text">Theme</span></button>
        <div class="theme-menu" id="theme-menu" role="menu" hidden>
          <button class="menu-item" role="menuitemradio" data-set="default">Default</button>
          <button class="menu-item" role="menuitemradio" data-set="brutal">Brutal</button>
        </div>
      </div>
    </div>
  </aside>
  <script>try { if (JSON.parse(localStorage.getItem("feedlab.collapsed"))) document.getElementById("sidebar").classList.add("collapsed"); } catch (e) {}</script>

  <div class="column">
    <header class="top">
      <div class="island">
        <span class="island-label">Account</span>
        <select id="account" aria-label="Account"></select>
        <button class="text-btn" id="bias-pill" title="Edit the bias" hidden></button>
      </div>
      <div class="island">
        <span class="island-label">System</span>
        <span class="chip" id="chip-llm" hidden></span>
        <span class="chip" id="chip-today" hidden></span>
        <span class="chip" id="chip-live" hidden></span>
        <span class="chip" id="chip-power" hidden></span>
      </div>
    </header>
    <p class="notice" id="no-account" hidden>No active account yet. <button id="go-accounts">Add one on the Accounts tab.</button></p>

    <main>
      <section id="tab-run">
        <div class="run-head card">
          <div class="state" id="run-state" role="status">&nbsp;</div>
          <p class="detail" id="run-status">Loading…</p>
          <div class="track" id="progress" hidden><i></i></div>
          <span class="clock" id="run-clock"></span>
        </div>
        <div class="deck card">
          <h3>Next session</h3>
          <div class="control">
            <span class="label">Engagement</span>
            <div class="seg" id="engagement"></div>
            <p class="hint" id="eng-hint"></p>
          </div>
          <div class="control">
            <label class="label" for="minutes">Minutes</label>
            <div class="inline"><input id="minutes" type="number" min="1" step="1"><span class="hint" id="minutes-hint"></span></div>
          </div>
          <div class="control">
            <label class="check"><input type="checkbox" id="headless"> Headless: no browser window</label>
            <p class="hint" id="headless-hint"></p>
            <label class="check"><input type="checkbox" id="start-llm"> Start LM Studio if it's off</label>
          </div>
          <p class="notice" id="skip" hidden></p>
          <div class="actions">
            <button class="btn primary" id="btn-run">Start session</button>
            <button class="btn danger" id="btn-stop" disabled>Stop</button>
          </div>
          <div class="actions quiet">
            <button class="linkish" id="btn-digest">Rebuild the digest only</button>
            <button class="linkish" id="btn-logs">Open the logs folder</button>
          </div>
        </div>
        <div class="monitor card" id="monitor">
          <div class="monitor-head">
            <span class="island-label">Reels</span>
            <span class="tally" id="tally"></span>
            <span class="spacer"></span>
            <div class="seg small" id="view" role="radiogroup" aria-label="Show">
              <label><input type="radio" name="view" value="reels"> Reels</label>
              <label><input type="radio" name="view" value="raw"> Raw output</label>
            </div>
          </div>
          <div class="stream" id="stream"></div>
          <pre class="console" id="console"></pre>
        </div>
      </section>

      <section id="tab-feed" hidden>
        <div class="card">
          <h2 class="headline" id="feed-state">Loading…</h2>
          <p class="lede" id="feed-curation"></p>
          <p class="notice" id="feed-uncat" hidden></p>
          <div class="actions">
            <button class="btn primary" id="btn-export-curated">Download the curated feed (JSONL)</button>
            <button class="btn" id="btn-export-curated-csv">Curated feed as CSV</button>
            <button class="btn" id="btn-export-all">Everything shown (JSONL)</button>
            <button class="btn" id="btn-categorize">Categorise now</button>
          </div>
        </div>
        <div class="card">
          <h3>What the feed shows</h3>
          <p class="hint" id="mix-sub"></p>
          <div class="legend" id="mix-legend"></div>
          <div class="chart" id="mix-chart"></div>
          <details><summary class="hint">Show as a table</summary><table class="data" id="mix-table"></table></details>
        </div>
        <div class="card">
          <h3>Creators the feed serves most</h3>
          <p class="hint" id="creators-sub"></p>
          <div class="table-scroll"><table class="data" id="creators"></table></div>
        </div>
        <div class="card">
          <h3>Every reel shown</h3>
          <div class="row"><select id="f-cat" aria-label="Category"></select>
            <select id="f-phase" aria-label="Phase"><option value="">Warm-up and curated</option>
              <option value="curated">Curated feed only</option><option value="warm-up">Warm-up only</option></select>
            <span class="hint" id="f-count"></span></div>
          <div id="reels"></div>
        </div>
      </section>

      <section id="tab-digest" hidden>
        <div class="row"><select id="digest-date" aria-label="Digest date"></select>
          <span class="hint" id="digest-stats"></span><span class="spacer"></span>
          <a id="digest-open" target="_blank" rel="noopener" hidden>Open the full page</a></div>
        <div id="stories"></div>
      </section>

      <section id="tab-report" hidden>
        <div class="card">
          <h2 class="headline">Is the feed curated yet?</h2>
          <div class="status"><span class="hint">The verdict needs 100 judged reels.</span>
            <button class="linkish" id="btn-report">Check again</button></div>
          <pre class="report" id="report"></pre>
        </div>
      </section>

      <section id="tab-bias" hidden>
        <div class="card">
          <div class="status"><h2 class="headline" id="bias-title">Loading…</h2>
            <span class="badge warn" id="bias-dirty" hidden>Unsaved changes</span></div>
          <p class="hint" id="bias-state"></p>
          <div class="field"><div class="name">Start from</div>
            <p class="hint">A preset only fills in the form below; nothing changes until you save.</p>
            <div class="presets" id="presets"></div></div>
          <div class="two">
            <div>
              <div class="field"><label for="bias-name">Name</label>
                <input type="text" id="bias-name" maxlength="40"></div>
              <div class="field"><div class="name">Scoring</div>
                <p class="hint">What counts as a hit for this bias.</p>
                <div class="seg" id="bias-scoring" role="radiogroup" aria-label="Scoring">
                  <label><input type="radio" name="scoring" value="informative"> What's informative</label>
                  <label><input type="radio" name="scoring" value="persona"> What this person would engage with</label>
                </div>
                <p class="hint" id="scoring-hint"></p></div>
              <div class="field"><label for="bias-interests" id="interests-label">What to look for</label>
                <p class="hint">The LLM scores every reel against this: say what's high, medium and low value to you.</p>
                <textarea id="bias-interests" rows="16"></textarea></div>
            </div>
            <div>
              <div class="field"><label for="bias-watch">Watch words</label>
                <p class="hint">When LM Studio is off, the quick rules watch reels whose caption or hashtags contain one of these. Separate with commas.</p>
                <textarea class="words" id="bias-watch" rows="4"></textarea></div>
              <div class="field"><label for="bias-skip">Skip words</label>
                <p class="hint">Reels whose caption or hashtags contain one of these are skipped at once, without the LLM. Single words.</p>
                <textarea class="words" id="bias-skip" rows="4"></textarea></div>
              <div class="field"><label for="bias-hints">Account-name hints</label>
                <p class="hint">Parts of account names that suggest a relevant page, like "news". The quick rules watch those accounts.</p>
                <input type="text" id="bias-hints"></div>
              <div class="field"><label for="bias-seeds">Accounts to follow first</label>
                <p class="hint">Optional. Following these gives a fresh account a head start. Only verified accounts get followed.</p>
                <textarea class="words" id="bias-seeds" rows="2"></textarea>
                <div class="actions"><button class="btn" id="btn-seed-check">Check them</button>
                  <button class="btn" id="btn-seed-follow">Follow them</button></div></div>
            </div>
          </div>
          <div class="actions"><button class="btn primary" id="btn-bias-save">Save bias</button></div>
          <p class="hint" id="bias-note"></p>
        </div>
      </section>

      <section id="tab-schedule" hidden>
        <div class="two">
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
          <div class="card"><h3>Installed</h3><div id="schedules"></div></div>
        </div>
      </section>

      <section id="tab-accounts" hidden>
        <div class="card">
          <h3>Instagram accounts</h3>
          <div class="table-scroll"><table class="data" id="accounts"></table></div>
          <p class="hint">Passwords are kept only in accounts.json on this Mac, readable only by you, and used only to log
            in. The panel never shows them again. Deleting an account keeps what it collected; add it again to pick up
            where it left off.</p>
        </div>
        <div class="card">
          <h3>Add an account</h3>
          <div class="row"><span class="label">Username</span>
            <input type="text" id="acc-user" autocomplete="off" spellcheck="false" placeholder="instagram_username"></div>
          <div class="row"><span class="label">Password</span>
            <input type="password" id="acc-pass" autocomplete="new-password"></div>
          <p class="hint">Optional: without one, PRAG asks for it, and for any security code, when it first logs in.
            Use accounts made for PRAG, not your personal one: in live runs the agent likes, saves and follows.</p>
          <div class="actions"><button class="btn primary" id="btn-acc-add">Add account</button></div>
        </div>
      </section>
    </main>
    <footer>Only this Mac can open this panel. Closing it (Ctrl+C in its terminal) doesn't stop a run in progress.</footer>
  </div>
</div>

<div class="overlay" id="acc-pass-modal" hidden>
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="acc-pass-title">
    <h3 id="acc-pass-title">Password</h3>
    <p class="hint" id="acc-pass-note"></p>
    <input type="password" id="acc-pass-new" autocomplete="new-password" aria-label="New password">
    <div class="actions"><button class="btn primary" id="acc-pass-save">Save password</button>
      <button class="btn danger" id="acc-pass-forget">Forget password</button>
      <button class="btn" id="acc-pass-cancel">Cancel</button></div>
  </div>
</div>
<div class="overlay" id="acc-del-modal" hidden>
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="acc-del-title">
    <h3 id="acc-del-title">Delete</h3>
    <p>PRAG stops using it and forgets its saved password and daily schedule. The reels, digests and bias it
      collected are kept, and come back if you add it again.</p>
    <label class="check"><input type="checkbox" id="acc-del-login"> Also delete its saved Instagram login
      (you'd log in again next time)</label>
    <div class="actions"><button class="btn danger" id="acc-del-confirm">Delete account</button>
      <button class="btn" id="acc-del-cancel">Cancel</button></div>
  </div>
</div>
<div class="overlay" id="prompt" hidden>
  <form class="modal" id="prompt-form" autocomplete="off" role="dialog" aria-modal="true" aria-labelledby="prompt-title">
    <h3 id="prompt-title">Instagram needs you</h3>
    <p id="prompt-message"></p>
    <input id="prompt-input" type="text" spellcheck="false" aria-label="Answer">
    <div class="actions"><button type="submit" class="btn primary">Send</button>
      <button type="button" class="btn" id="prompt-cancel">Cancel</button><span class="spacer"></span>
      <span class="hint" id="prompt-wait"></span></div>
    <p class="hint">A dialog box on the Mac asks the same thing: answer either one.</p>
  </form>
</div>
<div id="toast" role="status" hidden></div>

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
  if (spec) { el.textContent = spec[0]; el.className = "chip " + (spec[1] || ""); el.title = spec[2] || ""; }
}

const account = () => $("account").value || store.get("account", "");
const mode = () => (document.querySelector('input[name="eng"]:checked') || {}).value || DEFAULTS.engagement;
const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
const MODE_NAMES = {mixed: "Mixed", live: "Live", dry: "Dry run"};
const sentence = (text) => text.charAt(0).toUpperCase() + text.slice(1);
const isSession = (label) => /^(Session|Run) for/.test(label || "");  // a scrolling session, not a digest or a tool

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
  chip("chip-llm", s && (s.llm ? ["LLM ready", "ok"] : ["LLM off: quick rules steer", "warn",
       "LM Studio isn't serving the model. The quick rules steer, and the trained model scores reels if it was " +
       "trained for this bias."]));
  chip("chip-today", s && [`${s.minutes_used} of ${s.minutes_cap} min today`, s.minutes_used >= s.minutes_cap ? "warn" : ""]);
  chip("chip-live", s && [s.engaged_today ? "Live session done today" : "No live session yet today", ""]);
  chip("chip-power", s && s.battery.charge !== null &&
       [s.battery.on_battery ? `On battery, ${s.battery.charge}%` : "On power",
        s.battery.on_battery && s.battery.charge < DEFAULTS.min_battery ? "warn" : ""]);
  const pill = $("bias-pill");
  pill.hidden = !(s && s.bias);
  if (s && s.bias) pill.textContent = s.bias.name + (s.bias.persona ? ", persona" : "");
  renderForm();
  renderStatus();
  renderPrompt();

  const running = state.run.running;
  if (wasRunning && !running) {
    const code = state.run.exit, problem = code > 0 && code !== 130;
    toast(code === 130 || code < 0 ? "The run stopped." : problem ? "The run ended with a problem: see its output."
          : "The run finished.", problem);
    if (problem) setView("raw");
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
  $("eng-hint").textContent = sentence(MODES[m]) + "." + (m === "mixed" && s ? ` Next run: ${s.mixed_next}.` : "");
  if (!state) return;
  if (s) {
    const left = Math.max(0, s.minutes_cap - s.minutes_used);
    $("minutes-hint").textContent = `${left} of ${s.minutes_cap} min left today`;
    $("skip").hidden = !s.skip;
    $("skip").textContent = s.skip ? `Won't scroll now: ${s.skip}. Starting only rebuilds the digest.` : "";
  }
  const running = state.run.running, none = !state.account;
  $("btn-run").textContent = (effectiveMode() === "live" ? "Start a live session" : "Start a dry run") +
    ($("headless").checked ? ", headless" : "");
  $("btn-run").disabled = running || none;
  $("btn-digest").disabled = running || none;
  $("btn-stop").disabled = !running;
}

function renderStatus() {
  const r = state.run, bar = $("progress");
  const elapsed = r.started ? Math.max(0, Date.now() / 1000 - r.started) : 0;
  let word = "Ready", tone = "", detail = "No run yet. Pick a mode and start a session.", time = "";
  if (r.key && r.running) {
    word = isSession(r.label) ? "Scrolling" : "Working";
    tone = "live";
    detail = r.label;
    time = r.minutes ? `${clock(elapsed)} of ${clock(r.minutes * 60)}` : `${clock(elapsed)} so far`;
  } else if (r.key) {
    const code = r.exit;
    word = code === null || code === undefined || code === 0 ? "Finished" : code === 130 || code < 0 ? "Stopped" : "Problem";
    tone = word === "Problem" ? "bad" : "";
    detail = word === "Problem" ? `${r.label}: it ended with exit code ${code}. The raw output shows why.` : r.label;
    time = r.started ? "Started " + new Date(r.started * 1000).toLocaleString([], {weekday: "short", hour: "2-digit",
                                                                                  minute: "2-digit"}) : "";
  }
  if ($("run-state").textContent !== word) $("run-state").textContent = word;
  $("run-state").className = "state " + tone;
  $("run-status").textContent = detail;
  $("run-clock").textContent = time;
  // The line along the top of the window: how far into its minutes the session is, on every tab
  const total = r.running && r.minutes ? r.minutes * 60 : 0, determinate = total > 0 && elapsed < total;
  bar.hidden = !r.running;
  bar.classList.toggle("indeterminate", !!r.running && !determinate);
  bar.firstElementChild.style.width = determinate ? `${(100 * elapsed / total).toFixed(2)}%` : "";
  $("nav-run-meta").textContent = r.running ? clock(elapsed) : "";  // visible from every tab
  $("tab-run").classList.toggle("running", !!r.running);
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

// ── Output: the reels as rows, newest first, with the raw output a click away ──

// One line per reel, as main.py's describe() prints it, then (for reels it didn't skip) the summary line
const REEL = /^#(\d+)\s+(\S+)\s+(.*?)\s*→\s+(SKIP|WATCH|FULL_WATCH|REWATCH|OBSERVE)\s+([\d.]+)s\s*(.*)$/;
const SAID = /^ {8}“(.*)”\s*$/;
const KIND = /^([a-z_?]+) (\d+)s(?:\s+|$)(.*)$/;
const NOTE = /^\s*\p{Extended_Pictographic}️?\s+(.*)$/u;  // the run's milestones start with an emoji
const QUIET = /Database initialised|^Log: |python -m /;
const STAMPS = {SKIP: "Skipped", WATCH: "Watched", FULL_WATCH: "Watched fully", REWATCH: "Rewatched", OBSERVE: "Recorded"};
const AUDIO = {music: "Music, ", original: "Original audio, "};
const live ={last: null, reels: 0, watched: 0, liked: 0};

function verdictOf(text) {  // what main.py cut short ends in …
  let m;
  if ((m = /^(\d\.\d\d) (.*)$/.exec(text))) return {score: Number(m[1]), topic: m[2]};
  if ((m = /^rule (\d\.\d) · (.*)$/.exec(text))) return {score: Number(m[1]), topic: "Quick rule: " + m[2]};
  if ((m = /^gate: (.*)$/.exec(text))) return {topic: "Skip word: " + m[1]};
  if ((m = /^recorded(?:: (.*))?$/.exec(text))) return {topic: "Recorded for review" + (m[1] ? `: ${m[1]}` : "")};
  if ((m = /^error: (.*)$/.exec(text))) return {topic: "LLM error: " + m[1]};
  const plain = {"creator: never valuable": "A creator that's never been valuable", "not scored in time": "Not scored in time",
                 "no metadata": "No details to score", "recording …": "Recording", ad: ""};
  return {topic: Object.hasOwn(plain, text) ? plain[text] : text};
}

function engagementOf(marks) {
  const done = [["❤", "liked"], ["🔖", "saved"], ["➕", "followed"]].filter(([mark]) => marks.includes(mark))
    .map(([, word]) => word);
  const notes = done.length ? [sentence(done.join(", "))] : [];
  const would = /\(would ([a-z,]+)\)/.exec(marks);
  if (would) notes.push("Dry run: would " + would[1].split(",").join(", "));
  if (marks.includes("🧭")) notes.push("Watched to explore");
  return {text: notes.join(". "), liked: done.includes("liked")};
}

function meter(score) {
  const width = Math.round(Math.max(0, Math.min(1, score)) * 100);
  return h("div", {class: "meter", title: "Score against the bias, 0 to 1"},
    h("b", {}, h("i", {style: `width: ${width}%`})), h("span", {text: score.toFixed(2)}));
}

function reelRow(m) {
  const [, seq, who, middle, action, dwell, marks] = m;
  const kind = KIND.exec(middle), verdict = verdictOf((kind ? kind[3] : middle).trim());
  const ad = who === "(ad)", flicked = action === "SKIP" && marks.includes("⚡"), engaged = engagementOf(marks);
  const tone = ad || action === "OBSERVE" ? "other" : action === "SKIP" ? "skipped" : "watched";
  const handle = /^@[A-Za-z0-9._]{1,30}$/.test(who) ? who.slice(1) : null;  // not one that was cut short
  const name = ad ? "Ad" : who === "(unknown)" ? "Unknown creator" : who;
  const summary = h("div", {class: "summary", hidden: true});
  const row = h("div", {class: "v-row " + tone},
    h("span", {class: "v-seq", text: "#" + seq}),
    h("div", {class: "v-main"},
      h("div", {class: "v-head"},
        handle ? h("a", {class: "who", href: `https://www.instagram.com/${handle}/`, target: "_blank", rel: "noopener",
                         text: name}) : h("span", {class: "who", text: name}),
        kind ? h("span", {class: "kind", text: (AUDIO[kind[1]] || "") + clock(Number(kind[2]))}) : null),
      verdict.topic ? h("div", {class: "topic", text: verdict.topic}) : null,
      summary,
      engaged.text ? h("div", {class: "extras", text: engaged.text}) : null),
    h("div", {class: "v-verdict"},
      h("div", {class: "stamp", text: flicked ? "Flicked" : STAMPS[action],
                title: flicked ? "Known to be skippable before it appeared" : null}),
      h("div", {class: "dwell", text: `${dwell}s on screen`}),
      verdict.score === undefined ? null : meter(verdict.score)));
  row.summary = summary;
  return {row, tone, ad, liked: engaged.liked};
}

function clearStream() {
  $("stream").replaceChildren();
  Object.assign(live, {last: null, reels: 0, watched: 0, liked: 0});
  renderTally();
}

function addToStream(text, animate) {
  const box = $("stream"), added = [];
  for (const line of text.split("\n")) {
    let m;
    if ((m = REEL.exec(line))) {
      const reel = reelRow(m);
      if (!reel.ad) {
        live.reels += 1;
        live.watched += reel.tone === "watched";
        live.liked += reel.liked;
      }
      live.last = reel.row;
      added.push(reel.row);
    } else if ((m = SAID.exec(line)) && live.last && live.last.summary) {
      live.last.summary.textContent = `“${m[1]}”`;
      live.last.summary.hidden = false;
    } else if ((m = NOTE.exec(line)) && !QUIET.test(m[1])) {
      if (live.last && live.last.classList.contains("v-note")) {
        live.last.append(h("div", {text: m[1]}));  // milestones in a row read as one block, oldest first
      } else {
        live.last = h("div", {class: "v-note"}, h("div", {text: m[1]}));
        added.push(live.last);
      }
    } else if (/^ {4}\S/.test(line) && live.last && live.last.classList.contains("v-note") && !QUIET.test(line)) {
      const said = totals(line.trim());  // the lines under a milestone, like the session's totals
      if (said) live.last.append(h("div", {text: said}));
    }
  }
  for (const node of added) {
    if (animate) node.classList.add("fresh");
    box.prepend(node);
  }
  while (box.childElementCount > 400) box.lastElementChild.remove();
  renderTally();
}

function totals(text) {
  let m;
  if ((m = /^informative: (\d+) of (\d+) scored \(([^)]*)\) · valuable: (\d+)$/.exec(text))) {
    return `${m[1]} of ${m[2]} scored reels were informative (${m[3]}), ${m[4]} valuable`;
  }
  if ((m = /^engagement: \S+ (\d+) · \S+ (\d+) · \S+ (\d+)$/.exec(text))) {
    return `${m[1]} liked, ${m[2]} saved, ${m[3]} followed`;
  }
  return /^decisions:/.test(text) ? null : text;  // the watched and skipped counts are in the rows
}

function renderTally() {
  const n = live.reels;
  $("tally").textContent = n ? `${n} reel${n === 1 ? "" : "s"}, ${live.watched} watched` +
    (live.liked ? `, ${live.liked} liked` : "") : "";
}

function setView(view) {
  $("monitor").classList.toggle("show-raw", view === "raw");
  document.querySelectorAll("#view input").forEach((input) => {
    input.checked = input.value === view;
    input.closest("label").classList.toggle("on", input.checked);
  });
  if (view === "raw") $("console").scrollTop = $("console").scrollHeight;
}

async function pollOutput() {
  if (out.busy || !state || !state.run.key) return;
  out.busy = true;
  try {
    const d = await api(`/api/output?key=${encodeURIComponent(out.key || "")}&offset=${out.offset}`);
    const box = $("console");
    if (d.key !== out.key) setView(isSession(state.run.label) ? "reels" : "raw");  // a new run: its natural view
    if (d.reset) {
      box.textContent = "";
      clearStream();
    }
    out.key = d.key;
    out.offset = d.offset;
    if (d.text) {
      const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
      box.append(d.text);
      if (box.textContent.length > 400000) box.textContent = box.textContent.slice(-300000);
      if (atBottom) box.scrollTop = box.scrollHeight;
      addToStream(d.text, !d.reset);  // what arrives while you watch slides in; the backlog doesn't
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
  const news = (v.headlines || []).slice(0, 2).map((a) => h("li", {}, link(a.url, a.title),
                                                             h("span", {class: "src", text: a.source})));
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
    $("digest-stats").textContent = `${st.stories} stories from ${st.informative_reels} informative reels, out of ` +
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
      const note = item.loaded ? "Active." : "Not loaded: install it again.";
      return h("div", {class: "sched"},
        h("div", {}, h("div", {text: item.description}),
          h("div", {class: "hint", text: note + (item.last_exit ? ` The last run's exit code was ${item.last_exit}.` : "")})),
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

// ── Feed: what the account is being shown ──

let feedData = null;
const SVG_NS = "http://www.w3.org/2000/svg";
const pct = (x) => `${Math.round(x * 100)}%`;
const total = (counts) => Object.values(counts).reduce((a, b) => a + b, 0);

function svg(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function localTime(utc) {
  if (!utc) return "?";
  return new Date(utc.replace(" ", "T") + "Z").toLocaleString([], {weekday: "short", day: "numeric", month: "short",
                                                                   hour: "2-digit", minute: "2-digit"});
}

async function loadFeed() {
  try { feedData = await api("/api/feed?account=" + encodeURIComponent(account())); }
  catch (e) { toast(e.message, true); return; }
  renderFeed();
}

function renderFeed() {
  const d = feedData, c = d.curation, n = d.counts, fit = d.bias.persona ? "fit the persona" : "on target";
  const [headline, lede] = !n.reels
    ? ["No reels yet", "Run a session: every reel this account is shown lands here, split into the warm-up and the " +
       "curated feed."]
    : c.curated
    ? [`Curated since ${localTime(c.since.shown_at)}`, `It took ${c.since.index} judged reels. The ${n.curated} reels ` +
       `shown since then are the curated feed; the ${n["warm-up"]} before were the warm-up.`]
    : ["Still warming up", (c.enough
        ? `Of the latest ${c.window} judged reels, ${pct(c.share)} ${fit}; curated needs ${pct(c.target)}.`
        : `${c.judged} of the ${c.window} judged reels needed so far, ${pct(c.share)} ${fit} (curated needs ` +
          `${pct(c.target)}).`) + " Everything so far is warm-up."];
  $("feed-state").textContent = headline;
  $("feed-curation").textContent = lede;
  $("feed-uncat").hidden = !n.uncategorised;
  $("feed-uncat").textContent = `${n.uncategorised} reels aren't categorised yet. "Categorise now" asks the LLM ` +
    "(about a minute per 60 reels); daily runs also do it after each session.";
  $("btn-export-curated").disabled = $("btn-export-curated-csv").disabled = !c.curated;
  drawMix();
  drawCreators();
  fillFilters();
  drawReels();
}

function mixRows() {
  const d = feedData, warm = d.mix["warm-up"], cur = d.mix.curated;
  const wt = total(warm), ct = total(cur), both = d.curation.curated && ct > 0;
  const rows = Object.keys(d.categories).filter((k) => warm[k] || cur[k]).map((k) => ({
    label: d.categories[k], w: wt ? (warm[k] || 0) / wt : 0, c: ct ? (cur[k] || 0) / ct : 0,
    wn: warm[k] || 0, cn: cur[k] || 0,
  }));
  rows.sort((a, b) => (both ? b.c - a.c || b.w - a.w : b.w - a.w));
  return {rows, both, wt, ct};
}

function barPath(x, y, w, height, r) {  // rounded at the data end, square at the baseline
  r = Math.min(r, height / 2, w);
  return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + height - r} ` +
         `Q${x + w},${y + height} ${x + w - r},${y + height} H${x} Z`;
}

function drawMix() {
  const {rows, both, wt, ct} = mixRows(), box = $("mix-chart");
  $("mix-sub").textContent = both
    ? `Each category's share of the ${wt} warm-up and ${ct} curated reels (categorised, ads excluded), sorted by the curated feed.`
    : `Each category's share of the ${wt} categorised reels so far (ads excluded).`;
  $("mix-legend").replaceChildren(...(both ? [["--viz-before", "Warm-up"], ["--viz-after", "Curated"]] : []).map(
    ([token, label]) => h("span", {}, h("i", {style: `background: var(${token})`}), label)));
  box.replaceChildren();
  if (!rows.length) {
    box.append(h("p", {class: "hint", text: "No categorised reels yet."}));
    $("mix-table").replaceChildren();
    return;
  }
  const W = 900, labelW = 200, valueW = 52, rowH = 30, top = 6, axisH = 26;
  const x0 = labelW, x1 = W - valueW, plotH = rows.length * rowH, H = top + plotH + axisH;
  const max = Math.max(...rows.map((r) => Math.max(r.w, both ? r.c : 0)));
  const step = max > 0.5 ? 0.2 : max > 0.2 ? 0.1 : 0.05;
  const xmax = Math.max(step, Math.ceil(max / step - 1e-9) * step);
  const x = (v) => x0 + (x1 - x0) * v / xmax;
  const chart = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
                            "aria-label": "Share of each content category in the feed"});
  rows.forEach((r, i) => {  // row hit targets first, so the marks sit on top of the hover wash
    const hit = svg("rect", {x: 0, y: top + i * rowH, width: W, height: rowH, class: "hit", tabindex: "0",
      "aria-label": `${r.label}: ` + (both ? `warm-up ${pct(r.w)}, curated ${pct(r.c)}` : pct(r.w))});
    hit.addEventListener("pointermove", (e) => showTip(e, r, both));
    hit.addEventListener("focus", (e) => showTip(e, r, both));
    hit.addEventListener("pointerleave", hideTip);
    hit.addEventListener("blur", hideTip);
    chart.append(hit);
  });
  for (let i = 0, t = 0; t <= xmax + 1e-9; i++, t = i * step) {
    chart.append(svg("line", {x1: x(t), x2: x(t), y1: top, y2: top + plotH, "stroke-width": 1,
                              stroke: i === 0 ? "var(--viz-axis)" : "var(--viz-grid)", "pointer-events": "none"}));
    chart.append(svg("text", {x: x(t), y: H - 7, "text-anchor": "middle", class: "tick"}, pct(t)));
  }
  rows.forEach((r, i) => {
    const cy = top + i * rowH + rowH / 2;
    chart.append(svg("text", {x: labelW - 14, y: cy + 4, "text-anchor": "end", class: "cat", "pointer-events": "none"}, r.label));
    if (both) {
      chart.append(svg("line", {x1: x(r.w), x2: x(r.c), y1: cy, y2: cy, stroke: "var(--viz-link)", "stroke-width": 2,
                                "stroke-linecap": "round", "pointer-events": "none"}));
      for (const [value, token] of [[r.w, "--viz-before"], [r.c, "--viz-after"]]) {
        chart.append(svg("circle", {cx: x(value), cy, r: 5, fill: `var(${token})`, stroke: "var(--panel)",
                                    "stroke-width": 2, "pointer-events": "none"}));
      }
      chart.append(svg("text", {x: x1 + 12, y: cy + 4, class: "val", "pointer-events": "none"}, pct(r.c)));
    } else {
      const w = x(r.w) - x0;
      if (w > 0) chart.append(svg("path", {d: barPath(x0, cy - 7, w, 14, 4), fill: "var(--viz-bar)", "pointer-events": "none"}));
      chart.append(svg("text", {x: x(r.w) + 8, y: cy + 4, class: "val", "pointer-events": "none"}, pct(r.w)));
    }
  });
  box.append(chart, h("div", {class: "tip", id: "mix-tip", hidden: true}));
  $("mix-table").replaceChildren(
    h("tr", {}, h("th", {text: "Category"}), h("th", {class: "num", text: both ? "Warm-up" : "Share"}),
      both ? h("th", {class: "num", text: "Curated"}) : null),
    ...rows.map((r) => h("tr", {}, h("td", {text: r.label}), h("td", {class: "num", text: `${pct(r.w)} (${r.wn})`}),
      both ? h("td", {class: "num", text: `${pct(r.c)} (${r.cn})`}) : null)));
}

function showTip(event, r, both) {
  const tip = $("mix-tip"), box = $("mix-chart").getBoundingClientRect();
  const lines = both ? [["--viz-before", "warm-up", r.w, r.wn], ["--viz-after", "curated", r.c, r.cn]]
                     : [["--viz-bar", "so far", r.w, r.wn]];
  tip.replaceChildren(h("div", {class: "title", text: r.label}), ...lines.map(([token, name, share, count]) =>
    h("div", {class: "line"}, h("span", {class: "key", style: `background: var(${token})`}), h("b", {text: pct(share)}),
      h("span", {class: "name", text: `${name}, ${count} reels`}))));
  tip.hidden = false;
  const at = event.clientX !== undefined && event.type !== "focus" ? event
    : {clientX: box.left + box.width / 2, clientY: event.target.getBoundingClientRect().top};
  const left = Math.min(at.clientX - box.left + 14, box.width - tip.offsetWidth - 4);
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${at.clientY - box.top + 14}px`;
}

function hideTip() { const tip = $("mix-tip"); if (tip) tip.hidden = true; }

function drawCreators() {
  const d = feedData;
  $("creators-sub").textContent = d.curation.curated ? "In the curated feed." : "So far: the feed isn't curated yet.";
  $("creators").replaceChildren(
    h("tr", {}, h("th", {text: "Creator"}), h("th", {class: "num", text: "Reels"}), h("th", {text: "Usually"}),
      h("th", {class: "num", text: "Watched"})),
    ...d.creators.map((c) => h("tr", {}, h("td", {}, link(`https://www.instagram.com/${encodeURIComponent(c.creator)}/`, "@" + c.creator)),
      h("td", {class: "num", text: String(c.reels)}), h("td", {text: c.category ? d.categories[c.category] : "—"}),
      h("td", {class: "num", text: pct(c.watched)}))));
}

function fillFilters() {
  const select = $("f-cat"), keep = select.value;
  select.replaceChildren(h("option", {value: "", text: "All categories"}),
    ...Object.entries(feedData.categories).map(([key, label]) => h("option", {value: key, text: label})),
    h("option", {value: "_none", text: "Not categorised"}), h("option", {value: "_ad", text: "Ads"}));
  select.value = keep;
}

function drawReels() {
  const cat = $("f-cat").value, phase = $("f-phase").value;
  const shown = feedData.reels.filter((r) => (!phase || r.phase === phase) &&
    (!cat || (cat === "_ad" ? r.is_ad : cat === "_none" ? !r.category && !r.is_ad : r.category === cat)));
  $("f-count").textContent = `${shown.length} of the latest ${feedData.reels.length} reels`;
  $("reels").replaceChildren(...shown.slice(0, 250).map((r) => h("div", {class: "reel-row"},
    h("span", {class: "when", text: localTime(r.shown_at)}),
    link(r.url, "@" + (r.creator || "?")),
    h("span", {}, h("span", {class: "tag-cat", text: r.is_ad ? "Ad" : r.category ? feedData.categories[r.category] : "Not categorised"})),
    h("span", {}, r.topic ? h("strong", {text: r.topic + ": "}) : null, r.summary || ""),
    h("span", {class: "num"}, h("span", {text: STAMPS[r.decision] || ""}),
      h("span", {text: r.value_score == null ? "" : Number(r.value_score).toFixed(2)})))));
}

async function exportFeed(phase, format) {
  try {
    const res = await fetch(`/api/feed/export?account=${encodeURIComponent(account())}&phase=${phase}&format=${format}`,
                            {headers: {"X-PRAG-Token": TOKEN}, cache: "no-store"});
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
    const url = URL.createObjectURL(await res.blob());
    const a = h("a", {href: url, download: `${account()}-${phase}.${format}`});
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  } catch (e) { toast(e.message, true); }
}

async function categorizeNow() {
  try {
    await api("/api/feed/categorize", {account: account()});
    toast("Categorising: follow it on the Run tab.");
    showTab("run");
    await refresh();
    pollOutput();
  } catch (e) { toast(e.message, true); }
}

// ── Accounts ──

let accountTarget = null;

async function loadAccounts() {
  let items;
  try { items = (await api("/api/accounts")).items; } catch (e) { toast(e.message, true); return; }
  const head = h("tr", {}, ...["Account", "Password", "Status", "Bias", "Reels", "Last used", "Instagram login", ""]
    .map((text, i) => h("th", {class: i === 4 ? "num" : null, text})));
  $("accounts").replaceChildren(head, ...(items.length ? items.map((a) => {
    const change = h("button", {class: "linkish", text: "Change password"});
    change.onclick = () => openPasswordModal(a);
    const remove = h("button", {class: "linkish danger", text: "Delete"});
    remove.onclick = () => openDeleteModal(a);
    const status = a.status === "ACTIVE" ? "Active" : a.status.charAt(0) + a.status.slice(1).toLowerCase().replace("_", " ");
    return h("tr", {},
      h("td", {}, h("strong", {text: "@" + a.username})),
      h("td", {text: a.has_password ? "Saved" : "Not saved"}),
      h("td", {text: status}),
      h("td", {text: a.bias}),
      h("td", {class: "num", text: String(a.reels)}),
      h("td", {text: a.last_used ? localTime(a.last_used) : "Never"}),
      h("td", {text: a.login_saved ? "Saved" : "Not yet"}),
      h("td", {}, h("span", {class: "row-actions"}, change, remove)));
  }) : [h("tr", {}, h("td", {colspan: "8", class: "hint", text: "No accounts yet: add one below."}))]));
}

async function addAccount() {
  const username = $("acc-user").value.trim().replace(/^@/, ""), password = $("acc-pass").value;
  if (!username) { toast("Type the username first.", true); return; }
  try {
    await api("/api/accounts/add", {username, password});
    $("acc-user").value = "";
    $("acc-pass").value = "";
    toast(`Added @${username}.`);
    await refresh();
    loadAccounts();
  } catch (e) { toast(e.message, true); }
}

function openPasswordModal(a) {
  accountTarget = a.username;
  $("acc-pass-title").textContent = `Password for @${a.username}`;
  $("acc-pass-note").textContent = a.has_password ? "A password is saved. Type a new one to replace it."
    : "No password is saved: type one, or leave it to the login prompt.";
  $("acc-pass-forget").hidden = !a.has_password;
  $("acc-pass-new").value = "";
  $("acc-pass-modal").hidden = false;
  $("acc-pass-new").focus();
}

function closeModals() {
  $("acc-pass-new").value = "";
  $("acc-pass-modal").hidden = true;
  $("acc-del-modal").hidden = true;
  accountTarget = null;
}

async function savePassword(forget) {
  const password = $("acc-pass-new").value;
  if (!forget && !password) { $("acc-pass-new").focus(); return; }
  const username = accountTarget;
  try {
    await api("/api/accounts/password", forget ? {username, remove: true} : {username, password});
    toast(forget ? `Forgot the saved password for @${username}.` : `Saved the new password for @${username}.`);
    closeModals();
    loadAccounts();
  } catch (e) { toast(e.message, true); }
}

function openDeleteModal(a) {
  accountTarget = a.username;
  $("acc-del-title").textContent = `Delete @${a.username}?`;
  $("acc-del-login").checked = false;
  $("acc-del-login").closest("label").hidden = !a.login_saved;
  $("acc-del-modal").hidden = false;
}

async function deleteAccount() {
  const username = accountTarget;
  try {
    const done = await api("/api/accounts/delete", {username, delete_login: $("acc-del-login").checked});
    toast(`Deleted @${username}` + (done.login_deleted ? " and its saved login." : "."));
    closeModals();
    await refresh();
    loadAccounts();
  } catch (e) { toast(e.message, true); }
}

// ── Bias ──

let biasData = null;
const BIAS_FIELDS = {name: "bias-name", interests: "bias-interests", watch_words: "bias-watch",
                     skip_words: "bias-skip", creator_hints: "bias-hints", seeds: "bias-seeds"};
const LIST_FIELDS = ["watch_words", "skip_words", "creator_hints", "seeds"];

function biasForm() {
  const form = Object.fromEntries(Object.entries(BIAS_FIELDS).map(([key, id]) => [key, $(id).value]));
  form.persona = (document.querySelector('input[name="scoring"]:checked') || {}).value === "persona";
  return form;
}

function canonical(bias) {  // how the server will store it, to spot unsaved changes
  const list = (v) => (Array.isArray(v) ? v : String(v).split(/[,\n]/))
    .map((w) => w.trim().replace(/^[#@]+/, "")).filter(Boolean)
    .map((w) => (/^[\x00-\x7f]*$/.test(w) ? w.toLowerCase() : w));
  return JSON.stringify([String(bias.name).trim(), String(bias.interests).trim(), !!bias.persona,
                         ...LIST_FIELDS.map((k) => list(bias[k]))]);
}

function fillBias(bias) {
  $("bias-name").value = bias.name;
  $("bias-interests").value = bias.interests.trim();
  for (const key of LIST_FIELDS) $(BIAS_FIELDS[key]).value = bias[key].join(", ");
  document.querySelector(`input[name="scoring"][value="${bias.persona ? "persona" : "informative"}"]`).checked = true;
  biasChanged();
}

function biasChanged() {
  if (!biasData) return;
  const form = biasForm(), interests = form.interests.trim();
  document.querySelectorAll("#presets button").forEach((b) => {
    const preset = biasData.presets[b.dataset.key];
    b.classList.toggle("on", preset.interests.trim() === interests && !!preset.persona === form.persona);
  });
  document.querySelectorAll("#bias-scoring label").forEach((l) => l.classList.toggle("on", l.querySelector("input").checked));
  $("interests-label").textContent = form.persona ? "Who this person is and what they engage with" : "What to look for";
  $("scoring-hint").textContent = form.persona
    ? "Scores whatever this person would watch to the end and engage with, opinion and memes included: for studying " +
      "what the algorithm feeds someone like them. Describe the whole person below, not just one side of them."
    : "News and knowledge score high: for a feed of useful information.";
  $("bias-dirty").hidden = canonical(form) === canonical(biasData.bias);
  const fits = biasData.model_interests !== null && biasData.model_interests.trim() === interests &&
               !!biasData.model_persona === form.persona;
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
  $("bias-title").textContent = biasData.bias.name + (biasData.bias.persona ? ", a persona" : "");
  $("bias-state").textContent = biasData.own ? `@${account()}'s own bias. Sessions steer the feed towards it.`
    : `@${account()} uses the default bias, ${biasData.default_name}. Save to give it its own.`;
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
  document.querySelectorAll("nav button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
    if (b.dataset.tab === name) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  document.querySelectorAll("main > section").forEach((s) => { s.hidden = s.id !== "tab-" + name; });
  if (name === "digest") loadDigest();
  if (name === "report") loadReport();
  if (name === "schedule") loadSchedules();
  if (name === "bias") loadBias();
  if (name === "feed") loadFeed();
  if (name === "accounts") loadAccounts();
}

// ── The look: two themes, the collapsible rail, and the contour field behind it ──

const THEMES = ["default", "brutal"];  // black and white; cream, orange and navy

function applyTheme(name) {
  if (!THEMES.includes(name)) name = "default";
  if (name === "default") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", name);
  store.set("theme", name);
  document.querySelectorAll("#theme-menu .menu-item").forEach((b) => {
    b.classList.toggle("active", b.dataset.set === name);
    b.setAttribute("aria-checked", String(b.dataset.set === name));
  });
  Contours.recolor();
}

function toggleThemeMenu(open) {
  const menu = $("theme-menu"), show = open === undefined ? menu.hidden : open;
  menu.hidden = !show;
  $("theme-btn").setAttribute("aria-expanded", String(show));
}

function setCollapsed(collapsed) {
  const button = $("collapse");
  $("sidebar").classList.toggle("collapsed", collapsed);
  store.set("collapsed", collapsed);
  button.setAttribute("aria-expanded", String(!collapsed));
  button.title = collapsed ? "Expand the menu" : "Collapse the menu";
  button.setAttribute("aria-label", button.title);
}

// The drifting topography from the KIDRAAH_ workspace's rail (its shader.js), in plain WebGL: decoration only.
const CONTOUR_SHADER = `
  #ifdef GL_FRAGMENT_PRECISION_HIGH
  precision highp float;
  #else
  precision mediump float;
  #endif
  uniform float u_time; uniform vec2 u_resolution; uniform vec3 u_color; uniform float u_alpha;
  vec4 permute(vec4 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0); const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy)); vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz); vec3 l = 1.0 - g; vec3 i1 = min(g.xyz, l.zxy); vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx; vec3 x2 = x0 - i2 + 2.0 * C.xxx; vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
    i = mod(i, 289.0);
    vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                     + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 1.0 / 7.0; vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z); vec4 x_ = floor(j * ns.z); vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy; vec4 y = y_ * ns.x + ns.yyyy; vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy); vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0; vec4 s1 = floor(b1) * 2.0 + 1.0; vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy; vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x); vec3 p1 = vec3(a0.zw, h.y); vec3 p2 = vec3(a1.xy, h.z); vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m; return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
  }
  void main() {
    vec2 st = gl_FragCoord.xy / u_resolution.xy; st.x *= u_resolution.x / u_resolution.y;
    vec2 pos = st * 1.5; vec2 warp = vec2(0.0);
    warp.x = snoise(vec3(pos.x, pos.y, u_time * 0.005)) * 0.8;
    warp.y = snoise(vec3(pos.x + 5.2, pos.y + 1.3, u_time * 0.008)) * 0.8;
    pos += warp;
    float n = snoise(vec3(pos.x, pos.y, u_time * 0.01)) * 0.5 + 0.5;
    float line = fract(n * 8.0);
    float contour = smoothstep(0.0, 0.06, line) * (1.0 - smoothstep(0.88, 1.0, line));
    float a = contour * u_alpha;
    gl_FragColor = vec4(u_color * a, a);
  }`;

const Contours = {
  gl: null, u: null, time: 0, raf: null, last: 0, still: false,

  init() {
    const canvas = $("contours"), gl = canvas.getContext("webgl", {alpha: true, antialias: true});
    if (!gl) return;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
    };
    const vertex = compile(gl.VERTEX_SHADER, "attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }");
    const fragment = compile(gl.FRAGMENT_SHADER, CONTOUR_SHADER);
    if (!vertex || !fragment) return;
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const corner = gl.getAttribLocation(program, "p");
    gl.enableVertexAttribArray(corner);
    gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
    this.gl = gl;
    this.canvas = canvas;
    this.u = Object.fromEntries(["u_time", "u_resolution", "u_color", "u_alpha"].map((n) => [n, gl.getUniformLocation(program, n)]));
    this.still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.addEventListener("resize", () => this.resize());
    document.addEventListener("visibilitychange", () => (document.hidden ? this.stop() : this.start()));
    this.resize();
    this.recolor();
    this.start();
  },

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2), width = 248, height = window.innerHeight;
    this.canvas.width = width * ratio;
    this.canvas.height = height * ratio;
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.gl.uniform2f(this.u.u_resolution, width, height);  // CSS pixels, as the original does
    this.draw();
  },

  recolor() {
    if (!this.gl) return;
    const css = getComputedStyle(document.documentElement);
    const [r, g, b] = css.getPropertyValue("--contour").split(",").map((n) => parseFloat(n) / 255);
    this.gl.uniform3f(this.u.u_color, r, g, b);
    this.gl.uniform1f(this.u.u_alpha, parseFloat(css.getPropertyValue("--contour-alpha")) || 0.2);
    this.draw();
  },

  draw() {
    if (!this.gl) return;
    this.gl.uniform1f(this.u.u_time, this.time);
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
  },

  start() {  // it drifts, slowly; paused in a hidden tab, and still when motion is reduced
    if (!this.gl || this.raf || this.still) return;
    this.last = performance.now();
    const tick = (now) => {
      this.raf = requestAnimationFrame(tick);
      if (now - this.last < 40) return;  // 25 frames a second is plenty for a drift this slow
      this.time += (now - this.last) / 1000;
      this.last = now;
      this.draw();
    };
    this.raf = requestAnimationFrame(tick);
  },

  stop() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
  },
};

// ── Start ──

function init() {
  Contours.init();
  applyTheme(store.get("theme", "default"));
  setCollapsed(!!store.get("collapsed", false));
  $("collapse").onclick = () => setCollapsed(!$("sidebar").classList.contains("collapsed"));
  $("theme-btn").onclick = (e) => { e.stopPropagation(); toggleThemeMenu(); };
  document.querySelectorAll("#theme-menu .menu-item").forEach((b) => {
    b.onclick = () => { applyTheme(b.dataset.set); toggleThemeMenu(false); $("theme-btn").focus(); };
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".sidebar-bottom")) toggleThemeMenu(false); });

  const seg = $("engagement");
  for (const [value, label] of Object.entries(MODE_NAMES)) {
    seg.append(h("label", {title: sentence(MODES[value])}, h("input", {type: "radio", name: "eng", value}), label));
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
  $("sch-eng").replaceChildren(...Object.entries(MODES).map(([m, what]) => h("option", {value: m, text: `${MODE_NAMES[m] || m}: ${what}`})));
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
  document.querySelectorAll('input[name="scoring"]').forEach((r) => r.addEventListener("change", biasChanged));
  $("btn-categorize").onclick = categorizeNow;
  $("btn-export-curated").onclick = () => exportFeed("curated", "jsonl");
  $("btn-export-curated-csv").onclick = () => exportFeed("curated", "csv");
  $("btn-export-all").onclick = () => exportFeed("all", "jsonl");
  $("f-cat").onchange = drawReels;
  $("f-phase").onchange = drawReels;
  $("btn-acc-add").onclick = addAccount;
  $("acc-pass-save").onclick = () => savePassword(false);
  $("acc-pass-forget").onclick = () => savePassword(true);
  $("acc-del-confirm").onclick = deleteAccount;
  for (const id of ["acc-pass-cancel", "acc-del-cancel"]) $(id).onclick = closeModals;
  $("acc-pass-new").addEventListener("keydown", (e) => { if (e.key === "Enter") savePassword(false); });
  $("acc-pass").addEventListener("keydown", (e) => { if (e.key === "Enter") addAccount(); });
  document.querySelectorAll("nav button").forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });
  $("bias-pill").onclick = () => showTab("bias");
  $("go-accounts").onclick = () => showTab("accounts");
  $("view").addEventListener("change", (e) => setView(e.target.value));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && (!$("acc-pass-modal").hidden || !$("acc-del-modal").hidden)) closeModals();
    if (e.key === "Escape" && !$("theme-menu").hidden) { toggleThemeMenu(false); $("theme-btn").focus(); }
  });

  setView("reels");
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
