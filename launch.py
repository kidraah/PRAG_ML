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
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from dotenv import load_dotenv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import daily
from agent import bias as biases
from automation import prompt
from dashboard import feed as feed_study
from ml import feed_model
from tools import headtohead
from database import accounts as account_store
from config import DAILY_ENGAGEMENT, DAILY_MAX_MINUTES, ENGAGEMENT_MODES, LLM_MODEL_NAME, MIN_BATTERY, SESSION_MINUTES
from database.db import DEMO_DB_PATH, SOURCE, current_db, engaged_today, get_connection, init_db, list_accounts, minutes_today
from ml.scorer import ModelScorer
from perception.local_evaluator import llm_serving
from stories.digest import DIGEST_DIR
from tools import schedule
from gradient import api as gradient_api
from gradient import simulate as demo
from gradient import panel as gradient_panel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))  # OPENROUTER_API_KEY, for the Jev head-to-head
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


def feed_rows(account: str | None) -> list[dict]:
    """Every reel the account (None: every account) was shown, with its phase, kept until the database changes, so
    paging through 12,000 reels doesn't reload them for every page."""
    path = current_db()
    stamp = tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in (path, path + "-wal"))
    key = f"feed-rows|{path}|{account}"
    hit = _cache.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    conn = get_connection()
    try:
        if account:
            rows, _ = feed_study.load_feed(conn, account)
            for row in rows:
                row["account"] = account
        else:
            rows = [r for _a, found, _s in feed_study.all_feeds(conn) for r in found]
    finally:
        conn.close()
    _cache[key] = (stamp, rows)
    return rows


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
        "bias": {"name": bias.name, "persona": bias.persona, "own": biases.has_own(account)},
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
    demo_source = SOURCE.get() == "demo"
    folder = os.path.join(demo.DEMO_DIGEST_DIR if demo_source else DIGEST_DIR, account)
    dates = sorted((os.path.basename(p)[:-5] for p in glob.glob(os.path.join(folder, "*.json"))), reverse=True)
    if not dates:
        return {"dates": [], "date": None, "digest": None, "page": None}
    date = date if date in dates else dates[0]
    with open(os.path.join(folder, f"{date}.json")) as fh:
        return {"dates": dates, "date": date, "digest": json.load(fh), "demo": demo_source,
                "page": f"/{'demo-digest' if demo_source else 'digest'}/{account}/{date}.html"}


def bias_info(account: str) -> dict:
    current, presets = biases.load(account), biases.presets()
    trained = ModelScorer.trained_for(bias=current)
    model_bias = next((b for b in [current, biases.DEFAULT_BIAS, *presets.values(), *biases.BUILT_IN.values()]
                       if b.fingerprint == trained), None)
    return {
        "bias": current.to_dict(),
        "own": biases.has_own(account),
        "presets": {key: preset.to_dict() for key, preset in presets.items()},
        "presets_edited": os.path.exists(biases.presets_path()),
        "model_interests": model_bias.interests if model_bias else None,  # the fallback model scores only for these
        "model_name": model_bias.name if model_bias else None,
        "model_persona": model_bias.persona if model_bias else False,
        "default_name": biases.DEFAULT_BIAS.name,
    }


def report_text(account: str) -> str:
    db_args = ["--db", DEMO_DB_PATH] if SOURCE.get() == "demo" else []
    proc = subprocess.run([sys.executable, "-m", "dashboard.report", "--account", account, *db_args], cwd=PROJECT_ROOT,
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

    def _source(self) -> str:
        """Which data this request reads: "demo" (the simulated sessions) or "live" (the real ones)."""
        return "demo" if self.headers.get("X-PRAG-Source") == "demo" else "live"

    def _handle(self, routes: dict, path: str, arg) -> None:
        handler = routes.get(path)
        if handler is None:
            return self._json({"error": "Not found"}, 404)
        token = SOURCE.set(self._source())
        try:
            self._json(handler(arg))
        except (ValueError, TypeError, RuntimeError, argparse.ArgumentTypeError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001 — show it on the page rather than drop the request
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        finally:
            SOURCE.reset(token)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if not self._allowed(api=url.path.startswith("/api/")):
            return
        if url.path == "/":
            return self._send(200, page(), "text/html; charset=utf-8", {"Content-Security-Policy": PAGE_CSP})
        if url.path.startswith("/digest/"):
            return self._digest_file(url.path)
        if url.path.startswith("/demo-digest/"):
            return self._digest_file(url.path, demo.DEMO_DIGEST_DIR)
        if url.path.startswith("/fonts/"):
            return self._font_file(url.path[len("/fonts/"):])
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path == "/api/feed/export":
            token = SOURCE.set(self._source())
            try:
                return self._feed_export(query)
            finally:
                SOURCE.reset(token)
        self._handle({"/api/state": self.state, "/api/output": self.output, "/api/digest": self.digest,
                      "/api/report": self.report, "/api/schedules": self.schedules, "/api/bias": self.bias,
                      "/api/feed": self.feed, "/api/feed/reels": self.feed_reels, "/api/feed/reel": self.feed_reel,
                      "/api/accounts": self.account_list, "/api/h2h": self.h2h,
                      "/api/demo/sessions": self.demo_sessions, "/api/demo/log": self.demo_log,
                      **gradient_api.GET}, url.path, query)

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
                      "/api/feed/categorize": self.categorize, "/api/feed/model": self.train_feed_model,
                      "/api/accounts/add": self.add_account,
                      "/api/accounts/password": self.account_password, "/api/accounts/delete": self.delete_account,
                      "/api/presets/save": self.save_preset, "/api/presets/delete": self.delete_preset,
                      "/api/presets/restore": self.restore_presets,
                      "/api/h2h/label": self.h2h_label, "/api/h2h/run": self.h2h_run,
                      "/api/schedule/install": self.install_schedule,
                      "/api/schedule/remove": self.remove_schedule, **gradient_api.POST}, url.path, body)

    def _digest_file(self, path: str, root: str = DIGEST_DIR) -> None:
        parts = path.split("/")  # "", "digest", account, file
        if len(parts) == 4 and parts[2] in active_accounts() and DIGEST_FILE.fullmatch(parts[3]):
            file = os.path.join(root, parts[2], parts[3])
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
        """One account's feed, or (scope=all) every account's side by side; with the feed model's topics."""
        account = None if query.get("scope") == "all" else checked_account(query.get("account"))
        day = query.get("day") or None
        if day and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("day must look like 2026-10-14")
        demo_source = SOURCE.get() == "demo"
        conn = get_connection()
        try:
            info = feed_study.summary(conn, account, day=day) if account else feed_study.summary_all(conn)
            info["topics"] = (feed_model.analysis(conn, account, path=demo.DEMO_FEED_MODEL) if demo_source
                              else feed_model.analysis(conn, account))
            info["demo"] = demo_source
            return info
        finally:
            conn.close()

    def feed_reels(self, query: dict) -> dict:
        """One page of every reel shown (newest first), filtered by day, phase, category, account or text."""
        account = None if query.get("scope") == "all" else checked_account(query.get("account"))
        day = query.get("day") or None
        if day and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("day must look like 2026-10-14")
        who = query.get("who") or None
        if who and who not in active_accounts():
            raise ValueError("Unknown account")
        return feed_study.reel_page(feed_rows(account), day=day, phase=query.get("phase") or None,
                                    category=query.get("cat") or None, account=who, query=(query.get("q") or "")[:100],
                                    offset=int(query.get("offset") or 0), limit=min(int(query.get("limit") or 50), 200))

    def feed_reel(self, query: dict) -> dict:
        """Everything recorded about one reel shown: what was perceived, how it was judged, what was done."""
        try:
            session, seq = int(query.get("session", "")), int(query.get("seq", ""))
        except ValueError:
            raise ValueError("Pick a reel.") from None
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT i.*, s.account, s.scorer, s.model FROM impressions i JOIN sessions s ON "
                               "s.id = i.session_id WHERE i.session_id = ? AND i.seq = ?", (session, seq)).fetchone()
            if row is None:
                raise ValueError("No such reel.")
            out = {k: row[k] for k in row.keys()}
            simulated = SOURCE.get() == "demo"
            if simulated:
                source = conn.execute("SELECT * FROM demo_items WHERE code = ?", (row["code"],)).fetchone()
                out["source"] = dict(source) if source else None
                out["url"] = out["source"]["news_url"] if source and source["news_url"] else None
        finally:
            conn.close()
        out["visual"] = json.loads(out["visual"]) if out.get("visual") else None
        out["simulated"] = simulated
        return out

    def _feed_export(self, query: dict) -> None:
        """The account's dataset as a download: the curated feed or everything, as JSONL or CSV."""
        try:
            account = None if query.get("scope") == "all" else checked_account(query.get("account"))
            fmt = query.get("format", "jsonl")
            if fmt not in ("jsonl", "csv"):
                raise ValueError("format must be jsonl or csv")
            curated = query.get("phase", "curated") == "curated"
            conn = get_connection()
            try:
                records = (feed_study.dataset(conn, account, curated_only=curated) if account
                           else feed_study.dataset_all(conn, curated_only=curated))
            finally:
                conn.close()
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        body = feed_study.to_jsonl(records) if fmt == "jsonl" else feed_study.to_csv(records)
        name = f"{account or 'all-accounts'}-{'curated' if curated else 'all'}.{fmt}"
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
        profile = body["profile"] if "profile" in body else biases.load(account).profile  # the form doesn't edit it
        biases.save(account, biases.validated({**body, "persona": True, "profile": profile}))
        return bias_info(account)

    @staticmethod
    def _presets(key: str | None = None) -> dict:
        return {"key": key, "presets": {k: p.to_dict() for k, p in biases.presets().items()},
                "presets_edited": os.path.exists(biases.presets_path())}

    def save_preset(self, body: dict) -> dict:
        key = body.get("key")
        if key is not None and not biases.PRESET_KEY.fullmatch(str(key)):
            raise ValueError("Unknown preset")
        preset = body.get("preset")
        if not isinstance(preset, dict):
            raise ValueError("Expected the persona to keep as a preset")
        return self._presets(biases.save_preset(biases.validated({**preset, "persona": True}), key))

    def delete_preset(self, body: dict) -> dict:
        biases.delete_preset(str(body.get("key", "")))
        return self._presets()

    def restore_presets(self, _body: dict) -> dict:
        biases.restore_presets()
        return self._presets()

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

    @staticmethod
    def _demo_conn():
        if not os.path.exists(DEMO_DB_PATH):
            raise ValueError("No simulated sessions yet: run python -m gradient.simulate")
        conn = get_connection(DEMO_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def demo_sessions(self, query: dict) -> dict:
        """Every simulated session of the account, by simulated day."""
        account = checked_account(query.get("account"))
        conn = self._demo_conn()
        try:
            days: dict[str, list] = {}
            for r in conn.execute("SELECT * FROM demo_sessions WHERE account = ? ORDER BY simulated_start", (account,)):
                days.setdefault(r["simulated_date"], []).append({
                    "id": r["session_id"], "start": r["simulated_start"], "end": r["simulated_end"], "kind": r["kind"],
                    "generated_at": r["generated_at"], **json.loads(r["stats"] or "{}")})
            return {"account": account, "days": [{"date": d, "sessions": v} for d, v in days.items()]}
        finally:
            conn.close()

    def demo_log(self, query: dict) -> dict:
        """One simulated session's console log, each line with its simulated time."""
        try:
            session = int(query.get("session", ""))
        except ValueError:
            raise ValueError("Pick a session.") from None
        conn = self._demo_conn()
        try:
            info = conn.execute("SELECT * FROM demo_sessions WHERE session_id = ?", (session,)).fetchone()
            if info is None:
                raise ValueError("No such simulated session.")
            lines = [{"at": r["at"], "line": r["line"]} for r in
                     conn.execute("SELECT at, line FROM demo_log WHERE session_id = ? ORDER BY n", (session,))]
            return {"session": dict(info), "lines": lines}
        finally:
            conn.close()

    def h2h(self, _query: dict) -> dict:
        """The head-to-head sample for labelling (the models' scores stay hidden), your labels, and the results."""
        run = headtohead.load_run()
        if run is None:
            return {"exists": False}
        labels = headtohead.load_labels()
        keep = ("code", "url", "creator", "caption", "cover_text", "transcript", "transcript_language", "audio_type",
                "duration", "topic", "summary", "category", "hindi")
        return {"exists": True, "personas": run["personas"], "labels": labels,
                "reels": [{k: r.get(k) for k in keep} for r in run["reels"]],
                "results": headtohead.results(run, labels), "errors": len(run.get("errors", []))}

    def h2h_label(self, body: dict) -> dict:
        label = body.get("label")
        headtohead.save_label(str(body.get("persona", "")), str(body.get("code", "")),
                              None if label is None else str(label))
        return self.h2h({})

    def h2h_run(self, _body: dict) -> dict:
        return self.runner._launch([sys.executable, "-m", "tools.headtohead"], "Scoring the head-to-head")

    def train_feed_model(self, _body: dict) -> dict:
        return self.runner._launch([sys.executable, "-m", "ml.feed_model", "--train"], "Training the feed model")

    def seed(self, body: dict) -> dict:
        account = checked_account(body.get("account"))
        handles = biases.load(account).seeds
        if not handles:
            raise ValueError("Add accounts to follow first, and save the persona.")
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
    return gradient_panel.inject(PAGE.replace("__TOKEN__", script_json(TOKEN)).replace("__MODES__", script_json(ENGAGEMENT_MODES))
                                 .replace("__DEFAULTS__", script_json(defaults)), LLM_MODEL_NAME)


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
.reel-row { cursor: pointer; }
.reel-row:hover, .reel-row.open { background: var(--hover); }
.pager { display: flex; align-items: center; justify-content: center; gap: 18px; margin-top: 16px; }
.inspect { border: 2px solid var(--line); border-top: 0; background: var(--panel); padding: 14px 18px 18px; margin-bottom: 8px;
           display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 4px 30px; font-size: 15px; }
.inspect h4 { font: 500 10px var(--display); letter-spacing: 1.4px; text-transform: uppercase; color: var(--muted); margin: 12px 0 4px; }
.inspect p { margin: 0; overflow-wrap: anywhere; white-space: pre-wrap; }
.inspect dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 14px; margin: 0; }
.inspect dt { color: var(--muted); font: 500 12px var(--mono); }
.inspect dd { margin: 0; font: 500 13px var(--mono); overflow-wrap: anywhere; }
.swatches { display: flex; gap: 6px; margin-top: 6px; }
.swatches i { display: block; height: 22px; border: 1px solid var(--line); }
@media (max-width: 900px) { .inspect { grid-template-columns: minmax(0, 1fr); } }
#feed-scope { margin-bottom: 16px; }
.persona-text { white-space: pre-wrap; max-width: 80ch; color: var(--muted); }
.h2h-field { margin: 10px 0 0; max-width: 90ch; }
.h2h-field b { display: block; font: 500 10px var(--display); letter-spacing: 1.4px; text-transform: uppercase; color: var(--muted); }
.h2h-field span { overflow-wrap: anywhere; }
h4.sub { font: 500 11px/1.3 var(--display); letter-spacing: 1.6px; text-transform: uppercase; color: var(--muted); margin: 28px 0 8px; }
ul.plain { list-style: none; margin: 8px 0 0; padding: 0; }
#alike th { font: 700 14px var(--body); letter-spacing: 0; text-transform: none; color: var(--text); }
ul.plain li { display: flex; justify-content: space-between; gap: 16px; padding: 9px 0; border-top: 1px solid var(--rule); }
ul.plain li b { font: 500 13px var(--mono); white-space: nowrap; }
.reel-row .when .to { display: block; font: 500 12px var(--body); color: var(--text); }
.tip .note { color: var(--muted); margin: -2px 0 6px; }
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
.demo-banner { margin: 0 0 22px; padding: 10px 16px; border: 2px solid var(--warn); border-left-width: 8px;
               background: var(--glass); font-size: 15px; color: var(--text); }
.demo-banner b { color: var(--warn); font: 700 12px var(--display); letter-spacing: 1.4px; margin-right: 6px; }
.demo-list { display: flex; flex-direction: column; gap: 6px; }
.demo-list button { text-align: left; font: 600 15px var(--body); padding: 8px 12px; border: 2px solid var(--line);
                    background: var(--glass); color: var(--text); cursor: pointer; }
.demo-list button small { display: block; font: 500 11.5px var(--mono); color: var(--muted); }
.demo-list button.on { background: var(--accent); color: var(--on-accent); border-color: var(--accent); }
.demo-list button.on small { color: var(--on-accent); }
body.demo .deck:not(#demo-deck) { display: none; }
/*GRADIENT_CSS*/
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
        <button class="nav-btn" data-tab="bias" title="Persona"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 11.75a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5zm6 0a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8 0-.29.02-.58.05-.86 2.36-1.05 4.23-2.98 5.21-5.37C11.07 8.33 14.05 10 17.42 10c.78 0 1.53-.09 2.25-.26.21.71.33 1.47.33 2.26 0 4.41-3.59 8-8 8z"/></svg><span class="btn-text">Persona</span></button>
        <button class="nav-btn" data-tab="label" title="Label"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM10 17H5v-2h5v2zm0-4H5v-2h5v2zm0-4H5V7h5v2zm4.82 6L12 12.16l1.41-1.41 1.41 1.42L17.99 9l1.42 1.42L14.82 15z"/></svg><span class="btn-text">Label</span></button>
        <!--GRADIENT_NAV-->
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
        <button class="text-btn" id="bias-pill" title="Edit the persona" hidden></button>
        <span class="island-label">Data</span>
        <select id="source" aria-label="Data source"><option value="live">Live</option><option value="demo">Simulated</option></select>
      </div>
<!--GRADIENT_ISLAND-->
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
      <div class="demo-banner" id="demo-banner" role="status" hidden><b>Simulated sessions.</b>
        Run, Feed, Digest and Report show 14 days of simulated sessions for both personas, under simulated dates.
        Nothing here was observed on Instagram; switch Data to Live for the real sessions.</div>
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
        <div class="deck card" id="demo-deck" hidden>
          <h3>Simulated sessions</h3>
          <p class="hint">Pick a simulated day and one of its sessions to see its log, or replay it as it ran.</p>
          <div class="control"><label class="label" for="demo-day">Simulated day</label>
            <select id="demo-day"></select></div>
          <div class="control"><span class="label">Sessions</span><div id="demo-list" class="demo-list"></div></div>
          <div class="actions">
            <button class="btn primary" id="btn-demo-replay" disabled>Replay</button>
            <button class="btn" id="btn-demo-show" disabled>Show the whole log</button>
          </div>
          <p class="hint" id="demo-meta"></p>
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
          <div class="seg small" id="feed-scope" role="radiogroup" aria-label="Whose feed">
            <label><input type="radio" name="feed-scope" value="account" checked> This account</label>
            <label><input type="radio" name="feed-scope" value="all"> All accounts</label>
          </div>
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
        <div class="card" id="personas-card" hidden>
          <h3>Each persona's feed</h3>
          <p class="hint">Every account scrolls as its own persona. "Top of its feed" is the curated feed once there is
            one, and everything so far until then.</p>
          <div class="table-scroll"><table class="data" id="personas"></table></div>
        </div>
        <div class="card" id="progress-card" hidden>
          <h3>Day by day</h3>
          <p class="hint" id="progress-sub"></p>
          <div class="chart" id="progress-chart"></div>
          <details><summary class="hint">Show as a table</summary><table class="data" id="progress-table"></table></details>
        </div>
        <div class="card">
          <h3>What the feed shows</h3>
          <p class="hint" id="mix-sub"></p>
          <div class="legend" id="mix-legend"></div>
          <div class="chart" id="mix-chart"></div>
          <details><summary class="hint">Show as a table</summary><table class="data" id="mix-table"></table></details>
        </div>
        <div class="card">
          <div class="status"><h3>Topics the algorithm serves</h3><span class="spacer"></span>
            <button class="btn" id="btn-train">Retrain now</button></div>
          <p class="hint" id="topics-sub"></p>
          <div id="topics-body" hidden>
            <p class="hint">The feed model learns these from every account's reels together, so each persona's feed
              is described in the same topics and can be compared.</p>
            <div class="legend" id="topic-legend"></div>
            <div class="chart" id="topic-chart"></div>
            <details><summary class="hint">Show as a table</summary><table class="data" id="topic-table"></table></details>
            <div class="two">
              <div>
                <h4 class="sub">How narrow the feed is</h4>
                <p id="narrow"></p>
                <div class="table-scroll"><table class="data" id="days"></table></div>
              </div>
              <div>
                <div id="rising-box"><h4 class="sub">Rising</h4>
                  <p class="hint">More of these in the last three days than before.</p><ul class="plain" id="rising"></ul></div>
                <div id="apart-box"><h4 class="sub">What sets this persona apart</h4>
                  <p class="hint">Topics it's served more than the accounts overall.</p><ul class="plain" id="apart"></ul></div>
                <div id="alike-box"><h4 class="sub">How alike the feeds are</h4>
                  <p class="hint">How much each pair of accounts' feeds share: 100% is the same mix of topics, 0% is
                    nothing in common.</p><div class="table-scroll"><table class="data" id="alike"></table></div></div>
              </div>
            </div>
          </div>
        </div>
        <div class="card">
          <h3>Creators the feed serves most</h3>
          <p class="hint" id="creators-sub"></p>
          <div class="table-scroll"><table class="data" id="creators"></table></div>
        </div>
        <div class="card">
          <h3>Every reel shown</h3>
          <div class="row"><select id="f-acct" aria-label="Account" hidden></select>
            <select id="f-day" aria-label="Day"><option value="">Every day</option></select>
            <select id="f-cat" aria-label="Category"></select>
            <select id="f-phase" aria-label="Phase"><option value="">Warm-up and curated</option>
              <option value="curated">Curated feed only</option><option value="warm-up">Warm-up only</option></select>
            <input type="search" id="f-q" placeholder="Search creator, topic, caption" aria-label="Search the reels">
            <span class="hint" id="f-count"></span></div>
          <p class="hint">Click a reel to inspect what was perceived, how it was judged and what was done.</p>
          <div id="reels"></div>
          <div class="pager" id="f-pager" hidden><button class="btn" id="f-prev">Newer</button>
            <span class="hint" id="f-page"></span><button class="btn" id="f-next">Older</button></div>
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
          <div class="field"><div class="name">Presets</div>
            <p class="hint">Picking one fills in the form below; nothing changes until you save. The presets are yours
              to edit: keep what's in the form as a new preset, or update or delete the one you picked.</p>
            <div class="presets" id="presets"></div>
            <div class="actions quiet">
              <button class="linkish" id="btn-preset-new">Save as a new preset</button>
              <button class="linkish" id="btn-preset-update" hidden>Update</button>
              <button class="linkish danger" id="btn-preset-delete" hidden>Delete</button>
              <button class="linkish" id="btn-preset-restore" hidden>Bring back the built-in presets</button>
            </div></div>
          <div class="two">
            <div>
              <div class="field"><label for="bias-name">Persona name</label>
                <input type="text" id="bias-name" maxlength="40"></div>
              <div class="field"><label for="bias-interests">Who this person is, and what they engage with</label>
                <p class="hint">The LLM scores every reel by how likely this person is to watch it to the end, like and
                  follow. Describe the whole person, not just one side of them: a real feed is never only politics.</p>
                <textarea id="bias-interests" rows="20"></textarea></div>
            </div>
            <div>
              <div class="field"><label for="bias-watch">Watch words</label>
                <p class="hint">When LM Studio is off, the quick rules watch reels whose caption or hashtags contain one of these. Separate with commas.</p>
                <textarea class="words" id="bias-watch" rows="4"></textarea></div>
              <div class="field"><label for="bias-skip">Skip words</label>
                <p class="hint">Reels whose caption or hashtags contain one of these are skipped at once, without the LLM. Leave out anything this person enjoys.</p>
                <textarea class="words" id="bias-skip" rows="4"></textarea></div>
              <div class="field"><label for="bias-hints">Account-name hints</label>
                <p class="hint">Parts of account names that suggest a page this person would follow, like "news". The quick rules watch those accounts.</p>
                <input type="text" id="bias-hints"></div>
              <div class="field"><label for="bias-seeds">Accounts to follow first</label>
                <p class="hint">Pages this person would really follow. Following them gives a fresh account a head start; only verified accounts get followed.</p>
                <textarea class="words" id="bias-seeds" rows="2"></textarea>
                <div class="actions"><button class="btn" id="btn-seed-check">Check them</button>
                  <button class="btn" id="btn-seed-follow">Follow them</button></div></div>
            </div>
          </div>
          <div class="actions"><button class="btn primary" id="btn-bias-save">Save persona</button></div>
          <p class="hint" id="bias-note"></p>
        </div>
<!--GRADIENT_PROFILE-->
      </section>

      <section id="tab-label" hidden>
        <div class="card">
          <h2 class="headline">Jev or the local LLM?</h2>
          <p class="lede">Both models scored the same reels as each persona. Say how the person would react to each
            reel; you won't see either model's answer. The one that agrees with you more judges the persona better.</p>
          <p class="notice" id="h2h-none" hidden>No head-to-head yet. <button id="btn-h2h-run">Score the sample now</button>
            (needs LM Studio running; it takes a few minutes).</p>
          <div id="h2h-body" hidden>
            <div class="row"><div class="seg small" id="h2h-persona" role="radiogroup" aria-label="Persona"></div>
              <span class="hint" id="h2h-progress"></span></div>
            <details><summary class="hint">Who this person is</summary><p class="persona-text" id="h2h-who"></p></details>
          </div>
        </div>
        <div class="card" id="h2h-reel-card" hidden>
          <div class="status"><h3 id="h2h-count"></h3><span class="spacer"></span>
            <button class="linkish" id="h2h-prev">Previous</button><button class="linkish" id="h2h-next">Next</button>
            <button class="linkish" id="h2h-todo">Next unlabelled</button></div>
          <div id="h2h-reel"></div>
          <div class="actions">
            <button class="btn primary" id="h2h-watch">Would watch to the end (W)</button>
            <button class="btn" id="h2h-skip">Would scroll past (S)</button>
            <button class="btn" id="h2h-unsure">Not sure (U)</button>
            <span class="hint" id="h2h-yours"></span>
          </div>
        </div>
        <div class="card" id="h2h-results-card" hidden>
          <h3>Who agrees with you</h3>
          <p class="hint" id="h2h-results-sub"></p>
          <div class="table-scroll"><table class="data" id="h2h-results"></table></div>
          <div id="h2h-misses"></div>
        </div>
      </section>

<!--GRADIENT_SECTIONS-->
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
let SOURCE = store.get("source", "live") === "demo" ? "demo" : "live";
const igProfile = (handle) => (SOURCE === "demo" ? null : `https://www.instagram.com/${encodeURIComponent(handle)}/`);
const out = {key: null, offset: 0, busy: false};

async function api(path, body) {
  const init = {headers: {"X-PRAG-Token": TOKEN, "X-PRAG-Source": SOURCE}, cache: "no-store"};
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
  if (s && s.bias) pill.textContent = s.bias.own ? s.bias.name : "No persona yet";
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
  if (SOURCE === "demo") return renderDemoStatus();
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
const FRAME = /^ {8}🖼 (.*)$/;  // the cover frame's descriptors, under the reel
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
  if (marks.includes("🖼")) notes.push("Jev saw the cover frame with the text");
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
        handle && igProfile(handle) ? h("a", {class: "who", href: igProfile(handle), target: "_blank", rel: "noopener",
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
    } else if ((m = FRAME.exec(line)) && live.last && live.last.summary) {
      live.last.summary.after(h("div", {class: "extras", text: m[1]}));
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
  if (SOURCE === "demo" || out.busy || !state || !state.run.key) return;
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

// ── Feed: what the algorithm shows one persona, or every account ──

let feedData = null;
const SVG_NS = "http://www.w3.org/2000/svg";
const pct = (x) => `${Math.round(x * 100)}%`;
const total = (counts) => Object.values(counts).reduce((a, b) => a + b, 0);
const feedScope = () => (document.querySelector('input[name="feed-scope"]:checked') || {}).value || "account";

function svg(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function localTime(utc) {
  if (!utc) return "?";
  return new Date(utc.replace(" ", "T").replace(/(\+00:00|Z)?$/, "Z")).toLocaleString([], {weekday: "short",
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"});
}

async function loadFeed() {
  const all = feedScope() === "all", day = all ? "" : $("f-day").value;
  document.querySelectorAll("#feed-scope label").forEach((l) => l.classList.toggle("on", l.querySelector("input").checked));
  try { feedData = await api(all ? "/api/feed?scope=all" : "/api/feed?account=" + encodeURIComponent(account()) +
                                   (day ? "&day=" + day : "")); }
  catch (e) { toast(e.message, true); return; }
  renderFeed();
}

function renderFeed() {
  const d = feedData, n = d.counts, all = d.scope === "all";
  let headline, lede;
  if (all) {
    [headline, lede] = !n.reels
      ? ["No reels yet", "Run sessions: every reel each account is shown lands here, persona by persona."]
      : [`${n.curated_accounts} of ${n.accounts} feeds curated`, `${n.reels} reels shown to ${n.accounts} accounts, ` +
         `each scrolling as its own persona, ${n.curated} of them in curated feeds. The accounts side by side first, ` +
         "then everything together."];
  } else {
    const c = d.curation, fit = d.bias.persona ? "fit the persona" : "on target";
    [headline, lede] = !n.reels
      ? ["No reels yet", "Run a session: every reel this account is shown lands here, split into the warm-up and the " +
         "curated feed."]
      : c.curated
      ? [`Curated since ${localTime(c.since.shown_at)}`, `It took ${c.since.index} judged reels. The ${n.curated} ` +
         `reels shown since then are the curated feed; the ${n["warm-up"]} before were the warm-up.`]
      : ["Still warming up", (c.enough
          ? `Of the latest ${c.window} judged reels, ${pct(c.share)} ${fit}; curated needs ${pct(c.target)}.`
          : `${c.judged} of the ${c.window} judged reels needed so far, ${pct(c.share)} ${fit} (curated needs ` +
            `${pct(c.target)}).`) + " Everything so far is warm-up."];
  }
  $("feed-state").textContent = headline;
  $("feed-curation").textContent = lede;
  $("feed-uncat").hidden = !n.uncategorised;
  $("feed-uncat").textContent = `${n.uncategorised} reels aren't categorised yet. ` + (all
    ? "Each account's daily run categorises its reels; to do it now, pick the account and use \"Categorise now\"."
    : "\"Categorise now\" asks the LLM (about a minute per 60 reels); daily runs also do it after each session.");
  $("btn-categorize").hidden = all;
  $("btn-export-curated").disabled = $("btn-export-curated-csv").disabled = !n.curated;
  $("btn-export-curated").textContent = all ? "Download every curated feed (JSONL)" : "Download the curated feed (JSONL)";
  $("btn-export-curated-csv").textContent = all ? "Every curated feed as CSV" : "Curated feed as CSV";
  $("btn-export-all").textContent = all ? "Everything, every account (JSONL)" : "Everything shown (JSONL)";
  $("personas-card").hidden = !all;
  if (all) drawPersonas();
  drawMix();
  drawTopics();
  drawCreators();
  fillFilters();
  drawProgress();
  loadReels(0);
}

// ── Day by day: the account's reels per day and how many fit the persona ──

function drawProgress() {
  const days = (feedData.scope === "all" ? [] : feedData.days) || [], card = $("progress-card");
  const select = $("f-day"), keep = feedData.day || "";
  select.replaceChildren(h("option", {value: "", text: "Every day"}),
    ...[...days].reverse().map((d) => h("option", {value: d.day, text: `${d.day} (${d.reels} reels)`})));
  select.value = keep;
  select.hidden = !days.length;
  card.hidden = days.length < 2;
  if (card.hidden) return;
  const W = 900, H = 220, P = {l: 44, r: 10, t: 14, b: 36}, n = days.length, bw = (W - P.l - P.r) / n;
  const s = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Share of each day's reels that fit the persona"});
  for (const f of [0, 0.5, 1]) {
    const y = P.t + (1 - f) * (H - P.t - P.b);
    s.append(svg("line", {x1: P.l, x2: W - P.r, y1: y, y2: y, stroke: "var(--viz-grid)"}),
      svg("text", {class: "tick", x: 4, y: y + 4}, `${Math.round(f * 100)}%`));
  }
  days.forEach((d, i) => {
    const share = d.reels ? d.fit / d.reels : 0, height = share * (H - P.t - P.b), x = P.l + i * bw + 2;
    const y = H - P.b - height;
    s.append(svg("path", {d: barPath(x, y, bw - 4, height, 3), fill: d.day === keep ? "var(--viz-after)" : "var(--viz-bar)"}));
    if (n <= 16 || i % 2 === 0) s.append(svg("text", {class: "tick", x: x + (bw - 4) / 2, y: H - 14, "text-anchor": "middle"}, d.day.slice(5)));
    const hit = svg("rect", {class: "hit", x: P.l + i * bw, y: P.t, width: bw, height: H - P.t - P.b, tabindex: 0});
    hit.append(svg("title", {}, `${d.day}: ${d.fit} of ${d.reels} reels fit the persona (${pct(share)}), ` +
      `${d.watched} watched, ${d.liked} liked`));
    hit.onclick = () => { $("f-day").value = d.day; loadReels(0); $("reels").scrollIntoView({behavior: "smooth", block: "start"}); };
    s.append(hit);
  });
  $("progress-chart").replaceChildren(s);
  $("progress-sub").textContent = "Share of each day's reels that fit the persona (Jev's score at or above the watch line). " +
    "Click a day to list its reels.";
  $("progress-table").replaceChildren(h("tr", {}, ...["Day", "Reels", "Fit the persona", "Watched", "Liked"].map((t, i) =>
      h("th", {class: i ? "num" : null, text: t}))),
    ...days.map((d) => h("tr", {}, h("td", {text: d.day}), ...[d.reels, d.fit, d.watched, d.liked].map((v) =>
      h("td", {class: "num", text: v})))));
}

// ── The simulated sessions on the Run tab ──

const demoState = {days: [], day: null, session: null, timer: null, lines: []};

function applySource() {
  document.body.classList.toggle("demo", SOURCE === "demo");
  $("demo-banner").hidden = SOURCE !== "demo";
  $("demo-deck").hidden = SOURCE !== "demo";
  $("source").value = SOURCE;
}

function renderDemoStatus() {
  $("run-state").textContent = "Simulated";
  $("run-state").className = "state";
  $("run-status").textContent = demoState.session ? `Simulated session ${demoState.session.session_id}, ` +
    `simulated ${localTime(demoState.session.simulated_start)}` : "Simulated sessions: pick a day and a session.";
  $("progress").hidden = true;
  $("nav-run-meta").textContent = "";
}

async function loadDemoSessions() {
  if (SOURCE !== "demo") return;
  try { demoState.days = (await api("/api/demo/sessions?account=" + encodeURIComponent(account()))).days; }
  catch (e) { toast(e.message, true); demoState.days = []; }
  const select = $("demo-day");
  select.replaceChildren(...[...demoState.days].reverse().map((d) => h("option", {value: d.date,
    text: new Date(d.date + "T12:00").toLocaleDateString([], {weekday: "short", day: "numeric", month: "short", year: "numeric"})})));
  if (!demoState.days.some((d) => d.date === demoState.day)) demoState.day = demoState.days.length ? demoState.days[demoState.days.length - 1].date : null;
  select.value = demoState.day || "";
  drawDemoList();
}

function drawDemoList() {
  const day = demoState.days.find((d) => d.date === demoState.day);
  $("demo-list").replaceChildren(...(day ? day.sessions : []).map((x) => {
    const b = h("button", {class: demoState.session && demoState.session.session_id === x.id ? "on" : null},
      `${new Date(x.start.replace(" ", "T") + "Z").toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})} · ${x.reels} reels` +
        (x.kind ? ` · ${x.kind}` : ""),
      h("small", {text: `watched ${x.WATCH + x.FULL_WATCH + x.REWATCH} · ❤ ${x.liked} · 🔖 ${x.saved} · ➕ ${x.followed}`}));
    b.onclick = () => openDemoSession(x.id, false);
    return b;
  }));
  $("btn-demo-replay").disabled = $("btn-demo-show").disabled = !demoState.session;
}

async function openDemoSession(id, replay) {
  clearInterval(demoState.timer);
  let d;
  try { d = await api("/api/demo/log?session=" + id); } catch (e) { toast(e.message, true); return; }
  demoState.session = d.session;
  demoState.lines = d.lines;
  drawDemoList();
  renderDemoStatus();
  $("demo-meta").textContent = `Simulated ${localTime(d.session.simulated_start)} to ${localTime(d.session.simulated_end)}; ` +
    `generated ${localTime(d.session.generated_at)}. ${d.lines.length} log lines.`;
  showDemoLog(replay);
}

function showDemoLog(replay) {
  clearInterval(demoState.timer);
  const box = $("console"), lines = demoState.lines;
  box.textContent = "";
  clearStream();
  const stamp = (l) => `[${localTime(l.at)}] ${l.line}\n`;
  if (!replay) {
    box.textContent = lines.map(stamp).join("");
    addToStream(lines.map((l) => l.line).join("\n"), false);
    return;
  }
  let i = 0;  // a replay: the lines arrive in simulated order, thirty times faster than they happened
  const t0 = Date.parse(lines[0].at.replace(" ", "T") + "Z"), started = Date.now();
  demoState.timer = setInterval(() => {
    const now = t0 + (Date.now() - started) * 30, batch = [];
    while (i < lines.length && Date.parse(lines[i].at.replace(" ", "T") + "Z") <= now) batch.push(lines[i++]);
    if (batch.length) {
      box.append(batch.map(stamp).join(""));
      box.scrollTop = box.scrollHeight;
      addToStream(batch.map((l) => l.line).join("\n"), true);
      $("run-clock").textContent = "Simulated " + localTime(batch[batch.length - 1].at);
    }
    if (i >= lines.length) clearInterval(demoState.timer);
  }, 200);
}

function drawPersonas() {
  const d = feedData;
  $("personas").replaceChildren(
    h("tr", {}, ...["Account", "Persona", "Reels", "Curated", "Top of its feed"].map((text, i) =>
      h("th", {class: i === 2 ? "num" : null, text}))),
    ...d.accounts.map((a) => {
      const size = total(a.mix), top = Object.entries(a.mix).sort((x, y) => y[1] - x[1]).slice(0, 3)
        .map(([key, count]) => `${d.categories[key]} ${pct(count / size)}`).join(", ");
      return h("tr", {},
        h("td", {}, h("strong", {text: "@" + a.account})),
        h("td", {text: a.own ? a.persona : "None yet"}),
        h("td", {class: "num", text: String(a.reels)}),
        h("td", {text: a.curated ? `Since ${localTime(a.since.shown_at)}`
          : `${a.judged} of ${a.window} judged, ${pct(a.share)} ${a.is_persona ? "fit" : "on target"}`}),
        h("td", {text: size ? top + (a.mix_of === "curated" ? "" : " (so far)") : "Not categorised yet"}));
    }));
}

function shareRows(mixes, labels, notes) {  // one row per key: its share of the warm-up and of the curated feed
  const warm = mixes["warm-up"], cur = mixes.curated, wt = total(warm), ct = total(cur), both = ct > 0;
  const rows = Object.keys(labels).filter((k) => warm[k] || cur[k]).map((k) => ({
    label: labels[k], note: notes ? notes[k] : "", w: wt ? (warm[k] || 0) / wt : 0, c: ct ? (cur[k] || 0) / ct : 0,
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

function drawShares(id, {rows, both}, head, aria) {  // a dumbbell per row (warm-up to curated), or bars before curation
  const box = $(id + "-chart");
  $(id + "-legend").replaceChildren(...(both ? [["--viz-before", "Warm-up"], ["--viz-after", "Curated"]] : []).map(
    ([token, label]) => h("span", {}, h("i", {style: `background: var(${token})`}), label)));
  box.replaceChildren();
  if (!rows.length) {
    box.append(h("p", {class: "hint", text: "Nothing to show yet."}));
    $(id + "-table").replaceChildren();
    return;
  }
  const W = 900, labelW = 230, valueW = 52, rowH = 30, top = 6, axisH = 26;
  const x0 = labelW, x1 = W - valueW, plotH = rows.length * rowH, H = top + plotH + axisH;
  const max = Math.max(...rows.map((r) => Math.max(r.w, both ? r.c : 0)));
  const step = max > 0.5 ? 0.2 : max > 0.2 ? 0.1 : 0.05;
  const xmax = Math.max(step, Math.ceil(max / step - 1e-9) * step);
  const x = (v) => x0 + (x1 - x0) * v / xmax;
  const chart = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": aria});
  rows.forEach((r, i) => {  // row hit targets first, so the marks sit on top of the hover wash
    const hit = svg("rect", {x: 0, y: top + i * rowH, width: W, height: rowH, class: "hit", tabindex: "0",
      "aria-label": `${r.label}: ` + (both ? `warm-up ${pct(r.w)}, curated ${pct(r.c)}` : pct(r.w))});
    hit.addEventListener("pointermove", (e) => showTip(e, r, both, id));
    hit.addEventListener("focus", (e) => showTip(e, r, both, id));
    hit.addEventListener("pointerleave", () => hideTip(id));
    hit.addEventListener("blur", () => hideTip(id));
    chart.append(hit);
  });
  for (let i = 0, t = 0; t <= xmax + 1e-9; i++, t = i * step) {
    chart.append(svg("line", {x1: x(t), x2: x(t), y1: top, y2: top + plotH, "stroke-width": 1,
                              stroke: i === 0 ? "var(--viz-axis)" : "var(--viz-grid)", "pointer-events": "none"}));
    chart.append(svg("text", {x: x(t), y: H - 7, "text-anchor": "middle", class: "tick"}, pct(t)));
  }
  rows.forEach((r, i) => {
    const cy = top + i * rowH + rowH / 2, label = r.label.length > 32 ? r.label.slice(0, 31) + "…" : r.label;
    chart.append(svg("text", {x: labelW - 14, y: cy + 4, "text-anchor": "end", class: "cat", "pointer-events": "none"}, label));
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
  box.append(chart, h("div", {class: "tip", id: id + "-tip", hidden: true}));
  $(id + "-table").replaceChildren(
    h("tr", {}, h("th", {text: head}), h("th", {class: "num", text: both ? "Warm-up" : "Share"}),
      both ? h("th", {class: "num", text: "Curated"}) : null),
    ...rows.map((r) => h("tr", {}, h("td", {text: r.label + (r.note ? ` (${r.note})` : "")}),
      h("td", {class: "num", text: `${pct(r.w)} (${r.wn})`}), both ? h("td", {class: "num", text: `${pct(r.c)} (${r.cn})`}) : null)));
}

function showTip(event, r, both, id) {
  const tip = $(id + "-tip"), box = $(id + "-chart").getBoundingClientRect();
  const lines = both ? [["--viz-before", "warm-up", r.w, r.wn], ["--viz-after", "curated", r.c, r.cn]]
                     : [["--viz-bar", "so far", r.w, r.wn]];
  tip.replaceChildren(h("div", {class: "title", text: r.label}), r.note ? h("div", {class: "note", text: r.note}) : null,
    ...lines.map(([token, name, share, count]) => h("div", {class: "line"},
      h("span", {class: "key", style: `background: var(${token})`}), h("b", {text: pct(share)}),
      h("span", {class: "name", text: `${name}, ${count} reels`}))));
  tip.hidden = false;
  const at = event.clientX !== undefined && event.type !== "focus" ? event
    : {clientX: box.left + box.width / 2, clientY: event.target.getBoundingClientRect().top};
  const left = Math.min(at.clientX - box.left + 14, box.width - tip.offsetWidth - 4);
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${at.clientY - box.top + 14}px`;
}

function hideTip(id) { const tip = $(id + "-tip"); if (tip) tip.hidden = true; }

function drawMix() {
  const d = feedData, rows = shareRows(d.mix, d.categories), all = d.scope === "all";
  const whose = all ? "every account's" : "the";
  $("mix-sub").textContent = rows.both
    ? `Each category's share of ${whose} ${rows.wt} warm-up and ${rows.ct} curated reels (categorised, ads excluded), ` +
      "sorted by the curated feed."
    : `Each category's share of ${whose} ${rows.wt} categorised reels so far (ads excluded).`;
  drawShares("mix", rows, "Category", "Share of each content category in the feed");
}

function drawTopics() {
  const t = feedData.topics, all = feedData.scope === "all", one = (x) => Number(x).toFixed(1);
  $("topics-body").hidden = !t.trained;
  $("btn-train").textContent = t.trained ? "Retrain now" : "Train the feed model";
  if (!t.trained) {
    $("btn-train").disabled = t.reels < t.needed;
    $("topics-sub").textContent = t.reels < t.needed
      ? `It learns the topics once there are ${t.needed} different reels (there are ${t.reels}), then after every daily run.`
      : `Not trained yet. It learns from every account's reels (${t.reels} so far), after every daily run or now.`;
    return;
  }
  $("btn-train").disabled = false;
  $("topics-sub").textContent = `Learned ${t.k} topics from ${t.trained_reels} reels across ` +
    `${t.trained_accounts.length} accounts, ${localTime(t.created)}. ` + (t.newer ? `${t.newer} newer reels ` +
    "join at the next training (after the next daily run)." : "Every reel shown is in it.");
  const labels = Object.fromEntries(t.topics.map((x) => [String(x.id), x.label]));
  const notes = Object.fromEntries(t.topics.map((x) => [String(x.id), x.category ? feedData.categories[x.category] : ""]));
  drawShares("topic", shareRows(t.mix, labels, notes), "Topic", "Share of each topic in the feed");

  const first = t.daily[0], last = t.daily[t.daily.length - 1];
  $("narrow").textContent = `Spread over about ${one(t.effective_topics)} topics` +
    (t.of === "curated" ? " in the curated feed" : "") + (t.daily.length > 1 && first.effective_topics !== last.effective_topics
      ? `, ${last.effective_topics < first.effective_topics ? "down" : "up"} from ${one(first.effective_topics)} on ` +
        `the first day. Fewer means the algorithm is narrowing the feed around the persona.`
      : ". Fewer over time means the algorithm is narrowing the feed around the persona.");
  $("days").replaceChildren(
    h("tr", {}, h("th", {text: "Day"}), h("th", {class: "num", text: "Reels"}), h("th", {class: "num", text: "Topics"}),
      h("th", {class: "num", text: "Changed"})),
    ...t.daily.slice(-14).reverse().map((day) => h("tr", {},
      h("td", {text: new Date(day.day + "T12:00:00Z").toLocaleDateString([], {weekday: "short", day: "numeric", month: "short"})}),
      h("td", {class: "num", text: String(day.reels)}), h("td", {class: "num", text: one(day.effective_topics)}),
      h("td", {class: "num", text: day.moved === null ? "" : pct(day.moved)}))));
  const name = (id) => labels[String(id)];
  $("rising-box").hidden = !t.rising.length;
  $("rising").replaceChildren(...t.rising.map((r) => h("li", {}, h("span", {text: name(r.topic)}),
    h("b", {text: `${pct(r.from)} to ${pct(r.to)}`}))));
  $("apart-box").hidden = all || !(t.apart || []).length;
  $("apart").replaceChildren(...(t.apart || []).map((a) => h("li", {}, h("span", {text: name(a.topic)}),
    h("b", {text: `${pct(a.share)} here, ${pct(a.overall)} overall`}))));
  const alike = t.alike;
  $("alike-box").hidden = !all || !alike || alike.accounts.length < 2;
  if (alike) {
    $("alike").replaceChildren(h("tr", {}, h("th", {text: ""}), ...alike.accounts.map((a) => h("th", {class: "num", text: "@" + a}))),
      ...alike.accounts.map((a, i) => h("tr", {}, h("td", {}, h("strong", {text: "@" + a})),
        ...alike.matrix[i].map((v, j) => h("td", {class: "num", text: i === j ? "" : pct(v)})))));
  }
}

function drawCreators() {
  const d = feedData, all = d.scope === "all";
  $("creators-sub").textContent = d.counts.curated ? "In the curated feed" + (all ? "s." : ".")
    : "So far: " + (all ? "no feed is curated yet." : "the feed isn't curated yet.");
  $("creators").replaceChildren(
    h("tr", {}, h("th", {text: "Creator"}), h("th", {class: "num", text: "Reels"}), h("th", {text: "Usually"}),
      h("th", {class: "num", text: "Watched"}), all ? h("th", {text: "Reached"}) : null),
    ...d.creators.map((c) => h("tr", {}, h("td", {}, link(igProfile(c.creator), "@" + c.creator)),
      h("td", {class: "num", text: String(c.reels)}), h("td", {text: c.category ? d.categories[c.category] : "—"}),
      h("td", {class: "num", text: pct(c.watched)}),
      all ? h("td", {text: (c.accounts || []).map((a) => "@" + a).join(", ")}) : null)));
}

function fillFilters() {
  const select = $("f-cat"), keep = select.value, accounts = $("f-acct"), keepAccount = accounts.value;
  select.replaceChildren(h("option", {value: "", text: "All categories"}),
    ...Object.entries(feedData.categories).map(([key, label]) => h("option", {value: key, text: label})),
    h("option", {value: "_none", text: "Not categorised"}), h("option", {value: "_ad", text: "Ads"}));
  select.value = keep;
  const names = (feedData.accounts || []).map((a) => a.account).sort();
  accounts.replaceChildren(h("option", {value: "", text: "Every account"}),
    ...names.map((a) => h("option", {value: a, text: "@" + a})));
  accounts.value = names.includes(keepAccount) ? keepAccount : "";
  accounts.hidden = feedData.scope !== "all";
}

// ── Every reel shown: a page at a time from the server (12,000+ reels stay quick), each one inspectable ──

const reelPage = {offset: 0, limit: 50, total: 0, open: null, seq: 0};

async function loadReels(offset) {
  const all = feedData.scope === "all", q = new URLSearchParams();
  if (all) q.set("scope", "all"); else q.set("account", account());
  const day = all ? "" : $("f-day").value, who = all ? $("f-acct").value : "";
  for (const [k, v] of [["day", day], ["who", who], ["cat", $("f-cat").value], ["phase", $("f-phase").value],
                        ["q", $("f-q").value.trim()]]) if (v) q.set(k, v);
  q.set("offset", Math.max(0, offset));
  q.set("limit", reelPage.limit);
  const mine = ++reelPage.seq;  // a slower, older request mustn't draw over a newer one
  let d;
  try { d = await api("/api/feed/reels?" + q); } catch (e) { toast(e.message, true); return; }
  if (mine !== reelPage.seq) return;
  Object.assign(reelPage, {offset: d.offset, total: d.total, open: null});
  drawReels(d.items);
}

function drawReels(items) {
  const all = feedData.scope === "all", {offset, total, limit} = reelPage;
  $("f-count").textContent = total ? `${(offset + 1).toLocaleString()}–${Math.min(offset + limit, total).toLocaleString()} of ` +
    `${total.toLocaleString()} reels` : "No reels match";
  $("f-pager").hidden = total <= limit;
  $("f-page").textContent = `Page ${Math.floor(offset / limit) + 1} of ${Math.max(1, Math.ceil(total / limit))}`;
  $("f-prev").disabled = offset === 0;
  $("f-next").disabled = offset + limit >= total;
  $("reels").replaceChildren(...items.map((r) => {
    const row = h("div", {class: "reel-row", tabindex: "0", role: "button", "aria-expanded": "false"},
      h("span", {class: "when"}, localTime(r.shown_at), all ? h("span", {class: "to", text: "to @" + r.account}) : null),
      SOURCE === "demo" || !r.url ? h("span", {text: "@" + (r.creator || "?")}) : link(r.url, "@" + (r.creator || "?")),
      h("span", {}, h("span", {class: "tag-cat", text: r.is_ad ? "Ad" : r.category ? feedData.categories[r.category] : "Not categorised"})),
      h("span", {}, r.topic ? h("strong", {text: r.topic + ": "}) : null, r.summary || ""),
      h("span", {class: "num"}, h("span", {text: (r.image_used ? "🖼 " : "") + (STAMPS[r.decision] || "")}),
        h("span", {text: r.value_score == null ? "" : Number(r.value_score).toFixed(2)})));
    const toggle = (e) => { if (e.target.closest("a")) return; inspectReel(row, r); };
    row.onclick = toggle;
    row.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(e); } };
    return row;
  }));
}

async function inspectReel(row, r) {
  const open = row.nextElementSibling && row.nextElementSibling.classList.contains("inspect");
  document.querySelectorAll("#reels .inspect").forEach((x) => x.remove());
  document.querySelectorAll("#reels .reel-row.open").forEach((x) => { x.classList.remove("open"); x.setAttribute("aria-expanded", "false"); });
  if (open) return;
  let d;
  try { d = await api(`/api/feed/reel?session=${r.session_id}&seq=${r.seq}`); } catch (e) { toast(e.message, true); return; }
  row.classList.add("open");
  row.setAttribute("aria-expanded", "true");
  const text = (title, value) => value ? [h("h4", {text: title}), h("p", {text: value})] : [];
  const facts = (pairs) => h("dl", {}, ...pairs.filter(([, v]) => v !== null && v !== undefined && v !== "")
    .flatMap(([k, v]) => [h("dt", {text: k}), h("dd", {text: String(v)})]));
  const v = d.visual, sim = d.simulated;
  const vision = v ? [h("h4", {text: "Cover frame, measured"}), facts([
      ["size", `${v.width}×${v.height}${v.portrait ? " (portrait)" : ""}`], ["brightness", v.brightness], ["contrast", v.contrast],
      ["saturation", v.saturation], ["colourfulness", v.colourfulness], ["edge density", v.edge_density]]),
    h("div", {class: "swatches"}, ...(v.palette || []).map((c) => h("i", {style: `background:${c.hex};width:${Math.max(18, c.share * 160)}px`, title: `${c.hex} ${gPct(c.share)}`})))]
    : [h("h4", {text: "Cover frame"}), h("p", {class: "hint", text: sim ? "None: a simulated session's reels are written text, with no image."
        : "Not measured: the cover wasn't downloaded for this reel (sessions before cover measuring, or OCR and vision both off)."})];
  const src = d.source && d.source.news_title ? [h("h4", {text: "Written about this article"}),
    h("p", {}, d.source.news_url ? link(d.source.news_url, d.source.news_title) : d.source.news_title,
      ` · ${d.source.news_source || ""} ${d.source.news_date ? "· " + d.source.news_date.slice(0, 10) : ""}`)] : [];
  const panel = h("div", {class: "inspect"},
    h("div", {}, ...text("Caption", d.caption), ...text("Spoken (transcript)", d.transcript), ...text("Text on the cover", d.cover_text),
      ...text("Summary", d.summary), ...text("Why this score", d.reasoning), ...src),
    h("div", {}, h("h4", {text: sim ? "Simulated impression" : "Impression"}), facts([
        ["account", "@" + d.account], ["session", `${d.session_id} · reel #${d.seq}`],
        [sim ? "simulated time" : "shown", localTime(d.shown_at)], ["creator", d.creator && "@" + d.creator],
        ["audio", [d.audio_type, d.duration && clock(d.duration)].filter(Boolean).join(", ")],
        ["topic", d.topic], ["category", d.category && (feedData.categories[d.category] || d.category)],
        ["stance", d.stance], ["score", d.value_score == null ? null : Number(d.value_score).toFixed(2)], ["scored by", d.scorer],
        ["image input", d.image_used == null ? (sim ? "no image" : "vision off") : d.image_used ? "yes: Jev saw the cover" : "no: text only"],
        ["decision", STAMPS[d.decision] || d.decision], ["on screen", d.dwell_seconds == null ? null : d.dwell_seconds + " s"],
        ["engagement", ["liked", "saved", "followed"].filter((k) => d[k]).join(", ") || "none"], ["error", d.error]]),
      ...vision));
  row.after(panel);
}

async function exportFeed(phase, format) {
  const all = feedScope() === "all";
  try {
    const where = all ? "scope=all" : `account=${encodeURIComponent(account())}`;
    const res = await fetch(`/api/feed/export?${where}&phase=${phase}&format=${format}`,
                            {headers: {"X-PRAG-Token": TOKEN, "X-PRAG-Source": SOURCE}, cache: "no-store"});
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
    const url = URL.createObjectURL(await res.blob());
    const a = h("a", {href: url, download: `${all ? "all-accounts" : account()}-${phase}.${format}`});
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  } catch (e) { toast(e.message, true); }
}

async function runTask(path, body, message) {  // a job that runs like a session: its output is on the Run tab
  try {
    await api(path, body);
    toast(message);
    showTab("run");
    await refresh();
    pollOutput();
  } catch (e) { toast(e.message, true); }
}

const categorizeNow = () => runTask("/api/feed/categorize", {account: account()}, "Categorising: follow it on the Run tab.");
const trainFeedModel = () => runTask("/api/feed/model", {}, "Training the feed model: follow it on the Run tab.");

// ── Label: your calls for the Jev / local LLM head-to-head ──

let h2hData = null, h2hIndex = 0;
const h2hPersona = () => (document.querySelector('input[name="h2h-persona"]:checked') || {}).value;

async function loadH2H() {
  try { h2hData = await api("/api/h2h"); } catch (e) { toast(e.message, true); return; }
  renderH2H();
}

function renderH2H() {
  const d = h2hData;
  $("h2h-none").hidden = d.exists;
  $("h2h-body").hidden = $("h2h-reel-card").hidden = $("h2h-results-card").hidden = !d.exists;
  if (!d.exists) return;
  const pick = $("h2h-persona"), keys = Object.keys(d.personas);
  if (pick.childElementCount !== keys.length) {
    pick.replaceChildren(...keys.map((k, i) => h("label", {}, h("input", {type: "radio", name: "h2h-persona", value: k,
      checked: i === 0}), d.personas[k].name)));
  }
  pick.querySelectorAll("label").forEach((l) => l.classList.toggle("on", l.querySelector("input").checked));
  const persona = h2hPersona(), mine = d.labels[persona] || {};
  $("h2h-who").textContent = d.personas[persona].interests;
  $("h2h-progress").textContent = `${Object.keys(mine).length} of ${d.reels.length} labelled`;
  drawH2HReel();
  drawH2HResults();
}

function drawH2HReel() {
  const d = h2hData, persona = h2hPersona(), r = d.reels[h2hIndex], mine = (d.labels[persona] || {})[r.code];
  $("h2h-count").textContent = `Reel ${h2hIndex + 1} of ${d.reels.length}`;
  const field = (name, text) => text ? h("p", {class: "h2h-field"}, h("b", {text: name}), h("span", {text})) : null;
  const clip = (t, n) => (t || "").length > n ? t.slice(0, n) + "…" : t;
  const audio = r.audio_type === "music" ? "Music" : r.audio_type === "original" ? "Original audio" : "";
  $("h2h-reel").replaceChildren(
    h("div", {class: "v-head"}, link(r.url, "@" + (r.creator || "?")),
      h("span", {class: "kind", text: [audio, r.duration ? clock(r.duration) : ""].filter(Boolean).join(", ")}),
      r.hindi ? h("span", {class: "badge", text: "Hindi"}) : null),
    field("What it's about (written earlier by the LLM)", [r.topic, r.summary].filter(Boolean).join(": ")),
    field("Caption", clip(r.caption, 700)), field("Text on the cover", clip(r.cover_text, 300)),
    field("Speech", clip(r.transcript, 600)));
  $("h2h-yours").textContent = mine ? `Your call: ${{watch: "would watch to the end", skip: "would scroll past",
    unsure: "not sure"}[mine]}` : "Not labelled yet";
  for (const [id, label] of [["h2h-watch", "watch"], ["h2h-skip", "skip"], ["h2h-unsure", "unsure"]]) {
    $(id).classList.toggle("primary", mine === label || (!mine && label === "watch"));
  }
}

async function labelH2H(label) {
  const r = h2hData.reels[h2hIndex];
  try { h2hData = await api("/api/h2h/label", {persona: h2hPersona(), code: r.code, label}); }
  catch (e) { toast(e.message, true); return; }
  const mine = h2hData.labels[h2hPersona()] || {};
  const next = h2hData.reels.findIndex((x, i) => i > h2hIndex && !mine[x.code]);
  if (next >= 0) h2hIndex = next;
  else if (Object.keys(mine).length === h2hData.reels.length) toast("Every reel is labelled for this persona.");
  renderH2H();
}

function drawH2HResults() {
  const res = h2hData.results, persona = h2hPersona(), mine = res.personas[persona];
  const cell = (a) => a && a.n ? `${pct(a.agree)} of ${a.n}` : "";
  $("h2h-results-sub").textContent = "How often each model's watch-or-scroll call matches yours (reels marked " +
    "\"not sure\" are left out). Kappa is agreement beyond chance: 1 is perfect, 0 is no better than guessing. " +
    `Median time per reel: Jev ${res.seconds.jev ?? "?"} s, local LLM ${res.seconds.llm ?? "?"} s.` +
    (h2hData.errors ? ` ${h2hData.errors} scoring calls failed; run the head-to-head again to retry them.` : "");
  const rows = [];
  for (const [name, entry] of [[h2hData.personas[persona].name, mine], ["Every persona", res.overall]]) {
    for (const [model, label] of [["jev", "Jev"], ["llm", "Local LLM"]]) {
      const m = entry[model] || {};
      rows.push(h("tr", {}, h("td", {text: name}), h("td", {}, h("strong", {text: label})),
        h("td", {class: "num", text: cell(m)}), h("td", {class: "num", text: m.kappa == null ? "" : m.kappa.toFixed(2)}),
        h("td", {class: "num", text: cell(m.hindi)})));
    }
  }
  $("h2h-results").replaceChildren(h("tr", {}, ...["Persona", "Model", "Agrees with you", "Kappa", "Hindi reels"]
    .map((t, i) => h("th", {class: i > 1 ? "num" : null, text: t}))), ...rows);
  const done = mine.labelled === mine.of;  // the models' answers stay hidden until you've labelled every reel
  $("h2h-misses").replaceChildren(done && mine.misses.length ? h("h4", {class: "sub", text: "Where a model disagreed with you"}) : null,
    ...(done ? mine.misses : []).map((x) => h("p", {class: "hint", text: `@${x.creator}: you said ${x.you}; ` +
      `Jev ${x.jev.toFixed(2)}, local LLM ${x.llm.toFixed(2)} (watch at 0.40 or more)`})));
}

// ── Accounts ──

let accountTarget = null;

async function loadAccounts() {
  let items;
  try { items = (await api("/api/accounts")).items; } catch (e) { toast(e.message, true); return; }
  const head = h("tr", {}, ...["Account", "Password", "Status", "Persona", "Reels", "Last used", "Instagram login", ""]
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
      h("td", {text: a.persona_set ? a.bias : "None yet"}),
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

// ── Persona: who the account scrolls as, and the presets it can start from ──

let biasData = null, presetKey = null;  // presetKey: the preset last picked, which Update and Delete act on
const BIAS_FIELDS = {name: "bias-name", interests: "bias-interests", watch_words: "bias-watch",
                     skip_words: "bias-skip", creator_hints: "bias-hints", seeds: "bias-seeds"};
const LIST_FIELDS = ["watch_words", "skip_words", "creator_hints", "seeds"];

function biasForm() {
  const form = Object.fromEntries(Object.entries(BIAS_FIELDS).map(([key, id]) => [key, $(id).value]));
  form.persona = true;  // everything the tab saves is a persona
  return form;
}

function canonical(bias) {  // how the server will store it, to spot unsaved changes
  const list = (v) => (Array.isArray(v) ? v : String(v).split(/[,\n]/))
    .map((w) => w.trim().replace(/^[#@]+/, "")).filter(Boolean)
    .map((w) => (/^[\x00-\x7f]*$/.test(w) ? w.toLowerCase() : w));
  return JSON.stringify([String(bias.name).trim(), String(bias.interests).trim(), !!bias.persona,
                         ...LIST_FIELDS.map((k) => list(bias[k]))]);
}

function fillBias(bias, key) {
  $("bias-name").value = bias.name;
  $("bias-interests").value = bias.interests.trim();
  for (const field of LIST_FIELDS) $(BIAS_FIELDS[field]).value = bias[field].join(", ");
  if (key !== undefined) presetKey = key;
  biasChanged();
}

function renderPresets() {
  $("presets").replaceChildren(...Object.entries(biasData.presets).map(([key, preset]) => {
    const b = h("button", {type: "button", text: preset.name, "data-key": key});
    b.onclick = () => fillBias(preset, key);
    return b;
  }));
  if (presetKey && !biasData.presets[presetKey]) presetKey = null;
  $("btn-preset-restore").hidden = !biasData.presets_edited;
}

function biasChanged() {
  if (!biasData) return;
  const form = biasForm(), mine = canonical(form);
  document.querySelectorAll("#presets button").forEach((b) => {
    b.classList.toggle("on", canonical(biasData.presets[b.dataset.key]) === mine);
  });
  const picked = presetKey && biasData.presets[presetKey];
  $("btn-preset-update").hidden = $("btn-preset-delete").hidden = !picked;
  if (picked) {
    $("btn-preset-update").textContent = `Update “${picked.name}”`;
    $("btn-preset-delete").textContent = `Delete “${picked.name}”`;
  }
  $("bias-dirty").hidden = biasData.own && mine === canonical(biasData.bias);
  $("bias-dirty").textContent = biasData.own ? "Unsaved changes" : "Not saved yet";
  const fits = biasData.model_interests !== null && biasData.model_interests.trim() === form.interests.trim() &&
               !!biasData.model_persona;
  $("bias-note").textContent = "Changing who the persona is starts a fresh comparison in the report: earlier reels " +
    "stay, but aren't compared with the new ones. Keep one persona per account for the whole study." +
    (fits ? "" : " LM Studio has to be on to score reels for a persona; without it, reels are only recorded.");
}

async function loadBias() {
  try {
    biasData = await api("/api/bias?account=" + encodeURIComponent(account()));
  } catch (e) { toast(e.message, true); return; }
  renderPresets();
  $("btn-bias-save").textContent = `Save persona for @${account()}`;
  if (biasData.own) {
    $("bias-title").textContent = biasData.bias.name;
    $("bias-state").textContent = `@${account()} scrolls as this persona. Sessions keep it up, and the Feed tab shows ` +
      "what the algorithm serves it.";
    fillBias(biasData.bias, null);
  } else {
    $("bias-title").textContent = "No persona yet";
    $("bias-state").textContent = `@${account()} scrolls with the default ${biasData.default_name} interests until ` +
      "it has a persona. Pick a preset or describe a person, then save.";
    const start = biasData.presets["news-follower"] ? "news-follower" : Object.keys(biasData.presets)[0];
    if (start) fillBias(biasData.presets[start], start);
    else fillBias({...biasData.bias, name: "", interests: ""}, null);
  }
}

async function saveBias() {
  try {
    biasData = {...biasData, ...(await api("/api/bias", {account: account(), ...biasForm()}))};
    await loadBias();
    toast(`Saved. @${account()}'s next session scrolls as ${biasData.bias.name}.`);
    refresh();
  } catch (e) { toast(e.message, true); }
}

async function presetAction(kind) {
  const picked = presetKey && biasData.presets[presetKey];
  if (kind === "delete" && !confirm(`Delete the preset “${picked.name}”? Accounts already using it keep their persona.`)) return;
  if (kind === "restore" && !confirm("Bring back the built-in presets? Your edited and added presets are removed.")) return;
  try {
    const done = kind === "new" ? await api("/api/presets/save", {preset: biasForm()})
      : kind === "update" ? await api("/api/presets/save", {key: presetKey, preset: biasForm()})
      : kind === "delete" ? await api("/api/presets/delete", {key: presetKey})
      : await api("/api/presets/restore", {});
    biasData = {...biasData, presets: done.presets, presets_edited: done.presets_edited};
    presetKey = kind === "new" || kind === "update" ? done.key : null;
    renderPresets();
    biasChanged();
    const name = (done.key && done.presets[done.key] || picked || {}).name;
    toast({new: `Saved as a new preset: ${name}.`, update: `Updated the preset ${name}.`,
           delete: `Deleted the preset ${name}.`, restore: "The built-in presets are back."}[kind]);
  } catch (e) { toast(e.message, true); }
}

async function followSeeds(checkOnly) {
  if (!biasData) return;
  if (!$("bias-dirty").hidden) { toast("Save the persona first.", true); return; }
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
  if (name === "run") loadDemoSessions();
  if (name === "report") loadReport();
  if (name === "schedule") loadSchedules();
  if (name === "bias") loadBias();
  if (name === "feed") loadFeed();
  if (name === "accounts") loadAccounts();
  if (name === "label") loadH2H();
  gradientShow(name);
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
  $("account").addEventListener("change", () => { store.set("account", $("account").value); demoState.session = null;
    refresh().then(() => showTab(tab)); });
  applySource();
  $("source").addEventListener("change", () => {
    SOURCE = $("source").value === "demo" ? "demo" : "live";
    store.set("source", SOURCE);
    clearInterval(demoState.timer);
    demoState.session = null;
    clearStream();
    $("console").textContent = "";
    out.key = null;
    out.offset = 0;
    applySource();
    refresh().then(() => showTab(tab));
  });
  $("demo-day").onchange = () => { demoState.day = $("demo-day").value; demoState.session = null; drawDemoList(); };
  $("btn-demo-replay").onclick = () => { setView("reels"); showDemoLog(true); };
  $("btn-demo-show").onclick = () => showDemoLog(false);
  $("f-day").onchange = () => loadReels(0);

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
  $("btn-preset-new").onclick = () => presetAction("new");
  $("btn-preset-update").onclick = () => presetAction("update");
  $("btn-preset-delete").onclick = () => presetAction("delete");
  $("btn-preset-restore").onclick = () => presetAction("restore");
  $("btn-categorize").onclick = categorizeNow;
  $("btn-export-curated").onclick = () => exportFeed("curated", "jsonl");
  $("btn-export-curated-csv").onclick = () => exportFeed("curated", "csv");
  $("btn-export-all").onclick = () => exportFeed("all", "jsonl");
  $("f-cat").onchange = () => loadReels(0);
  $("f-acct").onchange = () => loadReels(0);
  $("btn-train").onclick = trainFeedModel;
  $("h2h-persona").addEventListener("change", () => { h2hIndex = 0; renderH2H(); });
  $("h2h-watch").onclick = () => labelH2H("watch");
  $("h2h-skip").onclick = () => labelH2H("skip");
  $("h2h-unsure").onclick = () => labelH2H("unsure");
  $("h2h-prev").onclick = () => { h2hIndex = Math.max(0, h2hIndex - 1); drawH2HReel(); };
  $("h2h-next").onclick = () => { h2hIndex = Math.min(h2hData.reels.length - 1, h2hIndex + 1); drawH2HReel(); };
  $("h2h-todo").onclick = () => {
    const mine = h2hData.labels[h2hPersona()] || {};
    const next = h2hData.reels.findIndex((x) => !mine[x.code]);
    if (next >= 0) { h2hIndex = next; drawH2HReel(); } else toast("Every reel is labelled for this persona.");
  };
  $("btn-h2h-run").onclick = () => runTask("/api/h2h/run", {}, "Scoring the head-to-head: follow it on the Run tab.");
  document.addEventListener("keydown", (e) => {
    if (tab !== "label" || !h2hData || !h2hData.exists || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    const key = e.key.toLowerCase();
    if (key === "w") labelH2H("watch");
    else if (key === "s") labelH2H("skip");
    else if (key === "u") labelH2H("unsure");
    else if (e.key === "ArrowRight") $("h2h-next").click();
    else if (e.key === "ArrowLeft") $("h2h-prev").click();
  });
  const scope = store.get("feed-scope", "account");
  document.querySelector(`input[name="feed-scope"][value="${scope === "all" ? "all" : "account"}"]`).checked = true;
  $("feed-scope").addEventListener("change", () => { store.set("feed-scope", feedScope()); loadFeed(); });
  $("f-phase").onchange = () => loadReels(0);
  { let t = null; $("f-q").oninput = () => { clearTimeout(t); t = setTimeout(() => loadReels(0), 300); }; }
  $("f-prev").onclick = () => loadReels(reelPage.offset - reelPage.limit);
  $("f-next").onclick = () => loadReels(reelPage.offset + reelPage.limit);
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
//GRADIENT_JS
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
