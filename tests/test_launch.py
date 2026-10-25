"""Tests for launch.py: the output reader, the run manager and the HTTP layer — no browser, no real runs."""

import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

import launch


class OutputTests(unittest.TestCase):
    def test_read_output_filters_noise_and_waits_for_whole_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.out")
            with open(path, "wb") as fh:
                fh.write("🚀  Session 9\n".encode())
                fh.write(b"/x/dataloader.py:759: UserWarning: 'pin_memory' argument is set\n")
                fh.write(b"  super().__init__(loader)\n")
                fh.write(b"Loading weights: 0%\rLoading weights: 100%\n")
                fh.write(b"\x1b[32m#1  @ndtv  0.95\x1b[0m\n")
                fh.write(b"#2  @partial")  # still being written
            text, offset, restarted = launch.read_output(path, 0)
            self.assertEqual(text, "🚀  Session 9\n#1  @ndtv  0.95\n")
            self.assertFalse(restarted)
            with open(path, "ab") as fh:
                fh.write(b" 0.10\n")
            self.assertEqual(launch.read_output(path, offset)[0], "#2  @partial 0.10\n")
            self.assertTrue(launch.read_output(path, 10**6)[2])  # the file is shorter than the offset: a new run
        self.assertEqual(launch.read_output(path, 5), ("", 0, True))  # gone


class RunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for name, value in (("RUN_OUTPUT", os.path.join(tmp.name, "panel-run.out")),
                            ("RUN_INFO", os.path.join(tmp.name, "panel-run.json"))):
            patcher = mock.patch.object(launch, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(launch.daily, "LOG_DIR", tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.elsewhere = None
        patcher = mock.patch.object(launch.daily, "current_run", side_effect=lambda: self.elsewhere)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_start_runs_daily_in_its_own_session_and_refuses_a_second_run(self):
        proc = mock.Mock(pid=4242, returncode=None)
        proc.poll.return_value = None
        runner = launch.Runner()
        with mock.patch.object(launch.subprocess, "Popen", return_value=proc) as popen:
            info = runner.start("acct", "mixed", 40, True, False)
            with self.assertRaisesRegex(ValueError, "already in progress"):
                runner.start("acct", "dry", 10, False, False)
        command, kwargs = popen.call_args.args[0], popen.call_args.kwargs
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[2:], ["--account", "acct", "--engagement", "mixed", "--minutes", "40", "--start-llm"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
        current, path = runner.current()
        self.assertEqual((current["running"], current["source"], current["key"], path),
                         (True, "panel", info["key"], launch.RUN_OUTPUT))
        with open(launch.RUN_INFO) as fh:
            self.assertEqual(json.load(fh)["pid"], 4242)  # a reopened panel finds its run again
        self.assertEqual(current["minutes"], 40)  # the page's progress line

        proc.poll.return_value, proc.returncode = 130, 130
        current, _ = runner.current()
        self.assertEqual((current["running"], current["exit"]), (False, 130))
        reopened, path = launch.Runner().current()  # the panel was closed and opened again
        self.assertEqual((reopened["key"], reopened["exit"], path), (info["key"], 130, launch.RUN_OUTPUT))

    def test_headless_session(self):
        proc = mock.Mock(pid=8)
        proc.poll.return_value = None
        with mock.patch.object(launch.subprocess, "Popen", return_value=proc) as popen:
            info = launch.Runner().start("acct", "dry", 20, False, False, headless=True)
        self.assertEqual(popen.call_args.args[0][2:],
                         ["--account", "acct", "--engagement", "dry", "--minutes", "20", "--headless"])
        self.assertIn("headless", info["label"])

    def test_digest_only(self):
        proc = mock.Mock(pid=7)
        proc.poll.return_value = None
        with mock.patch.object(launch.subprocess, "Popen", return_value=proc) as popen:
            info = launch.Runner().start("acct", "live", 40, True, True)
        self.assertEqual(popen.call_args.args[0][2:], ["--account", "acct", "--digest-only"])
        self.assertNotIn("minutes", info)  # no session, so no progress to show

    def test_follows_a_run_started_elsewhere(self):
        runner = launch.Runner()
        self.assertEqual(runner.current(), ({"key": None, "running": False}, None))
        self.elsewhere = {"pid": 999, "account": "acct", "log": "/logs/acct/run.log", "started": 5.0}
        current, path = runner.current()
        self.assertEqual((current["running"], current["source"], path), (True, "elsewhere", "/logs/acct/run.log"))
        self.assertNotIn("log", current)
        self.elsewhere = None  # it finished: still shown, as the latest run
        current, path = runner.current()
        self.assertEqual((current["running"], current["source"], path), (False, "elsewhere", "/logs/acct/run.log"))


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), launch.Panel)
        self.port = self.server.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        patches = [
            mock.patch.object(launch, "ALLOWED_HOSTS", {self.host}),
            mock.patch.object(launch, "active_accounts", return_value=["acct"]),
            mock.patch.object(launch, "account_status", return_value={"llm": True, "minutes_used": 5}),
            mock.patch.object(launch.Panel, "runner", mock.Mock(**{"current.return_value": ({"key": None}, None)})),
            mock.patch.object(launch.prompt, "pending", return_value=[]),
            mock.patch.object(launch.prompt, "answer", return_value=True),
            mock.patch.object(launch.biases, "BIAS_DIR", self.enterContext(tempfile.TemporaryDirectory())),
            mock.patch.object(launch.ModelScorer, "trained_for", return_value=launch.biases.DEFAULT_FINGERPRINT),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def request(self, method, path, body=None, token=True, host=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)
        headers = {"Host": host or self.host, "Content-Type": "application/json", **(headers or {})}
        if token:
            headers["X-PRAG-Token"] = launch.TOKEN
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, resp.getheaders(), data if path.startswith("/fonts/") else data.decode()

    def test_page(self):
        status, headers, body = self.request("GET", "/", token=False)
        headers = dict(headers)
        self.assertEqual(status, 200)
        self.assertIn(json.dumps(launch.TOKEN), body)
        self.assertNotIn("__TOKEN__", body)
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertIn("font-src 'self';", headers["Content-Security-Policy"])  # no font comes from elsewhere

    def test_fonts_are_served_from_this_mac(self):
        status, headers, body = self.request("GET", "/fonts/orbitron-latin.woff2", token=False)
        self.assertEqual((status, dict(headers)["Content-Type"], body[:4]), (200, "font/woff2", b"wOF2"))
        for path in ("/fonts/missing.woff2", "/fonts/..%2Flaunch.py", "/fonts/../launch.py", "/fonts/OFL-orbitron.txt"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path, token=False)[0], 404)
        self.assertEqual(self.request("GET", "/fonts/orbitron-latin.woff2", host=f"evil.example:{self.port}")[0], 403)

    def test_other_hosts_are_refused(self):  # a web page that re-points its own name at 127.0.0.1
        self.assertEqual(self.request("GET", "/", host=f"evil.example:{self.port}")[0], 403)
        self.assertEqual(self.request("GET", "/api/state", host=f"evil.example:{self.port}")[0], 403)

    def test_api_needs_the_token(self):
        status, _, body = self.request("GET", "/api/state", token=False)
        self.assertEqual(status, 403)
        self.assertIn("reload", json.loads(body)["error"])
        status, _, body = self.request("POST", "/api/run", {"account": "acct"}, token=False)
        self.assertEqual(status, 403)
        launch.Panel.runner.start.assert_not_called()

    def test_state(self):
        status, _, body = self.request("GET", "/api/state?account=nobody")
        data = json.loads(body)
        self.assertEqual((status, data["accounts"], data["account"], data["status"]["llm"]), (200, ["acct"], "acct", True))

    def test_questions_from_a_run(self):
        launch.prompt.pending.return_value = [{"id": "a" * 32, "account": "acct", "kind": "code", "secret": False}]
        state = json.loads(self.request("GET", "/api/state")[2])
        self.assertEqual(state["prompts"][0]["kind"], "code")
        self.assertEqual(self.request("POST", "/api/prompt", {"id": "a" * 32, "answer": " 123456 "})[0], 200)
        launch.prompt.answer.assert_called_with("a" * 32, "123456")
        self.request("POST", "/api/prompt", {"id": "a" * 32, "cancel": True})
        launch.prompt.answer.assert_called_with("a" * 32, None)
        self.assertEqual(self.request("POST", "/api/prompt", {"id": "a" * 32, "answer": "  "})[0], 400)
        launch.prompt.answer.return_value = False  # answered in the dialog meanwhile
        status, _, body = self.request("POST", "/api/prompt", {"id": "a" * 32, "answer": "1"})
        self.assertEqual((status, json.loads(body)["error"]), (400, "That question isn't waiting any more."))
        self.assertEqual(self.request("POST", "/api/prompt", {"id": "a" * 32, "answer": "1"}, token=False)[0], 403)

    def test_bias(self):
        status, _, body = self.request("GET", "/api/bias?account=acct")
        data = json.loads(body)
        self.assertEqual((status, data["own"], data["bias"]["name"]), (200, False, "News (India)"))
        self.assertEqual((data["model_name"], data["model_interests"]), ("News (India)", data["bias"]["interests"]))
        self.assertIn("tech", data["presets"])
        self.assertTrue(all(p["persona"] for p in data["presets"].values()))
        self.assertEqual(self.request("GET", "/api/bias?account=nobody")[0], 400)

        self.assertEqual(self.request("POST", "/api/bias", {"account": "acct", "interests": "x"})[0], 400)
        launch.Panel.runner.start_seed.return_value = {"key": "s"}
        self.assertEqual(self.request("POST", "/api/seed", {"account": "acct"})[0], 200)  # the default's news list
        self.assertIn("ndtv", launch.Panel.runner.start_seed.call_args.args[1])

        new = {"account": "acct", "name": "Tech", "interests": "AI launches, research and real gadget specs.",
               "watch_words": "AI, chip", "seeds": "@gadgets360, the_verge"}
        status, _, body = self.request("POST", "/api/bias", new)
        data = json.loads(body)
        self.assertEqual((status, data["own"], data["bias"]["watch_words"]), (200, True, ["ai", "chip"]))
        self.assertTrue(data["bias"]["persona"])  # the panel only saves personas
        self.assertEqual(self.request("POST", "/api/seed", {"account": "acct", "check_only": True})[0], 200)
        launch.Panel.runner.start_seed.assert_called_with("acct", ["gadgets360", "the_verge"], True)

        self.request("POST", "/api/bias", {**new, "seeds": ""})
        status, _, body = self.request("POST", "/api/seed", {"account": "acct"})
        self.assertEqual((status, json.loads(body)["error"]), (400, "Add accounts to follow first, and save the persona."))

    def test_presets_can_be_edited(self):
        mine = {"name": "Cricket fan", "interests": "A cricket fan from Pune who watches every IPL highlight."}
        status, _, body = self.request("POST", "/api/presets/save", {"preset": mine})
        data = json.loads(body)
        self.assertEqual((status, data["key"], data["presets_edited"]), (200, "cricket-fan", True))
        self.assertTrue(data["presets"]["cricket-fan"]["persona"])
        status, _, body = self.request("POST", "/api/presets/save", {"key": "tech", "preset": {**mine, "name": "Mine"}})
        self.assertEqual((status, json.loads(body)["presets"]["tech"]["name"]), (200, "Mine"))
        for body in ({"key": "../x", "preset": mine}, {"key": "gone", "preset": mine}, {"preset": "nope"},
                     {"preset": {"name": "x", "interests": "short"}}):
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", "/api/presets/save", body)[0], 400)
        status, _, body = self.request("POST", "/api/presets/delete", {"key": "cricket-fan"})
        self.assertEqual((status, "cricket-fan" in json.loads(body)["presets"]), (200, False))
        self.assertEqual(self.request("POST", "/api/presets/delete", {"key": "cricket-fan"})[0], 400)
        status, _, body = self.request("POST", "/api/presets/restore", {})
        data = json.loads(body)
        self.assertEqual((status, data["presets"]["tech"]["name"], data["presets_edited"]), (200, "Tech enthusiast", False))
        self.assertEqual(self.request("POST", "/api/presets/restore", {}, token=False)[0], 403)

    def test_feed_endpoints(self):
        topics = {"trained": False, "reels": 3, "needed": 30}
        with mock.patch.object(launch, "get_connection"), \
                mock.patch.object(launch.feed_study, "summary", return_value={"counts": {"reels": 3}}), \
                mock.patch.object(launch.feed_study, "summary_all", return_value={"scope": "all"}), \
                mock.patch.object(launch.feed_model, "analysis", return_value=topics) as analysis, \
                mock.patch.object(launch.feed_study, "dataset", return_value=[{"code": "A", "category": "sports"}]), \
                mock.patch.object(launch.feed_study, "dataset_all", return_value=[{"code": "A"}, {"code": "B"}]):
            status, _, body = self.request("GET", "/api/feed?account=acct")
            self.assertEqual((status, json.loads(body)), (200, {"counts": {"reels": 3}, "topics": topics, "demo": False}))
            self.assertEqual(analysis.call_args.args[1], "acct")  # one persona's feed
            self.assertEqual(self.request("GET", "/api/feed?account=nobody")[0], 400)
            status, _, body = self.request("GET", "/api/feed?scope=all")
            self.assertEqual((status, json.loads(body)), (200, {"scope": "all", "topics": topics, "demo": False}))
            self.assertIsNone(analysis.call_args.args[1])  # every account's
            status, headers, body = self.request("GET", "/api/feed/export?account=acct&phase=all&format=csv")
            self.assertEqual(status, 200)
            self.assertEqual(dict(headers)["Content-Disposition"], 'attachment; filename="acct-all.csv"')
            self.assertTrue(body.startswith("account,bias,persona,phase"))
            status, headers, body = self.request("GET", "/api/feed/export?scope=all&phase=curated&format=jsonl")
            self.assertEqual(dict(headers)["Content-Disposition"], 'attachment; filename="all-accounts-curated.jsonl"')
            self.assertEqual(len(body.splitlines()), 2)
            self.assertEqual(self.request("GET", "/api/feed/export?account=acct&format=xml")[0], 400)
            self.assertEqual(self.request("GET", "/api/feed/export?account=acct", token=False)[0], 403)

    def test_head_to_head_endpoints(self):
        run = {"personas": {"p": {"name": "P", "interests": "x"}}, "reels": [{"code": "A", "creator": "a", "hindi": False}],
               "scores": {"p": {"A": {"jev": {"score": 0.9, "seconds": 0.3}, "llm": {"score": 0.8, "seconds": 9}}}}}
        with mock.patch.object(launch.headtohead, "load_run", return_value=None):
            self.assertEqual(json.loads(self.request("GET", "/api/h2h")[2]), {"exists": False})
        with mock.patch.object(launch.headtohead, "load_run", return_value=run), \
                mock.patch.object(launch.headtohead, "load_labels", return_value={}), \
                mock.patch.object(launch.headtohead, "save_label") as save:
            data = json.loads(self.request("GET", "/api/h2h")[2])
            self.assertEqual((data["exists"], data["reels"][0]["code"]), (True, "A"))
            self.assertNotIn("scores", json.dumps(data["reels"]))  # the models' answers stay hidden
            self.assertEqual(self.request("POST", "/api/h2h/label", {"persona": "p", "code": "A", "label": "watch"})[0], 200)
            save.assert_called_with("p", "A", "watch")
            save.side_effect = ValueError("That reel isn't in the head-to-head sample.")
            self.assertEqual(self.request("POST", "/api/h2h/label", {"persona": "p", "code": "Z", "label": "watch"})[0], 400)
        launch.Panel.runner._launch.return_value = {"key": "h"}
        self.assertEqual(self.request("POST", "/api/h2h/run", {})[0], 200)
        self.assertEqual(launch.Panel.runner._launch.call_args.args[0][1:], ["-m", "tools.headtohead"])

    def test_gradient_endpoints(self):
        from gradient import api as gradient_api, store

        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(store, "DB_PATH", os.path.join(tmp, "corpus.db")))
        self.enterContext(mock.patch.object(store, "MODEL_DIR", os.path.join(tmp, "models")))
        conn = store.connect()
        batch = store.new_batch(conn, "generated", "reels written for the feed")
        item, _ = store.add_item(conn, {"origin": "generated", "caption": "Naya expressway", "topic": "infrastructure",
                                        "stance": "supportive", "language": "hinglish"}, batch)
        store.add_to_feed(conn, "mlproject01", item, batch, "generator")
        conn.commit()
        conn.close()
        data = json.loads(self.request("GET", "/api/gradient/overview")[2])
        self.assertEqual((data["items"], data["origins"][0]["name"]), (1, "Written by an LLM (Qwen 3.5) for the feed"))
        mine = next(p for p in data["personas"] if p["account"] == "mlproject01")
        self.assertEqual((mine["feed_total"], mine["topics"][0]["topic"]), (1, "infrastructure"))
        feed = json.loads(self.request("GET", "/api/gradient/items?account=mlproject01")[2])
        self.assertEqual(feed["items"][0]["text"], "Naya expressway")
        for method, path, body in (("GET", "/api/gradient/items?account=someone_else", None),
                                   ("POST", "/api/gradient/talk/start", {"kind": "chat", "account": "mlproject01", "text": " "}),
                                   ("POST", "/api/gradient/predict", {"text": "hi"}),
                                   ("POST", "/api/gradient/train", {"what": "rm -rf"})):
            with self.subTest(path=path):
                self.assertEqual(self.request(method, path, body)[0], 400)
        self.assertEqual(self.request("GET", "/api/gradient/overview", token=False)[0], 403)
        with mock.patch.object(gradient_api.retrieval, "search", return_value={"items": [], "model": None, "enough": False}):
            predicted = json.loads(self.request("POST", "/api/gradient/predict", {"text": "A long enough reel text"})[2])
        self.assertEqual((predicted["personas"][0]["p"], predicted["feed"]), (None, None))  # no model yet: says so

    def test_the_demo_source_reads_only_the_demo_database(self):
        from database import db
        from gradient import simulate

        tmp = self.enterContext(tempfile.TemporaryDirectory())
        live_path, demo_path = os.path.join(tmp, "live.db"), os.path.join(tmp, "demo.db")
        self.enterContext(mock.patch.object(db, "DB_PATH", live_path))
        self.enterContext(mock.patch.object(db, "DEMO_DB_PATH", demo_path))
        self.enterContext(mock.patch.object(launch, "DEMO_DB_PATH", demo_path))
        self.enterContext(mock.patch.object(launch.feed_model, "analysis", return_value={"trained": False}))
        for path, n in ((live_path, 1), (demo_path, 3)):
            conn = db.get_connection(path)
            db.init_db(conn)
            conn.executescript(simulate.TABLES)
            sid = conn.execute("INSERT INTO sessions (account, scorer) VALUES ('acct', 's')").lastrowid
            for seq in range(n):
                db.log_impression({"session_id": sid, "seq": seq, "code": f"c{path[-7:]}{seq}", "creator": "x",
                                   "score_basis": "score", "value_score": 0.9, "decision": "WATCH"}, conn)
            if path == demo_path:
                conn.execute("INSERT INTO demo_sessions VALUES (?, 'acct', '2026-10-14', '2026-10-14 03:00:00', "
                             "'2026-10-14 03:20:00', '2026-10-25 10:00:00', 's', 0.5, ?)", (sid, json.dumps({"reels": 3})))
                conn.execute("INSERT INTO demo_log VALUES (?, 0, '2026-10-14 03:00:00', '🧪  DEMO — simulated session')", (sid,))
            conn.commit()
            conn.close()
        live = json.loads(self.request("GET", "/api/feed?account=acct")[2])
        demo = json.loads(self.request("GET", "/api/feed?account=acct", headers={"X-PRAG-Source": "demo"})[2])
        self.assertEqual((live["counts"]["reels"], live["demo"]), (1, False))
        self.assertEqual((demo["counts"]["reels"], demo["demo"]), (3, True))
        self.assertEqual(len(demo["days"]), 1)
        days = json.loads(self.request("GET", "/api/demo/sessions?account=acct")[2])["days"]
        self.assertEqual((days[0]["date"], days[0]["sessions"][0]["reels"]), ("2026-10-14", 3))
        log = json.loads(self.request("GET", f"/api/demo/log?session={days[0]['sessions'][0]['id']}")[2])
        self.assertIn("DEMO", log["lines"][0]["line"])
        self.assertEqual(log["session"]["generated_at"], "2026-10-25 10:00:00")  # kept apart from the simulated time
        self.assertEqual(self.request("GET", "/api/demo/log?session=999")[0], 400)
        self.assertEqual(self.request("GET", "/api/feed?account=acct&day=yesterday")[0], 400)

    def test_train_the_feed_model(self):
        launch.Panel.runner._launch.return_value = {"key": "m"}
        self.assertEqual(self.request("POST", "/api/feed/model", {})[0], 200)
        self.assertEqual(launch.Panel.runner._launch.call_args.args[0][1:], ["-m", "ml.feed_model", "--train"])

    def test_categorize_needs_lm_studio(self):
        launch.Panel.runner._launch.return_value = {"key": "c"}
        with mock.patch.object(launch, "_cache", {}), mock.patch.object(launch, "llm_serving", return_value=False):
            status, _, body = self.request("POST", "/api/feed/categorize", {"account": "acct"})
        self.assertEqual(status, 400)
        self.assertIn("LM Studio", json.loads(body)["error"])
        with mock.patch.object(launch, "_cache", {}), mock.patch.object(launch, "llm_serving", return_value=True):
            self.assertEqual(self.request("POST", "/api/feed/categorize", {"account": "acct"})[0], 200)
        self.assertEqual(launch.Panel.runner._launch.call_args.args[0][1:],
                         ["-m", "dashboard.feed", "--account", "acct", "--categorize"])

    def test_accounts_endpoints(self):
        store = launch.account_store
        with mock.patch.object(store, "overview", return_value=[{"username": "acct", "has_password": True}]), \
                mock.patch.object(store, "add", return_value="new_acct") as add, \
                mock.patch.object(store, "set_password") as set_password:
            status, _, body = self.request("GET", "/api/accounts")
            self.assertEqual((status, json.loads(body)["items"][0]["has_password"]), (200, True))
            self.assertEqual(self.request("POST", "/api/accounts/add", {"username": "new_acct", "password": "pw"})[0], 200)
            add.assert_called_once_with("new_acct", "pw")
            add.side_effect = ValueError("@new_acct is already added.")
            status, _, body = self.request("POST", "/api/accounts/add", {"username": "new_acct"})
            self.assertEqual((status, json.loads(body)["error"]), (400, "@new_acct is already added."))
            self.assertEqual(self.request("POST", "/api/accounts/password", {"username": "acct"})[0], 400)
            self.request("POST", "/api/accounts/password", {"username": "acct", "remove": True})
            set_password.assert_called_with("acct", "")
            self.request("POST", "/api/accounts/password", {"username": "acct", "password": "new-pw"})
            set_password.assert_called_with("acct", "new-pw")
            self.assertEqual(self.request("GET", "/api/accounts", token=False)[0], 403)

    def test_delete_account(self):
        store = launch.account_store
        launch.Panel.runner.current.return_value = ({"key": "k", "running": True}, None)
        with mock.patch.object(store, "remove") as remove, mock.patch.object(launch.schedule, "remove", return_value=True):
            status, _, body = self.request("POST", "/api/accounts/delete", {"username": "acct"})
            self.assertEqual(status, 400)
            self.assertIn("stop it first", json.loads(body)["error"])
            remove.assert_not_called()
            launch.Panel.runner.current.return_value = ({"key": None, "running": False}, None)
            remove.return_value = {"username": "acct", "login_deleted": True}
            status, _, body = self.request("POST", "/api/accounts/delete", {"username": "acct", "delete_login": True})
            self.assertEqual(json.loads(body), {"username": "acct", "login_deleted": True, "schedule_removed": True})
            remove.assert_called_once_with("acct", delete_login=True)
            self.assertEqual(self.request("POST", "/api/accounts/delete", {"username": "../x"})[0], 400)

    def test_run_validates_its_input(self):
        for body in ({"account": "nobody"}, {"account": "acct", "engagement": "sometimes"},
                     {"account": "acct", "minutes": 999}, {"account": "acct", "minutes": "soon"}):
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", "/api/run", body)[0], 400)
        launch.Panel.runner.start.assert_not_called()
        launch.Panel.runner.start.return_value = {"key": "1-2"}
        status, _, body = self.request("POST", "/api/run", {"account": "acct", "engagement": "dry", "minutes": 20})
        self.assertEqual((status, json.loads(body)), (200, {"key": "1-2"}))
        launch.Panel.runner.start.assert_called_once_with("acct", "dry", 20.0, False, False, headless=False)
        self.request("POST", "/api/run", {"account": "acct", "headless": True})
        self.assertTrue(launch.Panel.runner.start.call_args.kwargs["headless"])

    def test_schedule_validation(self):
        for body in ({"account": "acct", "times": "25:00"}, {"account": "acct", "times": ""},
                     {"account": "acct", "times": "09:00", "engagement": "x"}):
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", "/api/schedule/install", body)[0], 400)
        self.assertEqual(self.request("POST", "/api/schedule/remove", {"account": "../evil"})[0], 400)

    def test_digest_files_stay_inside_the_digest_folder(self):
        for path in ("/digest/acct/../../accounts.json", "/digest/acct/..%2F..%2Faccounts.json",
                     "/digest/nobody/latest.html", "/digest/acct/notes.txt"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path, token=False)[0], 404)

    def test_bad_json(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)
        conn.request("POST", "/api/run", body="[1, 2", headers={"Host": self.host, "X-PRAG-Token": launch.TOKEN})
        self.assertEqual(conn.getresponse().status, 400)


if __name__ == "__main__":
    unittest.main()
