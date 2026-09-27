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
            launch.Runner().start("acct", "live", 40, True, True)
        self.assertEqual(popen.call_args.args[0][2:], ["--account", "acct", "--digest-only"])

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

    def request(self, method, path, body=None, token=True, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)
        headers = {"Host": host or self.host, "Content-Type": "application/json"}
        if token:
            headers["X-PRAG-Token"] = launch.TOKEN
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.getheaders(), resp.read().decode()

    def test_page(self):
        status, headers, body = self.request("GET", "/", token=False)
        headers = dict(headers)
        self.assertEqual(status, 200)
        self.assertIn(json.dumps(launch.TOKEN), body)
        self.assertNotIn("__TOKEN__", body)
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

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
        self.assertIn("tech-ai", data["presets"])
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
        self.assertEqual(self.request("POST", "/api/seed", {"account": "acct", "check_only": True})[0], 200)
        launch.Panel.runner.start_seed.assert_called_with("acct", ["gadgets360", "the_verge"], True)

        self.request("POST", "/api/bias", {**new, "seeds": ""})
        status, _, body = self.request("POST", "/api/seed", {"account": "acct"})
        self.assertEqual((status, json.loads(body)["error"]), (400, "Add accounts to follow first, and save the bias."))

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
