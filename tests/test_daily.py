"""Tests for daily.py: guards, the LLM → rules fallback and the run summary — no browser, no network."""

import argparse
import asyncio
import contextlib
import datetime
import io
import os
import sqlite3
import subprocess
import tempfile
import unittest
from unittest import mock

import daily
from config import DAILY_ENGAGEMENT, DAILY_MAX_MINUTES, MIN_BATTERY
from database.db import engaged_today, init_db, log_impression, minutes_today


class GuardTests(unittest.TestCase):
    def test_battery(self):
        on_battery = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t25%; discharging; 1:02 remaining\n"
        on_ac = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t80%; charging; 0:40 remaining\n"
        for out, expected in ((on_battery, (True, 25)), (on_ac, (False, 80)), ("Now drawing from 'AC Power'\n", (False, None))):
            with mock.patch.object(daily.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, out, "")):
                self.assertEqual(daily.battery(), expected)
        with mock.patch.object(daily.subprocess, "run", side_effect=OSError):
            self.assertEqual(daily.battery(), (False, None))

    def test_skip_reason(self):
        self.assertIsNone(daily.skip_reason("a", 0, False, None))
        self.assertIn("daily limit", daily.skip_reason("a", DAILY_MAX_MINUTES - 1, False, None))
        self.assertIn("on battery at", daily.skip_reason("a", 0, True, MIN_BATTERY - 1))
        self.assertIsNone(daily.skip_reason("a", 0, True, MIN_BATTERY))
        self.assertIsNone(daily.skip_reason("a", 0, False, 5))  # low but charging

    def test_minutes_today(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        init_db(conn)
        now = datetime.datetime(2026, 9, 26, 20, 0).astimezone()
        at = lambda h, m=0: now.replace(hour=h, minute=m)
        utc = lambda ts: ts.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        sessions = [
            ("acct", at(9), at(9, 30), None),                     # 30 min
            ("acct", at(12), None, at(12, 10)),                   # killed: counts up to its last reel
            ("acct", at(0) - datetime.timedelta(hours=2), at(0) - datetime.timedelta(hours=1), None),  # yesterday
            ("other", at(10), at(11), None),                      # another account
        ]
        for account, start, end, last_reel in sessions:
            sid = conn.execute("INSERT INTO sessions (account, started_at, ended_at) VALUES (?, ?, ?)",
                               (account, utc(start), utc(end) if end else None)).lastrowid
            if last_reel:
                log_impression(dict(session_id=sid, seq=1, code="X"), conn)
                conn.execute("UPDATE impressions SET shown_at = ? WHERE session_id = ?", (utc(last_reel), sid))
        self.assertEqual(minutes_today("acct", now=now, conn=conn), 40)
        self.assertEqual(minutes_today("nobody", now=now, conn=conn), 0)

    def test_engaged_today(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        init_db(conn)
        now = datetime.datetime(2026, 9, 26, 20, 0).astimezone()
        utc = lambda ts: ts.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        def session(account, dry_run, start, reels=True):
            sid = conn.execute("INSERT INTO sessions (account, dry_run, started_at) VALUES (?, ?, ?)",
                               (account, dry_run, utc(start))).lastrowid
            if reels:
                log_impression(dict(session_id=sid, seq=1, code="X"), conn)

        session("acct", 1, now.replace(hour=9))                       # a dry run
        session("acct", 0, now.replace(hour=10), reels=False)         # live, but failed before any reel
        session("acct", 0, now.replace(hour=0) - datetime.timedelta(hours=1))  # live yesterday
        session("other", 0, now.replace(hour=11))                     # another account
        self.assertFalse(engaged_today("acct", now=now, conn=conn))
        session("acct", 0, now.replace(hour=12))
        self.assertTrue(engaged_today("acct", now=now, conn=conn))


class ScrollTests(unittest.TestCase):
    def scroll(self, serving, outcomes, start_llm=False):
        run = mock.AsyncMock(side_effect=outcomes)
        with mock.patch.object(daily, "llm_serving", return_value=serving), \
                mock.patch.object(daily.agent, "run", run), \
                mock.patch.object(daily.ModelScorer, "available", return_value=True), \
                contextlib.redirect_stdout(io.StringIO()):
            outcome = asyncio.run(daily.scroll("acct", 30, True, start_llm))
        return outcome, [call.args[0] for call in run.await_args_list]

    def test_llm_when_lm_studio_serves_the_model(self):
        outcome, calls = self.scroll(True, [{"status": "ok"}])
        self.assertEqual((outcome["mode"], [c.rules for c in calls]), ("llm", [False]))
        self.assertEqual((calls[0].account, calls[0].minutes, calls[0].dry_run, calls[0].observe),
                         ("acct", 30, True, False))

    def test_rules_and_model_when_lm_studio_is_off(self):
        outcome, calls = self.scroll(False, [{"status": "ok"}])
        self.assertEqual((outcome["mode"], [c.rules for c in calls]), ("rules", [True]))

    def test_falls_back_when_the_llm_fails_to_warm_up(self):
        outcome, calls = self.scroll(True, [{"status": "llm_unavailable"}, {"status": "ok"}])
        self.assertEqual((outcome["status"], outcome["mode"], [c.rules for c in calls]), ("ok", "rules", [False, True]))

    def test_headless_reaches_the_session(self):
        run = mock.AsyncMock(return_value={"status": "ok"})
        with mock.patch.object(daily, "llm_serving", return_value=True), mock.patch.object(daily.agent, "run", run), \
                contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(daily.scroll("acct", 30, True, False, True))
        self.assertTrue(run.await_args.args[0].headless)

    def test_start_llm_tries_the_llm_when_it_is_off(self):
        _, calls = self.scroll(False, [{"status": "ok"}], start_llm=True)
        self.assertEqual([c.rules for c in calls], [False])


class EngagementTests(unittest.TestCase):
    def test_choose_dry_run(self):
        with mock.patch.object(daily, "engaged_today", return_value=False) as engaged:
            self.assertEqual(daily.choose_dry_run("live", "acct"), (False, "live"))
            self.assertEqual(daily.choose_dry_run("dry", "acct"), (True, "dry run"))
            engaged.assert_not_called()
            self.assertEqual(daily.choose_dry_run("mixed", "acct"), (False, "mixed → live (first session today)"))
            engaged.return_value = True
            self.assertEqual(daily.choose_dry_run("mixed", "acct"),
                             (True, "mixed → dry run (today's live session already ran)"))

    def test_command_line(self):
        self.assertEqual(daily.parse_args([]).engagement, DAILY_ENGAGEMENT)
        self.assertEqual(daily.parse_args(["--dry-run"]).engagement, "dry")
        self.assertEqual(daily.parse_args(["--engagement", "live"]).engagement, "live")
        self.assertEqual((daily.parse_args([]).headless, daily.parse_args(["--headless"]).headless), (False, True))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            daily.parse_args(["--engagement", "sometimes"])


class DailyTests(unittest.TestCase):
    def run_daily(self, used, outcome=None, digest_only=False, engagement="dry", engaged=False):
        args = argparse.Namespace(digest_only=digest_only, minutes=40, engagement=engagement, start_llm=False,
                                  open=False, headless=False)
        digest = mock.Mock(stories=[mock.Mock(score=0.9)], checks=[mock.Mock(label="In the news: 3 outlets")])
        scroll, notify, build = mock.AsyncMock(return_value=outcome), mock.Mock(), mock.Mock(return_value=digest)
        with mock.patch.multiple(daily, minutes_today=mock.Mock(return_value=used),
                                 engaged_today=mock.Mock(return_value=engaged),
                                 battery=mock.Mock(return_value=(False, None)), scroll=scroll,
                                 get_connection=mock.Mock(), build_digest=build,
                                 write_digest=mock.Mock(return_value=("/d/acct/2026-09-26.html", "/d/acct/x.json")),
                                 story_title=mock.Mock(return_value="UPI fee rules"), notify=notify), \
                contextlib.redirect_stdout(io.StringIO()):
            code = daily.daily(args, "acct")
        return code, scroll, notify, build

    def test_session_then_digest_and_notification(self):
        outcome = {"status": "ok", "account": "acct", "mode": "llm", "minutes": 39.6, "reels": 80,
                   "informative": 30, "scored": 70, "dry_run": True}
        code, scroll, notify, _ = self.run_daily(used=0, outcome=outcome)
        self.assertEqual(code, 0)
        scroll.assert_awaited_once_with("acct", 40, True, False, False)
        notify.assert_called_once_with("PRAG · @acct", "1 story. Top: UPI fee rules",
                                       subtitle="Scrolled 40 min (LLM, dry run): 80 reels, 30 of 70 informative.")

    def test_mixed_engages_only_in_the_first_session_of_the_day(self):
        _, first, _, _ = self.run_daily(used=0, outcome={"status": "ok"}, engagement="mixed", engaged=False)
        _, later, _, _ = self.run_daily(used=40, outcome={"status": "ok"}, engagement="mixed", engaged=True)
        self.assertEqual((first.await_args.args[2], later.await_args.args[2]), (False, True))

    def test_session_is_shortened_to_the_daily_budget(self):
        _, scroll, _, _ = self.run_daily(used=DAILY_MAX_MINUTES - 20, outcome={"status": "ok"})
        self.assertEqual(scroll.await_args.args[1], 20)

    def test_over_budget_skips_scrolling_but_refreshes_the_digest(self):
        code, scroll, notify, build = self.run_daily(used=DAILY_MAX_MINUTES)
        self.assertEqual(code, 0)
        scroll.assert_not_awaited()
        build.assert_called_once()
        self.assertTrue(notify.call_args.kwargs["subtitle"].startswith("Didn't scroll: @acct already scrolled"))

    def test_login_needed_is_reported(self):
        code, _, notify, _ = self.run_daily(used=0, outcome={"status": "login_needed", "account": "acct"})
        self.assertEqual(code, 1)
        self.assertIn("browser window (not headless)", notify.call_args.kwargs["subtitle"])

    def test_interrupted_session_stops_without_a_digest(self):
        code, _, notify, build = self.run_daily(used=0, outcome={"status": "interrupted", "account": "acct"})
        self.assertEqual(code, 130)
        build.assert_not_called()
        notify.assert_not_called()


class OutputTests(unittest.TestCase):
    def test_tee_writes_to_both(self):
        terminal, log = io.StringIO(), io.StringIO()
        tee = daily.Tee(terminal, log)
        print("hello", file=tee)
        self.assertEqual((terminal.getvalue(), log.getvalue(), tee.getvalue()), ("hello\n",) * 3)

    def test_prune_logs_keeps_the_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            for day in range(1, 6):
                open(os.path.join(tmp, f"2026-09-0{day}_0900.log"), "w").close()
            daily.prune_logs(tmp, keep=2)
            self.assertEqual(sorted(os.listdir(tmp)), ["2026-09-04_0900.log", "2026-09-05_0900.log"])


if __name__ == "__main__":
    unittest.main()
