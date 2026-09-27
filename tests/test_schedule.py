"""Tests for tools/schedule.py — plists go to a temporary folder and launchctl is never called."""

import argparse
import os
import plistlib
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import schedule


def fake_launchctl(loaded: bool = True):
    def run(*args):
        if args[0] == "print":
            return subprocess.CompletedProcess(args, 0 if loaded else 113, "\tlast exit code = 0\n" if loaded else "", "")
        return subprocess.CompletedProcess(args, 0, "", "")
    return mock.Mock(side_effect=run)


class ScheduleTests(unittest.TestCase):
    def test_parse_time(self):
        self.assertEqual(schedule.parse_time("9:05"), (9, 5))
        self.assertEqual(schedule.parse_time("21:30"), (21, 30))
        for bad in ("24:00", "9", "9:5", "noon"):
            with self.assertRaises(argparse.ArgumentTypeError):
                schedule.parse_time(bad)

    def test_plist(self):
        plist = schedule.build_plist("acct", [(21, 0), (9, 30), (9, 30)], minutes=30, engagement="dry",
                                     start_llm=False, python="/venv/bin/python", root="/proj")
        self.assertEqual(plist["Label"], "com.feedlab.daily.acct")
        self.assertEqual(plist["ProgramArguments"],
                         ["/venv/bin/python", "/proj/daily.py", "--account", "acct", "--engagement", "dry",
                          "--minutes", "30"])
        self.assertEqual(plist["StartCalendarInterval"], [{"Hour": 9, "Minute": 30}, {"Hour": 21, "Minute": 0}])
        self.assertEqual((plist["WorkingDirectory"], plist["StandardOutPath"]), ("/proj", "/proj/logs/launchd-acct.log"))
        self.assertIn("/opt/homebrew/bin", plist["EnvironmentVariables"]["PATH"].split(":"))
        self.assertEqual(plist["EnvironmentVariables"]["FEEDLAB_LOG_ONLY"], "1")
        self.assertEqual(schedule.read(plist), {"account": "acct", "times": ["09:30", "21:00"], "minutes": "30",
                                                "engagement": "dry", "headless": False, "start_llm": False})
        self.assertEqual(schedule.describe(plist),
                         "@acct at 09:30, 21:00 · 30 min · dry: watch time only, no likes, saves or follows")
        mixed = schedule.build_plist("acct", [(8, 0)], minutes=None, engagement="mixed", start_llm=True,
                                     headless=True, python="/venv/bin/python", root="/proj")
        self.assertNotIn("--minutes", mixed["ProgramArguments"])
        self.assertEqual(mixed["ProgramArguments"][-2:], ["--headless", "--start-llm"])
        self.assertEqual(schedule.describe(mixed), "@acct at 08:00 · default min · mixed: live for the day's first "
                                                   "session, dry runs after · headless · starts LM Studio if needed")

    def test_install_show_remove(self):
        with tempfile.TemporaryDirectory() as agents, tempfile.TemporaryDirectory() as root, \
                mock.patch.object(schedule, "_launchctl", fake_launchctl()) as launchctl:
            plist = schedule.build_plist("acct", [(9, 0)], minutes=None, engagement="live", start_llm=False,
                                         python="/venv/bin/python", root=root)
            path = schedule.install("acct", plist, agents_dir=agents)
            with open(path, "rb") as fh:
                self.assertEqual(plistlib.load(fh), plist)
            self.assertTrue(os.path.isdir(os.path.join(root, "logs")))
            self.assertEqual(launchctl.call_args_list[-1].args[:2], ("bootstrap", f"gui/{os.getuid()}"))
            self.assertEqual(schedule.show(agents_dir=agents),
                             ["@acct at 09:00 · default min · live: likes, saves and follows · loaded · last exit: 0"])
            self.assertTrue(schedule.remove("acct", agents_dir=agents))
            self.assertFalse(os.path.exists(path))
            self.assertFalse(schedule.remove("acct", agents_dir=agents))
            self.assertEqual(schedule.show(agents_dir=agents), [])

    def test_install_reports_a_launchctl_failure(self):
        failing = mock.Mock(return_value=subprocess.CompletedProcess([], 5, "", "Bootstrap failed: 5: Input/output error"))
        with tempfile.TemporaryDirectory() as agents, tempfile.TemporaryDirectory() as root, \
                mock.patch.object(schedule, "_launchctl", failing), mock.patch.object(schedule.time, "sleep"):
            plist = schedule.build_plist("acct", [(9, 0)], minutes=None, engagement="dry", start_llm=False,
                                         python="/venv/bin/python", root=root)
            with self.assertRaisesRegex(RuntimeError, "Input/output error"):
                schedule.install("acct", plist, agents_dir=agents)


if __name__ == "__main__":
    unittest.main()
