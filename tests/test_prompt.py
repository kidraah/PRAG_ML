"""Tests for automation/prompt.py — the dialog box is a stand-in process, so nothing appears on screen."""

import asyncio
import os
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock

from automation import prompt


def stand_in(code: str):
    """A fake dialog: a process running `code` (print = the typed answer)."""
    async def open_dialog(message, secret, timeout):
        return await asyncio.create_subprocess_exec(sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
    return open_dialog


async def no_dialog(message, secret, timeout):
    return None


NO_PANEL = object()  # the panel doesn't answer (None would be a Cancel there)


class PromptTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = os.path.join(tmp.name, "prompts")
        for name, value in (("PROMPT_DIR", self.dir), ("RESOLVED_EVERY", 0.1)):
            patcher = mock.patch.object(prompt, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def ask(self, dialog=no_dialog, panel=NO_PANEL, timeout=10.0, resolved=None):
        """Run ask(); `panel` (if given) answers from the panel once the question shows up there."""
        seen = {}

        async def scenario():
            with mock.patch.object(prompt, "_open_dialog", dialog):
                task = asyncio.create_task(prompt.ask("Code for @acct?", account="acct", kind="code",
                                                      timeout=timeout, resolved=resolved))
                if panel is not NO_PANEL:
                    while not prompt.pending():
                        await asyncio.sleep(0.02)
                    seen["pending"] = prompt.pending()
                    seen["mode"] = stat.S_IMODE(os.stat(self.dir).st_mode)
                    self.assertTrue(prompt.answer(seen["pending"][0]["id"], panel))
                started = time.monotonic()
                result = await task
                return result, time.monotonic() - started

        result, seconds = asyncio.run(scenario())
        self.assertEqual(os.listdir(self.dir) if os.path.isdir(self.dir) else [], [])  # nothing left behind
        return result, seconds, seen

    def test_answered_in_the_panel(self):
        result, _, seen = self.ask(panel=" 123456 ")
        self.assertEqual(result, "123456")
        waiting = seen["pending"][0]
        self.assertEqual((waiting["account"], waiting["kind"], waiting["secret"]), ("acct", "code", False))
        self.assertEqual(seen["mode"], 0o700)  # only this user can read the answers

    def test_answered_in_the_dialog(self):
        self.assertEqual(self.ask(dialog=stand_in("print('654321')"))[0], "654321")

    def test_cancel_in_the_dialog(self):
        cancel = "import sys; sys.stderr.write('execution error: User canceled. (-128)'); sys.exit(1)"
        result, seconds, _ = self.ask(dialog=stand_in(cancel))
        self.assertIsNone(result)
        self.assertLess(seconds, 5)

    def test_dialog_that_gives_up(self):
        self.assertIsNone(self.ask(dialog=stand_in(f"print({prompt.NO_ANSWER!r})"))[0])

    def test_the_panel_can_answer_when_the_dialog_cannot_open(self):
        self.assertEqual(self.ask(dialog=stand_in("import sys; sys.exit(1)"), panel="777")[0], "777")

    def test_a_panel_answer_closes_the_dialog(self):
        processes = []

        async def slow_dialog(message, secret, timeout):
            process = await stand_in("import time; time.sleep(60)")(message, secret, timeout)
            processes.append(process)
            return process

        result, seconds, _ = self.ask(dialog=slow_dialog, panel="4242")
        self.assertEqual(result, "4242")
        self.assertIsNotNone(processes[0].returncode)  # the dialog is gone
        self.assertLess(seconds, 5)

    def test_cancel_in_the_panel(self):
        result, seconds, _ = self.ask(dialog=stand_in("import time; time.sleep(60)"), panel=None)
        self.assertIsNone(result)
        self.assertLess(seconds, 5)

    def test_nobody_answers(self):
        result, seconds, _ = self.ask(timeout=0.6)
        self.assertIsNone(result)
        self.assertLess(seconds, 3)

    def test_stops_asking_once_it_no_longer_matters(self):
        async def logged_in_in_the_window():
            return True

        result, seconds, _ = self.ask(dialog=stand_in("import time; time.sleep(60)"), resolved=logged_in_in_the_window)
        self.assertIsNone(result)
        self.assertLess(seconds, 5)

    def test_pending_and_answer_housekeeping(self):
        self.assertEqual(prompt.pending(), [])
        self.assertFalse(prompt.answer("../../etc/passwd", "x"))
        self.assertFalse(prompt.answer("0" * 32, "x"))  # no such question
        prompt._write(prompt._path("request", "a" * 32), {"id": "a" * 32, "expires": time.time() - 1})
        self.assertEqual(prompt.pending(), [])  # expired: dropped
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
