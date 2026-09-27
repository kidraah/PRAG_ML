"""Tests for automation/login.py — the browser and the questions are stand-ins."""

import asyncio
import contextlib
import io
import unittest
from unittest import mock

from automation import login

CODE_BOX = object()


class LoginTests(unittest.TestCase):
    def ensure(self, *, headless=False, password="pw", attempts=("challenge",), answers=(), code_field=None,
               code_works=(True,)):
        """ensure_logged_in with Instagram's reply to each password try (attempts), what's typed into each
        question (answers), and whether each entered code logs in (code_works)."""
        page = mock.Mock()
        page.screenshot = mock.AsyncMock()
        mocks = {
            "log_in": mock.AsyncMock(side_effect=list(attempts)),
            "ask": mock.AsyncMock(side_effect=list(answers)),
            "enter_code": mock.AsyncMock(),
            "wait_for_login": mock.AsyncMock(side_effect=lambda page, timeout: timeout > 60 or code_results.pop(0)),
        }
        code_results = list(code_works)
        with mock.patch.multiple(login, is_logged_in=mock.AsyncMock(return_value=False),
                                 account_password=mock.Mock(return_value=password),
                                 find_code_field=mock.AsyncMock(return_value=code_field), notify=mock.DEFAULT,
                                 **mocks) as patched, contextlib.redirect_stdout(io.StringIO()):
            ok = asyncio.run(login.ensure_logged_in(page, "acct", manual_timeout=600, headless=headless))
        return ok, {**mocks, "notify": patched["notify"], "page": page}

    def test_the_password_is_enough(self):
        ok, m = self.ensure(attempts=("ok",))
        self.assertTrue(ok)
        m["ask"].assert_not_awaited()
        m["notify"].assert_not_called()

    def test_the_security_code_is_asked_for_and_typed_in(self):
        ok, m = self.ensure(code_field=CODE_BOX, answers=("123456",))
        self.assertTrue(ok)
        self.assertEqual(m["ask"].await_args.kwargs["kind"], "code")
        self.assertFalse(m["ask"].await_args.kwargs.get("secret", False))
        m["enter_code"].assert_awaited_once_with(m["page"], CODE_BOX, "123456")

    def test_a_wrong_code_is_asked_for_again(self):
        ok, m = self.ensure(code_field=CODE_BOX, answers=("111111", "222222"), code_works=(False, True))
        self.assertTrue(ok)
        self.assertIn("didn't work", m["ask"].await_args_list[1].args[0])
        self.assertEqual(m["enter_code"].await_args.args[2], "222222")

    def test_a_rejected_password_is_asked_for(self):
        ok, m = self.ensure(attempts=("wrong_credentials", "ok"), answers=("right-pw",))
        self.assertTrue(ok)
        self.assertEqual((m["ask"].await_args.kwargs["kind"], m["ask"].await_args.kwargs["secret"]), ("password", True))
        self.assertEqual(m["log_in"].await_args.args[2], "right-pw")

    def test_no_password_asks_for_one(self):
        ok, m = self.ensure(password="", attempts=("ok",), answers=("typed",))
        self.assertTrue(ok)
        self.assertIn("none in accounts.json", m["ask"].await_args.args[0])
        self.assertEqual(m["log_in"].await_args.args[2], "typed")

    def test_cancelling_leaves_it_to_the_window(self):
        ok, m = self.ensure(code_field=CODE_BOX, answers=(None,))
        self.assertTrue(ok)  # logged in by hand in the window
        m["enter_code"].assert_not_awaited()
        self.assertEqual(m["wait_for_login"].await_args.kwargs["timeout"], 600)
        self.assertIn("browser window", m["notify"].call_args.args[1])

    def test_headless_without_a_code_box_gives_up(self):
        ok, m = self.ensure(headless=True, code_field=None)
        self.assertFalse(ok)
        m["wait_for_login"].assert_not_awaited()  # no window to log in by hand
        m["page"].screenshot.assert_awaited_once()  # what Instagram showed, for later
        self.assertIn("headless off", m["notify"].call_args.args[1])

    def test_headless_can_still_enter_a_code(self):
        ok, m = self.ensure(headless=True, code_field=CODE_BOX, answers=("123456",))
        self.assertTrue(ok)
        m["notify"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
