"""Tests for database/accounts.py — a temporary accounts.json, database and browser-profile folder."""

import json
import os
import sqlite3
import stat
import tempfile
import unittest
from unittest import mock

import automation.browser_driver as browser_driver
from database import accounts
from database.db import init_db, log_impression
from database.seed import account_password

SECRET = "correct horse battery staple"


class AccountsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(self.dir, "accounts.json")
        with open(self.path, "w") as fh:  # an existing account (with a field PRAG doesn't use) and an empty slot
            json.dump([{"username": "old_acct", "password": "old-pass", "status": "ACTIVE", "note": "keep me"},
                       {"username": "", "password": "", "status": "ACTIVE"}], fh)
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        init_db(self.conn)
        self.conn.execute("INSERT INTO accounts (username, password, status) VALUES ('old_acct', '', 'ACTIVE')")
        patcher = mock.patch.object(browser_driver, "PROFILES_DIR", os.path.join(self.dir, "profiles"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.kw = {"conn": self.conn, "path": self.path}

    def entries(self):
        with open(self.path) as fh:
            return json.load(fh)

    def test_add_fills_the_empty_slot_privately_and_keeps_the_rest(self):
        self.assertEqual(accounts.add("@new_acct", SECRET, **self.kw), "new_acct")
        entries = self.entries()
        self.assertEqual(entries[0], {"username": "old_acct", "password": "old-pass", "status": "ACTIVE", "note": "keep me"})
        self.assertEqual(entries[1], {"username": "new_acct", "password": SECRET, "status": "ACTIVE"})
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)  # it holds passwords
        self.assertEqual(account_password("new_acct", self.path), SECRET)  # what login reads
        self.assertEqual(self.conn.execute("SELECT password, status FROM accounts WHERE username = 'new_acct'").fetchone(),
                         ("", "ACTIVE"))  # never the password in the database
        accounts.add("third_acct", None, **self.kw)  # no empty slot left: appended, without a password
        self.assertEqual(self.entries()[2], {"username": "third_acct", "password": "", "status": "ACTIVE"})

    def test_add_refuses_bad_names_and_duplicates(self):
        for bad in ("", "has space", "semi;colon", "x" * 31, "../etc"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "username"):
                accounts.add(bad, "pw", **self.kw)
        with self.assertRaisesRegex(ValueError, "already added"):
            accounts.add("old_acct", "pw", **self.kw)
        with self.assertRaisesRegex(ValueError, "longer than"):
            accounts.add("new_acct", "x" * 200, **self.kw)

    def test_overview_never_includes_passwords(self):
        accounts.add("new_acct", SECRET, **self.kw)
        items = {item["username"]: item for item in accounts.overview(**self.kw)}
        self.assertEqual(set(items), {"old_acct", "new_acct"})
        self.assertTrue(items["new_acct"]["has_password"])
        self.assertNotIn(SECRET, json.dumps(items))
        self.assertNotIn("old-pass", json.dumps(items))
        self.assertEqual((items["old_acct"]["reels"], items["old_acct"]["login_saved"]), (0, False))
        self.assertFalse(items["old_acct"]["persona_set"])  # scrolls with the default until given a persona

    def test_set_and_forget_a_password(self):
        accounts.set_password("old_acct", "new-pass", **self.kw)
        self.assertEqual(account_password("old_acct", self.path), "new-pass")
        accounts.set_password("old_acct", "", **self.kw)
        self.assertIsNone(account_password("old_acct", self.path))
        with self.assertRaisesRegex(ValueError, "isn't added"):
            accounts.set_password("nobody", "pw", **self.kw)

    def test_remove_keeps_the_collected_data(self):
        sid = self.conn.execute("INSERT INTO sessions (account, scorer) VALUES ('old_acct', 's')").lastrowid
        log_impression(dict(session_id=sid, seq=1, code="R1"), self.conn)
        profile = browser_driver.profile_dir("old_acct")
        os.makedirs(profile)
        self.assertEqual(accounts.remove("old_acct", **self.kw), {"username": "old_acct", "login_deleted": False})
        self.assertIsNone(accounts._entry(self.entries(), "old_acct"))
        self.assertIsNone(self.conn.execute("SELECT 1 FROM accounts WHERE username = 'old_acct'").fetchone())
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM impressions").fetchone()[0], 1)  # kept
        self.assertTrue(os.path.isdir(profile))  # the saved login stays unless asked
        with self.assertRaisesRegex(ValueError, "isn't added"):
            accounts.remove("old_acct", **self.kw)

    def test_remove_can_delete_the_saved_login(self):
        profile = browser_driver.profile_dir("old_acct")
        os.makedirs(profile)
        self.assertTrue(accounts.remove("old_acct", delete_login=True, **self.kw)["login_deleted"])
        self.assertFalse(os.path.exists(profile))

    def test_remove_waits_for_a_run_to_finish(self):
        with browser_driver.profile_lock("old_acct") as held:  # a session has the browser open
            self.assertTrue(held)
            with self.assertRaisesRegex(ValueError, "being used by a run"):
                accounts.remove("old_acct", **self.kw)
        self.assertIsNotNone(accounts._entry(self.entries(), "old_acct"))


if __name__ == "__main__":
    unittest.main()
