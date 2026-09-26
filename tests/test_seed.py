"""Tests for database/seed.py, on an in-memory DB and a temporary accounts file."""

import json
import os
import sqlite3
import tempfile
import unittest

from database.db import init_db
from database.seed import account_password, seed_accounts
from tools.follow_seed import read_handles


class SeedTests(unittest.TestCase):
    def test_skips_blank_slots_and_never_stores_passwords(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        init_db(conn)
        entries = [
            {"username": "main", "password": "secret-1", "status": "ACTIVE"},
            {"username": "@test1", "password": "secret-2"},
            {"username": "", "password": "", "status": "ACTIVE"},
            {"username": "   "},
            {"username": "old", "status": "BANNED"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "accounts.json")
            with open(path, "w") as fh:
                json.dump(entries, fh)
            self.assertEqual(seed_accounts(path, conn), ["main", "test1", "old"])
            seed_accounts(path, conn)  # re-seeding adds nothing

        rows = conn.execute("SELECT username, password, status FROM accounts ORDER BY username").fetchall()
        self.assertEqual(rows, [("main", "", "ACTIVE"), ("old", "", "BANNED"), ("test1", "", "ACTIVE")])

    def test_account_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "accounts.json")
            with open(path, "w") as fh:
                json.dump([{"username": "@with_pw", "password": "pw"}, {"username": "no_pw"}], fh)
            self.assertEqual(account_password("with_pw", path), "pw")
            self.assertIsNone(account_password("no_pw", path))
            self.assertIsNone(account_password("missing", path))
            self.assertIsNone(account_password("with_pw", os.path.join(tmp, "absent.json")))

    def test_read_handles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seeds.txt")
            with open(path, "w") as fh:
                fh.write("# comment\nndtv\n@thehindu  # with a note\n\nndtv\n")
            self.assertEqual(read_handles(path), ["ndtv", "thehindu"])


if __name__ == "__main__":
    unittest.main()
