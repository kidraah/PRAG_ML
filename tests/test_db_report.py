"""Tests for database/db.py (sessions, impressions) and dashboard/report.py, on an in-memory DB."""

import sqlite3
import unittest

from config import CURATION_TARGET_SHARE, CURATION_WINDOW
from dashboard.report import curation_verdict, on_target, render, window_stats
from database.db import (
    choose_account, init_db, load_creator_history, log_impression, update_impression_score,
)


def _db(test: unittest.TestCase) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    test.addCleanup(conn.close)
    init_db(conn)
    return conn


def _session(conn, scorer="v1", dry_run=0, account="acct") -> int:
    cur = conn.execute("INSERT INTO sessions (account, dry_run, model, scorer) VALUES (?, ?, 'm', ?)",
                       (account, dry_run, scorer))
    conn.commit()
    return cur.lastrowid


def _reel(conn, session_id, seq, *, score=None, basis="score", creator="c", code=None, decision="SKIP",
          dwell=2.0, is_ad=0, gate_audit=0, decision_basis=None, **extra):
    row = dict(session_id=session_id, seq=seq, code=code or f"R{session_id}_{seq}", creator=creator,
               is_ad=is_ad, score_basis=basis, value_score=score, decision=decision,
               decision_basis=decision_basis or basis, dwell_seconds=dwell, gate_audit=gate_audit,
               url=f"https://www.instagram.com/reels/{code or seq}/", **extra)
    if basis is not None:
        row["scored_at"] = "2026-09-26 10:00:00"
    return log_impression(row, conn)


class ImpressionTests(unittest.TestCase):
    def test_rejects_unknown_columns(self):
        conn = _db(self)
        with self.assertRaises(ValueError):
            log_impression({"session_id": _session(conn), "seq": 1, "bogus": 1}, conn)

    def test_late_score_fills_only_unscored_rows(self):
        conn = _db(self)
        sid = _session(conn)
        _reel(conn, sid, 1, code="LATE", basis=None, decision_basis="timeout")
        _reel(conn, sid, 2, code="DONE", score=0.9, summary="kept")
        self.assertEqual(update_impression_score(sid, "LATE", {"score_basis": "score", "value_score": 0.8,
                                                               "not_a_column": 1}, conn), 1)
        self.assertEqual(update_impression_score(sid, "DONE", {"value_score": 0.1}, conn), 0)
        rows = dict(conn.execute("SELECT code, value_score FROM impressions").fetchall())
        self.assertEqual(rows, {"LATE": 0.8, "DONE": 0.9})
        self.assertEqual(update_impression_score(sid, "LATE", {"value_score": 0.2}, conn), 0)  # only once

    def test_creator_history_is_per_scorer_and_account(self):
        conn = _db(self)
        old, new = _session(conn, scorer="old"), _session(conn, scorer="new")
        other = _session(conn, scorer="new", account="test_acct")
        _reel(conn, old, 1, score=0.9, creator="a", code="X")
        _reel(conn, new, 1, score=0.2, creator="b", code="Y")
        _reel(conn, new, 2, score=0.4, creator="b", code="Y")  # same reel shown twice → best score
        _reel(conn, new, 3, basis="keyword_gate", creator="c", code="Z", followed=1)
        _reel(conn, other, 1, score=0.95, creator="d", code="W", followed=1)
        scores, followed = load_creator_history("new", "acct", conn)
        self.assertEqual(scores, [("b", "Y", 0.4)])
        self.assertEqual(followed, {"c"})
        self.assertEqual(load_creator_history("new", "test_acct", conn), ([("d", "W", 0.95)], {"d"}))


class MigrationTests(unittest.TestCase):
    def test_older_database_gets_new_columns(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE impressions (id INTEGER PRIMARY KEY, session_id INTEGER, seq INTEGER, creator TEXT)")
        conn.execute("INSERT INTO impressions (session_id, seq) VALUES (1, 1)")
        init_db(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(impressions)")}
        self.assertTrue({"rule_score", "rule_reason", "ready_on_arrival"} <= columns)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM impressions").fetchone()[0], 1)  # data kept
        init_db(conn)  # running again is harmless


class ChooseAccountTests(unittest.TestCase):
    accounts = [{"username": "main", "status": "ACTIVE"}, {"username": "test1", "status": "ACTIVE"},
                {"username": "old", "status": "BANNED"}]

    def test_explicit_choice(self):
        self.assertEqual(choose_account(self.accounts, "test1"), ("test1", None))
        self.assertEqual(choose_account(self.accounts, "@main"), ("main", None))
        self.assertIsNone(choose_account(self.accounts, "old")[0])
        self.assertIn("Unknown account @nope", choose_account(self.accounts, "nope")[1])

    def test_never_picks_between_several_on_its_own(self):
        username, problem = choose_account(self.accounts, None)
        self.assertIsNone(username)
        self.assertIn("--account", problem)
        self.assertEqual(choose_account(self.accounts[:1], None), ("main", None))
        self.assertIn("No ACTIVE accounts", choose_account([], None)[1])


class ReportMetricTests(unittest.TestCase):
    def rows(self, conn):
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM impressions ORDER BY session_id, seq").fetchall()

    def test_on_target_rules(self):
        conn = _db(self)
        sid = _session(conn)
        _reel(conn, sid, 1, score=0.5)                         # informative
        _reel(conn, sid, 2, score=0.1)                         # not
        _reel(conn, sid, 3, basis="keyword_gate")              # gated → presumed not
        _reel(conn, sid, 4, basis="ad", is_ad=1)               # excluded
        _reel(conn, sid, 5, basis=None, decision_basis="timeout")  # never scored → excluded
        _reel(conn, sid, 6, basis="llm_error")                 # excluded
        self.assertEqual([on_target(r) for r in self.rows(conn)], [True, False, False, None, None, None])

    def test_windows_and_verdict(self):
        conn = _db(self)
        sid = _session(conn)
        n = CURATION_WINDOW * 2
        for i in range(n):
            # first half poor (10% informative), second half curated (80%)
            good = (i % 10 == 0) if i < n // 2 else (i % 10 < 8)
            _reel(conn, sid, i + 1, score=0.8 if good else 0.1)
        rows = self.rows(conn)
        windows = window_stats(rows, CURATION_WINDOW)
        self.assertEqual([round(w["informative"], 2) for w in windows], [0.1, 0.8])
        verdict = curation_verdict(rows)
        self.assertTrue(verdict["enough"])
        self.assertTrue(verdict["curated"])
        self.assertGreaterEqual(verdict["share"], CURATION_TARGET_SHARE)
        early = curation_verdict(rows[: CURATION_WINDOW // 2])
        self.assertFalse(early["enough"])
        self.assertFalse(early["curated"])

    def test_render(self):
        conn = _db(self)
        self.assertIn("No sessions yet", render(conn))

        sid = _session(conn)
        _reel(conn, sid, 1, score=0.92, creator="good", code="G1", decision="FULL_WATCH", dwell=30.0,
              summary="Government launched three websites for Gen Z.", topic="govt schemes", liked=1)
        _reel(conn, sid, 2, score=0.1, decision="SKIP", dwell=2.0, topic="devotional")
        _reel(conn, sid, 3, basis="keyword_gate", decision="SKIP", dwell=1.5)
        _reel(conn, sid, 4, score=0.6, decision="WATCH", dwell=9.0, gate_audit=1, topic="economy")
        _reel(conn, sid, 5, basis=None, decision_basis="timeout", decision="SKIP", dwell=6.0)
        text = render(conn, window=2)
        self.assertIn("reels     1–2", text)
        self.assertIn(f"need {CURATION_WINDOW} for a verdict", text)
        self.assertIn("SKIP 2.0s (n=3)", text)
        self.assertIn("FULL_WATCH 30.0s (n=1)", text)
        self.assertIn("decided before the score arrived: 20%", text)
        self.assertIn("Gate audit: 1 gated reels scored anyway — 1 were informative", text)
        self.assertIn("@good  “Government launched three websites for Gen Z.”", text)
        self.assertIn("https://www.instagram.com/reels/G1/", text)

    def test_render_is_per_account(self):
        conn = _db(self)
        main = _session(conn, account="main")
        _reel(conn, main, 1, score=0.9, creator="main_creator", code="M1", summary="main account reel")
        test = _session(conn, account="test1")
        _reel(conn, test, 1, score=0.1, creator="test_creator", code="T1")
        latest = render(conn)
        self.assertIn("curation report for @test1", latest)
        self.assertIn("Other accounts (see --account): @main (1 sessions)", latest)
        self.assertNotIn("main account reel", latest)
        chosen = render(conn, account="main")
        self.assertIn("curation report for @main", chosen)
        self.assertIn("main account reel", chosen)
        self.assertIn("No sessions for @ghost", render(conn, account="ghost"))

    def test_render_ignores_older_scorer(self):
        conn = _db(self)
        old = _session(conn, scorer="old")
        _reel(conn, old, 1, score=0.9)
        new = _session(conn, scorer="new")
        _reel(conn, new, 1, score=0.1)
        text = render(conn)
        self.assertIn("1 reels scored with an older prompt/INTERESTS/model not compared", text)
        self.assertIn("(old scorer)", text)


if __name__ == "__main__":
    unittest.main()
