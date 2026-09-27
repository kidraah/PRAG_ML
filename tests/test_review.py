"""Tests for tools/review.py and --observe recordings in the report, on an in-memory DB."""

import sqlite3
import unittest

from dashboard.report import render
from database.db import REVIEW_SCORER, init_db, log_impression
from tools.review import export_reels, import_judgements


def _recorded_session(test: unittest.TestCase) -> tuple[sqlite3.Connection, int]:
    conn = sqlite3.connect(":memory:")
    test.addCleanup(conn.close)
    init_db(conn)
    sid = conn.execute(
        "INSERT INTO sessions (account, dry_run, model, scorer) VALUES ('tester', 1, 'none', ?)", (REVIEW_SCORER,)
    ).lastrowid
    for seq, (code, caption, cover, transcript) in enumerate([
        ("R1", "RBI ने रेपो रेट घटाया #news", "RBI ने घटाया रेपो रेट", "आरबीआई ने रेपो रेट घटाया"),
        ("R2", "जय श्री राम 🚩", "", ""),
    ], start=1):
        log_impression(dict(session_id=sid, seq=seq, code=code, url=f"https://www.instagram.com/reels/{code}/",
                            creator="c", caption=caption, cover_text=cover, transcript=transcript,
                            taken_at=1790380800, score_basis="observed", decision="OBSERVE",
                            decision_basis="observe", dwell_seconds=3.2, scored_at="2026-09-26 10:00:00"), conn)
    log_impression(dict(session_id=sid, seq=3, code="AD", is_ad=1, score_basis="ad", decision="SKIP",
                        decision_basis="ad", dwell_seconds=1.4), conn)
    return conn, sid


class ReviewTests(unittest.TestCase):
    def test_export_lists_only_recorded_reels(self):
        conn, sid = _recorded_session(self)
        reels = export_reels(conn, "tester")
        self.assertEqual([r["code"] for r in reels], ["R1", "R2"])
        self.assertEqual(reels[0]["posted"], "2026-09-26")
        self.assertEqual(reels[0]["cover_text"], "RBI ने घटाया रेपो रेट")
        self.assertNotIn("cover_text", reels[1])          # empty fields are left out
        self.assertEqual(export_reels(conn, "someone_else"), [])
        self.assertEqual(len(export_reels(conn, "tester", session=sid + 1)), 0)

    def test_import_scores_reels_once_and_reports_problems(self):
        conn, sid = _recorded_session(self)
        judgements = [
            {"session": sid, "code": "R1", "value_score": 0.95, "is_informational": True,
             "topic": "RBI repo rate", "summary": "RBI cut the repo rate.", "reasoning": "Specific news."},
            {"session": sid, "code": "R2", "value_score": 0.0, "is_informational": False, "topic": "Devotional"},
            {"session": sid, "code": "NOPE", "value_score": 0.5},
            {"session": sid, "code": "R1", "value_score": 3},
            {"code": "R2"},
        ]
        updated, problems = import_judgements(conn, judgements)
        self.assertEqual(updated, 2)
        self.assertEqual(len(problems), 3)
        rows = conn.execute("SELECT code, score_basis, value_score, topic FROM impressions ORDER BY seq").fetchall()
        self.assertEqual(rows[:2], [("R1", "score", 0.95, "RBI repo rate"), ("R2", "score", 0.0, "Devotional")])
        self.assertEqual(import_judgements(conn, judgements[:1]), (0, [f"no recorded reel R1 awaiting review in session {sid}"]))

    def test_report_waits_for_review_then_counts(self):
        conn, sid = _recorded_session(self)
        before = render(conn)
        self.assertIn("2 recorded reels await review", before)
        self.assertIn("0 judged", before)
        self.assertIn("OBSERVE 3.2s (n=2)", before)
        import_judgements(conn, [{"session": sid, "code": "R1", "value_score": 0.9},
                                 {"session": sid, "code": "R2", "value_score": 0.0}])
        after = render(conn)
        self.assertNotIn("await review", after)
        self.assertIn("2 judged", after)

    def test_report_compares_rules_with_review_and_shows_preloading(self):
        conn, sid = _recorded_session(self)
        # The rules watched R1 (news word) and skipped R2 — both known before the reels appeared.
        conn.execute("UPDATE impressions SET rule_score = 0.6, rule_reason = 'news word: rbi', ready_on_arrival = 1 WHERE code = 'R1'")
        conn.execute("UPDATE impressions SET rule_score = 0.1, rule_reason = 'no news signal', ready_on_arrival = 1 WHERE code = 'R2'")
        conn.execute("UPDATE impressions SET ready_on_arrival = 0 WHERE code = 'AD'")
        conn.commit()
        import_judgements(conn, [{"session": sid, "code": "R1", "value_score": 0.9},
                                 {"session": sid, "code": "R2", "value_score": 0.5}])
        text = render(conn)
        self.assertIn("known before the reel appeared (preloaded feed data): 100%", text)  # ads don't count
        self.assertIn("Quick rules vs scores (2 scored reels the rules also judged)", text)
        self.assertIn("rules watched 1: 1 informative, 0 not · informative reels the rules skipped: 1 of 2", text)


if __name__ == "__main__":
    unittest.main()
