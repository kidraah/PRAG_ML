"""Tests for stories/: grouping, verification and the digest — no network, no embedding model."""

import datetime
import json
import os
import sqlite3
import tempfile
import unittest

import numpy as np

from config import VERIFY_DAYS, VERIFY_MAX_DAYS
from database.db import init_db, log_impression
from stories.digest import build_digest, render_html, to_json, write_digest
from stories.grouping import Reel, group
from stories.verify import Article, key_terms, parse_rss, search_window, verify

NOW = datetime.datetime(2026, 9, 26, 12, 0, tzinfo=datetime.timezone.utc)


def fake_embed(mapping: dict[str, list[float]]):
    """Embedding stand-in: texts containing a key get its (normalised) vector, others a far-away one."""
    def embed(texts):
        out = []
        for text in texts:
            vec = next((v for k, v in mapping.items() if k in text), [0.0, 0.0, 1.0])
            vec = np.asarray(vec, dtype=float)
            out.append(vec / np.linalg.norm(vec))
        return np.vstack(out)
    return embed


def _reel(code, topic, summary, score, minutes_ago=0, taken_at=None):
    return Reel(code=code, url=f"https://www.instagram.com/reels/{code}/", creator=f"c_{code}", topic=topic,
                summary=summary, caption="", value_score=score,
                first_seen=NOW - datetime.timedelta(minutes=minutes_ago), taken_at=taken_at)


class GroupingTests(unittest.TestCase):
    def test_same_story_reels_merge_and_order_by_value(self):
        embed = fake_embed({"Ghosh": [1, 0, 0], "Hormuz": [0, 1, 0]})
        reels = [
            _reel("A", "West Bengal politics", "TMC MLA Kunal Ghosh's car attacked", 0.85, minutes_ago=5),
            _reel("B", "West Bengal politics", "Kunal Ghosh alleges attack on car", 0.90, minutes_ago=30),
            _reel("C", "Strait of Hormuz", "Iran says Hormuz could reopen", 0.95),
        ]
        stories = group(reels, embed_fn=embed)
        self.assertEqual([len(s.reels) for s in stories], [1, 2])
        self.assertEqual(stories[0].title, "Strait of Hormuz")
        ghosh = stories[1]
        self.assertEqual((ghosh.lead.code, ghosh.score), ("B", 0.90))
        self.assertEqual(ghosh.first_seen, NOW - datetime.timedelta(minutes=30))
        self.assertEqual(ghosh.creators, ["c_A", "c_B"])

    def test_empty_and_single(self):
        self.assertEqual(group([], embed_fn=fake_embed({})), [])
        self.assertEqual(len(group([_reel("A", "t", "s", 0.5)], embed_fn=fake_embed({}))), 1)


class VerifyTests(unittest.TestCase):
    def test_key_terms(self):
        self.assertEqual(key_terms("US diesel export ban", "The US is weighing a 90-day diesel export ban; India says no."),
                         ["diesel", "export", "ban"])
        self.assertEqual(key_terms("Asian Games kabaddi", "India's women's kabaddi team won gold at the 2026 Asian Games."),
                         ["Asian", "Games", "kabaddi"])
        self.assertEqual(key_terms("ISRO EOS-05 launch", "Claims ISRO placed EOS-05 into orbit from Sriharikota."),
                         ["ISRO", "EOS-05", "Sriharikota", "launch"])

    def test_search_window_follows_posting_date(self):
        ts = lambda days: int((NOW - datetime.timedelta(days=days)).timestamp())
        self.assertEqual(search_window(None, NOW), VERIFY_DAYS)
        self.assertEqual(search_window(ts(0), NOW), VERIFY_DAYS)
        self.assertEqual(search_window(ts(23), NOW), 25)
        self.assertEqual(search_window(ts(300), NOW), VERIFY_MAX_DAYS)

    def test_parse_rss(self):
        xml = """<rss><channel>
          <item><title>US mulls 90-day diesel export ban - Bloomberg</title><link>https://x/1</link>
                <pubDate>Wed, 23 Sep 2026 10:00:00 GMT</pubDate><source url="https://bloomberg.com">Bloomberg</source></item>
          <item><title>No source here</title><link>https://x/2</link></item>
        </channel></rss>"""
        articles = parse_rss(xml)
        self.assertEqual((articles[0].title, articles[0].source, articles[0].published),
                         ("US mulls 90-day diesel export ban", "Bloomberg", "2026-09-23"))
        self.assertEqual(articles[1].title, "No source here")

    def _verify(self, articles, **kwargs):
        calls = []

        def search(query, days):
            calls.append((query, days))
            return list(articles)

        embed = fake_embed({"diesel export": [1, 0, 0], "Diesel": [0.95, 0.31, 0], "cricket": [0, 1, 0]})
        # "Washington" makes the first query differ from the broader, topic-only one.
        result = verify("US diesel export ban", "Washington is weighing a diesel export ban.", search=search,
                        embed_fn=embed, now=NOW, **kwargs)
        return result, calls

    def test_close_headlines_confirm_the_story(self):
        articles = [Article("US mulls Diesel ban", "Bloomberg", "u1"), Article("US mulls Diesel ban", "Bloomberg", "u1"),
                    Article("Diesel exports may stop", "Reuters", "u2"), Article("Diesel export curbs eyed", "Mint", "u3"),
                    Article("India wins cricket match", "ESPN", "u4"), Article("Diesel talk", "Semafor", "u5")]
        result, calls = self._verify(articles)
        self.assertEqual(result.status, "widely_reported")
        self.assertEqual(result.outlets, ["Bloomberg", "Reuters", "Mint", "Semafor"])
        self.assertEqual(len(result.headlines), 3)               # duplicates removed, best three kept
        self.assertEqual(len(calls), 1)                          # enough results: no broader search
        self.assertTrue(result.label.startswith("In the news: 4 outlets"))

    def test_no_close_headline_and_failures(self):
        result, calls = self._verify([Article("India wins cricket match", "ESPN", "u")])
        self.assertEqual((result.status, result.label), ("reels_only", "Not in the news yet"))
        self.assertEqual(len(calls), 2)                          # few results: broader search tried
        failing = verify("US diesel export ban", "The US weighs a diesel export ban.",
                         search=lambda q, d: (_ for _ in ()).throw(OSError("offline")), embed_fn=fake_embed({}))
        self.assertEqual(failing.status, "not_checked")
        self.assertEqual(verify("x", "", embed_fn=fake_embed({})).status, "not_checked")


class DigestTests(unittest.TestCase):
    def test_digest_from_database(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        init_db(conn)
        sid = conn.execute("INSERT INTO sessions (account, dry_run, model, scorer) VALUES ('acct', 1, 'm', 'review')").lastrowid
        stamp = lambda hours: (NOW - datetime.timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        rows = [
            ("K1", 0.90, "West Bengal politics", "Kunal Ghosh's car attacked <script>", 2),
            ("K2", 0.85, "West Bengal politics", "Kunal Ghosh alleges attack", 1),
            ("K2", 0.85, "West Bengal politics", "Kunal Ghosh alleges attack", 0.5),   # shown twice
            ("E1", 0.45, "CEO career advice", "Relationships matter more than skills", 1),
            ("J1", 0.10, "Devotional", "", 1),                                          # not informative
            ("OLD", 0.95, "Old story", "From two days ago", 48),                        # outside the window
        ]
        for seq, (code, score, topic, summary, hours) in enumerate(rows, start=1):
            log_impression(dict(session_id=sid, seq=seq, code=code, creator=f"c_{code}", score_basis="score",
                                value_score=score, topic=topic, summary=summary, decision="WATCH",
                                decision_basis="rules", dwell_seconds=30.0), conn)
            conn.execute("UPDATE impressions SET shown_at = ? WHERE session_id = ? AND seq = ?", (stamp(hours), sid, seq))
        conn.commit()

        digest = build_digest(conn, "acct", hours=24, verify=False, now=NOW,
                              embed_fn=fake_embed({"Ghosh": [1, 0, 0], "Relationships": [0, 1, 0]}))
        self.assertEqual((digest.reels_seen, digest.sessions, round(digest.minutes)), (5, 1, 2))
        self.assertEqual([s.title for s in digest.stories], ["West Bengal politics", "CEO career advice"])
        self.assertEqual(digest.informative_reels, 3)

        data = to_json(digest)
        self.assertEqual(data["stats"]["stories"], 2)
        self.assertEqual([r["code"] for r in data["stories"][0]["reels"]], ["K1", "K2"])
        self.assertEqual(data["stories"][0]["verification"]["status"], "not_checked")

        page = render_html(digest)
        self.assertIn("Top stories", page)
        self.assertIn("Also worth knowing", page)
        self.assertIn("&lt;script&gt;", page)          # reel text is escaped
        self.assertNotIn("<script>", page)

        with tempfile.TemporaryDirectory() as tmp:
            html_path, json_path = write_digest(digest, out_dir=tmp)
            self.assertTrue(os.path.exists(html_path))
            with open(json_path) as fh:
                self.assertEqual(json.load(fh)["account"], "acct")


if __name__ == "__main__":
    unittest.main()
