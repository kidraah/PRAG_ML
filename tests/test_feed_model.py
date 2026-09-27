"""Tests for ml/feed_model.py: topics learned from every account, described per persona and overall (no real model)."""

import os
import sqlite3
import tempfile
import unittest
from collections import Counter
from unittest import mock

import numpy as np

from dashboard import feed, report
from database.db import init_db, log_impression
from ml import feed_model

THEMES = {  # what each kind of reel says, and a direction in the fake embedding space
    "election": ("Election commission announces the vote count", 0),
    "cricket": ("Cricket match ends with a last-ball wicket", 1),
    "bhajan": ("Morning bhajan and aarti from the temple", 2),
    "gadget": ("Phone launch with a new gadget camera", 3),
}


def fake_embed(texts):
    """One direction per theme, so the topics are exactly the themes."""
    out = []
    for text in texts:
        axis = next(axis for words, axis in THEMES.values() if words.split()[0].lower() in text.lower())
        vector = np.zeros(8)
        vector[axis] = 1.0
        vector[4 + axis] = 0.05 * (len(text) % 3)  # a little spread within a theme
        out.append(vector)
    return np.array(out)


def _db(plan):
    """plan: (account, day, theme, how many) rows; every reel is new, shown in that order."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    sessions, seq, n = {}, Counter(), 0
    for account, day, theme, count in plan:
        if (account, day) not in sessions:
            sessions[(account, day)] = conn.execute("INSERT INTO sessions (account, scorer) VALUES (?, 's')",
                                                    (account,)).lastrowid
        sid = sessions[(account, day)]
        for _ in range(count):
            n += 1
            seq[sid] += 1
            log_impression(dict(session_id=sid, seq=seq[sid], code=f"R{n}", creator=f"{theme}_page",
                                caption=f"{THEMES[theme][0]} #{theme} part {n}", score_basis="score", value_score=0.5),
                           conn)
            category = {"election": "news_politics", "cricket": "sports", "bhajan": "religion",
                        "gadget": "tech_science"}[theme]
            conn.execute("UPDATE impressions SET shown_at = ?, category = ? WHERE code = ?",
                         (f"2026-09-{day:02d} 10:00:00", category, f"R{n}"))
    return conn


PLAN = [  # @left sees elections, then narrows onto them; @right sees bhajans and cricket
    ("left", 21, "election", 5), ("left", 21, "gadget", 5), ("left", 22, "election", 6),
    ("left", 23, "election", 8), ("left", 24, "election", 9), ("left", 24, "cricket", 1),
    ("right", 21, "bhajan", 6), ("right", 22, "cricket", 6), ("right", 23, "bhajan", 6), ("right", 24, "bhajan", 6),
]


class FeedModelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.model_path = os.path.join(tmp.name, "feed-model.joblib")
        self.vectors_path = os.path.join(tmp.name, "feed-vectors.npz")
        self.conn = _db(PLAN)
        # nothing counts as curated here: the whole feed is the warm-up
        patcher = mock.patch.object(feed, "curated_since", lambda rows: report.curated_since(rows, 1000, 0.5))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = []

        def embed(texts):
            self.calls.append(len(texts))
            return fake_embed(texts)
        self.embed = embed

    def train(self):
        return feed_model.train(self.conn, embed_fn=self.embed, path=self.model_path, vectors_path=self.vectors_path,
                                progress=lambda message: None)

    def test_learns_named_topics_from_every_account(self):
        model = self.train()
        self.assertEqual((model["reels"], model["accounts"]), (58, ["left", "right"]))
        self.assertEqual(model["k"], feed_model.topic_count(58))
        themes = {}
        for code, topic in model["assign"].items():
            caption = self.conn.execute("SELECT caption FROM impressions WHERE code = ?", (code,)).fetchone()[0]
            themes.setdefault(topic, set()).add(caption.split("#")[1].split()[0])
        self.assertTrue(all(len(t) == 1 for t in themes.values()))  # a topic never mixes two themes
        named = {t["id"]: t for t in model["topics"]}
        election = next(t for t, theme in themes.items() if theme == {"election"})
        self.assertIn("election", named[election]["keywords"])
        self.assertEqual(named[election]["category"], "news_politics")
        self.assertEqual(model["topics"][0]["size"], max(t["size"] for t in model["topics"]))  # biggest first

    def test_embeddings_are_computed_once(self):
        self.train()
        self.assertEqual(self.calls, [58])
        self.train()
        self.assertEqual(self.calls, [58])  # all cached
        log_impression(dict(session_id=1, seq=99, code="NEW", caption="Cricket final tonight", creator="c",
                            score_basis="score", value_score=0.5), self.conn)
        self.train()
        self.assertEqual(self.calls, [58, 1])

    def test_too_few_reels(self):
        conn = _db([("left", 21, "election", 5)])
        self.assertIsNone(feed_model.train(conn, embed_fn=fake_embed, path=self.model_path,
                                           vectors_path=self.vectors_path, progress=lambda message: None))
        info = feed_model.analysis(conn, path=self.model_path)
        self.assertEqual(info, {"trained": False, "reels": 5, "needed": feed_model.MIN_REELS})

    def test_one_personas_feed(self):
        self.train()
        info = feed_model.analysis(self.conn, "left", path=self.model_path)
        self.assertEqual((info["reels"], info["known"], info["newer"], info["of"]), (34, 34, 0, "all"))
        self.assertEqual([d["day"] for d in info["daily"]], ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"])
        first, *_rest, last = info["daily"]
        self.assertEqual((first["effective_topics"], first["moved"]), (2.0, None))  # half elections, half gadgets
        self.assertLess(last["effective_topics"], 1.5)  # then it narrows onto elections
        self.assertGreater(info["daily"][1]["moved"], 0.5)  # the gadgets went away
        keywords = {t["id"]: t["keywords"] for t in info["topics"]}
        self.assertIn("election", keywords[info["apart"][0]["topic"]])  # the most above the accounts overall
        self.assertNotIn("alike", info)

    def test_every_accounts_feed(self):
        self.train()
        info = feed_model.analysis(self.conn, path=self.model_path)
        self.assertEqual(info["reels"], 58)
        alike = info["alike"]
        self.assertEqual(alike["accounts"], ["left", "right"])
        self.assertEqual([alike["matrix"][0][0], alike["matrix"][1][1]], [1.0, 1.0])
        self.assertEqual(alike["matrix"][0][1], alike["matrix"][1][0])
        self.assertLess(alike["matrix"][0][1], 0.3)  # the two personas are shown very different feeds
        self.assertNotIn("apart", info)

    def test_rising_topics(self):
        conn = _db(PLAN + [("right", 25, "cricket", 8), ("right", 26, "cricket", 8)])
        feed_model.train(conn, embed_fn=fake_embed, path=self.model_path, vectors_path=self.vectors_path,
                         progress=lambda message: None)
        info = feed_model.analysis(conn, "right", path=self.model_path)
        keywords = {t["id"]: t["keywords"] for t in info["topics"]}
        self.assertIn("cricket", keywords[info["rising"][0]["topic"]])
        self.assertGreater(info["rising"][0]["to"], info["rising"][0]["from"])

    def test_newer_reels_wait_for_the_next_training(self):
        self.train()
        log_impression(dict(session_id=1, seq=99, code="NEW", caption="Cricket final tonight", creator="c",
                            score_basis="score", value_score=0.5), self.conn)
        info = feed_model.analysis(self.conn, "left", path=self.model_path)
        self.assertEqual((info["reels"], info["newer"]), (35, 1))

    def test_measures(self):
        self.assertEqual(feed_model.effective_topics(Counter({0: 5, 1: 5})), 2.0)
        self.assertEqual(feed_model.effective_topics(Counter({3: 7})), 1.0)
        self.assertEqual(feed_model.effective_topics(Counter()), 0.0)
        same, other = np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.0, 1.0])
        self.assertEqual(feed_model.js_distance(same, same), 0.0)
        self.assertEqual(feed_model.js_distance(same, other), 1.0)
        self.assertEqual(feed_model.tokens({"caption": "The #Election vote in दिल्ली is here, https://x.y @page"}),
                         {"election", "vote", "दिल्ली"})


if __name__ == "__main__":
    unittest.main()
