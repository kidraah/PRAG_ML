"""Tests for tools/headtohead.py and perception/jev.py — no network, no LLM."""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from database.db import init_db, log_impression
from perception import jev
from tools import headtohead

RUN = {
    "personas": {"p": {"name": "P", "interests": "x" * 30}},
    "reels": [{"code": c, "creator": c.lower(), "hindi": c == "D"} for c in "ABCD"],
    "scores": {"p": {"A": {"jev": {"score": 0.9, "seconds": 0.4, "cost": 1e-5}, "llm": {"score": 0.8, "seconds": 9}},
                     "B": {"jev": {"score": 0.1, "seconds": 0.3, "cost": 1e-5}, "llm": {"score": 0.7, "seconds": 11}},
                     "C": {"jev": {"score": 0.2, "seconds": 0.5, "cost": 1e-5}, "llm": {"score": 0.1, "seconds": 8}},
                     "D": {"jev": {"score": 0.6, "seconds": 0.4, "cost": 1e-5}, "llm": {"score": 0.2, "seconds": 10}}}},
}


class HeadToHeadTests(unittest.TestCase):
    def test_results_count_who_agrees_with_you(self):
        labels = {"p": {"A": "watch", "B": "skip", "C": "skip", "D": "watch"}}
        r = headtohead.results(RUN, labels)
        jev_, llm = r["personas"]["p"]["jev"], r["personas"]["p"]["llm"]
        self.assertEqual((jev_["agree"], jev_["n"]), (1.0, 4))
        self.assertEqual(llm["agree"], 0.5)  # it would watch B and skip D
        self.assertEqual(r["personas"]["p"]["jev"]["hindi"]["n"], 1)
        self.assertEqual({m["code"] for m in r["personas"]["p"]["misses"]}, {"B", "D"})
        self.assertEqual((r["seconds"]["jev"], r["seconds"]["llm"]), (0.4, 10))
        self.assertEqual(r["overall"]["jev"]["n"], 4)

    def test_unsure_and_unlabelled_are_left_out(self):
        r = headtohead.results(RUN, {"p": {"A": "unsure", "B": "skip"}})
        self.assertEqual((r["personas"]["p"]["labelled"], r["personas"]["p"]["jev"]["n"]), (1, 1))

    def test_save_label(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(headtohead, "load_run", return_value=RUN):
            path = os.path.join(tmp, "labels.json")
            self.assertEqual(headtohead.save_label("p", "A", "watch", path), {"p": {"A": "watch"}})
            self.assertEqual(headtohead.save_label("p", "A", None, path), {"p": {}})
            for persona, code, label in (("p", "Z", "watch"), ("nobody", "A", "watch"), ("p", "A", "maybe")):
                with self.subTest(code=code, label=label), self.assertRaises(ValueError):
                    headtohead.save_label(persona, code, label, path)

    def test_sample_is_varied_and_keeps_news(self):
        conn = sqlite3.connect(":memory:")
        init_db(conn)
        sid = conn.execute("INSERT INTO sessions (account, scorer) VALUES ('a', 's')").lastrowid
        n = 0
        for category, count in (("news_politics", 30), ("religion", 30), ("comedy_memes", 5), ("sports", 5)):
            for _ in range(count):
                n += 1
                log_impression(dict(session_id=sid, seq=n, code=f"R{n}", caption=f"caption {n}", creator="c"), conn)
                conn.execute("UPDATE impressions SET category = ? WHERE code = ?", (category, f"R{n}"))
        log_impression(dict(session_id=sid, seq=999, code="EMPTY", creator="c"), conn)  # no text: never picked
        sample = headtohead.pick_sample(conn, size=20)
        cats = [r["category"] for r in sample]
        self.assertEqual(len(sample), 20)
        self.assertEqual(cats.count("news_politics"), 6)  # a third
        # the other 14 go round the rest in turn, so the rare categories get as many as the common one
        self.assertEqual(sorted(cats.count(c) for c in ("religion", "comedy_memes", "sports")), [4, 5, 5])
        self.assertNotIn("EMPTY", {r["code"] for r in sample})
        self.assertIn("Caption: caption", sample[0]["text"])
        self.assertEqual(sample, headtohead.pick_sample(conn, size=20))  # the same sample every time


class JevTests(unittest.TestCase):
    def test_parse_scales_the_level_to_0_1(self):
        data = {"answers": {"value": {"type": "score", "score": 3.0, "confidence": 0.8}}, "usage": {"cost": 2e-05}}
        score = jev.parse(data, seconds=0.4)
        self.assertEqual((score.value_score, score.confidence, score.cost), (0.75, 0.8, 2e-05))
        with self.assertRaises(jev.JevError):
            jev.parse({"error": "nope"})

    def test_state_carries_the_persona_and_the_reel(self):
        state = jev.build_state("Caption: hello", "A cricket fan from Pune.")
        self.assertIn("A cricket fan from Pune.", state)
        self.assertTrue(state.endswith("Caption: hello"))
        self.assertEqual(jev.QUESTION["value"]["criteria"], jev.LEVELS)


if __name__ == "__main__":
    unittest.main()
