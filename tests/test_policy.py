"""Tests for agent/bandit.py and agent/creators.py."""

import random
import unittest

from agent.bandit import decide
from agent.creators import CreatorMemory
from config import (
    AD_DWELL, EXPLORATION_RATE, FULL_WATCH_BOUNDS, SKIP_DWELL, WATCH_DWELL_BOUNDS,
)


def _decide(value, basis="score", duration=30.0, has_liked=False, has_saved=False, follow_ok=False, seed=1):
    return decide(value_score=value, basis=basis, duration=duration, has_liked=has_liked,
                  has_saved=has_saved, follow_ok=follow_ok, rng=random.Random(seed))


class DecideTests(unittest.TestCase):
    def test_non_score_bases_skip_fast(self):
        for basis in ("timeout", "keyword_gate", "creator_prior", "llm_error", "no_metadata"):
            d = _decide(0.99, basis=basis)
            self.assertEqual(d.action, "SKIP", basis)
            self.assertTrue(SKIP_DWELL[0] <= d.dwell <= SKIP_DWELL[1], basis)
            self.assertEqual(d.planned, "")
        ad = _decide(None, basis="ad")
        self.assertEqual(ad.action, "SKIP")
        self.assertTrue(AD_DWELL[0] <= ad.dwell <= AD_DWELL[1])

    def test_valuable_reel_watched_through_liked_saved(self):
        d = _decide(0.9, duration=40.0, follow_ok=True)
        self.assertEqual(d.action, "FULL_WATCH")
        self.assertTrue(36.0 <= d.dwell <= 40.0)          # 90-100% of the reel
        self.assertEqual(d.planned, "like,save,follow")

    def test_like_without_save_below_save_threshold(self):
        d = _decide(0.78)
        self.assertEqual((d.action, d.like, d.save), ("FULL_WATCH", True, False))

    def test_never_re_likes_or_re_saves(self):
        d = _decide(0.95, has_liked=True, has_saved=True)
        self.assertEqual((d.like, d.save), (False, False))

    def test_full_watch_bounds(self):
        self.assertEqual(_decide(0.9, duration=500.0).dwell, FULL_WATCH_BOUNDS[1])
        self.assertEqual(_decide(0.9, duration=3.0).dwell, FULL_WATCH_BOUNDS[0])
        self.assertTrue(15.0 <= _decide(0.9, duration=None).dwell <= 30.0)

    def test_informative_reel_watched_partly(self):
        d = _decide(0.55, duration=30.0)
        self.assertEqual(d.action, "WATCH")
        self.assertTrue(WATCH_DWELL_BOUNDS[0] <= d.dwell <= WATCH_DWELL_BOUNDS[1])
        self.assertFalse(d.like or d.save or d.follow)

    def test_low_score_skips_and_borderline_sometimes_explores(self):
        self.assertEqual(_decide(0.05).action, "SKIP")
        rng = random.Random(7)
        runs = [decide(value_score=0.3, basis="score", duration=20.0, has_liked=False,
                       has_saved=False, follow_ok=False, rng=rng) for _ in range(2000)]
        explored = [d for d in runs if d.explored]
        self.assertTrue(all(d.action == "WATCH" for d in explored))
        self.assertAlmostEqual(len(explored) / len(runs), EXPLORATION_RATE, delta=0.03)
        below = [decide(value_score=0.1, basis="score", duration=20.0, has_liked=False,
                        has_saved=False, follow_ok=False, rng=rng) for _ in range(500)]
        self.assertFalse(any(d.explored for d in below))


class CreatorMemoryTests(unittest.TestCase):
    def test_skip_needs_enough_consistently_low_reels(self):
        memory = CreatorMemory()
        for i in range(3):
            memory.record("dull", f"c{i}", 0.1)
        self.assertFalse(memory.should_skip("dull"))
        memory.record("dull", "c3", 0.2)
        self.assertTrue(memory.should_skip("dull"))
        memory.record("dull", "c4", 0.6)
        self.assertFalse(memory.should_skip("dull"))

    def test_same_reel_counts_once(self):
        memory = CreatorMemory()
        for _ in range(5):
            memory.record("dup", "same", 0.1)
        self.assertEqual(memory.seen("dup"), 1)
        self.assertFalse(memory.should_skip("dup"))

    def test_follow_after_three_valuable_reels_unless_following(self):
        memory = CreatorMemory.from_history([("great", "a", 0.8), ("great", "b", 0.9)], followed=set())
        self.assertFalse(memory.qualifies_for_follow("great"))
        memory.record("great", "c", 0.76)
        self.assertTrue(memory.qualifies_for_follow("great"))
        memory.mark_following("great")
        self.assertFalse(memory.qualifies_for_follow("great"))
        self.assertFalse(CreatorMemory().qualifies_for_follow(""))


if __name__ == "__main__":
    unittest.main()
