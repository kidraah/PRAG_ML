"""Tests for agent/bandit.py and agent/creators.py."""

import random
import unittest

from agent.bandit import decide
from agent.creators import CreatorMemory
from config import (
    AD_DWELL, EXPLORATION_RATE, FULL_WATCH_BOUNDS, LIKE_PROBABILITY, LIKE_THRESHOLD,
    REWATCH_FACTOR, REWATCH_MAX_SECONDS, REWATCH_PROBABILITY, SAVE_PROBABILITY,
    SKIP_DWELL, WATCH_DWELL_BOUNDS,
)


class FixedRandom(random.Random):
    """random() always returns `value` (so uniform(a, b) is a + (b - a) * value)."""

    def __init__(self, value: float):
        super().__init__(0)
        self.value = value

    def random(self):
        return self.value


EVERY_CHANCE = FixedRandom(0.0)     # every probabilistic like / save / rewatch happens
NO_CHANCE = FixedRandom(0.999)      # none of them do


def _decide(value, basis="score", duration=30.0, has_liked=False, has_saved=False, follow_ok=False,
            likes_left=10, saves_left=10, rng=None):
    return decide(value_score=value, basis=basis, duration=duration, has_liked=has_liked,
                  has_saved=has_saved, follow_ok=follow_ok, likes_left=likes_left,
                  saves_left=saves_left, rng=rng or random.Random(1))


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

    def test_top_reel_rewatched_liked_saved_when_chances_allow(self):
        d = _decide(0.95, duration=40.0, follow_ok=True, rng=EVERY_CHANCE)
        self.assertEqual(d.action, "REWATCH")
        self.assertAlmostEqual(d.dwell, 40.0 * REWATCH_FACTOR[0])   # plays through again
        self.assertEqual(d.planned, "like,save,follow")

    def test_top_reel_watched_once_without_engagement_when_chances_decline(self):
        d = _decide(0.95, duration=40.0, follow_ok=True, rng=NO_CHANCE)
        self.assertEqual(d.action, "FULL_WATCH")
        self.assertTrue(36.0 <= d.dwell <= 40.0)                     # 90-100% of the reel
        self.assertEqual(d.planned, "follow")                        # follows are decided by the caller

    def test_saves_and_rewatches_only_for_the_best(self):
        d = _decide(0.80, rng=EVERY_CHANCE)
        self.assertEqual((d.action, d.like, d.save), ("FULL_WATCH", True, False))

    def test_session_caps_and_existing_state(self):
        capped = _decide(0.95, likes_left=0, saves_left=0, rng=EVERY_CHANCE)
        self.assertEqual((capped.like, capped.save), (False, False))
        already = _decide(0.95, has_liked=True, has_saved=True, rng=EVERY_CHANCE)
        self.assertEqual((already.like, already.save), (False, False))

    def test_engagement_rates_rise_with_score(self):
        def rates(score, n=4000):
            rng = random.Random(3)
            runs = [_decide(score, duration=20.0, rng=rng) for _ in range(n)]
            return (sum(d.like for d in runs) / n, sum(d.save for d in runs) / n,
                    sum(d.action == "REWATCH" for d in runs) / n)

        like_low, save_low, rewatch_low = rates(LIKE_THRESHOLD)
        like_top, save_top, rewatch_top = rates(1.0)
        self.assertAlmostEqual(like_low, LIKE_PROBABILITY[0], delta=0.03)
        self.assertAlmostEqual(like_top, LIKE_PROBABILITY[1], delta=0.03)
        self.assertEqual((save_low, rewatch_low), (0.0, 0.0))        # below SAVE / REWATCH thresholds
        self.assertAlmostEqual(save_top, SAVE_PROBABILITY[1], delta=0.03)
        self.assertAlmostEqual(rewatch_top, REWATCH_PROBABILITY, delta=0.03)

    def test_no_rewatch_for_long_or_unknown_length_reels(self):
        long_reel = REWATCH_MAX_SECONDS / REWATCH_FACTOR[0] + 1
        d = _decide(0.95, duration=long_reel, rng=EVERY_CHANCE)
        self.assertEqual((d.action, d.dwell), ("FULL_WATCH", FULL_WATCH_BOUNDS[1]))
        unknown = _decide(0.95, duration=None, rng=EVERY_CHANCE)
        self.assertEqual(unknown.action, "FULL_WATCH")
        self.assertTrue(15.0 <= unknown.dwell <= 30.0)

    def test_rewatch_capped(self):
        d = _decide(0.95, duration=REWATCH_MAX_SECONDS / REWATCH_FACTOR[0] - 1, rng=FixedRandom(0.49))
        self.assertEqual(d.action, "REWATCH")
        self.assertLessEqual(d.dwell, REWATCH_MAX_SECONDS)

    def test_full_watch_bounds(self):
        self.assertEqual(_decide(0.9, duration=500.0, rng=NO_CHANCE).dwell, FULL_WATCH_BOUNDS[1])
        self.assertEqual(_decide(0.9, duration=3.0, rng=NO_CHANCE).dwell, FULL_WATCH_BOUNDS[0])

    def test_informative_reel_watched_partly(self):
        d = _decide(0.55, duration=30.0)
        self.assertEqual(d.action, "WATCH")
        self.assertTrue(WATCH_DWELL_BOUNDS[0] <= d.dwell <= WATCH_DWELL_BOUNDS[1])
        self.assertFalse(d.like or d.save or d.follow)

    def test_low_score_skips_and_borderline_sometimes_explores(self):
        self.assertEqual(_decide(0.05).action, "SKIP")
        rng = random.Random(7)
        runs = [_decide(0.3, duration=20.0, rng=rng) for _ in range(2000)]
        explored = [d for d in runs if d.explored]
        self.assertTrue(all(d.action == "WATCH" for d in explored))
        self.assertAlmostEqual(len(explored) / len(runs), EXPLORATION_RATE, delta=0.03)
        below = [_decide(0.1, duration=20.0, rng=rng) for _ in range(500)]
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
