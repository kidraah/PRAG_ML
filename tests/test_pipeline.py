"""Tests for agent/pipeline.py (queue order, gates, worker) and main.py's logging glue — no network, no models."""

import asyncio
import random
import unittest
from unittest import mock

import agent.pipeline as pipeline_module
from agent.bandit import Decision
from agent.creators import CreatorMemory
from agent.pipeline import ScoreResult, ScoringPipeline
from automation.feed_capture import FeedCatalog, ReelMeta
from main import Performed, impression_row
from perception.local_evaluator import ScoringError, ValueJudgement
from tests.fixtures import feed_page, make_media


class _AlwaysExplore(random.Random):
    def random(self):
        return 0.0


class _NeverExplore(random.Random):
    def random(self):
        return 0.99


def _pipeline(catalog, creators=None, rng=None, on_result=None) -> ScoringPipeline:
    return ScoringPipeline(catalog, creators or CreatorMemory(), rng=rng or _NeverExplore(),
                           user_agent="test", on_result=on_result)


def _judgement(score: float) -> ValueJudgement:
    return ValueJudgement(value_score=score, is_informational=score > 0.4, topic="t",
                          summary="s", reasoning="r")


class QueueOrderTests(unittest.TestCase):
    def test_current_first_then_upcoming_then_recent_past(self):
        async def scenario():
            catalog = FeedCatalog()
            p = _pipeline(catalog)
            catalog.on_new_feed_reel = p.enqueue
            catalog.ingest_doc(feed_page(*(make_media(f"R{i}") for i in range(12))))
            order = []
            p.set_current("R5", catalog.get("R5"))
            while (meta := p._pick_next()) is not None:
                order.append(meta.code)
                p._queued.discard(meta.code)
                p._results[meta.code] = ScoreResult(meta.code, basis="score", value_score=0.5)
            await p.close()
            return order

        order = asyncio.run(scenario())
        # R5 on screen, then R6..R11 upcoming, then back-fill R4..R0 newest first (window 8).
        self.assertEqual(order, ["R5", "R6", "R7", "R8", "R9", "R10", "R11", "R4", "R3", "R2", "R1", "R0"])

    def test_scored_reels_are_not_requeued(self):
        async def scenario():
            catalog = FeedCatalog()
            p = _pipeline(catalog)
            p._results["A"] = ScoreResult("A", basis="score", value_score=0.1)
            p.enqueue(ReelMeta(code="A"))
            queued = set(p._queued)
            await p.close()
            return queued

        self.assertEqual(asyncio.run(scenario()), set())


class GateTests(unittest.TestCase):
    def run_score(self, meta, creators=None, rng=None, judgement=None, error=None):
        async def scenario():
            p = _pipeline(FeedCatalog(), creators=creators, rng=rng)
            perceive = mock.AsyncMock(return_value=("speech", "hi", "cover", []))
            evaluate = mock.AsyncMock(side_effect=error or None, return_value=judgement or _judgement(0.9))
            with mock.patch.object(p, "_perceive", perceive), \
                 mock.patch.object(pipeline_module, "evaluate_value_async", evaluate):
                result = await p._score(meta)
            await p.close()
            return result, perceive.await_count

        return asyncio.run(scenario())

    def test_ads_are_never_scored(self):
        result, perceived = self.run_score(ReelMeta(code="AD", is_ad=True))
        self.assertEqual((result.basis, perceived), ("ad", 0))

    def test_keyword_gate_skips_without_perception(self):
        result, perceived = self.run_score(ReelMeta(code="K", caption="Best recipe ever"))
        self.assertEqual((result.basis, result.gate_keyword, perceived), ("keyword_gate", "recipe", 0))

    def test_gate_audit_scores_a_gated_reel(self):
        result, perceived = self.run_score(ReelMeta(code="K", caption="recipe"), rng=_AlwaysExplore())
        self.assertEqual((result.basis, result.gate_audit, result.gate_keyword), ("score", True, "recipe"))
        self.assertEqual((result.transcript, result.cover_text, perceived), ("speech", "cover", 1))

    def test_creator_prior(self):
        creators = CreatorMemory.from_history([("dull", f"c{i}", 0.05) for i in range(4)], set())
        result, _ = self.run_score(ReelMeta(code="D", creator="dull"), creators=creators)
        self.assertEqual(result.basis, "creator_prior")

    def test_llm_error_is_an_error_not_a_zero(self):
        result, _ = self.run_score(ReelMeta(code="E"), error=ScoringError("LM Studio unreachable"))
        self.assertEqual(result.basis, "llm_error")
        self.assertIsNone(result.value_score)
        self.assertIn("LM Studio unreachable", result.error)


class WorkerTests(unittest.TestCase):
    def test_worker_scores_in_order_and_records_creators(self):
        async def scenario():
            catalog = FeedCatalog()
            creators = CreatorMemory()
            seen = []
            p = _pipeline(catalog, creators=creators, on_result=lambda m, r: seen.append((m.code, r.basis)))
            catalog.on_new_feed_reel = p.enqueue
            scores = {"A": 0.9, "B": 0.2}
            evaluate = mock.AsyncMock(side_effect=lambda _s, meta, *a, **k: _judgement(scores[meta.code]))
            with mock.patch.object(p, "_perceive", mock.AsyncMock(return_value=("", None, "", []))), \
                 mock.patch.object(pipeline_module, "evaluate_value_async", evaluate):
                worker = asyncio.create_task(p.run())
                catalog.ingest_doc(feed_page(make_media("A", username="u1"), make_media("B", username="u2")))
                p.set_current("A", catalog.get("A"))
                a = await p.wait_for("A", timeout=2)
                b = await p.wait_for("B", timeout=2)
                missing = await p.wait_for("NOPE", timeout=0.1)
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await p.close()
            return a, b, missing, seen, creators

        a, b, missing, seen, creators = asyncio.run(scenario())
        self.assertEqual((a.value_score, b.value_score), (0.9, 0.2))
        self.assertIsNone(missing)
        self.assertEqual(seen, [("A", "score"), ("B", "score")])
        self.assertEqual((creators.high_count("u1"), creators.seen("u2")), (1, 1))


class ImpressionRowTests(unittest.TestCase):
    def test_ad_row(self):
        row = impression_row(1, 3, "AD", None, {"creator": "brand", "sponsored": True}, True, None,
                             Decision("SKIP", 1.5, "ad"), 1.61, Performed())
        self.assertEqual((row["score_basis"], row["is_ad"], row["creator"], row["decision_basis"]),
                         ("ad", 1, "brand", "ad"))

    def test_timeout_row_has_no_score_yet(self):
        meta = ReelMeta(code="T", creator="u", hashtags=["a", "b"], duration=20.0)
        row = impression_row(1, 4, "T", meta, {}, False, None, Decision("SKIP", 2.0, "timeout"), 6.02,
                             Performed())
        self.assertNotIn("score_basis", row)
        self.assertNotIn("scored_at", row)
        self.assertEqual((row["hashtags"], row["decision_basis"], row["dwell_seconds"]), ("a b", "timeout", 6.02))

    def test_scored_row(self):
        meta = ReelMeta(code="S", creator="u")
        result = ScoreResult("S", basis="score", value_score=0.8, topic="t", summary="s", seconds=3.14159)
        row = impression_row(1, 5, "S", meta, {}, False, result,
                             Decision("FULL_WATCH", 20.0, "score", like=True, save=False), 20.3,
                             Performed(liked=True))
        self.assertEqual((row["value_score"], row["planned"], row["liked"], row["score_seconds"]),
                         (0.8, "like", 1, 3.14))
        self.assertIn("scored_at", row)


if __name__ == "__main__":
    unittest.main()
