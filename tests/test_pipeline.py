"""Tests for agent/pipeline.py (queue order, gates, worker) and main.py's logging glue — no network, no models."""

import argparse
import asyncio
import contextlib
import io
import random
import time
import unittest
from unittest import mock

import agent.pipeline as pipeline_module
import main as main_module
from agent.bandit import Decision
from agent.bias import Bias
from agent.creators import CreatorMemory
from agent.pipeline import ScoreResult, ScoringPipeline
from automation.feed_capture import FeedCatalog, ReelMeta
from agent.rules import RuleVerdict
from main import Performed, await_score, describe, impression_row
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

    def test_observe_mode_records_without_judging(self):
        async def scenario():
            p = ScoringPipeline(FeedCatalog(), CreatorMemory(), rng=_NeverExplore(), user_agent="test",
                                use_llm=False)
            evaluate = mock.AsyncMock()
            with mock.patch.object(p, "_perceive", mock.AsyncMock(return_value=("बोले", "hi", "headline", []))), \
                 mock.patch.object(pipeline_module, "evaluate_value_async", evaluate):
                result = await p._score(ReelMeta(code="O", caption="best recipe"))  # gates don't apply
            await p.close()
            return result, evaluate.await_count

        result, llm_calls = asyncio.run(scenario())
        self.assertEqual((result.basis, result.transcript, result.cover_text, result.value_score),
                         ("observed", "बोले", "headline", None))
        self.assertEqual(llm_calls, 0)

    def test_rules_mode_scores_recordings_with_the_trained_model(self):
        class FakeModel:
            def score(self, meta, transcript, cover_text):
                return (0.9, "model p=0.90") if "news" in transcript else (0.1, "model p=0.10")

            @staticmethod
            def informative(value):
                return value >= 0.4

        async def scenario():
            p = ScoringPipeline(FeedCatalog(), CreatorMemory(), rng=_NeverExplore(), user_agent="test",
                                use_llm=False, fallback_scorer=FakeModel())
            evaluate = mock.AsyncMock()
            with mock.patch.object(p, "_perceive", mock.AsyncMock(return_value=("news tonight", "en", "", []))), \
                 mock.patch.object(pipeline_module, "evaluate_value_async", evaluate):
                result = await p._score(ReelMeta(code="M", caption="Parliament passes the data bill. #news"))
            await p.close()
            return result, evaluate.await_count

        result, llm_calls = asyncio.run(scenario())
        self.assertEqual((result.basis, result.value_score, result.is_informational), ("score", 0.9, True))
        self.assertEqual((result.summary, result.reasoning, result.topic),
                         ("Parliament passes the data bill.", "model p=0.90", ""))
        self.assertEqual((result.transcript, llm_calls), ("news tonight", 0))

    def test_the_bias_steers_the_gate_and_the_llm(self):
        tech = Bias(name="Tech", interests="AI launches and research results, with real details.",
                    skip_words=["cricket"])

        async def scenario():
            p = ScoringPipeline(FeedCatalog(), CreatorMemory(), rng=_NeverExplore(), user_agent="test", bias=tech)
            evaluate = mock.AsyncMock(return_value=_judgement(0.9))
            with mock.patch.object(p, "_perceive", mock.AsyncMock(return_value=("speech", "en", "", []))), \
                 mock.patch.object(pipeline_module, "evaluate_value_async", evaluate):
                gated = await p._score(ReelMeta(code="C", caption="Cricket highlights"))
                scored = await p._score(ReelMeta(code="R", caption="Easy recipe"))  # skipped by default, not by Tech
            await p.close()
            return gated, scored, evaluate

        gated, scored, evaluate = asyncio.run(scenario())
        self.assertEqual((gated.basis, gated.gate_keyword), ("keyword_gate", "cricket"))
        self.assertEqual(scored.basis, "score")
        self.assertEqual(evaluate.await_args.kwargs["interests"], tech.interests)

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
        self.assertEqual(sorted(seen), [("A", "score"), ("B", "score")])
        self.assertEqual((creators.high_count("u1"), creators.seen("u2")), (1, 1))

    def test_workers_score_reels_concurrently(self):
        async def scenario():
            catalog = FeedCatalog()
            p = _pipeline(catalog)
            catalog.on_new_feed_reel = p.enqueue
            in_flight = peak = 0
            both_started = asyncio.Event()

            async def slow_evaluate(_session, meta, *args, **kwargs):
                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                if in_flight >= 2:
                    both_started.set()
                # Times out (→ llm_error) if reels are scored one at a time.
                await asyncio.wait_for(both_started.wait(), timeout=1.0)
                in_flight -= 1
                return _judgement(0.5)

            with mock.patch.object(p, "_perceive", mock.AsyncMock(return_value=("", None, "", []))), \
                 mock.patch.object(pipeline_module, "evaluate_value_async", slow_evaluate), \
                 mock.patch.object(pipeline_module, "SCORING_WORKERS", 2):
                worker = asyncio.create_task(p.run())
                catalog.ingest_doc(feed_page(make_media("A"), make_media("B")))
                results = [await p.wait_for(code, timeout=2) for code in ("A", "B")]
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await p.close()
            return results, peak

        results, peak = asyncio.run(scenario())
        self.assertEqual(peak, 2)
        self.assertEqual([r.basis for r in results], ["score", "score"])


class AwaitScoreTests(unittest.TestCase):
    """A slow score: news-like reels keep playing while it finishes, the rest are skipped at MAX_DECISION_WAIT."""

    def wait(self, meta, arrives_on_call):
        waits = []

        class SlowPipeline:
            async def wait_for(self, code, timeout):
                waits.append(round(timeout, 1))
                return ScoreResult(code, basis="score", value_score=0.9) if len(waits) == arrives_on_call else None

        result, rule = asyncio.run(await_score(SlowPipeline(), meta.code, meta, time.monotonic()))
        return result, rule, waits

    def test_ready_in_time(self):
        result, rule, waits = self.wait(ReelMeta(code="A", caption="dance"), arrives_on_call=1)
        self.assertEqual((result.value_score, rule, waits), (0.9, None, [3.5]))

    def test_news_like_reel_waits_longer_for_its_score(self):
        result, rule, waits = self.wait(ReelMeta(code="N", creator="the_hindu", creator_following=True),
                                        arrives_on_call=2)
        self.assertEqual((result.value_score, rule.score, waits), (0.9, 0.8, [3.5, 25.0]))

    def test_other_reels_are_not_waited_for(self):
        result, rule, waits = self.wait(ReelMeta(code="J", caption="dance challenge"), arrives_on_call=2)
        self.assertIsNone(result)
        self.assertLess(rule.score, 0.4)
        self.assertEqual(waits, [3.5])


class OutcomeTests(unittest.TestCase):
    """run() reports why a session didn't happen, so the daily runner can say so."""

    def run_session(self, **patches):
        args = argparse.Namespace(account=None, observe=False, rules=False, dry_run=True, minutes=1, max_reels=1)
        quiet = dict(init_db=mock.DEFAULT, reset_cooldown_accounts=mock.DEFAULT, list_accounts=mock.DEFAULT,
                     mark_account_used=mock.DEFAULT, start_session=mock.DEFAULT, profile_lock=mock.DEFAULT)
        with mock.patch.multiple(main_module, **{**quiet, **patches}) as mocks, \
                contextlib.redirect_stdout(io.StringIO()):
            return asyncio.run(main_module.run(args)), mocks

    def test_no_account(self):
        outcome, _ = self.run_session(choose_account=mock.Mock(return_value=(None, "No ACTIVE account")))
        self.assertEqual(outcome, {"status": "no_account", "detail": "No ACTIVE account"})

    def test_llm_unavailable_stops_before_a_session_starts(self):
        outcome, mocks = self.run_session(choose_account=mock.Mock(return_value=("acct", None)),
                                          ensure_llm_ready=mock.Mock(side_effect=RuntimeError("LM Studio is off")))
        self.assertEqual(outcome, {"status": "llm_unavailable", "account": "acct", "detail": "LM Studio is off"})
        mocks["start_session"].assert_not_called()

    def test_busy_when_another_run_holds_the_profile(self):
        held_elsewhere = mock.MagicMock()
        held_elsewhere.return_value.__enter__.return_value = False
        outcome, mocks = self.run_session(choose_account=mock.Mock(return_value=("acct", None)),
                                          profile_lock=held_elsewhere)
        self.assertEqual(outcome, {"status": "busy", "account": "acct"})
        mocks["mark_account_used"].assert_not_called()


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

    def test_console_line_shows_rule_and_flick(self):
        meta = ReelMeta(code="F", creator="dull_page", duration=12.0, audio_type="music")
        flick = describe(7, meta, {}, False, None, Decision("SKIP", 0.9, "rules"), 0.93, Performed(), True,
                         RuleVerdict(0.1, "no news signal"), ready=True)
        self.assertIn("rule 0.1 · no news signal", flick)
        self.assertIn("⚡", flick)
        watched = describe(8, meta, {}, False, None, Decision("WATCH", 9.0, "rules"), 9.1, Performed(), True,
                           RuleVerdict(0.8, "follows @aajtak"), ready=True)
        self.assertIn("rule 0.8 · follows @aajtak", watched)
        self.assertNotIn("⚡", watched)

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
