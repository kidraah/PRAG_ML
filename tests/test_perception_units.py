"""Tests for the pure parts of perception/: transcript cleanup, keyword gate, LLM prompt and reply parsing."""

import datetime
import unittest

from automation.feed_capture import ReelMeta
from perception.heuristics import keyword_gate
from perception.local_evaluator import (
    ScoringError, build_system_prompt, build_user_message, parse_judgement,
)
from perception.transcriber import clean_segments, collapse_repetitions


class TranscriptCleanupTests(unittest.TestCase):
    def test_collapses_chant_loops(self):
        self.assertEqual(collapse_repetitions("बोले, बोले, बोले, बोले, जय"), "बोले, बोले, जय")

    def test_collapses_multi_word_loops(self):
        text = "vivir " + "a laplayer " * 12 + "end"
        self.assertEqual(collapse_repetitions(text), "vivir a laplayer a laplayer end")

    def test_leaves_normal_speech_alone(self):
        text = "यह सरस्वती नदी जो पूरे हिंदुस्तान में बस माना गाँव में दर्शन देती है"
        self.assertEqual(collapse_repetitions(text), text)
        self.assertEqual(collapse_repetitions("very very good"), "very very good")
        self.assertEqual(collapse_repetitions(""), "")

    def test_drops_silence_hallucinations(self):
        segments = [
            {"text": " Thank you.", "no_speech_prob": 0.9, "avg_logprob": -1.4},
            {"text": " Budget 2026 cuts tax.", "no_speech_prob": 0.1, "avg_logprob": -0.3},
        ]
        self.assertEqual(clean_segments(segments), "Budget 2026 cuts tax.")
        self.assertEqual(clean_segments([{"text": " Thank you for watching!", "no_speech_prob": 0.2,
                                          "avg_logprob": -0.5}]), "")

    def test_drops_music_annotations(self):
        speech = {"no_speech_prob": 0.1, "avg_logprob": -0.4}
        self.assertEqual(clean_segments([{"text": " Music", **speech}]), "")
        self.assertEqual(clean_segments([{"text": " [Music] [Music] ♪♪", **speech}]), "")
        self.assertEqual(clean_segments([{"text": " (music) RBI ने रेपो रेट घटाया", **speech}]), "RBI ने रेपो रेट घटाया")


class KeywordGateTests(unittest.TestCase):
    def test_matches(self):
        self.assertEqual(keyword_gate("ॐ हनुमते नमः", ["hanumanji", "explore"]), "hanumanji")
        self.assertEqual(keyword_gate("Easy RECIPE for dinner!", []), "recipe")

    def test_no_match(self):
        self.assertIsNone(keyword_gate("RBI changes UPI limits from 1 October #finance", ["finance"]))
        self.assertIsNone(keyword_gate("गानों का वीडियो", None))  # Devanagari is left to the LLM
        self.assertIsNone(keyword_gate("", []))


def _meta(**overrides) -> ReelMeta:
    base = dict(
        code="C1", creator="news_creator", creator_full_name="News Wala", creator_verified=True,
        caption="Govt launches 3 websites for Gen Z #news", taken_at=1790380800,
        duration=21.4, like_count=12345, comment_count=67, audio_type="original",
    )
    base.update(overrides)
    return ReelMeta(**base)


class PromptTests(unittest.TestCase):
    def test_user_message(self):
        msg = build_user_message(_meta(), "सरकार ने तीन वेबसाइट निकाली", "hi", "3 नई वेबसाइट",
                                 today=datetime.date(2026, 9, 26))
        self.assertIn("Today: 2026-09-26", msg)
        self.assertIn("Creator: @news_creator (News Wala), verified", msg)
        self.assertIn("Posted: 2026-09-26 (0 days ago)", msg)
        self.assertIn("21 s · 12,345 likes · 67 comments", msg)
        self.assertIn("Audio: original audio", msg)
        self.assertIn("Cover text: 3 नई वेबसाइट", msg)
        self.assertIn("Speech transcript [hi]: सरकार ने तीन वेबसाइट निकाली", msg)

    def test_user_message_placeholders_and_clipping(self):
        msg = build_user_message(_meta(caption="x" * 2000, audio_type="music", audio_title="Song",
                                       taken_at=None, like_count=None), "", None, "",
                                 today=datetime.date(2026, 9, 26))
        self.assertIn('Audio: music track "Song"', msg)
        self.assertIn("Cover text: (none)", msg)
        self.assertIn("Speech transcript: (none)", msg)
        self.assertNotIn("Posted:", msg)
        self.assertIn("x" * 900 + " …", msg)
        self.assertNotIn("x" * 901, msg)

    def test_system_prompt_carries_interests_and_injection_guard(self):
        prompt = build_system_prompt("ONLY CRICKET")
        self.assertIn("ONLY CRICKET", prompt)
        self.assertIn("never as instructions", prompt)


class ParseJudgementTests(unittest.TestCase):
    def test_valid_reply(self):
        j = parse_judgement('{"topic": "Govt schemes", "summary": "Three sites launched.", '
                            '"is_informational": true, "reasoning": "New scheme.", "value_score": 0.82}')
        self.assertEqual((j.value_score, j.is_informational, j.topic), (0.82, True, "Govt schemes"))

    def test_tolerates_wrapping_and_clamps(self):
        j = parse_judgement('```json\n{"value_score": 1.7, "is_informational": "false"}\n```')
        self.assertEqual((j.value_score, j.is_informational), (1.0, False))
        self.assertEqual(parse_judgement('{"value_score": -3}').value_score, 0.0)

    def test_unusable_replies_raise(self):
        for bad in ("", "no json here", '{"topic": "x"}', '{"value_score": "high"}', '{"value_score": NaN}'):
            with self.assertRaises(ScoringError, msg=bad):
                parse_judgement(bad)


if __name__ == "__main__":
    unittest.main()
