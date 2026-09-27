"""Tests for agent/bias.py and how a bias steers the rules, the gate and the LLM prompt."""

import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import config
from agent import bias
from agent.rules import rule_score
from automation.feed_capture import ReelMeta, extract_hashtags
from perception.heuristics import keyword_gate
from perception.local_evaluator import scorer_version


def _meta(caption="", creator="someone") -> ReelMeta:
    return ReelMeta(code="X", creator=creator, caption=caption, hashtags=extract_hashtags(caption))


TECH = bias.Bias(name="Tech", interests="AI launches and research, gadgets with real specs.",
                 watch_words=["ai", "supreme court", "खबर"], skip_words=["cricket"], creator_hints=["gadget"])


class BiasTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(bias, "BIAS_DIR", tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_is_config_and_keeps_its_fingerprint(self):
        default = bias.load("no_such_account")
        self.assertIs(default, bias.PRESETS[bias.DEFAULT])
        self.assertEqual(default.interests, config.INTERESTS.strip())
        self.assertEqual(set(default.watch_words), config.NEWS_KEYWORDS)
        self.assertEqual(set(default.skip_words), config.SKIP_KEYWORDS)
        self.assertEqual(scorer_version(default.interests), scorer_version())  # old scores stay comparable
        self.assertIn("ndtv", default.seeds)

    def test_presets_are_valid(self):
        for key, preset in bias.PRESETS.items():
            with self.subTest(key):
                self.assertEqual(bias.validated(preset.to_dict()), preset)
                self.assertEqual(bias.preset_for(preset), key)
        fingerprints = {preset.fingerprint for preset in bias.PRESETS.values()}
        self.assertEqual(len(fingerprints), len(bias.PRESETS))

    def test_words(self):
        self.assertEqual(bias.words("AI, #ChatGPT\nखबर, ai,  , @ndtv"), ["ai", "chatgpt", "खबर", "ndtv"])
        self.assertEqual(bias.words(["Supreme  Court"]), ["supreme court"])
        self.assertEqual(bias.words(None), [])

    def test_validation(self):
        good = {"name": "  My   bias ", "interests": "Policy changes and new government schemes.",
                "watch_words": "RBI, gst", "seeds": "@NDTV, the_hindu"}
        made = bias.validated(good)
        self.assertEqual((made.name, made.watch_words, made.seeds), ("My bias", ["rbi", "gst"], ["ndtv", "the_hindu"]))
        for broken, message in (({"interests": "news"}, "at least a sentence"),
                                ({**good, "seeds": "not a handle!"}, "isn't an Instagram handle"),
                                ({**good, "skip_words": "x" * 50}, "too long"),
                                ({**good, "watch_words": ",".join(f"w{i}" for i in range(301))}, "at most"),
                                ({**good, "interests": "x" * 5000}, "under")):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                bias.validated(broken)
        self.assertEqual(bias.validated({**good, "name": ""}).name, "Custom")

    def test_save_and_load(self):
        self.assertFalse(bias.has_own("acct"))
        path = bias.save("acct", TECH)
        self.assertTrue(path.endswith("acct.json"))
        self.assertTrue(bias.has_own("acct"))
        self.assertEqual(bias.load("acct"), TECH)
        with self.assertRaises(ValueError):
            bias.path_for("../accounts")
        with open(path, "w") as fh:
            fh.write("{broken")
        with redirect_stdout(StringIO()) as out:
            self.assertIs(bias.load("acct"), bias.PRESETS[bias.DEFAULT])  # a broken file falls back, loudly
        self.assertIn("can't be used", out.getvalue())


class SteeringTests(unittest.TestCase):
    def test_rules_follow_the_bias(self):
        self.assertEqual(rule_score(_meta("New AI model tops the benchmarks"), TECH).reason, "watch word: ai")
        self.assertEqual(rule_score(_meta("The Supreme Court ruled today"), TECH).reason, "watch word: supreme court")
        self.assertEqual(rule_score(_meta("आज की बड़ी खबर"), TECH).reason, "watch word: खबर")
        self.assertEqual(rule_score(_meta("Cricket highlights"), TECH).score, 0.0)
        self.assertEqual(rule_score(_meta("hello", creator="gadgetguru"), TECH).reason, "account name looks relevant")
        # The same reels under the default bias:
        self.assertEqual(rule_score(_meta("New AI model tops the benchmarks")).reason, "no watch signal")
        self.assertEqual(rule_score(_meta("Cricket highlights")).reason, "no watch signal")

    def test_gate_uses_the_bias_skip_words(self):
        self.assertEqual(keyword_gate("Cricket highlights", [], TECH.skip_words), "cricket")
        self.assertIsNone(keyword_gate("Easy recipe", [], TECH.skip_words))
        self.assertEqual(keyword_gate("Easy recipe", []), "recipe")  # default words


if __name__ == "__main__":
    unittest.main()
