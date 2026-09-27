"""Tests for agent/rules.py — the quick stand-in scorer."""

import unittest

from agent.rules import (
    DEFAULT_SCORE, FOLLOWED_SCORE, NEWS_CREATOR_SCORE, NEWS_WORD_SCORE, news_word, rule_score,
)
from automation.feed_capture import ReelMeta, extract_hashtags


def _meta(caption="", creator="someone", following=False) -> ReelMeta:
    return ReelMeta(code="X", creator=creator, caption=caption, hashtags=extract_hashtags(caption),
                    creator_following=following)


class RuleTests(unittest.TestCase):
    def test_followed_accounts_come_first(self):
        verdict = rule_score(_meta("Bigg Boss promo #funny", creator="aajtak", following=True))
        self.assertEqual((verdict.score, verdict.reason), (FOLLOWED_SCORE, "follows @aajtak"))

    def test_news_words(self):
        self.assertEqual(rule_score(_meta("RBI cuts repo rate, EMIs to fall #finance")).score, NEWS_WORD_SCORE)
        self.assertEqual(rule_score(_meta("UP से उत्तराखंड तक आफत की बारिश! IMD ने जारी किया अलर्ट")).reason,
                         "watch word: imd")
        self.assertEqual(news_word("केंद्र सरकार की नई योजना"), "केंद्र सरकार")
        self.assertEqual(news_word("", ["मौसमविभाग"]), "मौसम")  # inside a Hindi hashtag

    def test_ambiguous_words_are_not_news(self):
        for caption in (
            "Two Italians on the practice court sharpening each other up",   # tennis court
            "तेरे होते मेरी हार कैसे होगी सरकार 🙏",                        # devotional title
            "P e a c e 💝 #softaniai.art AI art",                            # 'ai' isn't a keyword
            "🌿⛰️ #rain #naturelovers",                                      # scenery, not a weather alert
        ):
            self.assertEqual(rule_score(_meta(caption)).score, DEFAULT_SCORE, caption)

    def test_skip_words_and_news_like_accounts(self):
        self.assertEqual(rule_score(_meta("Easy recipe for dinner")).score, 0.0)
        self.assertEqual(rule_score(_meta("🇮🇳🤝", creator="nationupdate3")).score, NEWS_CREATOR_SCORE)
        self.assertEqual(rule_score(_meta("hello", creator="dancer_girl")).score, DEFAULT_SCORE)


if __name__ == "__main__":
    unittest.main()
