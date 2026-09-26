"""
FeedLab — Quick Rules (a stand-in scorer while the LLM is off)

Scores a reel instantly from its feed data, so `main.py --rules` can steer
in real time without LM Studio: reels from accounts this account follows,
and reels whose caption or hashtags read like news, get watched; the rest
are skipped. Deliberately crude — every reel is still recorded and
reviewed afterwards, and the report shows how often the rules agreed with
the review.
"""

import re
from dataclasses import dataclass

from config import NEWS_CREATOR_HINTS, NEWS_KEYWORDS
from perception.heuristics import keyword_gate

FOLLOWED_SCORE = 0.80        # an account this account chose to follow (the news seed list)
NEWS_WORD_SCORE = 0.60       # caption / hashtags read like news
NEWS_CREATOR_SCORE = 0.50    # the account name looks like a news page
DEFAULT_SCORE = 0.10         # no news signal

_LATIN_KEYWORDS = {k for k in NEWS_KEYWORDS if k.isascii()}
_OTHER_KEYWORDS = sorted(k for k in NEWS_KEYWORDS if not k.isascii())


@dataclass
class RuleVerdict:
    score: float
    reason: str


def news_word(caption: str, hashtags: list[str] | None = None) -> str | None:
    """First NEWS_KEYWORDS match in the caption or hashtags, else None."""
    text = caption or ""
    words = set(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()) | set(hashtags or [])
    for word in sorted(words & _LATIN_KEYWORDS):
        return word
    for keyword in _OTHER_KEYWORDS:
        if keyword in text or any(keyword in tag for tag in hashtags or []):
            return keyword
    return None


def rule_score(meta) -> RuleVerdict:
    """Score a ReelMeta from its feed data alone (no audio, OCR or LLM)."""
    if meta.creator_following:
        return RuleVerdict(FOLLOWED_SCORE, f"follows @{meta.creator}")
    gate = keyword_gate(meta.caption, meta.hashtags)
    if gate:
        return RuleVerdict(0.0, f"skip word: {gate}")
    word = news_word(meta.caption, meta.hashtags)
    if word:
        return RuleVerdict(NEWS_WORD_SCORE, f"news word: {word}")
    handle = (meta.creator or "").lower()
    if any(hint in handle for hint in NEWS_CREATOR_HINTS):
        return RuleVerdict(NEWS_CREATOR_SCORE, "news-like account name")
    return RuleVerdict(DEFAULT_SCORE, "no news signal")
