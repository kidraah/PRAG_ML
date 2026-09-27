"""
PRAG — Quick Rules (a stand-in scorer while the LLM is off)

Scores a reel instantly from its feed data, so `main.py --rules` can steer
in real time without LM Studio: reels from accounts this account follows,
and reels whose caption or hashtags have one of the bias's watch words
(agent/bias.py; the news words in config.py by default), get watched; the
rest are skipped. Deliberately crude — every reel is still recorded and
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


def news_word(caption: str, hashtags: list[str] | None = None, words=None) -> str | None:
    """First watch word (`words`, default NEWS_KEYWORDS) in the caption or hashtags, else None.

    Latin words match whole words and hashtags; Latin phrases ("supreme court") match as whole words in
    order; other scripts (Devanagari) match anywhere.
    """
    latin = _LATIN_KEYWORDS if words is None else {w for w in words if w.isascii()}
    other = _OTHER_KEYWORDS if words is None else sorted(w for w in words if not w.isascii())
    text = caption or ""
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    for word in sorted((set(tokens) | set(hashtags or [])) & latin):
        return word
    joined = f" {' '.join(tokens)} "
    for phrase in sorted(p for p in latin if " " in p):
        if f" {phrase} " in joined:
            return phrase
    for keyword in other:
        if keyword in text or any(keyword in tag for tag in hashtags or []):
            return keyword
    return None


def rule_score(meta, bias=None) -> RuleVerdict:
    """Score a ReelMeta from its feed data alone (no audio, OCR or LLM), for `bias` (default: config.py's)."""
    if meta.creator_following:
        return RuleVerdict(FOLLOWED_SCORE, f"follows @{meta.creator}")
    gate = keyword_gate(meta.caption, meta.hashtags, bias.skip_words if bias else None)
    if gate:
        return RuleVerdict(0.0, f"skip word: {gate}")
    word = news_word(meta.caption, meta.hashtags, bias.watch_words if bias else None)
    if word:
        return RuleVerdict(NEWS_WORD_SCORE, f"watch word: {word}")
    handle = (meta.creator or "").lower()
    if any(hint in handle for hint in (bias.creator_hints if bias else NEWS_CREATOR_HINTS)):
        return RuleVerdict(NEWS_CREATOR_SCORE, "account name looks relevant")
    return RuleVerdict(DEFAULT_SCORE, "no watch signal")
