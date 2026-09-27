"""
PRAG — Fast-Path Keyword Gate

Instant check of a reel's caption and hashtags against the bias's skip
words (config.SKIP_KEYWORDS by default), so obviously off-topic reels
(entertainment, devotional, ...) are skipped in the time a person would
take — without waiting on audio, OCR or the LLM.
"""

import re

from config import SKIP_KEYWORDS


def keyword_gate(caption: str, hashtags: list[str] | None = None, words=None) -> str | None:
    """Return the first skip word (`words`, default SKIP_KEYWORDS) in the caption or hashtags, else None.

    Matching is on whole Latin-script words, case-insensitive; hashtags are
    also matched whole ("#harharmahadev" matches "harharmahadev").
    Devanagari text never matches and is left to the LLM.
    """
    vocabulary = SKIP_KEYWORDS if words is None else set(words)
    for tag in hashtags or []:
        if tag in vocabulary:
            return tag
    normalized = re.sub(r"[^a-z0-9\s]", " ", (caption or "").lower())
    for word in normalized.split():
        if word in vocabulary:
            return word
    return None
