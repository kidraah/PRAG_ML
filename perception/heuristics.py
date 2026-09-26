"""
FeedLab — Fast-Path Keyword Gate

Instant check of a reel's caption and hashtags against SKIP_KEYWORDS, so
obviously off-topic reels (entertainment, devotional, ...) are skipped in
the time a person would take — without waiting on audio, OCR or the LLM.
"""

import re

from config import SKIP_KEYWORDS


def keyword_gate(caption: str, hashtags: list[str] | None = None) -> str | None:
    """Return the first SKIP_KEYWORDS match in the caption or hashtags, else None.

    Matching is on whole Latin-script words, case-insensitive; hashtags are
    also matched whole ("#harharmahadev" matches "harharmahadev").
    Devanagari text never matches and is left to the LLM.
    """
    for tag in hashtags or []:
        if tag in SKIP_KEYWORDS:
            return tag
    normalized = re.sub(r"[^a-z0-9\s]", " ", (caption or "").lower())
    for word in normalized.split():
        if word in SKIP_KEYWORDS:
            return word
    return None
