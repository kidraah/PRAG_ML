"""
FeedLab — Fast-Path DOM Heuristics

Provides instant, keyword-based evaluation of DOM captions to aggressively
filter out completely irrelevant or unaligned reels before wasting time on
video capture or LLM inference.
"""

import re

# Keywords indicating domains entirely separate from the target bias
# (e.g. fashion, food, makeup, gaming, generic pop culture).
EXCLUDE_KEYWORDS = {
    "cooking", "recipe", "foodie", "baking", "kitchen", "yummy", "tasty",
    "gaming", "pubg", "bgmi", "freefire", "valorant", "gta", "gamer", "streamer",
    "makeup", "beauty", "skincare", "cosmetics", "lipstick", "glowup",
    "fashion", "ootd", "style", "outfit", "wardrobe",
    "cricket", "ipl", "virat", "dhoni", "rohit", "match",
    "dance", "choreography", "dancer",
    "bollywood", "cinema", "movie", "actor", "actress", "song",
    "standup", "comedy", "funny", "joke", "meme",
    "fitness", "workout", "gym", "bodybuilding", "yoga"
}

def fast_heuristic_check(caption: str) -> bool:
    """Evaluate whether the caption likely belongs to an irrelevant domain.

    Parameters
    ----------
    caption : str
        The DOM-extracted caption string.

    Returns
    -------
    bool
        True if the caption passes heuristics (is not definitively excluded),
        False if it contains any EXCLUDE_KEYWORDS (Instant Reject).
    """
    if not caption:
        return True
    
    # Normalize caption to lowercase, replace punctuation with spaces to allow word boundary matching
    normalized = caption.lower()
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    
    words = set(normalized.split())
    
    # If there is any overlap between caption words and excluded keywords, reject immediately
    if EXCLUDE_KEYWORDS.intersection(words):
        return False
        
    return True
