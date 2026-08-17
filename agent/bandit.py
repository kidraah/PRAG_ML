"""
FeedLab — Contextual Bandit RL Policy

A threshold-based policy that decides which action to take on a reel
and how long to watch it, based on alignment score.
"""

import random


def decide_timing_and_action(alignment_score: float, heuristic_pass: bool = True) -> tuple[str, float]:
    """Choose an engagement action and watch duration for a reel.

    Parameters
    ----------
    alignment_score : float
        A value between 0.0 and 1.0 indicating how relevant the reel
        is to the target bias.
    heuristic_pass : bool
        Whether the reel passed the fast-path heuristics check.

    Returns
    -------
    tuple[str, float]
        ``(action_name, watch_seconds)``
    """
    if not heuristic_pass:
        return ("FAST_SKIP", random.uniform(0.35, 0.65))

    if alignment_score >= 0.75:
        return ("LIKE_AND_WATCH", random.uniform(12.3, 22.7))
    elif alignment_score >= 0.40:
        return ("WATCH", random.uniform(4.5, 8.2))
    else:
        return ("SKIP", random.uniform(1.2, 2.8))
