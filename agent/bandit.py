"""
FeedLab — Engagement Policy

Turns a reel's value score into what a person with those interests would
do: skip within a couple of seconds, watch part of it, or watch it
through and like / save it (and follow a creator who keeps delivering).
Watch time is the strongest signal the Reels ranker gets, so dwell differs
sharply by decision and is counted from the moment the reel appeared.

With probability EXPLORATION_RATE a borderline reel is watched anyway, so
the feed keeps surfacing new creators instead of collapsing onto a few.
"""

import random
from dataclasses import dataclass

from config import (
    AD_DWELL,
    EXPLORATION_RATE,
    EXPLORE_MIN_SCORE,
    FULL_WATCH_BOUNDS,
    FULL_WATCH_FRACTION,
    LIKE_THRESHOLD,
    SAVE_THRESHOLD,
    SKIP_DWELL,
    WATCH_DWELL_BOUNDS,
    WATCH_FRACTION,
    WATCH_THRESHOLD,
)


@dataclass
class Decision:
    action: str            # SKIP | WATCH | FULL_WATCH
    dwell: float           # target seconds on screen, from when the reel appeared
    basis: str             # score | ad | no_metadata | timeout | keyword_gate | creator_prior | llm_error
    like: bool = False
    save: bool = False
    follow: bool = False
    explored: bool = False

    @property
    def planned(self) -> str:
        """Engagement the policy asked for, e.g. 'like,save'."""
        return ",".join(name for name, on in (("like", self.like), ("save", self.save), ("follow", self.follow)) if on)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def watch_dwell(duration: float | None, rng: random.Random) -> float:
    if duration:
        return _clamp(duration * rng.uniform(*WATCH_FRACTION), WATCH_DWELL_BOUNDS)
    return rng.uniform(WATCH_DWELL_BOUNDS[0], 12.0)


def full_watch_dwell(duration: float | None, rng: random.Random) -> float:
    if duration:
        return _clamp(duration * rng.uniform(*FULL_WATCH_FRACTION), FULL_WATCH_BOUNDS)
    return rng.uniform(15.0, 30.0)


def decide(
    *,
    value_score: float | None,
    basis: str,
    duration: float | None,
    has_liked: bool,
    has_saved: bool,
    follow_ok: bool,
    rng: random.Random,
) -> Decision:
    """Choose an action and watch duration for the reel on screen.

    value_score is only used when basis == "score"; every other basis
    (ad, gate, creator memory, missing metadata, timeout, LLM error) skips.
    follow_ok says whether this reel's creator qualifies for a follow
    (creator memory + per-session cap, decided by the caller).
    """
    if basis != "score" or value_score is None:
        return Decision("SKIP", rng.uniform(*(AD_DWELL if basis == "ad" else SKIP_DWELL)), basis)

    if value_score >= LIKE_THRESHOLD:
        return Decision(
            "FULL_WATCH",
            full_watch_dwell(duration, rng),
            basis,
            like=not has_liked,
            save=value_score >= SAVE_THRESHOLD and not has_saved,
            follow=follow_ok,
        )
    if value_score >= WATCH_THRESHOLD:
        return Decision("WATCH", watch_dwell(duration, rng), basis)
    if value_score >= EXPLORE_MIN_SCORE and rng.random() < EXPLORATION_RATE:
        return Decision("WATCH", watch_dwell(duration, rng), basis, explored=True)
    return Decision("SKIP", rng.uniform(*SKIP_DWELL), basis)
