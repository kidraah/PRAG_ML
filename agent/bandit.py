"""
PRAG — Engagement Policy

Turns a reel's value score into what a person with those interests would
do: skip within a couple of seconds, watch part of it, or watch it
through — the very best sometimes twice, since the player loops them —
and like, save or follow some of it. Watch time is the strongest signal
the Reels ranker gets, so dwell differs sharply by decision and is
counted from the moment the reel appeared.

People like only a share of what they enjoy, so likes and saves are a
chance that rises with the score, capped per session (the caller tracks
the caps). With probability EXPLORATION_RATE a borderline reel is watched
anyway, so the feed keeps surfacing new creators instead of collapsing
onto a few.
"""

import random
from dataclasses import dataclass

from config import (
    AD_DWELL,
    EXPLORATION_RATE,
    EXPLORE_MIN_SCORE,
    FLICK_DWELL,
    FULL_WATCH_BOUNDS,
    FULL_WATCH_FRACTION,
    LIKE_PROBABILITY,
    LIKE_THRESHOLD,
    REWATCH_FACTOR,
    REWATCH_MAX_SECONDS,
    REWATCH_PROBABILITY,
    REWATCH_THRESHOLD,
    SAVE_PROBABILITY,
    SAVE_THRESHOLD,
    SKIP_DWELL,
    WATCH_DWELL_BOUNDS,
    WATCH_FRACTION,
    WATCH_THRESHOLD,
)


@dataclass
class Decision:
    action: str            # SKIP | WATCH | FULL_WATCH | REWATCH
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


# Skip reasons that can be known before a reel appears (the others — timeout,
# no metadata, LLM error — only happen once it's on screen).
PRE_DECIDABLE = ("keyword_gate", "creator_prior")


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def _ramp(score: float, start: float, probabilities: tuple[float, float]) -> float:
    """Chance rising linearly from probabilities[0] at `start` to probabilities[1] at a score of 1.0."""
    low, high = probabilities
    if start >= 1.0:
        return high
    return low + (high - low) * _clamp((score - start) / (1.0 - start), (0.0, 1.0))


def watch_dwell(duration: float | None, rng: random.Random) -> float:
    if duration:
        return _clamp(duration * rng.uniform(*WATCH_FRACTION), WATCH_DWELL_BOUNDS)
    return rng.uniform(WATCH_DWELL_BOUNDS[0], 12.0)


def full_watch_dwell(duration: float | None, rng: random.Random) -> float:
    if duration:
        return _clamp(duration * rng.uniform(*FULL_WATCH_FRACTION), FULL_WATCH_BOUNDS)
    return rng.uniform(15.0, 30.0)


def rewatch_dwell(value_score: float, duration: float | None, rng: random.Random) -> float | None:
    """Dwell that lets a top reel play through again, or None to watch it once."""
    if not duration or value_score < REWATCH_THRESHOLD or duration * REWATCH_FACTOR[0] > REWATCH_MAX_SECONDS:
        return None
    if rng.random() >= REWATCH_PROBABILITY:
        return None
    return min(duration * rng.uniform(*REWATCH_FACTOR), REWATCH_MAX_SECONDS)


def decide(
    *,
    value_score: float | None,
    basis: str,
    duration: float | None,
    has_liked: bool,
    has_saved: bool,
    follow_ok: bool,
    likes_left: int,
    saves_left: int,
    rng: random.Random,
    pre_analyzed: bool = False,
) -> Decision:
    """Choose an action and watch duration for the reel on screen.

    value_score is only used when basis == "score"; every other basis
    (ad, gate, creator memory, missing metadata, timeout, LLM error) skips.
    follow_ok says whether this reel's creator qualifies for a follow, and
    likes_left / saves_left are what remains of this session's caps —
    all tracked by the caller. pre_analyzed means the verdict was ready
    before the reel reached the screen (from the preloaded feed data), so a
    skip is a quick flick rather than a considered one.
    """
    skip_dwell = FLICK_DWELL if pre_analyzed else SKIP_DWELL
    if basis != "score" or value_score is None:
        if basis == "ad":
            return Decision("SKIP", rng.uniform(*AD_DWELL), basis)
        return Decision("SKIP", rng.uniform(*(skip_dwell if basis in PRE_DECIDABLE else SKIP_DWELL)), basis)

    if value_score >= LIKE_THRESHOLD:
        like = (
            not has_liked and likes_left > 0
            and rng.random() < _ramp(value_score, LIKE_THRESHOLD, LIKE_PROBABILITY)
        )
        save = (
            value_score >= SAVE_THRESHOLD and not has_saved and saves_left > 0
            and rng.random() < _ramp(value_score, SAVE_THRESHOLD, SAVE_PROBABILITY)
        )
        rewatch = rewatch_dwell(value_score, duration, rng)
        if rewatch is not None:
            return Decision("REWATCH", rewatch, basis, like=like, save=save, follow=follow_ok)
        return Decision("FULL_WATCH", full_watch_dwell(duration, rng), basis, like=like, save=save, follow=follow_ok)
    if value_score >= WATCH_THRESHOLD:
        return Decision("WATCH", watch_dwell(duration, rng), basis)
    if value_score >= EXPLORE_MIN_SCORE and rng.random() < EXPLORATION_RATE:
        return Decision("WATCH", watch_dwell(duration, rng), basis, explored=True)
    return Decision("SKIP", rng.uniform(*skip_dwell), basis)
