"""
PRAG — Creator Memory

Per-creator score history across sessions, used two ways: follow creators
who keep posting valuable reels (a follow is one of the strongest signals
a feed gets), and skip creators who never do without spending an LLM
call. Each reel counts once per creator, however often it is shown.
"""

from config import (
    CREATOR_SKIP_MAX_SCORE,
    CREATOR_SKIP_MIN_SEEN,
    FOLLOW_MIN_HIGH_REELS,
    LIKE_THRESHOLD,
)


class CreatorMemory:
    def __init__(self) -> None:
        self._scores: dict[str, dict[str, float]] = {}
        self._following: set[str] = set()

    @classmethod
    def from_history(cls, scores: list[tuple[str, str, float]], followed: set[str]) -> "CreatorMemory":
        """Build from (creator, code, value_score) rows and the set of followed creators."""
        memory = cls()
        for creator, code, score in scores:
            memory.record(creator, code, score)
        memory._following |= {c for c in followed if c}
        return memory

    def record(self, creator: str, code: str, score: float) -> None:
        if creator and code and score is not None:
            self._scores.setdefault(creator, {})[code] = score

    def mark_following(self, creator: str) -> None:
        if creator:
            self._following.add(creator)

    def is_following(self, creator: str) -> bool:
        return creator in self._following

    def seen(self, creator: str) -> int:
        return len(self._scores.get(creator, {}))

    def high_count(self, creator: str) -> int:
        return sum(1 for s in self._scores.get(creator, {}).values() if s >= LIKE_THRESHOLD)

    def should_skip(self, creator: str) -> bool:
        """Enough reels seen from this creator, and none came close to valuable."""
        scores = self._scores.get(creator, {})
        return len(scores) >= CREATOR_SKIP_MIN_SEEN and max(scores.values()) < CREATOR_SKIP_MAX_SCORE

    def qualifies_for_follow(self, creator: str) -> bool:
        return bool(creator) and not self.is_following(creator) and self.high_count(creator) >= FOLLOW_MIN_HIGH_REELS
