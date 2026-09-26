"""
FeedLab — Story Grouping (the "Grouping" in P.R.A.G.)

Several reels often report the same story (Aaj Tak and Indian Express on
the same incident). Each informative reel's topic and summary is embedded
with a small multilingual sentence model, and reels closer than
STORY_DISTANCE (cosine distance, average linkage) become one story.
"""

import datetime
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import EMBEDDING_MODEL, STORY_DISTANCE

_model = None


@dataclass
class Reel:
    """One informative reel (a scored impression, deduplicated by shortcode)."""
    code: str
    url: str
    creator: str
    topic: str
    summary: str
    caption: str
    value_score: float
    first_seen: datetime.datetime      # when the agent first saw it (UTC)
    taken_at: int | None = None        # when it was posted (unix time)

    @property
    def text(self) -> str:
        """What gets embedded: the scorer's topic and one-line summary (English)."""
        return f"{self.topic}. {self.summary or self.caption[:300]}"


@dataclass
class Story:
    reels: list[Reel] = field(default_factory=list)   # most valuable first

    @property
    def lead(self) -> Reel:
        return self.reels[0]

    @property
    def title(self) -> str:
        return self.lead.topic

    @property
    def summary(self) -> str:
        return self.lead.summary

    @property
    def score(self) -> float:
        return self.lead.value_score

    @property
    def first_seen(self) -> datetime.datetime:
        return min(r.first_seen for r in self.reels)

    @property
    def creators(self) -> list[str]:
        return sorted({r.creator for r in self.reels})


def embed(texts: list[str]) -> np.ndarray:
    """Unit-length sentence embeddings (the model downloads once, ~470 MB)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return np.asarray(_model.encode(texts, normalize_embeddings=True))


def group(
    reels: list[Reel],
    distance: float = STORY_DISTANCE,
    embed_fn: Callable[[list[str]], np.ndarray] = embed,
) -> list[Story]:
    """Cluster reels into stories, most valuable story first."""
    if not reels:
        return []
    if len(reels) == 1:
        labels = [0]
    else:
        from sklearn.cluster import AgglomerativeClustering

        vectors = embed_fn([r.text for r in reels])
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance
        ).fit_predict(vectors)
    stories: dict[int, Story] = {}
    for label, reel in zip(labels, reels):
        stories.setdefault(int(label), Story()).reels.append(reel)
    for story in stories.values():
        story.reels.sort(key=lambda r: (-r.value_score, r.first_seen))
    return sorted(stories.values(), key=lambda s: (-s.score, s.first_seen))
