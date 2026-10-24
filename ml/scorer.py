"""
PRAG — Model Scorer (the fallback when LM Studio isn't available)

Scores a recorded reel with the saved preference model (python -m
ml.train --save) instead of the LLM, so `main.py --rules` still produces
scores — and the digest still has content — when LM Studio is off. The
model has no words of its own, so the summary is the caption's first line
(or the cover text / transcript when there is no caption).
"""

import os
import re

from agent.bias import DEFAULT_FINGERPRINT
from config import WATCH_THRESHOLD
from ml.dataset import reel_text
from stories.grouping import clip

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "informative-embedding.joblib")  # the single shared file older versions saved
FOLLOWED_FLOOR = 0.80  # a followed account's reel counts at least this much (the quick rules' follow signal)
TRAILING_TAGS = re.compile(r"(?:(?:^|\s+)[#@][^\s#@]+)+\s*$")  # "... #news #india @ndtv" at a line's end
ABBREVIATION = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Ms|St|Jr|Sr|Prof|Gen|Col|Lt|Capt|Gov|Sen|Rep|No|Rs|vs|etc)|\b[A-Z])\.$")


def lead_line(caption: str | None, cover_text: str | None = None, transcript: str | None = None,
              limit: int = 200) -> str:
    """A one-line summary without an LLM: the caption's first line or sentence, without links or tag lists."""
    for text in (caption, cover_text, transcript):
        for line in re.sub(r"https?://\S+", " ", text or "").splitlines():  # a caption's first line is its headline
            line = TRAILING_TAGS.sub("", line)
            line = " ".join(re.sub(r"#([^\s#@]+)", r"\1", line).split())  # inline #tags are words: "#India vs #Pakistan"
            if len(line) >= 12:
                return clip(first_sentence(line), limit)
    return ""


def first_sentence(line: str) -> str:
    """Up to the first full stop — not one after "Dr." or an initial, and not a short opener like "Big news!"."""
    parts = re.split(r"(?<=[.!?।])\s", line)
    sentence = parts[0]
    for part in parts[1:]:
        if len(sentence) >= 12 and not ABBREVIATION.search(sentence):
            break
        sentence += " " + part
    return sentence


def path_for(bias=None, model_dir: str | None = None) -> str:
    """The saved model for this bias: its own file (ml/train.py saves one per bias fingerprint and kind, so training
    for one persona never overwrites another's), the newest of the two kinds; else the older shared file."""
    model_dir = model_dir or MODEL_DIR
    fingerprint = bias.fingerprint if bias is not None else DEFAULT_FINGERPRINT
    own = [os.path.join(model_dir, f"informative-{fingerprint}-{kind}.joblib") for kind in ("embedding", "tfidf")]
    own = [p for p in own if os.path.exists(p)]
    if own:
        return max(own, key=os.path.getmtime)
    return os.path.join(model_dir, os.path.basename(MODEL_PATH))


class ModelScorer:
    """Loads the saved model once; score() returns the probability a reel is informative."""

    def __init__(self, path: str | None = None, bias=None):
        import joblib

        saved = joblib.load(path or path_for(bias))
        self.kind = saved["kind"]
        self.model = saved["model"]
        self.version = f"model-{saved['kind']}-{saved['created']}"

    @staticmethod
    def trained_for(path: str | None = None, bias=None) -> str | None:
        """The interests fingerprint of the bias the saved model was trained for; None without a model."""
        path = path or path_for(bias)
        if not os.path.exists(path):
            return None
        import joblib

        return joblib.load(path).get("bias") or DEFAULT_FINGERPRINT  # saved before biases: the default's

    @staticmethod
    def available(path: str | None = None, bias=None) -> bool:
        """Whether a saved model exists — and, given a bias, was trained for its interests (labels judged
        under one bias say nothing about another)."""
        trained = ModelScorer.trained_for(path, bias)
        return trained is not None and (bias is None or trained == bias.fingerprint)

    def probability(self, text: str) -> float:
        if self.kind == "tfidf":
            return float(self.model.predict_proba([text])[0, 1])
        from stories.grouping import embed

        return float(self.model.predict_proba(embed([text or "(empty)"]))[0, 1])

    def score(self, meta, transcript: str, cover_text: str) -> tuple[float, str]:
        """(value score, reasoning) for a ReelMeta: the model's probability, floored for followed accounts."""
        probability = self.probability(reel_text(meta.caption, cover_text, transcript))
        value = max(probability, FOLLOWED_FLOOR if meta.creator_following else 0.0)
        reason = f"model p={probability:.2f}" + (", followed account" if meta.creator_following else "")
        return value, reason

    @staticmethod
    def informative(value: float) -> bool:
        return value >= WATCH_THRESHOLD
