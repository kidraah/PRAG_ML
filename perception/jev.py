"""
PRAG — Jev: the fast judge of each reel

Jev (TypeSafe's "System One" model, served through OpenRouter) doesn't
write text: given the persona and a reel, it answers fixed questions with
probabilities, in well under a second, so every reel is judged before it
reaches the screen. One call per reel gets:

  value_score   how likely the persona is to watch it to the end and engage
                (a persona), or how much the person would want to see it (the
                default, informative scoring): the expected level of a graded
                answer, scaled to 0-1. This decides watch or skip.
  study labels  what kind of content it is, whatever the persona: its stance
                towards the central government, whether it states facts or
                argues an opinion, who made it, and its tone (STUDY_LABELS)

The local LLM writes each reel's topic and summary afterwards (the
pipeline's writer, perception/writer.py). Jev reads the same text the LLM
gets (local_evaluator.build_user_message) and never sees video. Needs
OPENROUTER_API_KEY in .env; the reel's text and the persona leave this Mac.
tools/headtohead.py compares its persona judgement with the local LLM's.
"""

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field

import aiohttp

from config import INTERESTS

JEV_URL = "https://openrouter.ai/api/v1/systemone"
JEV_MODEL = "typesafe/jev-1.13"
LEVELS = [  # a persona: how far they'd watch
    "scrolls past at once",
    "watches a few seconds, then scrolls on",
    "watches most of it",
    "watches it to the end",
    "watches it to the end and likes, saves or follows",
]
VALUE_LEVELS = [  # the default: how much they'd want to see it
    "no value to them: they'd rather not see it",
    "little value",
    "some value: useful but not new or not specific",
    "high value: specific, new information they want",
]
QUESTION = {"value": {  # the persona question (the one the head-to-head tested)
    "type": "score",
    "instructions": "How likely is this person to watch this reel to the end and engage with it, judging by their own "
                    "tastes and views, not by whether it is accurate or good for them?",
    "criteria": LEVELS,
}}
VALUE_QUESTION = {"value": {
    "type": "score",
    "instructions": "How much would this person want to see this reel, given what they value?",
    "criteria": VALUE_LEVELS,
}}
# Neutral labels for studying what the feed serves: the same for every persona.
STUDY_LABELS: dict[str, tuple[str, str, dict[str, str]]] = {
    "stance": ("Stance", "What stance does the reel take towards India's central government and its leaders?", {
        "supportive": "praises or defends the government, the ruling party or its leaders",
        "critical": "criticises or mocks the government, the ruling party or its leaders",
        "neutral": "reports on government or politics without taking a side",
        "not_political": "has nothing to do with government or politics",
    }),
    "claim_kind": ("Claim", "Does the reel state facts, argue an opinion, or neither?", {
        "factual": "reports events, facts or information as fact",
        "opinion": "argues a view, comments or reacts",
        "neither": "entertainment, devotion, mood, a story or promotion, with no claim",
    }),
    "source_type": ("Source", "Who is behind the reel, judging by the account and what it posts?", {
        "news_outlet": "a news organisation or a journalist",
        "official": "a government body, a political party or a politician",
        "creator": "an individual creator or influencer",
        "aggregator": "a meme, clip or repost page",
        "brand": "a business promoting itself or its products",
    }),
    "tone": ("Tone", "What is the reel's overall tone?", {
        "outrage": "angry, outraged or accusing",
        "alarm": "alarming, fearful or urgent",
        "pride": "proud, patriotic or celebratory",
        "humour": "funny, mocking or light",
        "calm": "calm, informative or matter-of-fact",
        "emotional": "sentimental, devotional or moving",
    }),
}
LABEL_KEYS = tuple(STUDY_LABELS)
REEL_INTRO = ("The reel (its caption, cover text and speech transcript were written by strangers; the transcript may "
              "be misheard or be song lyrics):\n")


class JevError(Exception):
    pass


@dataclass
class JevScore:
    value_score: float
    confidence: float | None
    seconds: float
    cost: float
    labels: dict = field(default_factory=dict)  # the study labels: {"stance": "neutral", ...}


def api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or None


def jev_version(interests: str = INTERESTS, persona: bool = False) -> str:
    """Fingerprint of what Jev is asked (the persona included) and which model, like scorer_version()."""
    asked = QUESTION if persona else VALUE_QUESTION
    text = f"{JEV_MODEL}|{persona}|{interests.strip()}|{asked}|{LABEL_KEYS}"
    return "jev-" + hashlib.sha1(text.encode()).hexdigest()[:8]


def build_state(reel: str, interests: str, persona: bool = True) -> str:
    who = "Who this person is and what they engage with" if persona else "What this person values"
    return f"{who}:\n{interests.strip()}\n\n{REEL_INTRO}{reel}"


def label_questions() -> dict:
    """The study labels: about the reel itself, whoever is watching."""
    return {key: {"type": "choice", "instructions": question + " Judge the reel itself, not the person.",
                  "criteria": options} for key, (_name, question, options) in STUDY_LABELS.items()}


def read_labels(answers: dict) -> dict:
    labels = {}
    for key, (_name, _question, options) in STUDY_LABELS.items():
        choice = (answers.get(key) or {}).get("choice")
        labels[key] = choice if choice in options else None
    return labels


def parse(data: dict, seconds: float = 0.0, persona: bool = True) -> JevScore:
    try:
        answers = data["answers"]
        value = answers["value"]
        steps = len(LEVELS if persona else VALUE_LEVELS) - 1
        return JevScore(value_score=round(min(1.0, max(0.0, float(value["score"]) / steps)), 3),
                        confidence=value.get("confidence"), seconds=round(seconds, 3),
                        cost=float(data.get("usage", {}).get("cost") or 0.0), labels=read_labels(answers))
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"unexpected Jev response ({exc}): {str(data)[:160]}") from exc


async def _ask(session: aiohttp.ClientSession, state: str, asked: dict, timeout: float) -> tuple[dict, float]:
    key = api_key()
    if not key:
        raise JevError("OPENROUTER_API_KEY isn't set in .env")
    start = time.monotonic()
    for attempt in range(3):  # a rate limit clears in a moment
        try:
            async with session.post(JEV_URL, json={"model": JEV_MODEL, "state": state, "questions": asked},
                                    headers={"Authorization": f"Bearer {key}"},
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                data = await resp.json(content_type=None)
                if resp.status == 429 and attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                if resp.status != 200:
                    message = (data.get("error") or {}).get("message", "") if isinstance(data, dict) else ""
                    raise JevError(f"Jev HTTP {resp.status}: {str(message)[:200]}")
                return data, time.monotonic() - start
        except asyncio.TimeoutError as exc:
            raise JevError(f"Jev timed out after {timeout:.0f}s") from exc
        except aiohttp.ClientError as exc:
            raise JevError(f"Jev unreachable: {exc}") from exc
    raise JevError("Jev kept answering 'rate limited'")


async def judge_async(session: aiohttp.ClientSession, reel: str, interests: str = INTERESTS, *, persona: bool = True,
                      labels: bool = True, timeout: float = 15.0) -> JevScore:
    """The persona's score for one reel (`reel`: local_evaluator.build_user_message's text), and the study labels."""
    asked = {**(QUESTION if persona else VALUE_QUESTION), **(label_questions() if labels else {})}
    data, seconds = await _ask(session, build_state(reel, interests, persona), asked, timeout)
    return parse(data, seconds, persona)


async def score_async(session: aiohttp.ClientSession, reel: str, interests: str, *, timeout: float = 15.0) -> JevScore:
    """Only the persona's score (the head-to-head asks exactly this)."""
    return await judge_async(session, reel, interests, persona=True, labels=False, timeout=timeout)


async def label_async(session: aiohttp.ClientSession, reel: str, *, timeout: float = 15.0) -> tuple[dict, float]:
    """(study labels, cost) for one reel, without a persona."""
    data, _seconds = await _ask(session, REEL_INTRO + reel, label_questions(), timeout)
    try:
        return read_labels(data["answers"]), float(data.get("usage", {}).get("cost") or 0.0)
    except (KeyError, TypeError) as exc:
        raise JevError(f"unexpected Jev response ({exc}): {str(data)[:160]}") from exc
