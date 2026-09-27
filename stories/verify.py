"""
PRAG — Story Verification

Checks whether a story seen on Reels is also covered by news outlets,
using Google News' public RSS search (one or two queries per story, a
second apart):

1. search for the story's distinctive words — names from its summary plus
   its topic label, without generic news words — back to the reel's
   posting date, and for the topic's own words too when that finds
   little or nothing close;
2. keep the headlines whose meaning is close to the story's summary or
   its first clause — headlines state the gist, not the details (the
   same sentence embeddings used for grouping).

Word overlap alone was too brittle: headlines rarely repeat a reel's
numbers or phrasing, and generic words ("Asian Games") matched unrelated
reports. A story with no close headline is "Not in the news yet" —
sometimes a scoop or an original explainer, sometimes not true. The
matched headlines are kept: they often carry a later development.
"""

import datetime
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Callable

import numpy as np

from config import VERIFY_DAYS, VERIFY_MAX_DAYS, VERIFY_SIMILARITY
from stories.grouping import embed

SEARCH_URL = "https://news.google.com/rss/search"
STOPWORDS = set("""
a an the and or but yet so if then of to in on at for with by from as is are was were be been being has have
had this that these those its it his her their our your after before over under into about than more most
some any each which who whom whose what when where why how not no also only just new says said may could
would will can amid during while between against without within across per via vs versus there here all
""".split())
# Words that appear in too many headlines to identify a story.
GENERIC = set("""
india indian indias us usa u.s america american government govt minister ministry state states national
news today report reports reported says said official officials people world country city day days week
year years first latest update updates new big major top live video chief leader leaders party police
court case explainer
""".split())
# Verbs the scorer's summaries often start with.
SUMMARY_VERBS = {"claims", "shows", "explains", "lists", "asks", "argues", "reports", "describes",
                 "discusses", "examines", "says", "highlights", "urges", "warns"}


@dataclass
class Article:
    title: str
    source: str
    url: str
    published: str = ""
    similarity: float = 0.0


@dataclass
class Verification:
    status: str                         # widely_reported | reported | reels_only | not_checked
    query: str = ""
    outlets: list[str] = field(default_factory=list)
    headlines: list[Article] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.status == "widely_reported":
            return f"In the news: {len(self.outlets)} outlets"
        if self.status == "reported":
            return "In the news: " + ", ".join(self.outlets)
        if self.status == "reels_only":
            return "Not in the news yet"
        return "Not checked"


def key_terms(topic: str, summary: str, limit: int = 4) -> list[str]:
    """Distinctive words for a news search: names from the summary first, then the topic's words."""
    terms: list[str] = []

    def add(word: str) -> None:
        low = word.lower()
        if len(word) < 2 or low in STOPWORDS or low in GENERIC or low in SUMMARY_VERBS:
            return
        if low not in (t.lower() for t in terms):
            terms.append(word)

    clean = lambda text: (text or "").replace("’s", "").replace("'s", "")
    for word in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", clean(summary)):
        if word[0].isupper():
            add(word)
            if len(terms) >= limit - 1:
                break
    for word in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", clean(topic)):
        add(word)
    return terms[:limit]


def search_window(taken_at: int | None, now: datetime.datetime | None = None) -> int:
    """Days of news to search: back to the reel's posting date (plus a margin), within limits."""
    if not taken_at:
        return VERIFY_DAYS
    now = now or datetime.datetime.now(datetime.timezone.utc)
    age = (now - datetime.datetime.fromtimestamp(taken_at, datetime.timezone.utc)).days
    return max(VERIFY_DAYS, min(VERIFY_MAX_DAYS, age + 2))


def parse_rss(xml_text: bytes | str) -> list[Article]:
    """Articles from a Google News RSS response (headline suffix " - Outlet" removed)."""
    root = ET.fromstring(xml_text)
    articles = []
    for item in root.iter("item"):
        source = (item.findtext("source") or "").strip()
        title = (item.findtext("title") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        published = item.findtext("pubDate") or ""
        try:
            published = parsedate_to_datetime(published).date().isoformat()
        except (TypeError, ValueError):
            pass
        articles.append(Article(title=title, source=source, url=(item.findtext("link") or "").strip(),
                                published=published))
    return articles


def search_news(query: str, days: int = VERIFY_DAYS) -> list[Article]:
    params = {"q": f"{query} when:{days}d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    request = urllib.request.Request(f"{SEARCH_URL}?{urllib.parse.urlencode(params)}",
                                     headers={"User-Agent": "Mozilla/5.0 (PRAG digest)"})
    with urllib.request.urlopen(request, timeout=20) as resp:
        return parse_rss(resp.read())


def verify(
    topic: str,
    summary: str,
    *,
    taken_at: int | None = None,
    search: Callable[[str, int], list[Article]] = search_news,
    embed_fn: Callable[[list[str]], np.ndarray] = embed,
    now: datetime.datetime | None = None,
) -> Verification:
    terms = key_terms(topic, summary)
    if len(terms) < 2 or not summary:
        return Verification("not_checked", query=" ".join(terms))
    query, days = " ".join(terms), search_window(taken_at, now)
    try:
        articles = _unique(search(query, days))
    except Exception:  # noqa: BLE001 — offline or rate-limited: say so rather than guess
        return Verification("not_checked", query=query)
    matched = close_headlines(summary, articles, embed_fn)
    broader = key_terms(topic, "")
    if (len(articles) < 5 or not matched) and len(broader) >= 2 and broader != terms:
        try:  # too narrow, or nothing close: widen to the topic's words
            articles = _unique(articles + search(" ".join(broader), days))
            matched = close_headlines(summary, articles, embed_fn)
        except Exception:  # noqa: BLE001 — keep what the first search found
            pass
    if not articles:
        return Verification("reels_only", query=query)
    outlets: list[str] = []
    for article in matched:
        if article.source and article.source not in outlets:
            outlets.append(article.source)
    status = "widely_reported" if len(outlets) >= 3 else "reported" if outlets else "reels_only"
    return Verification(status, query=query, outlets=outlets, headlines=matched[:3])


def gist(summary: str) -> str:
    """The summary's first clause: "X and Y traded charges at the UN, with X accusing …" → "X and Y traded charges at the UN"."""
    first = re.split(r"(?<=[.!?])\s", summary, maxsplit=1)[0]
    return re.split(r",\s(?:with|while|as|after|amid|which)\s|;\s", first, maxsplit=1)[0]


def close_headlines(summary: str, articles: list[Article], embed_fn) -> list[Article]:
    """Articles whose headline is close in meaning to the summary or its gist, closest first."""
    if not articles:
        return []
    vectors = embed_fn([summary, gist(summary)] + [a.title for a in articles])
    similarity = (vectors[2:] @ vectors[:2].T).max(axis=1)
    matched = []
    for article, score in sorted(zip(articles, similarity), key=lambda x: -x[1]):
        if score < VERIFY_SIMILARITY:
            break
        article.similarity = round(float(score), 2)
        matched.append(article)
    return matched


def _unique(articles: list[Article]) -> list[Article]:
    """The broader search can return the same headline again."""
    unique: dict[tuple[str, str], Article] = {}
    for article in articles:
        unique.setdefault((article.title.lower(), article.source), article)
    return list(unique.values())


def verify_all(stories, pause: float = 1.0, **kwargs) -> list[Verification]:
    """Verify stories one at a time, pausing between queries."""
    results = []
    for i, story in enumerate(stories):
        if i:
            time.sleep(pause)
        results.append(verify(story.title, story.summary, taken_at=story.lead.taken_at, **kwargs))
    return results
