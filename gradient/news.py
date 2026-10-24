"""
PRAG — the latest news, for feeds that are current.

Fetches the newest stories from Indian outlets' own RSS feeds (no search
API needed, nothing paid), stores each as an RSS item with its link and
publication time, and picks the stories each persona's feed would serve:
drawn by that persona's trained preference model (prefs.py), squared, with
some exploration, the way the algorithm keeps serving what someone engages
with. generate.run_news then writes reels about each picked story, in the
persona's feed, told to stick to the facts in the headline and summary.

    python -m gradient.news [--days 4] [--stories 150]   fetch, pick and write for both personas
"""

import argparse
import asyncio
import datetime
import hashlib
import html
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np

from gradient import generate, prefs, store

FEEDS = [
    ("https://www.thehindu.com/news/national/feeder/default.rss", "The Hindu"),
    ("https://www.thehindu.com/business/Economy/feeder/default.rss", "The Hindu"),
    ("https://www.thehindu.com/sci-tech/science/feeder/default.rss", "The Hindu"),
    ("https://www.thehindu.com/sci-tech/energy-and-environment/feeder/default.rss", "The Hindu"),
    ("https://www.thehindu.com/sci-tech/technology/gadgets/feeder/default.rss", "The Hindu"),
    ("https://www.thehindu.com/sport/cricket/feeder/default.rss", "The Hindu"),
    ("https://www.thehindu.com/sport/football/feeder/default.rss", "The Hindu"),
    ("https://www.thehindu.com/food/feeder/default.rss", "The Hindu"),
    ("https://www.thehindu.com/entertainment/movies/feeder/default.rss", "The Hindu"),
    ("https://indianexpress.com/section/india/feed/", "The Indian Express"),
    ("https://indianexpress.com/section/political-pulse/feed/", "The Indian Express"),
    ("https://indianexpress.com/section/business/feed/", "The Indian Express"),
    ("https://indianexpress.com/section/sports/feed/", "The Indian Express"),
    ("https://indianexpress.com/section/technology/feed/", "The Indian Express"),
    ("https://indianexpress.com/section/entertainment/feed/", "The Indian Express"),
    ("https://indianexpress.com/section/cities/pune/feed/", "The Indian Express"),
    ("https://www.livemint.com/rss/politics", "Mint"),
    ("https://www.livemint.com/rss/news", "Mint"),
    ("https://www.livemint.com/rss/money", "Mint"),
    ("https://www.livemint.com/rss/technology", "Mint"),
    ("https://feeds.feedburner.com/ndtvnews-top-stories", "NDTV"),
    ("https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml", "Hindustan Times"),
    ("https://timesofindia.indiatimes.com/rssfeedstopstories.cms", "The Times of India"),
    ("https://feeds.bbci.co.uk/hindi/rss.xml", "BBC News Hindi"),
    ("https://feeds.bbci.co.uk/news/world/asia/india/rss.xml", "BBC News"),
]
TAG = re.compile(r"<[^>]+>")


def clean(text: str | None) -> str:
    return " ".join(html.unescape(TAG.sub(" ", text or "")).split())


def parse(xml_text: bytes, source: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter("item"):
        title, link = clean(item.findtext("title")), (item.findtext("link") or "").strip()
        if not title or not link.startswith("http"):
            continue
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(datetime.timezone.utc)
        except (TypeError, ValueError):
            published = None
        summary = clean(item.findtext("description"))[:600]
        out.append({"title": title, "url": link, "summary": summary if summary != title else "", "source": source,
                    "published": published})
    return out


def fetch(days: float = 4, feeds=FEEDS, progress=print) -> list[dict]:
    """Every story from the last `days` days, newest first, one per link."""
    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    stories, failed = {}, []
    for url, source in feeds:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PRAG research)"})
            with urllib.request.urlopen(request, timeout=20) as resp:
                for story in parse(resp.read(), source):
                    if story["published"] is None or story["published"] >= since:
                        stories.setdefault(story["url"].split("?")[0], story)
        except Exception as exc:  # noqa: BLE001 - one dead feed doesn't stop the rest
            failed.append(f"{source}: {exc.__class__.__name__}")
    if failed:
        progress(f"  {len(failed)} feeds failed: {', '.join(failed)}")
    return sorted(stories.values(), key=lambda s: s["published"] or since, reverse=True)


def story_group(story: dict) -> str:
    return "news-" + hashlib.sha1(story["url"].split("?")[0].encode()).hexdigest()[:12]


def store_stories(conn, stories: list[dict]) -> tuple[str, list[dict]]:
    """Each story as an RSS item (its link, outlet and publication time); (batch, stories with their item id)."""
    batch = store.new_batch(conn, "news", f"the latest news: {len(stories)} stories from {len(FEEDS)} feeds",
                            batch_id=f"news-{datetime.datetime.now():%Y%m%d-%H%M}")
    for story in stories:
        text = story["title"] + (f". {story['summary']}" if story["summary"] else "")
        story["item_id"], _ = store.add_item(conn, {
            "origin": "rss", "source_name": story["source"], "source_url": story["url"], "title": story["title"],
            "caption": story["title"], "text": text, "group_id": story_group(story),
            "language": "hi" if re.search(r"[ऀ-ॿ]", story["title"]) else "en",
            "meta": {"published_at": story["published"].isoformat() if story["published"] else None,
                     "license": "headline and summary, linked to the source"}}, batch)
        story["group"] = story_group(story)
    conn.commit()
    return batch, stories


def pick(stories: list[dict], account: str, n: int, seed: int = 5) -> list[dict]:
    """n stories for this persona's feed: weighted by its preference model's probability squared, 15% at random."""
    model = prefs.load(account)
    p = prefs.predict(model, [s["title"] + ". " + s["summary"] for s in stories]) if model else np.full(len(stories), .5)
    rng = random.Random(f"{seed}|{account}|{len(stories)}")
    pool = list(range(len(stories)))
    picked = []
    while pool and len(picked) < n:
        weights = [1.0 if rng.random() < 0.15 else 0.02 + p[i] ** 2 for i in pool]
        picked.append(pool.pop(rng.choices(range(len(pool)), weights=weights)[0]))
    return [{**stories[i], "p": float(p[i])} for i in picked]


def main() -> None:
    parser = argparse.ArgumentParser(description="The latest news into both personas' feeds")
    parser.add_argument("--days", type=float, default=4)
    parser.add_argument("--stories", type=int, default=150, help="stories per persona")
    parser.add_argument("--reels", type=int, default=6, help="reels written per story")
    args = parser.parse_args()
    say = lambda m: print(m, flush=True)  # noqa: E731
    stories = fetch(args.days, progress=say)
    say(f"{len(stories)} stories from the last {args.days:g} days")
    conn = store.connect()
    _batch, stories = store_stories(conn, stories)
    conn.close()
    for account in store.ACCOUNTS:
        chosen = pick(stories, account, args.stories)
        say(f"{account}: {len(chosen)} stories picked by its preference model, e.g. "
            + "; ".join(s["title"][:60] for s in chosen[:3]))
        asyncio.run(generate.run_news(account, chosen, args.reels, progress=say))


if __name__ == "__main__":
    main()
