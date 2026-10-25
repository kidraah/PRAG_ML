"""
PRAG — the latest news, for feeds that are current.

Fetches the newest stories from Indian outlets' own RSS feeds (no search
API needed, nothing paid), stores each as an RSS item with its link and
publication time, and picks the stories each persona's feed would serve:
drawn by that persona's trained preference model (prefs.py), squared, with
some exploration, the way the algorithm keeps serving what someone engages
with. generate.run_news then writes reels about each picked story, in the
persona's feed, told to stick to the facts in the headline and summary.

    python -m gradient.news [--days 4] [--stories 150] [--search]   fetch, pick and write for both personas
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
# Google News searches, per persona topic, for stories further back than the outlets' feeds keep (headlines only)
SEARCH_URL = "https://news.google.com/rss/search"
SEARCHES = {
    "national_security": ["Indian Army", "India border China Pakistan", "Indian Navy Air Force defence"],
    "infrastructure": ["India expressway highway", "Vande Bharat railway", "metro project India"],
    "government_schemes": ["government scheme yojana India", "PM Kisan Ayushman Bharat"],
    "party_politics": ["BJP Congress", "India election campaign", "Lok Sabha Rajya Sabha"],
    "economy": ["India economy inflation GDP", "petrol diesel price India", "rupee RBI"],
    "labour": ["workers strike India", "gig workers India", "trade union India wages"],
    "public_services": ["government school India", "government hospital India", "public transport fares India"],
    "civil_liberties": ["internet shutdown India", "journalist arrested India", "Supreme Court bail India"],
    "environment": ["air pollution Delhi AQI", "India climate flood heatwave", "India renewable solar"],
    "crime_safety": ["cyber fraud India", "police India case"],
    "world_news": ["India foreign policy", "Indians abroad"],
    "cricket": ["India cricket", "IPL", "Indian women cricket"],
    "football": ["Indian football ISL", "Premier League", "Champions League"],
    "other_sports": ["India hockey badminton chess", "Neeraj Chopra athletics"],
    "gadgets": ["smartphone launch India", "phone review India price"],
    "tech_ai": ["AI India startup", "UPI digital India"],
    "science_space": ["ISRO", "India science research"],
    "hindi_comedy": ["stand-up comedy India comedian"],
    "street_food": ["street food India"],
    "indie_music": ["indie music India", "Indian music festival"],
    "film_tv": ["Bollywood box office", "OTT web series India"],
    "photography": ["photography India exhibition"],
    "gaming": ["BGMI esports India", "video game India"],
    "religion": ["temple festival India"],
    "finance": ["Sensex Nifty", "mutual fund SIP India"],
    "history_explainers": ["India history anniversary"],
    "travel_nature": ["India tourism travel", "India wildlife tiger"],
}
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


def search(days: int = 14, progress=print) -> list[dict]:
    """Stories from the last `days` days found by the topic searches, one per link (headline and outlet only)."""
    from urllib.parse import urlencode

    from stories.verify import parse_rss

    found, failed = {}, 0
    for topic, queries in SEARCHES.items():
        for query in queries:
            params = urlencode({"q": f"{query} when:{days}d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
            try:
                request = urllib.request.Request(f"{SEARCH_URL}?{params}", headers={"User-Agent": "Mozilla/5.0 (PRAG research)"})
                with urllib.request.urlopen(request, timeout=20) as resp:
                    articles = parse_rss(resp.read())
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            for a in articles:
                try:
                    published = datetime.datetime.fromisoformat(a.published).replace(tzinfo=datetime.timezone.utc)
                except ValueError:
                    published = None
                found.setdefault(a.title.lower(), {"title": a.title, "url": a.url, "summary": "", "source": a.source,
                                                   "published": published, "topic_hint": topic})
    if failed:
        progress(f"  {failed} searches failed")
    return sorted(found.values(), key=lambda s: s["published"] or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
                  reverse=True)


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
    parser.add_argument("--search", action="store_true", help="also search Google News by topic (goes further back)")
    args = parser.parse_args()
    say = lambda m: print(m, flush=True)  # noqa: E731
    stories = fetch(args.days, progress=say)
    if args.search:
        known = {s["title"].lower() for s in stories}
        stories += [s for s in search(int(args.days), progress=say) if s["title"].lower() not in known]
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
