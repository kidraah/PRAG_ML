"""
PRAG — Daily Digest

What you read instead of scrolling: the informative reels an account's
agent saw in the last DIGEST_HOURS, grouped into stories, checked against
the news, and written to

    digests/<account>/<date>.html   the page to read
    digests/<account>/<date>.json   the same stories, for the dataset
    digests/<account>/latest.html   always the newest page

    python -m stories.digest --account NAME [--hours 24] [--no-verify]
"""

import argparse
import datetime
import html
import json
import os
import sqlite3
from dataclasses import dataclass, field

from agent.bias import load as load_bias
from config import DIGEST_HOURS, LIKE_THRESHOLD, WATCH_THRESHOLD
from database.db import choose_account, get_connection, list_accounts
from stories.grouping import Reel, Story, group
from stories.verify import Verification, verify_all

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIGEST_DIR = os.path.join(PROJECT_ROOT, "digests")

REELS_SQL = """
SELECT i.code, i.url, i.creator, i.topic, i.summary, i.caption, i.value_score, i.shown_at, i.taken_at
FROM impressions i JOIN sessions s ON s.id = i.session_id
WHERE s.account = ? AND i.shown_at >= ? AND i.shown_at <= ? AND i.score_basis = 'score' AND {keep}
ORDER BY i.shown_at
"""
# What makes a reel digest material: worth watching for the bias, or — for a persona bias, whose scores
# say what the person would enjoy — any news or information the feed pushed to them.
KEEP_VALUABLE = "i.value_score >= ?"
KEEP_INFORMATIONAL = "i.is_informational = 1 AND ? IS NOT NULL"
STATS_SQL = """
SELECT COUNT(*) AS reels, COUNT(DISTINCT i.session_id) AS sessions, COALESCE(SUM(i.dwell_seconds), 0) AS seconds
FROM impressions i JOIN sessions s ON s.id = i.session_id
WHERE s.account = ? AND i.shown_at >= ? AND i.shown_at <= ?
"""


@dataclass
class Digest:
    account: str
    generated_at: datetime.datetime
    hours: float
    reels_seen: int
    sessions: int
    minutes: float
    stories: list[Story] = field(default_factory=list)
    checks: list[Verification] = field(default_factory=list)

    @property
    def informative_reels(self) -> int:
        return sum(len(s.reels) for s in self.stories)


def _utc(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text).replace(tzinfo=datetime.timezone.utc)


def load_reels(conn: sqlite3.Connection, account: str, since: datetime.datetime,
               persona: bool | None = None, until: datetime.datetime | None = None) -> list[Reel]:
    """Informative reels shown to the account since `since`, one per shortcode (best score, first sighting).

    For a persona account (default: from its bias), the news and information the feed showed it.
    """
    if persona is None:
        persona = load_bias(account).persona
    conn.row_factory = sqlite3.Row
    reels: dict[str, Reel] = {}
    stamp = since.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end = (until or datetime.datetime.now(datetime.timezone.utc)).astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sql = REELS_SQL.format(keep=KEEP_INFORMATIONAL if persona else KEEP_VALUABLE)
    for row in conn.execute(sql, (account, stamp, end, WATCH_THRESHOLD)):
        seen = reels.get(row["code"])
        if seen is None or row["value_score"] > seen.value_score:
            reels[row["code"]] = Reel(
                code=row["code"], url=row["url"] or f"https://www.instagram.com/reels/{row['code']}/",
                creator=row["creator"] or "", topic=row["topic"] or "", summary=row["summary"] or "",
                caption=row["caption"] or "", value_score=row["value_score"],
                first_seen=seen.first_seen if seen else _utc(row["shown_at"]), taken_at=row["taken_at"],
            )
    return list(reels.values())


def build_digest(conn: sqlite3.Connection, account: str, hours: float = DIGEST_HOURS, *,
                 verify: bool = True, now: datetime.datetime | None = None, **group_kwargs) -> Digest:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    since = now - datetime.timedelta(hours=hours)
    conn.row_factory = sqlite3.Row
    stamp = lambda t: t.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    stats = conn.execute(STATS_SQL, (account, stamp(since), stamp(now))).fetchone()
    stories = group(load_reels(conn, account, since, until=now), **group_kwargs)
    checks = verify_all(stories) if verify else [Verification("not_checked") for _ in stories]
    return Digest(account=account, generated_at=now, hours=hours, reels_seen=stats["reels"],
                  sessions=stats["sessions"], minutes=stats["seconds"] / 60, stories=stories, checks=checks)


# ── Output ────────────────────────────────────────────────────────────

def story_title(story: Story, check: Verification) -> str:
    """The scorer's topic; for model-scored reels (no topic) the closest news headline when one was found."""
    if not story.lead.topic and check.headlines:
        return check.headlines[0].title
    return story.title


def to_json(digest: Digest) -> dict:
    def when(ts: datetime.datetime) -> str:
        return ts.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")

    return {
        "account": digest.account,
        "generated_at": when(digest.generated_at),
        "window_hours": digest.hours,
        "stats": {"reels_seen": digest.reels_seen, "sessions": digest.sessions,
                  "agent_minutes": round(digest.minutes, 1), "informative_reels": digest.informative_reels,
                  "stories": len(digest.stories)},
        "stories": [{
            "title": story_title(s, c), "summary": s.summary, "score": s.score, "first_seen": when(s.first_seen),
            "creators": s.creators,
            "reels": [{"code": r.code, "url": r.url, "creator": r.creator, "topic": r.topic, "summary": r.summary,
                       "value_score": r.value_score, "first_seen": when(r.first_seen)} for r in s.reels],
            "verification": {"status": c.status, "label": c.label, "query": c.query, "outlets": c.outlets,
                             "headlines": [vars(a) for a in c.headlines]},
        } for s, c in zip(digest.stories, digest.checks)],
    }


PAGE_CSS = """
:root { --bg:#f7f6f2; --card:#fff; --text:#1c1c1e; --muted:#6c6c72; --line:#e5e3dc;
        --ok:#0b7a53; --warn:#a85d00; --link:#2856a3; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#131315; --card:#1c1c1f; --text:#ececef; --muted:#9b9ba3; --line:#2d2d32;
          --ok:#46c393; --warn:#e3a54c; --link:#86a9ec; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans", "Noto Sans Devanagari", sans-serif; }
main { max-width:720px; margin:0 auto; padding:28px 16px 56px; }
h1 { font-size:1.55rem; margin:0; }
.stats { color:var(--muted); font-size:.9rem; margin-top:4px; }
h2 { font-size:.82rem; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); margin:34px 0 10px; }
.story { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 18px; margin:12px 0; }
.story h3 { margin:0 0 4px; font-size:1.08rem; }
.story p { margin:0; }
.badge { display:inline-block; font-size:.76rem; font-weight:600; padding:2px 9px; border-radius:999px;
         border:1px solid currentColor; margin-top:10px; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .none { color:var(--muted); }
ul.news { margin:8px 0 0; padding-left:18px; font-size:.9rem; }
a { color:var(--link); text-decoration:none; } a:hover { text-decoration:underline; }
.meta { color:var(--muted); font-size:.84rem; margin-top:10px; }
.empty { color:var(--muted); }
footer { color:var(--muted); font-size:.8rem; margin-top:44px; }
"""


def _story_html(story: Story, check: Verification) -> str:
    e = html.escape
    badge = {"widely_reported": "ok", "reported": "ok", "reels_only": "warn"}.get(check.status, "none")
    news = "".join(
        f'<li><a href="{e(a.url)}" target="_blank" rel="noopener">{e(a.title)}</a> — {e(a.source)}</li>'
        for a in check.headlines[:2]
    )
    seen = story.first_seen.astimezone().strftime("%d %b, %H:%M")
    reels = " · ".join(f'<a href="{e(r.url)}" target="_blank" rel="noopener">@{e(r.creator)}</a>' for r in story.reels)
    count = f"{len(story.reels)} reels" if len(story.reels) > 1 else "1 reel"
    title = story_title(story, check)
    summary = f"<p>{e(story.summary)}</p>" if story.summary and story.summary != title else ""
    return (
        f'<article class="story"><h3>{e(title)}</h3>{summary}'
        f'<span class="badge {badge}">{e(check.label)}</span>'
        + (f'<ul class="news">{news}</ul>' if news else "")
        + f'<div class="meta">First seen {e(seen)} · {count}: {reels}</div></article>'
    )


def render_html(digest: Digest) -> str:
    e = html.escape
    pairs = list(zip(digest.stories, digest.checks))
    top = [p for p in pairs if p[0].score >= LIKE_THRESHOLD]
    more = [p for p in pairs if p[0].score < LIKE_THRESHOLD]
    date = digest.generated_at.astimezone().strftime("%A, %d %B %Y")
    stats = (f"{len(digest.stories)} stories from {digest.informative_reels} informative reels, out of "
             f"{digest.reels_seen} reels the agent scrolled in {digest.minutes:.0f} min "
             f"({digest.sessions} sessions, last {digest.hours:g} h)")
    sections = ""
    for heading, items in (("Top stories", top), ("Also worth knowing", more)):
        if items:
            sections += f"<h2>{heading}</h2>" + "".join(_story_html(s, c) for s, c in items)
    if not sections:
        sections = '<p class="empty">Nothing informative in this window yet.</p>'
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>PRAG Digest</title><style>{PAGE_CSS}</style></head><body><main>"
        f"<header><h1>Digest for @{e(digest.account)}</h1><div class=\"stats\">{e(date)} · {e(stats)}</div></header>"
        f"{sections}"
        "<footer>Summaries come from the reels themselves. “Not in the news yet” means no news headline "
        "close to the story was found — it may be a scoop or an original explainer, or it may not be true, "
        "so treat it as unconfirmed.</footer></main></body></html>"
    )


def write_digest(digest: Digest, out_dir: str = DIGEST_DIR) -> tuple[str, str]:
    """digests/<account>/<date>.html and .json; latest.html is always the newest page (one to bookmark)."""
    folder = os.path.join(out_dir, digest.account)
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, digest.generated_at.astimezone().strftime("%Y-%m-%d"))
    page = render_html(digest)
    for name in (stem + ".html", os.path.join(folder, "latest.html")):
        with open(name, "w") as fh:
            fh.write(page)
    with open(stem + ".json", "w") as fh:
        json.dump(to_json(digest), fh, ensure_ascii=False, indent=1)
    return stem + ".html", stem + ".json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the daily digest for one account")
    parser.add_argument("--account", help="Account whose feed to digest")
    parser.add_argument("--hours", type=float, default=DIGEST_HOURS, help="How far back to look")
    parser.add_argument("--no-verify", action="store_true", help="Skip the news check")
    args = parser.parse_args()

    account, problem = choose_account(list_accounts(), args.account)
    if problem:
        print(f"⚠️  {problem}")
        return
    conn = get_connection()
    try:
        digest = build_digest(conn, account, args.hours, verify=not args.no_verify)
    finally:
        conn.close()
    page, data = write_digest(digest)
    print(f"📰  @{account}: {len(digest.stories)} stories from {digest.informative_reels} informative reels "
          f"({digest.reels_seen} scrolled)")
    for story, check in zip(digest.stories, digest.checks):
        print(f"  {story.score:.2f}  {story_title(story, check)[:40]:40}  {len(story.reels)} reel(s)  · {check.label}")
    print(f"\n  Page: {page}\n  Data: {data}")


if __name__ == "__main__":
    main()
