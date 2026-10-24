"""
PRAG — the Gradient's reel writer: an LLM writes reels for a feed.

Each account's feed is built brief by brief. A brief is one topic and an
angle on it, drawn from the persona's profile the way a curated feed leans:
mostly what the persona's weights favour, some exploration, and the other
side of political stories too. For each brief the writer writes REELS_PER_BRIEF
different reels (a creator, caption, on-screen text, what is said), each
with the stance, format and language the brief asks for, plus how this
persona reacts: skip, watch, like, save or share, and a line in their own
words (the takes the persona language model learns from).

Briefs are numbered and seeded, so a run can stop and resume: a brief that
is done is never written again. The creators are invented handles; the
writer is told not to put invented quotes in real people's mouths.

The writer is config.GEN_MODEL on OpenRouter (GEN_BACKEND = "openrouter",
OPENROUTER_API_KEY in .env: the briefs and the persona descriptions leave
this Mac), or the LM Studio model (GEN_BACKEND = "local"). A run stops
before it spends more than GEN_MAX_SPEND_USD.

    python -m gradient.generate --account mlproject01 --reels 6000 [--parallel 16]
    python -m gradient.generate --account mlproject01 --takes    (takes for session reels and headlines)
"""

import argparse
import asyncio
import json
import os
import random
import time

import aiohttp
from dotenv import load_dotenv

from agent import bias as biases
from config import GEN_BACKEND, GEN_MAX_SPEND_USD, GEN_MODEL, LLM_MODEL_NAME, LM_STUDIO_URL
from gradient import store, taxonomy

load_dotenv(os.path.join(store.PROJECT_ROOT, ".env"))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REELS_PER_BRIEF = 8
GENERATOR = f"{GEN_MODEL if GEN_BACKEND == 'openrouter' else LLM_MODEL_NAME}|gen-v1"
EXPLORE = 0.18  # a brief's topic drawn at random instead of by the persona's weights
ACTIONS = ("skip", "watch", "like", "save", "share")

SCHEMA = {
    "name": "reels",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"reels": {"type": "array", "minItems": REELS_PER_BRIEF, "maxItems": REELS_PER_BRIEF, "items": {
            "type": "object",
            "properties": {
                "slot": {"type": "integer"},
                "creator": {"type": "string"},
                "caption": {"type": "string"},
                "on_screen": {"type": "string"},
                "transcript": {"type": "string"},
                "viewer_action": {"type": "string", "enum": list(ACTIONS)},
                "viewer_take": {"type": "string"},
            },
            "required": ["slot", "creator", "caption", "on_screen", "transcript", "viewer_action", "viewer_take"],
            "additionalProperties": False,
        }}},
        "required": ["reels"],
        "additionalProperties": False,
    },
}

SYSTEM = """You write realistic Indian Instagram Reels, as text: who posted it, its caption, the words burned \
on screen, and what is said in it. Each reel is a different creator with their own voice.

Rules:
- Creators are invented handles (like @desi_infra_diaries or @nidhi.talks.money), never real accounts or outlets.
- Real public figures and places can be named, but never invent a quote, an allegation or a statistic and attach \
it to a real person or organisation. Opinions belong to the creator.
- Keep it how reels really sound: short sentences, hooks in the first line, hashtags in captions, Hindi, \
English or Hinglish as asked. Hindi in Devanagari; Hinglish in Roman letters.
- transcript: 35-80 words of what is spoken (for a skit or meme, the dialogue; for music, the lyric hook).
- Then, for each reel, the viewer described below reacts in character: viewer_action is what they do \
(skip, watch, like, save or share) and viewer_take is one or two sentences in their own words, first person, \
in the language they would use. They scroll past what doesn't interest them, and can disagree with a reel \
they watch."""


def viewer_text(bias) -> str:
    profile = bias.profile or {}
    return f"The viewer:\n{bias.interests.strip()}\n\nIn short: {profile.get('summary', '')}"


def weighted(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[max(0.0, weights[k]) for k in keys])[0]


def brief(account: str, n: int, bias) -> dict:
    """Brief number n for this account: the same every time (seeded)."""
    rng = random.Random(f"{account}|{n}")
    profile = bias.profile or {}
    topic_weights = {t: w for t, w in (profile.get("topic_weights") or {}).items() if t in taxonomy.TOPICS}
    if not topic_weights or rng.random() < EXPLORE:
        topic = rng.choice(list(taxonomy.TOPICS))
    else:
        topic = weighted(rng, {t: w ** 2 for t, w in topic_weights.items()})
    category, political, angles = taxonomy.TOPICS[topic]
    stance_w = {s: (profile.get("stance") or {}).get(s, 0.5) + 0.2 for s in taxonomy.STANCES}
    lang_w = {lang: w for lang, w in (profile.get("languages") or {"en": 1, "hi": 1, "hinglish": 1}).items()
              if lang in taxonomy.LANGUAGES}
    slots = []
    for slot in range(1, REELS_PER_BRIEF + 1):
        slots.append({"slot": slot,
                      "stance": weighted(rng, stance_w) if political else "not_political",
                      "format": rng.choice(taxonomy.FORMATS),
                      "language": weighted(rng, lang_w)})
    return {"id": f"{account}-{n:05d}", "account": account, "topic": topic, "category": category,
            "angle": rng.choice(angles), "political": political, "slots": slots}


def user_message(b: dict) -> str:
    lines = [f"Topic: {b['topic'].replace('_', ' ')} — {b['angle']}.",
             f"Write {REELS_PER_BRIEF} different reels on it, one per slot:"]
    for s in b["slots"]:
        stance = {"supportive": "supportive of the central government / ruling party",
                  "critical": "critical of the central government / ruling party",
                  "neutral": "neutral reporting, no side taken",
                  "not_political": "not political"}[s["stance"]]
        lines.append(f"{s['slot']}. {s['format']}, {taxonomy.LANGUAGES[s['language']]}, {stance}")
    lines.append('Answer in JSON: {"reels": [...]}, one object per slot, each with exactly these keys: slot, creator, '
                 'caption, on_screen, transcript, viewer_action, viewer_take.')
    return "\n".join(lines)


def chat_request(messages: list[dict], schema: dict, backend: str = GEN_BACKEND) -> tuple[str, dict, dict]:
    """(url, headers, body) for one structured-output call to the writer."""
    body = {"messages": messages, "temperature": 0.9, "max_tokens": 4000,
            "response_format": {"type": "json_schema", "json_schema": schema}}
    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY isn't set in .env")
        return OPENROUTER_URL, {"Authorization": f"Bearer {key}"}, {
            **body, "model": GEN_MODEL, "reasoning": {"enabled": False}, "usage": {"include": True},
            "provider": {"require_parameters": True}}
    return LM_STUDIO_URL, {}, {**body, "model": LLM_MODEL_NAME, "reasoning_effort": "none"}


def request(b: dict, viewer: str, backend: str = GEN_BACKEND) -> tuple[str, dict, dict]:
    """(url, headers, body) for one brief."""
    return chat_request([{"role": "system", "content": SYSTEM + "\n\n" + viewer},
                         {"role": "user", "content": user_message(b)}], SCHEMA, backend)


ALIASES = {"creator": ("creator_handle", "handle", "username", "account"),
           "on_screen": ("burned_text", "on_screen_text", "onscreen_text", "text_on_screen", "overlay"),
           "viewer_action": ("action",), "viewer_take": ("take", "viewer_reaction", "reaction")}


def normalised(reel: dict) -> dict:
    """The schema's keys, whatever close name the writer used; text fields as stripped strings."""
    out = dict(reel)
    for key, names in ALIASES.items():
        if not out.get(key):
            out[key] = next((reel[n] for n in names if reel.get(n)), "")
    for key in ("creator", "caption", "on_screen", "transcript", "viewer_take"):
        value = out.get(key) or ""
        out[key] = (" ".join(value) if isinstance(value, list) else str(value)).strip()
    action = str(out.get("viewer_action") or "").lower().strip()
    out["viewer_action"] = action if action in ACTIONS else "watch"
    return out


async def _ask(session: aiohttp.ClientSession, url: str, headers: dict, payload: dict, timeout: float,
               spend: dict | None) -> dict:
    """The writer's JSON answer."""
    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"{GEN_BACKEND} HTTP {resp.status}: {(await resp.text())[:200]}")
        data = await resp.json(content_type=None)
    if spend is not None:
        spend["usd"] = spend.get("usd", 0.0) + float((data.get("usage") or {}).get("cost") or 0.0)
    message = data["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning_content") or ""
    return json.loads(content[content.index("{"):content.rindex("}") + 1])


async def write_brief(session: aiohttp.ClientSession, b: dict, viewer: str, timeout: float = 240,
                      spend: dict | None = None) -> list[dict]:
    url, headers, payload = request(b, viewer)
    reels = (await _ask(session, url, headers, payload, timeout, spend))["reels"]
    by_slot = {s["slot"]: s for s in b["slots"]}
    out = []
    for i, reel in enumerate(normalised(r) for r in reels if isinstance(r, dict)):
        slot = by_slot.get(reel.get("slot")) or b["slots"][min(i, len(b["slots"]) - 1)]
        if not (reel.get("caption") or "").strip() or len((reel.get("transcript") or "").split()) < 8:
            continue
        out.append({**reel, **{k: slot[k] for k in ("stance", "format", "language")}})
    return out


def save_brief(conn, b: dict, reels: list[dict], bias, batch: str) -> int:
    new = 0
    for reel in reels:
        creator = "@" + (reel["creator"] or "creator").strip().lstrip("@").split()[0][:40]
        item_id, is_new = store.add_item(conn, {
            "origin": "generated", "source_name": GENERATOR, "creator": creator,
            "caption": reel["caption"].strip(), "on_screen": reel["on_screen"].strip(),
            "transcript": reel["transcript"].strip(), "language": reel["language"], "topic": b["topic"],
            "category": b["category"], "stance": reel["stance"], "format": reel["format"],
            "group_id": b.get("group", b["id"]), "source_url": b.get("source_url"),
            "meta": {"brief": b["id"], "angle": b["angle"], **b.get("meta", {})},
        }, batch)
        new += is_new
        store.add_to_feed(conn, b["account"], item_id, batch, "generator")
        take = (reel.get("viewer_take") or "").strip()
        if take:
            conn.execute("INSERT OR REPLACE INTO takes VALUES (?, ?, ?, ?)",
                         (item_id, b["account"], reel["viewer_action"], take))
        action = reel["viewer_action"]
        store.add_label(conn, item_id, b["account"], bias.fingerprint, "generator", GENERATOR,
                        {"skip": 0.1, "watch": 0.6, "like": 0.8, "save": 0.9, "share": 0.9}[action],
                        int(action != "skip"))
    conn.execute("INSERT OR REPLACE INTO gen_briefs VALUES (?, ?, ?, ?, NULL)", (b["id"], b["account"], store.now(),
                                                                                 len(reels)))
    conn.commit()
    return new


async def run(account: str, reels: int, parallel: int = 16, progress=print, should_stop=lambda: False,
              max_spend: float = GEN_MAX_SPEND_USD) -> dict:
    bias = biases.load(account)
    conn = store.connect()
    briefs_needed = -(-reels // REELS_PER_BRIEF)
    done = {row[0] for row in conn.execute("SELECT id FROM gen_briefs WHERE account = ? AND done_at IS NOT NULL",
                                           (account,))}
    todo = [b for b in (brief(account, n, bias) for n in range(briefs_needed)) if b["id"] not in done]
    if not todo:
        progress(f"{account}: all {briefs_needed} briefs are written.")
        return {"new": 0, "failed": 0, "left": 0}
    batch = store.new_batch(conn, "generated", f"{account}: reels written for its feed",
                            batch_id=f"gen-{account}-{time.strftime('%Y%m%d-%H%M')}")
    conn.commit()
    viewer = viewer_text(bias)
    gate = asyncio.Semaphore(parallel)
    stats = {"new": 0, "failed": 0, "briefs": 0}
    spend: dict = {"usd": 0.0}
    stop = lambda: should_stop() or spend["usd"] >= max_spend
    started = time.monotonic()

    async with aiohttp.ClientSession() as session:
        async def one(b):
            if stop():
                return
            async with gate:
                if stop():
                    return
                for attempt in range(2):
                    try:
                        written = await write_brief(session, b, viewer, spend=spend)
                        break
                    except Exception as exc:  # noqa: BLE001 - a bad brief is retried once, then left for next run
                        if attempt:
                            stats["failed"] += 1
                            conn.execute("INSERT OR REPLACE INTO gen_briefs VALUES (?, ?, NULL, 0, ?)",
                                         (b["id"], account, str(exc)[:300]))
                            conn.commit()
                            return
                stats["new"] += save_brief(conn, b, written, bias, batch)
                stats["briefs"] += 1
                if stats["briefs"] % 5 == 0:
                    rate = stats["new"] / (time.monotonic() - started) * 60
                    left = (len(todo) - stats["briefs"]) * REELS_PER_BRIEF / max(rate, 0.1)
                    progress(f"{account}: {len(done) + stats['briefs']}/{briefs_needed} briefs, +{stats['new']} reels "
                             f"({rate:.0f}/min, ~{left:.0f} min left, ${spend['usd']:.3f})")

        await asyncio.gather(*(one(b) for b in todo))
    if spend["usd"] >= max_spend:
        progress(f"{account}: stopped at the ${max_spend:.2f} spending cap (config.GEN_MAX_SPEND_USD)")
    store.close_batch(conn, batch, {"generated": {"new": stats["new"], "failed_briefs": stats["failed"],
                                                  "usd": round(spend["usd"], 4), "writer": GENERATOR}})
    total = conn.execute("SELECT COUNT(*) FROM feed WHERE account = ? AND via = 'generator'", (account,)).fetchone()[0]
    conn.close()
    progress(f"{account}: +{stats['new']} reels this run, {total} written for its feed in all"
             + (f"; {stats['failed']} briefs failed (tried again next run)" if stats["failed"] else ""))
    return {**stats, "usd": spend["usd"], "left": len(todo) - stats["briefs"]}


# ── Reels about the latest real news (news.py picks the stories) ─────

NEWS_RULES = ("Base every reel on this real story. Don't add facts, numbers, names or quotes that aren't in the "
              "headline or summary: a creator's opinion, reaction, joke or question is fine, and they can say what "
              "isn't known yet. If the story has nothing to do with politics or government, ignore the political "
              "lean and set political to false.")


def news_schema(n: int) -> dict:
    reel = SCHEMA["schema"]["properties"]["reels"]["items"]
    reel = {**reel, "properties": {**reel["properties"], "political": {"type": "boolean"}},
            "required": [*reel["required"], "political"]}
    return {"name": "news_reels", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["topic", "reels"],
        "properties": {"topic": {"type": "string", "enum": list(taxonomy.TOPICS)},
                       "reels": {"type": "array", "minItems": n, "maxItems": n, "items": reel}}}}


def news_brief(account: str, story: dict, bias, n: int) -> dict:
    rng = random.Random(f"{account}|{story['url']}")
    profile = bias.profile or {}
    stance_w = {st: (profile.get("stance") or {}).get(st, 0.5) + 0.2 for st in taxonomy.STANCES}
    lang_w = {lang: w for lang, w in (profile.get("languages") or {"en": 1}).items() if lang in taxonomy.LANGUAGES}
    slots = [{"slot": i, "stance": weighted(rng, stance_w), "format": rng.choice(taxonomy.FORMATS),
              "language": weighted(rng, lang_w)} for i in range(1, n + 1)]
    return {"id": f"{account}-{story['group']}", "account": account, "group": story["group"],
            "angle": story["title"][:200], "source_url": story["url"], "story": story, "slots": slots,
            "meta": {"news_title": story["title"], "news_source": story["source"],
                     "news_published_at": story["published"].isoformat() if story.get("published") else None}}


def news_message(b: dict) -> str:
    story = b["story"]
    when = f", {story['published']:%d %B %Y}" if story.get("published") else ""
    lines = [f"Real news story ({story['source']}{when}): {story['title']}",
             f"Summary: {story['summary'] or '(only the headline)'}", "", NEWS_RULES, "",
             f"Write {len(b['slots'])} different reels reacting to it, one per slot (the lean applies only if it's "
             "political):"]
    for slot in b["slots"]:
        lean = {"supportive": "supportive of the central government / ruling party",
                "critical": "critical of the central government / ruling party",
                "neutral": "neutral, no side taken"}[slot["stance"]]
        lines.append(f"{slot['slot']}. {slot['format']}, {taxonomy.LANGUAGES[slot['language']]}, lean: {lean}")
    lines.append(f"Also give the story's topic, one of: {', '.join(taxonomy.TOPICS)}.")
    lines.append('Answer in JSON: {"topic": "...", "reels": [...]}, one reel per slot, each with exactly these keys: '
                 "slot, creator, caption, on_screen, transcript, political, viewer_action, viewer_take.")
    return "\n".join(lines)


async def write_news(session, b: dict, viewer: str, spend: dict) -> list[dict]:
    url, headers, payload = chat_request([{"role": "system", "content": SYSTEM + "\n\n" + viewer},
                                          {"role": "user", "content": news_message(b)}], news_schema(len(b["slots"])))
    answer = await _ask(session, url, headers, payload, 240, spend)
    topic = answer.get("topic") if answer.get("topic") in taxonomy.TOPICS else "party_politics"
    b["topic"], b["category"] = topic, taxonomy.category(topic)
    by_slot = {sl["slot"]: sl for sl in b["slots"]}
    out = []
    for i, reel in enumerate(normalised(r) for r in answer.get("reels", []) if isinstance(r, dict)):
        slot = by_slot.get(reel.get("slot")) or b["slots"][min(i, len(b["slots"]) - 1)]
        if not reel["caption"] or len(reel["transcript"].split()) < 8:
            continue
        political = reel.get("political") is True
        out.append({**reel, "stance": slot["stance"] if political else "not_political",
                    "format": slot["format"], "language": slot["language"]})
    return out


async def run_news(account: str, stories: list[dict], per_story: int = 6, parallel: int = 12, progress=print,
                   max_spend: float = GEN_MAX_SPEND_USD) -> dict:
    """Reels about these real stories, in this persona's feed (a story already written about is skipped)."""
    bias = biases.load(account)
    conn = store.connect()
    done = {r[0] for r in conn.execute("SELECT id FROM gen_briefs WHERE account = ? AND done_at IS NOT NULL",
                                       (account,))}
    todo = [b for b in (news_brief(account, s, bias, per_story) for s in stories) if b["id"] not in done]
    stats = {"new": 0, "failed": 0, "stories": 0}
    if not todo:
        conn.close()
        return stats
    batch = store.new_batch(conn, "news-reels", f"{account}: reels about the latest news",
                            batch_id=f"newsreels-{account}-{time.strftime('%Y%m%d-%H%M')}")
    conn.commit()
    viewer, gate, spend = viewer_text(bias), asyncio.Semaphore(parallel), {"usd": 0.0}
    async with aiohttp.ClientSession() as session:
        async def one(b):
            async with gate:
                if spend["usd"] >= max_spend:
                    return
                for attempt in range(2):
                    try:
                        written = await write_news(session, b, viewer, spend)
                        break
                    except Exception as exc:  # noqa: BLE001 - retried once, then left for the next run
                        if attempt:
                            stats["failed"] += 1
                            conn.execute("INSERT OR REPLACE INTO gen_briefs VALUES (?, ?, NULL, 0, ?)",
                                         (b["id"], account, str(exc)[:300]))
                            conn.commit()
                            return
                stats["new"] += save_brief(conn, b, written, bias, batch)
                stats["stories"] += 1
                if stats["stories"] % 25 == 0:
                    progress(f"{account}: {stats['stories']}/{len(todo)} stories, +{stats['new']} reels (${spend['usd']:.3f})")

        await asyncio.gather(*(one(b) for b in todo))
    store.close_batch(conn, batch, {"generated": {"new": stats["new"], "failed_briefs": stats["failed"],
                                                  "usd": round(spend["usd"], 4), "writer": GENERATOR}})
    conn.close()
    progress(f"{account}: +{stats['new']} reels about {stats['stories']} real stories (${spend['usd']:.3f})"
             + (f"; {stats['failed']} stories failed" if stats["failed"] else ""))
    return {**stats, "usd": spend["usd"]}


# ── Takes for reels the writer didn't write ───────────────────────────
# Reels from the Instagram sessions and headlines the recommender served come without the persona's reaction;
# the persona language model learns from takes, so the writer adds them, TAKES_PER_CALL reels at a time.

TAKES_PER_CALL = 10
TAKES_SYSTEM = """You play the viewer described below, scrolling Instagram Reels. For each numbered reel, say what \
they do (viewer_action: skip, watch, like, save or share) and give viewer_take: one or two sentences in their own \
words, first person, in the language they'd use. React to what the reel actually says. They scroll past what doesn't \
interest them, and can disagree with a reel they watch."""
TAKES_SCHEMA = {"name": "takes", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["takes"],
    "properties": {"takes": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["n", "viewer_action", "viewer_take"],
        "properties": {"n": {"type": "integer"}, "viewer_action": {"type": "string", "enum": list(ACTIONS)},
                       "viewer_take": {"type": "string"}}}}}}}


def takes_request(items: list[dict], viewer: str) -> tuple[str, dict, dict]:
    reels = "\n\n".join(f"{n}. {it.get('creator') or it.get('source_name') or ''}: {it['text'][:700]}"
                         for n, it in enumerate(items, 1))
    return chat_request([{"role": "system", "content": TAKES_SYSTEM + "\n\n" + viewer},
                         {"role": "user", "content": f"{reels}\n\nAnswer in JSON: {{\"takes\": [...]}}, one object "
                                                     "per reel, each with exactly these keys: n, viewer_action, "
                                                     "viewer_take."}], TAKES_SCHEMA)


async def write_takes(account: str, parallel: int = 12, progress=print, max_spend: float = GEN_MAX_SPEND_USD) -> dict:
    """The persona's take on every reel in its feed that doesn't have one yet."""
    bias = biases.load(account)
    conn = store.connect()
    items = [dict(r) for r in conn.execute(
        "SELECT i.* FROM feed f JOIN items i ON i.id = f.item_id WHERE f.account = ? AND NOT EXISTS "
        "(SELECT 1 FROM takes t WHERE t.item_id = i.id AND t.account = f.account) ORDER BY i.id", (account,))]
    stats = {"done": 0, "failed": 0, "usd": 0.0}
    if not items:
        conn.close()
        return stats
    progress(f"{account}: writing the persona's take on {len(items)} reels")
    viewer, gate, spend = viewer_text(bias), asyncio.Semaphore(parallel), {"usd": 0.0}
    async with aiohttp.ClientSession() as session:
        async def one(chunk):
            async with gate:
                if spend["usd"] >= max_spend:
                    return
                try:
                    takes = (await _ask(session, *takes_request(chunk, viewer), 180, spend))["takes"]
                except Exception:  # noqa: BLE001 - left for the next run
                    stats["failed"] += len(chunk)
                    return
                for t in (normalised(x) for x in takes if isinstance(x, dict)):
                    n = t.get("n")
                    if isinstance(n, int) and 1 <= n <= len(chunk) and t["viewer_take"]:
                        conn.execute("INSERT OR REPLACE INTO takes VALUES (?, ?, ?, ?)",
                                     (chunk[n - 1]["id"], account, t["viewer_action"], t["viewer_take"]))
                        stats["done"] += 1
                conn.commit()

        await asyncio.gather(*(one(items[i:i + TAKES_PER_CALL]) for i in range(0, len(items), TAKES_PER_CALL)))
    conn.close()
    stats["usd"] = spend["usd"]
    progress(f"{account}: {stats['done']} takes written (${spend['usd']:.3f})"
             + (f", {stats['failed']} left for next time" if stats["failed"] else ""))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="An LLM writes reels for an account's feed")
    parser.add_argument("--account", required=True, choices=store.ACCOUNTS)
    parser.add_argument("--reels", type=int, default=6000)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--takes", action="store_true", help="only write takes for feed reels that have none")
    args = parser.parse_args()
    if args.takes:
        asyncio.run(write_takes(args.account, args.parallel, progress=lambda m: print(m, flush=True)))
    else:
        asyncio.run(run(args.account, args.reels, args.parallel, progress=lambda m: print(m, flush=True)))


if __name__ == "__main__":
    main()
