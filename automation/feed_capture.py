"""
FeedLab — Feed Capture (network JSON → reel metadata)

The Reels web app receives every reel as JSON before showing it: caption,
creator, counts, a DASH manifest with a separate audio-only track, and the
cover image. FeedCatalog listens to those responses (plus the first batch,
which is embedded in the page HTML) and keeps a ReelMeta per shortcode in
feed order, so upcoming reels can be scored before they reach the screen.

Where Instagram puts reels (checked 2026-09 with `python -m tools.inspect_feed`):
  data.xdt_api__v1__clips__home__connection_v2.edges[].node.media   organic feed
  data.xdt_injected_reels_units.ad_media_items[].items[]             ads
"""

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Iterator

FEED_KEYS = ("xdt_api__v1__clips__home__connection_v2",)
AD_KEYS = ("xdt_injected_reels_units",)
GRAPHQL_PATHS = ("/graphql/query", "/api/graphql")

DASH_NS = "{urn:mpeg:dash:schema:mpd:2011}"
# `\w` would split Devanagari hashtags at vowel signs, so take everything up to whitespace.
HASHTAG_RE = re.compile(r"#([^\s#]+)")
HASHTAG_TRAILING = ".,!?:;…)]}\"'“”‘’।"
ISO_DURATION_RE = re.compile(
    r"^PT(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?$"
)

EMBEDDED_JSON_JS = """
() => Array.from(document.querySelectorAll('script[type="application/json"]'))
    .map(s => s.textContent || '')
    .filter(t => t.includes('xdt_api__v1__clips__home__connection_v2'))
"""


@dataclass
class ReelMeta:
    code: str
    creator: str = ""
    creator_full_name: str = ""
    creator_verified: bool = False
    creator_following: bool = False
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    like_count: int | None = None
    comment_count: int | None = None
    repost_count: int | None = None
    taken_at: int | None = None
    duration: float | None = None
    audio_type: str | None = None          # "original" | "music" | None
    audio_title: str = ""
    has_audio: bool = True
    audio_url: str | None = None           # audio-only DASH track
    cover_url: str | None = None
    has_liked: bool = False
    has_saved: bool = False
    is_ad: bool = False
    inventory_source: str | None = None
    feed_index: int | None = None          # position in the Reels feed; None for ads / other surfaces

    @property
    def url(self) -> str:
        return f"https://www.instagram.com/reels/{self.code}/"


# ── Parsing (pure) ────────────────────────────────────────────────────

def parse_body(text: str) -> list:
    """Parse a GraphQL response body into JSON documents.

    Handles the `for (;;);` anti-hijacking prefix and newline-delimited
    streams (one JSON document per line). Unparseable input yields [].
    """
    body = text.strip()
    if body.startswith("for (;;);"):
        body = body[len("for (;;);"):]
    if not body:
        return []
    try:
        return [json.loads(body)]
    except ValueError:
        docs = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except ValueError:
                continue
        return docs


def _is_media(obj: dict) -> bool:
    return isinstance(obj.get("code"), str) and (
        "video_dash_manifest" in obj or "video_versions" in obj
    )


def find_media(doc, source: str = "other") -> Iterator[tuple[dict, str]]:
    """Yield (media, source) for every video media object in a JSON document.

    source is "feed" under the Reels feed connection, "ad" under injected
    ad units, and "other" anywhere else (e.g. the home timeline, which the
    web app also loads on the Reels page).
    """
    if isinstance(doc, dict):
        if _is_media(doc):
            yield doc, source
            return
        for key, value in doc.items():
            child = "feed" if key in FEED_KEYS else "ad" if key in AD_KEYS else source
            yield from find_media(value, child)
    elif isinstance(doc, list):
        for item in doc:
            yield from find_media(item, source)


def parse_iso_duration(value: str) -> float | None:
    """'PT47.949207S' → 47.949207, 'PT1M2.5S' → 62.5; None if unparseable."""
    match = ISO_DURATION_RE.match(value or "")
    if not match or not any(match.groupdict().values()):
        return None
    hours, minutes, seconds = (float(match.group(g) or 0) for g in ("h", "m", "s"))
    return hours * 3600 + minutes * 60 + seconds


def parse_dash_manifest(manifest: str) -> tuple[str | None, float | None]:
    """Return (audio_track_url, duration_seconds) from a DASH manifest."""
    if not manifest:
        return None, None
    try:
        root = ET.fromstring(manifest)
    except ET.ParseError:
        return None, None

    duration = parse_iso_duration(root.get("mediaPresentationDuration", ""))
    for aset in root.iter(f"{DASH_NS}AdaptationSet"):
        reps = aset.findall(f"{DASH_NS}Representation")
        is_audio = (
            aset.get("contentType") == "audio"
            or (aset.get("mimeType") or "").startswith("audio/")
            or any((r.get("mimeType") or "").startswith("audio/") for r in reps)
        )
        if not is_audio:
            continue
        # Lowest bitrate is plenty for speech recognition and downloads fastest.
        for rep in sorted(reps, key=lambda r: int(r.get("bandwidth") or 0)):
            base = rep.find(f"{DASH_NS}BaseURL")
            if base is not None and (base.text or "").strip():
                return base.text.strip(), duration
    return None, duration


def pick_cover_url(image_versions: dict, target_width: int = 480) -> str | None:
    """Choose a portrait cover close to target_width (large enough for OCR, small to fetch)."""
    candidates = [c for c in image_versions.get("candidates") or [] if c.get("url")]
    if not candidates:
        return None
    portrait = [c for c in candidates if (c.get("height") or 0) > (c.get("width") or 0)]
    pool = portrait or candidates
    return min(pool, key=lambda c: abs((c.get("width") or 0) - target_width))["url"]


def extract_hashtags(caption: str) -> list[str]:
    """Lower-cased hashtags in order of appearance, without duplicates."""
    tags: list[str] = []
    for raw in HASHTAG_RE.findall(caption or ""):
        tag = raw.rstrip(HASHTAG_TRAILING).lower()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _int_or_none(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_media(media: dict, source: str) -> ReelMeta:
    """Normalise one media object into a ReelMeta."""
    user = media.get("user") or {}
    caption = ((media.get("caption") or {}).get("text") or "").strip()

    clips = media.get("clips_metadata") or {}
    original = clips.get("original_sound_info")
    music = clips.get("music_info")
    if original:
        audio_type, audio_title = "original", original.get("original_audio_title") or ""
    elif music:
        audio_type = "music"
        audio_title = (music.get("music_asset_info") or {}).get("title") or ""
    else:
        audio_type, audio_title = None, ""

    audio_url, duration = parse_dash_manifest(media.get("video_dash_manifest") or "")
    if duration is None and media.get("video_duration"):
        duration = float(media["video_duration"])

    counts_hidden = bool(media.get("like_and_view_counts_disabled"))
    return ReelMeta(
        code=media["code"],
        creator=user.get("username") or "",
        creator_full_name=user.get("full_name") or "",
        creator_verified=bool(user.get("is_verified")),
        creator_following=bool((user.get("friendship_status") or {}).get("following")),
        caption=caption,
        hashtags=extract_hashtags(caption),
        like_count=None if counts_hidden else _int_or_none(media.get("like_count")),
        comment_count=_int_or_none(media.get("comment_count")),
        repost_count=_int_or_none(media.get("media_repost_count")),
        taken_at=_int_or_none(media.get("taken_at")),
        duration=duration,
        audio_type=audio_type,
        audio_title=audio_title,
        has_audio=media.get("has_audio") is not False,
        audio_url=audio_url,
        cover_url=pick_cover_url(media.get("image_versions2") or {}),
        has_liked=bool(media.get("has_liked")),
        has_saved=bool(media.get("has_viewer_saved")),
        is_ad=source == "ad" or media.get("product_type") == "ad",
        inventory_source=media.get("inventory_source"),
    )


# ── Catalog ───────────────────────────────────────────────────────────

class FeedCatalog:
    """Every reel the web app has received this session, keyed by shortcode."""

    def __init__(self) -> None:
        self._metas: dict[str, ReelMeta] = {}
        self._feed_count = 0
        self.on_new_feed_reel: Callable[[ReelMeta], None] | None = None
        self.parse_errors = 0

    def __len__(self) -> int:
        return len(self._metas)

    def get(self, code: str | None) -> ReelMeta | None:
        return self._metas.get(code) if code else None

    async def wait_for(self, code: str | None, timeout: float) -> ReelMeta | None:
        """Return the reel's metadata, waiting briefly if its JSON is still in flight."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            meta = self.get(code)
            if meta is not None or not code or loop.time() >= deadline:
                return meta
            await asyncio.sleep(0.1)

    def ingest_doc(self, doc) -> int:
        """Add every reel found in a JSON document; returns how many were new."""
        added = 0
        for media, source in find_media(doc):
            try:
                meta = parse_media(media, source)
            except Exception:  # noqa: BLE001 — one odd object must not drop the batch
                self.parse_errors += 1
                continue
            if self._add(meta, source):
                added += 1
        return added

    def _add(self, meta: ReelMeta, source: str) -> bool:
        existing = self._metas.get(meta.code)
        if existing is not None:
            existing.has_liked = existing.has_liked or meta.has_liked
            existing.has_saved = existing.has_saved or meta.has_saved
            existing.creator_following = existing.creator_following or meta.creator_following
            if existing.feed_index is None and source == "feed":
                self._assign_feed_index(existing)
            return False
        self._metas[meta.code] = meta
        if source == "feed":
            self._assign_feed_index(meta)
        return True

    def _assign_feed_index(self, meta: ReelMeta) -> None:
        meta.feed_index = self._feed_count
        self._feed_count += 1
        if self.on_new_feed_reel is not None:
            self.on_new_feed_reel(meta)

    # ── Browser wiring ──

    def attach(self, page) -> None:
        """Start capturing reel JSON from the page's GraphQL responses."""
        page.on("response", self._on_response)

    async def _on_response(self, response) -> None:
        try:
            if response.request.resource_type not in ("xhr", "fetch"):
                return
            if not any(path in response.url for path in GRAPHQL_PATHS) or response.status != 200:
                return
            text = await response.text()
        except Exception:  # noqa: BLE001 — aborted requests, navigations, closed page
            return
        if '"video_dash_manifest"' not in text and '"video_versions"' not in text:
            return
        for doc in parse_body(text):
            self.ingest_doc(doc)

    async def ingest_embedded(self, page) -> int:
        """Ingest the first batch of reels, which ships inside the page HTML."""
        try:
            scripts = await page.evaluate(EMBEDDED_JSON_JS)
        except Exception:  # noqa: BLE001
            return 0
        return sum(self.ingest_doc(doc) for text in scripts for doc in parse_body(text))
