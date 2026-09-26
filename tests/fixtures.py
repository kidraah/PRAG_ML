"""
Synthetic Instagram web payloads, shaped like the real ones recorded with
`python -m tools.inspect_feed` (2026-09) but with made-up content.
"""

import json


def make_manifest(duration: float = 21.2, audio_url: str | None = "https://cdn.example/audio.mp4") -> str:
    audio = (
        '<AdaptationSet id="1" contentType="audio">'
        '<Representation id="a2" bandwidth="96000" codecs="mp4a.40.2" mimeType="audio/mp4">'
        "<BaseURL>https://cdn.example/audio-hi.mp4</BaseURL></Representation>"
        '<Representation id="a1" bandwidth="64000" codecs="mp4a.40.5" mimeType="audio/mp4">'
        f"<BaseURL>{audio_url}</BaseURL></Representation></AdaptationSet>"
        if audio_url else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" '
        f'mediaPresentationDuration="PT{duration}S"><Period id="0">'
        '<AdaptationSet id="0" contentType="video"><Representation id="v" bandwidth="300000" '
        'mimeType="video/mp4"><BaseURL>https://cdn.example/video.mp4</BaseURL></Representation>'
        f"</AdaptationSet>{audio}</Period></MPD>"
    )


def make_media(
    code: str,
    *,
    username: str = "news_creator",
    caption: str = "",
    audio: str | None = "original",
    audio_title: str = "Original audio",
    duration: float = 21.2,
    product_type: str = "clips",
    following: bool = False,
    likes: int = 1234,
    comments: int = 56,
    has_liked: bool = False,
    counts_hidden: bool = False,
) -> dict:
    clips = {"is_shared_to_fb": False, "music_info": None, "original_sound_info": None}
    if audio == "original":
        clips["original_sound_info"] = {"original_audio_title": audio_title}
    elif audio == "music":
        clips["music_info"] = {"music_asset_info": {"title": audio_title}}
    return {
        "__typename": "XDTMediaDict",
        "code": code,
        "pk": "3938308612128508443",
        "product_type": product_type,
        "media_type": 2,
        "user": {
            "pk": "1", "username": username, "full_name": "Some Name", "is_verified": True,
            "friendship_status": {"following": following, "is_feed_favorite": False},
        },
        "caption": {"text": caption, "pk": "9"} if caption else None,
        "clips_metadata": clips,
        "video_dash_manifest": make_manifest(duration),
        "video_versions": [{"width": 720, "height": 1280, "url": "https://cdn.example/v.mp4", "type": 101}],
        "image_versions2": {"candidates": [
            {"width": 640, "height": 1136, "url": "https://cdn.example/cover-640.jpg"},
            {"width": 480, "height": 852, "url": "https://cdn.example/cover-480.jpg"},
            {"width": 480, "height": 480, "url": "https://cdn.example/square-480.jpg"},
        ]},
        "like_count": likes,
        "comment_count": comments,
        "media_repost_count": 7,
        "like_and_view_counts_disabled": counts_hidden,
        "taken_at": 1783703326,
        "has_audio": True,
        "has_liked": has_liked,
        "has_viewer_saved": None,
        "inventory_source": "recommended_clips_chaining_model",
    }


def feed_page(*medias: dict) -> dict:
    return {"data": {"xdt_api__v1__clips__home__connection_v2": {
        "edges": [{"node": {"media": m, "__typename": "XDTClipsItem"}, "cursor": "c"} for m in medias],
        "page_info": {"end_cursor": "abc", "has_next_page": True},
    }}}


def ads_page(*medias: dict) -> dict:
    return {"data": {"xdt_injected_reels_units": {"ad_media_items": [{"items": list(medias)}]}}}


def timeline_page(*medias: dict) -> dict:
    return {"data": {"xdt_api__v1__feed__timeline__connection": {
        "edges": [{"node": {"media": m}} for m in medias],
    }}}


def embedded_script(*medias: dict) -> str:
    """The first batch of reels, nested the way the page HTML ships it."""
    return json.dumps({"require": [["ScheduledServerJS", "handle", None, [{"__bbox": {"require": [
        ["RelayPrefetchedStreamCache", "next", [], ["adp_key", {"__bbox": {"result": feed_page(*medias)}}]]
    ]}}]]]})
