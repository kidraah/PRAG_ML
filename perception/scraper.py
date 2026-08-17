"""
FeedLab — yt-dlp Media Extractor

Downloads Instagram Reel video + metadata using yt-dlp.
Extracts caption, like/comment counts alongside the media file.
"""

import os
import yt_dlp

# Suppress HuggingFace Hub token warning (used by mlx-whisper)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def download_reel_data(url: str, output_path: str = "temp_reel.mp4") -> dict:
    """Download a Reel video and extract its metadata.

    Parameters
    ----------
    url : str
        Full Instagram Reel URL (e.g. ``https://www.instagram.com/reels/xxx/``).
    output_path : str
        Where to save the downloaded video file.

    Returns
    -------
    dict
        ``{"file_path", "caption", "view_count", "like_count", "comment_count"}``
        Falls back to empty/zero values on any download error.
    """
    fallback = {
        "file_path": "",
        "caption": "",
        "view_count": 0,
        "like_count": 0,
        "comment_count": 0,
    }

    ydl_opts = {
        "format": "best[height<=480]/best",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "overwrites": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if info is None:
                print("  ⚠️  yt-dlp returned no info for this URL.")
                return fallback

            # Instagram doesn't expose view_count via yt-dlp — use 0 as fallback
            return {
                "file_path": output_path,
                "caption": info.get("description", "") or "",
                "view_count": int(info.get("view_count") or 0),
                "like_count": int(info.get("like_count") or 0),
                "comment_count": int(info.get("comment_count") or 0),
            }

    except yt_dlp.utils.DownloadError as exc:
        print(f"  ⚠️  yt-dlp download failed: {exc}")
        return fallback
    except Exception as exc:
        print(f"  ⚠️  yt-dlp unexpected error: {exc}")
        return fallback


def cleanup_temp_file(path: str = "temp_reel.mp4") -> None:
    """Safely remove the temporary video file."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
