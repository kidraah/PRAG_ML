"""
PRAG — Media Downloads

Fetches a reel's audio-only DASH track and its cover image straight from
the CDN, using the signed URLs in the feed JSON (no cookies needed). Both
are small — roughly 70-400 KB of audio and 50 KB of image per reel.
"""

import aiohttp

MAX_BYTES = 25 * 1024 * 1024


def new_session(user_agent: str) -> aiohttp.ClientSession:
    """HTTP session that presents the same browser identity as the page."""
    return aiohttp.ClientSession(
        headers={"User-Agent": user_agent, "Referer": "https://www.instagram.com/"},
    )


async def download(session: aiohttp.ClientSession, url: str, dest: str, timeout: float = 20.0) -> str:
    """Stream url to dest; returns dest. Raises on HTTP errors or oversized files."""
    size = 0
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError(f"media larger than {MAX_BYTES} bytes")
                fh.write(chunk)
    return dest
