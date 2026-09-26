"""
FeedLab — Feed Inspector (diagnostic)

Opens Instagram Reels with the agent's browser profile, scrolls a few reels
WITHOUT liking, saving or following, and dumps what the web app receives:
the JSON responses (where reel captions, creators, counts and media URLs
live), the media requests, and the active reel's action-button DOM.

Instagram's web data format is undocumented and changes; run this when the
capture layer stops finding reel metadata, then compare the dump against
`automation/feed_capture.py`.

Usage:
    python -m tools.inspect_feed [--reels 4] [--out debug/feed_dump]
"""

import argparse
import asyncio
import json
import os
import time

from automation.browser_driver import close_browser, init_browser

ACTIVE_REEL_DOM_JS = """
() => {
    const vids = Array.from(document.querySelectorAll('video'));
    const vh = window.innerHeight;
    const ranked = vids.map(v => {
        const r = v.getBoundingClientRect();
        const vis = Math.max(0, Math.min(vh, r.bottom) - Math.max(0, r.top));
        return { v, r, vis };
    }).sort((a, b) => b.vis - a.vis);
    const out = { url: location.href, videoCount: vids.length };
    if (!ranked.length) return out;

    const active = ranked[0].v;
    out.active = {
        rect: ranked[0].r, src: (active.currentSrc || '').slice(0, 60),
        muted: active.muted, paused: active.paused,
        duration: active.duration, currentTime: active.currentTime,
    };

    let node = active, depth = 0, container = null;
    while (node && depth < 30) {
        if (node.querySelector && node.querySelector('svg[aria-label="Like"], svg[aria-label="Unlike"]')) {
            container = node;
            break;
        }
        node = node.parentElement;
        depth++;
    }
    out.containerDepth = container ? depth : null;
    if (container) {
        out.ariaLabels = Array.from(container.querySelectorAll('[aria-label]')).map(e => ({
            tag: e.tagName, label: e.getAttribute('aria-label'), role: e.getAttribute('role'),
        }));
        out.clickables = Array.from(container.querySelectorAll('[role="button"], button, a')).map(b => ({
            tag: b.tagName, text: (b.innerText || '').trim().slice(0, 60),
            href: b.getAttribute('href'), aria: b.getAttribute('aria-label'),
        })).slice(0, 80);
    }
    out.likeSvgsInDocument = Array.from(
        document.querySelectorAll('svg[aria-label="Like"], svg[aria-label="Unlike"]')
    ).map(s => {
        const r = s.getBoundingClientRect();
        return { label: s.getAttribute('aria-label'), top: Math.round(r.top), left: Math.round(r.left) };
    });
    return out;
}
"""

EMBEDDED_JSON_JS = """
() => Array.from(document.querySelectorAll('script[type="application/json"]'))
    .map(s => s.textContent || '')
    .filter(t => t.includes('video_versions') || t.includes('clips') || t.includes('xdt_api'))
"""


async def inspect(num_reels: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    context, page = await init_browser()
    counter = {"json": 0}
    media_log: list[dict] = []

    async def on_response(response) -> None:
        try:
            req = response.request
            url = response.url
            rtype = req.resource_type
            ctype = response.headers.get("content-type", "")
            if rtype in ("xhr", "fetch", "document") and "instagram.com" in url:
                body = await response.text()
                counter["json"] += 1
                idx = counter["json"]
                meta = {
                    "idx": idx, "url": url, "status": response.status,
                    "resource_type": rtype, "content_type": ctype,
                    "method": req.method, "post_data": (req.post_data or "")[:2000],
                    "body_len": len(body),
                }
                with open(os.path.join(out_dir, f"resp_{idx:03d}.meta.json"), "w") as fh:
                    json.dump(meta, fh, indent=2)
                with open(os.path.join(out_dir, f"resp_{idx:03d}.body.txt"), "w") as fh:
                    fh.write(body)
            elif rtype == "media" or "video" in ctype or "audio" in ctype:
                media_log.append({
                    "t": round(time.time(), 2), "url": url[:300], "status": response.status,
                    "content_type": ctype, "range": req.headers.get("range", ""),
                    "content_length": response.headers.get("content-length", ""),
                })
        except Exception as exc:  # noqa: BLE001 — diagnostics must never crash the run
            media_log.append({"handler_error": str(exc), "url": response.url[:200]})

    page.on("response", on_response)

    try:
        await page.goto("https://www.instagram.com/reels/", wait_until="domcontentloaded")
        await asyncio.sleep(6)

        if "accounts/login" in page.url or await page.query_selector('input[name="username"]'):
            print("🔐  Not logged in — log into the agent account in the browser first.")
            return

        embedded = await page.evaluate(EMBEDDED_JSON_JS)
        with open(os.path.join(out_dir, "embedded_scripts.json"), "w") as fh:
            json.dump(embedded, fh)
        print(f"  Embedded JSON scripts mentioning reels: {len(embedded)}")

        dom_snapshots = []
        for i in range(num_reels):
            dom = await page.evaluate(ACTIVE_REEL_DOM_JS)
            dom_snapshots.append(dom)
            print(f"  Reel {i + 1}: {dom.get('url')}  (videos in DOM: {dom.get('videoCount')})")
            await page.keyboard.press("ArrowDown")
            await asyncio.sleep(4)

        with open(os.path.join(out_dir, "dom_snapshots.json"), "w") as fh:
            json.dump(dom_snapshots, fh, indent=2)
        with open(os.path.join(out_dir, "media_requests.json"), "w") as fh:
            json.dump(media_log, fh, indent=2)
        print(f"  Saved {counter['json']} responses, {len(media_log)} media requests → {out_dir}")
    finally:
        await close_browser()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reels", type=int, default=4, help="How many reels to step through")
    parser.add_argument("--out", default="debug/feed_dump", help="Output directory")
    args = parser.parse_args()
    asyncio.run(inspect(args.reels, args.out))


if __name__ == "__main__":
    main()
