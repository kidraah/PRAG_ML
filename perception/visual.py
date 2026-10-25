"""
PRAG — what a reel's cover frame looks like, measured.

The pipeline already downloads each reel's cover once (for OCR, and for Jev
when config.JEV_VISION sends it the image). describe() measures that frame
locally, with Pillow and numpy, into a few descriptors the session log, the
impressions table and the panel's reel inspector show next to the score:

  size, aspect       the frame's pixels, and portrait (9:16) or not
  brightness         mean luma, 0 (black) to 1 (white)
  contrast           spread of the luma (its standard deviation, 0 to about 0.5)
  saturation         mean HSV saturation, 0 (grey) to 1
  colourfulness      Hasler & Süsstrunk's metric, scaled to 0-1 (1 ≈ 150, very colourful)
  edge_density       share of pixels on a strong edge: how busy the frame is (text overlays, clutter)
  palette            the three dominant colours and their shares

These are measurements of one frame, the cover Instagram shows in the grid,
not of the video, and they are descriptors, not a score: Jev's judgement is
still the value score. They're computed only for reels whose cover was
downloaded; a simulated session's reels have no image, so they have none.
"""

import io

import numpy as np

VERSION = "visual-v1"
EDGE = 0.18  # gradient magnitude (on 0-1 luma) that counts as a strong edge
WORK_SIDE = 256  # measured on a copy this size: the descriptors barely change, and it takes a few milliseconds


def describe(data: bytes) -> dict | None:
    """The cover's descriptors, or None when it isn't a readable image."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        if getattr(image, "is_animated", False):
            image.seek(0)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None
    width, height = image.size
    small = image.copy()
    small.thumbnail((WORK_SIDE, WORK_SIDE))
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0] * 255, rgb[..., 1] * 255, rgb[..., 2] * 255
    rg, yb = r - g, 0.5 * (r + g) - b
    colourfulness = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    gy, gx = np.gradient(luma)
    edges = float((np.hypot(gx, gy) > EDGE / 2).mean())  # np.gradient halves the step: compare with half the threshold
    return {
        "version": VERSION,
        "width": width, "height": height, "aspect": round(width / height, 3) if height else None,
        "portrait": height > width * 1.3,
        "brightness": round(float(luma.mean()), 3),
        "contrast": round(float(luma.std()), 3),
        "saturation": round(float(hsv[..., 1].mean()), 3),
        "colourfulness": round(min(colourfulness / 150.0, 1.0), 3),
        "edge_density": round(edges, 3),
        "palette": palette(small),
    }


def palette(image, colours: int = 3) -> list[dict]:
    """The dominant colours, most common first: [{"hex", "share"}]."""
    quantised = image.quantize(colors=colours, method=1)  # 1 = maximum coverage: deterministic, no k-means seed
    counts = sorted(quantised.getcolors() or [], reverse=True)
    lookup = quantised.getpalette() or []
    total = sum(n for n, _ in counts) or 1
    return [{"hex": "#{:02x}{:02x}{:02x}".format(*lookup[i * 3:i * 3 + 3]), "share": round(n / total, 3)}
            for n, i in counts[:colours]]


def short(visual: dict | None) -> str:
    """The descriptors in a few words, for the session log."""
    if not visual:
        return ""
    words = [("bright" if visual["brightness"] > 0.6 else "dark" if visual["brightness"] < 0.3 else "mid-tone"),
             ("vivid" if visual["colourfulness"] > 0.45 else "muted" if visual["colourfulness"] < 0.15 else "colour"),
             ("busy" if visual["edge_density"] > 0.12 else "clean")]
    return " ".join(words)
