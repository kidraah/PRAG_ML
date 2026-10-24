"""
PRAG — cover images for Jev (config.JEV_VISION).

Jev 1.13 and 1.14 accept images through OpenRouter's Decisions API, as
inline base64 data URLs in the request's state (perception/jev.py). The
documented limits: PNG or JPEG only (no WebP), each side 256-1568 px, 78,400
to 2,458,624 pixels in all, EXIF orientation not applied (so rotate first),
remote URLs never fetched. prepared() turns a downloaded cover into a JPEG
inside those limits: rotated upright, scaled so its longer side is at most
JEV_IMAGE_MAX_SIDE, its shorter side upscaled to JEV_IMAGE_MIN_SIDE, or
rejected when it's too small to be worth sending.

Sessions keep each reel's cover bytes (save_cover, a bounded cache), so the
head-to-head and re-scoring can show Jev the image without downloading it
again. A cover is not the video: Jev sees one frame, the one Instagram shows
in the grid.
"""

import base64
import hashlib
import io
import os

from config import JEV_IMAGE_CACHE_MB, JEV_IMAGE_JPEG_QUALITY, JEV_IMAGE_MAX_SIDE, JEV_IMAGE_MIN_SIDE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "datasets", "jev-images")
API_MIN_SIDE, API_MAX_SIDE = 256, 1568
API_MIN_PIXELS, API_MAX_PIXELS = 78_400, 2_458_624
TINY = 120  # a cover smaller than this a side isn't worth upscaling
PREPROCESSING = f"jpeg-q{JEV_IMAGE_JPEG_QUALITY}-max{JEV_IMAGE_MAX_SIDE}-min{JEV_IMAGE_MIN_SIDE}-exif"


class ImageRejected(ValueError):
    pass


def prepared(data: bytes) -> bytes:
    """A JPEG of the cover that fits Jev's limits, or ImageRejected."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        if getattr(image, "is_animated", False):
            image.seek(0)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageRejected(f"not an image ({exc.__class__.__name__})") from exc
    width, height = image.size
    if min(width, height) < TINY:
        raise ImageRejected(f"too small ({width}×{height})")
    scale = min(1.0, JEV_IMAGE_MAX_SIDE / max(width, height))
    if min(width, height) * scale < JEV_IMAGE_MIN_SIDE:
        scale = JEV_IMAGE_MIN_SIDE / min(width, height)
    size = (round(width * scale), round(height * scale))
    if max(size) > API_MAX_SIDE:  # a very long, thin cover: it can't satisfy both sides
        raise ImageRejected(f"aspect ratio too extreme ({width}×{height})")
    if size != (width, height):
        image = image.resize(size, Image.LANCZOS)
    w, h = image.size
    if not (API_MIN_SIDE <= min(w, h) and max(w, h) <= API_MAX_SIDE and API_MIN_PIXELS <= w * h <= API_MAX_PIXELS):
        raise ImageRejected(f"outside Jev's size limits after scaling ({w}×{h})")
    out = io.BytesIO()
    image.save(out, "JPEG", quality=JEV_IMAGE_JPEG_QUALITY, optimize=True)
    return out.getvalue()


def image_part(jpeg: bytes) -> dict:
    """The state entry OpenRouter's Decisions API reads as an image."""
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def _path(code: str) -> str:
    return os.path.join(CACHE_DIR, hashlib.sha1(code.encode()).hexdigest()[:20] + ".jpg")


def save_cover(code: str, data: bytes, cache_dir: str | None = None) -> None:
    """Keep a reel's cover bytes, dropping the oldest when the cache is over JEV_IMAGE_CACHE_MB."""
    folder = cache_dir or CACHE_DIR
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, os.path.basename(_path(code)))
    with open(path + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(path + ".tmp", path)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
    total = sum(os.path.getsize(f) for f in files)
    for old in sorted(files, key=os.path.getmtime):
        if total <= JEV_IMAGE_CACHE_MB * 1024 * 1024:
            break
        total -= os.path.getsize(old)
        os.remove(old)


def load_cover(code: str, cache_dir: str | None = None) -> bytes | None:
    path = os.path.join(cache_dir or CACHE_DIR, os.path.basename(_path(code)))
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None
