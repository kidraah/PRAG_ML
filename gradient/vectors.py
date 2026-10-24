"""
PRAG — the Gradient's embeddings: one multilingual sentence vector per item
(config.EMBEDDING_MODEL, the same model the feed model and stories use),
computed once and cached by item id. Hindi, Hinglish and English land in one
space, so a Hindi reel and an English one about the same thing sit close.
"""

import os

import numpy as np

from config import EMBEDDING_MODEL
from gradient import store

VECTORS_PATH = os.path.join(store.DATA_DIR, "vectors.npz")
_cache: dict[str, np.ndarray] | None = None
_cache_path: str | None = None


def _load(path: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["model"]) != EMBEDDING_MODEL:
                return {}
            return dict(zip(data["ids"].tolist(), data["vectors"]))
    except (OSError, KeyError, ValueError):
        return {}


def embed_texts(texts: list[str]) -> np.ndarray:
    from stories.grouping import embed

    out = np.asarray(embed([t[:1000] or "(empty)" for t in texts]), dtype=np.float32)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def for_items(items: list[dict], path: str | None = None, embed_fn=None, progress=None) -> np.ndarray:
    """Unit-length vectors for these items (rows in order), embedding only the ones not cached."""
    global _cache, _cache_path
    path = path or VECTORS_PATH
    embed_fn = embed_fn or embed_texts
    if _cache is None or path != _cache_path:
        _cache, _cache_path = _load(path), path
    missing = [i for i in items if i["id"] not in _cache]
    if missing:
        if progress:
            progress(f"Embedding {len(missing)} items …")
        for start in range(0, len(missing), 2048):
            chunk = missing[start:start + 2048]
            for item, vector in zip(chunk, embed_fn([i["text"] for i in chunk])):
                _cache[item["id"]] = np.asarray(vector, dtype=np.float32)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ids = list(_cache)
        np.savez(path + ".tmp.npz", model=np.array(EMBEDDING_MODEL), ids=np.array(ids),
                 vectors=np.stack([_cache[i] for i in ids]))
        os.replace(path + ".tmp.npz", path)
    if not items:
        return np.zeros((0, 384), dtype=np.float32)
    return np.stack([_cache[i["id"]] for i in items])
