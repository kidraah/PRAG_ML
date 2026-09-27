"""
PRAG — Speech Transcription (mlx-whisper on Apple Silicon)

Transcribes a reel's full audio track. Whisper often labels spoken Hindi
as Urdu and prints Urdu script, so a detected language outside
WHISPER_LANGUAGES triggers one re-run in WHISPER_FALLBACK_LANGUAGE.
Temperature fallback is disabled to keep latency bounded on chants and
songs (it re-decodes repetitive audio several times); repeated phrases
are collapsed afterwards instead.

MLX work must stay on one thread — callers run these functions on a
dedicated single-thread executor (see agent/pipeline.py).
"""

import os
import re
from dataclasses import dataclass

from config import (
    MAX_TRANSCRIBE_SECONDS,
    WHISPER_FALLBACK_LANGUAGE,
    WHISPER_LANGUAGES,
    WHISPER_MODEL,
)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Whisper's stock outputs for silence or music.
HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching",
    "subscribe", "please subscribe", "you", "music",
}
ANNOTATION_RE = re.compile(r"[\[(]\s*(?:music|applause|laughter|silence|singing)\s*[\])]|♪+", re.IGNORECASE)
PUNCTUATION = ",.!?;:।…-\"'"


@dataclass
class Transcript:
    text: str
    language: str | None


def transcribe(path: str, duration: float | None = None) -> Transcript:
    """Transcribe an audio file (any format ffmpeg reads)."""
    import mlx_whisper

    options = {
        "path_or_hf_repo": WHISPER_MODEL,
        "verbose": None,
        "temperature": 0.0,
        "condition_on_previous_text": False,
    }
    if duration and duration > MAX_TRANSCRIBE_SECONDS:
        options["clip_timestamps"] = f"0,{MAX_TRANSCRIBE_SECONDS}"

    result = mlx_whisper.transcribe(path, **options)
    language = result.get("language")
    if language not in WHISPER_LANGUAGES and WHISPER_FALLBACK_LANGUAGE:
        result = mlx_whisper.transcribe(path, language=WHISPER_FALLBACK_LANGUAGE, **options)
        language = WHISPER_FALLBACK_LANGUAGE
    return Transcript(text=clean_segments(result.get("segments") or []), language=language)


def clean_segments(segments: list[dict]) -> str:
    """Join segments, dropping silence hallucinations and collapsing loops."""
    kept = []
    for seg in segments:
        silent = seg.get("no_speech_prob", 0.0) > 0.6 and seg.get("avg_logprob", 0.0) < -1.0
        if not silent:
            kept.append((seg.get("text") or "").strip())
    text = ANNOTATION_RE.sub(" ", " ".join(t for t in kept if t))
    text = collapse_repetitions(" ".join(text.split()))
    if text.lower().strip(PUNCTUATION + " ") in HALLUCINATIONS:
        return ""
    return text


def collapse_repetitions(text: str, max_repeats: int = 2, max_ngram: int = 8) -> str:
    """Cap consecutive repeats of any 1..max_ngram word phrase at max_repeats.

    'बोले, बोले, बोले, बोले' → 'बोले, बोले'. Words are compared without
    surrounding punctuation.
    """
    words = text.split()
    for n in range(1, max_ngram + 1):
        keys = [w.strip(PUNCTUATION).lower() for w in words]
        out: list[str] = []
        i = 0
        while i < len(words):
            gram = keys[i:i + n]
            if len(gram) < n:
                out.extend(words[i:])
                break
            repeats = 1
            while keys[i + repeats * n:i + (repeats + 1) * n] == gram:
                repeats += 1
            if repeats > max_repeats:
                out.extend(words[i:i + n * max_repeats])
                i += repeats * n
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def warm_up() -> None:
    """Load the Whisper model (a few seconds) before the first reel needs it."""
    import mlx_whisper
    import numpy as np

    mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=WHISPER_MODEL, verbose=None)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m perception.transcriber <audio_file>")
        sys.exit(1)

    result = transcribe(sys.argv[1])
    print(f"[{result.language}] {result.text}")
