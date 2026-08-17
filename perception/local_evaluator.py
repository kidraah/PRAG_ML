"""
FeedLab — Local Perception Engine (Multimodal Audio & Async LLM)

1. Audio Extraction: Uses local ffmpeg to extract audio from video in ~50ms.
2. Multilingual Transcription: Uses `mlx-whisper` (whisper-large-v3-turbo)
   natively accelerated on Apple Silicon (supporting Hindi, English, Hinglish).
3. Local Alignment Scoring: Asynchronously scores the speech transcript +
   caption against TARGET_BIAS using local LM Studio via aiohttp.
"""

import json
import os
import subprocess
import asyncio
import aiohttp

from config import LM_STUDIO_URL, LLM_MODEL_NAME, TARGET_BIAS

# Suppress HF hub warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


# ── Audio Extraction ──────────────────────────────────────────────────

def extract_audio(video_path: str, audio_path: str = "temp_audio.mp3") -> str | None:
    """Extract audio track from video file using local ffmpeg.

    Parameters
    ----------
    video_path : str
        Input video path (.webm or .mp4).
    audio_path : str
        Output mp3 audio path.

    Returns
    -------
    str | None
        Path to audio file on success, None on failure.
    """
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return None

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "4",
            "-loglevel", "error",
            audio_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            return audio_path
        return None
    except Exception as exc:
        print(f"  ⚠️  Audio extraction failed: {exc}")
        return None


# ── Speech Transcription (mlx-whisper on Apple Silicon) ───────────────

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio locally using mlx-whisper (supports Hindi, English, Hinglish).

    Parameters
    ----------
    audio_path : str
        Path to audio file (.mp3 or .wav).

    Returns
    -------
    str
        Full speech transcription.
    """
    if not audio_path or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return ""

    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=WHISPER_MODEL,
        )
        text = result.get("text", "").strip()
        return text
    except Exception as exc:
        print(f"  ⚠️  Local transcription error: {exc}")
        return ""


# ── Local LLM Alignment Scoring (Async) ───────────────────────────────

SYSTEM_PROMPT = (
    "You are a JSON-only evaluator. You receive a speech transcript and a "
    "caption from an Instagram Reel, along with a target bias. "
    "You must return ONLY a valid JSON object with exactly two keys:\n"
    '  • "alignment_score": a float between 0.0 (completely unrelated/opposed) '
    "and 1.0 (perfectly aligned with the target bias).\n"
    '  • "reasoning": a brief one-sentence explanation of the score.\n'
    "Do not include any markdown, backticks, or other text outside the JSON object."
)


async def evaluate_alignment_async(
    transcript: str,
    caption: str,
    target_bias: str = TARGET_BIAS,
) -> dict:
    """Asynchronously evaluate reel alignment using local LM Studio.

    Parameters
    ----------
    transcript : str
        Speech transcript extracted via mlx-whisper.
    caption : str
        Reel caption / on-screen text.
    target_bias : str
        The target ideological/topic bias to score against.

    Returns
    -------
    dict
        ``{"alignment_score": float, "reasoning": str}``
    """
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Target Bias:\n{target_bias}\n\n"
                    f"Caption:\n{caption}\n\n"
                    f"Speech Transcript:\n{transcript}\n\n"
                    "Evaluate how well this content aligns with the target bias."
                ),
            },
        ],
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "alignment_evaluation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "alignment_score": {
                            "type": "number",
                            "description": "Float between 0.0 and 1.0",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of the score",
                        },
                    },
                    "required": ["alignment_score", "reasoning"],
                },
            },
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                LM_STUDIO_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    return {
                        "alignment_score": 0.0,
                        "reasoning": f"LM Studio returned status {response.status}: {text[:100]}",
                    }

                data = await response.json()
                raw_content = data["choices"][0]["message"]["content"]
                result = json.loads(raw_content)

                return {
                    "alignment_score": float(result.get("alignment_score", 0.0)),
                    "reasoning": str(result.get("reasoning", "No reasoning provided.")),
                }

    except Exception as exc:
        return {
            "alignment_score": 0.0,
            "reasoning": f"Async evaluation error: {exc}",
        }


def cleanup_audio_file(path: str = "temp_audio.mp3") -> None:
    """Safely remove the temporary audio file."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
