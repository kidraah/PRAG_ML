"""
FeedLab — Two-Layer Alignment Evaluator

Layer 1 (Eyes & Ears — Gemini API):
    Uploads the captured video to Gemini Flash for neutral sensory extraction.
    Returns a raw transcript (Hindi/English) + visual description + text with zero bias.
    Includes auto-retry and model fallback on transient 503 errors.

Layer 2 (Judgment Brain — Local LM Studio):
    Receives the sensory summary + caption + TARGET_BIAS and returns
    a scored alignment JSON.
"""

import json
import os
import time

import requests
from google import genai

from config import LM_STUDIO_URL, LLM_MODEL_NAME, TARGET_BIAS


# ── Gemini Client ────────────────────────────────────────────────────

_gemini_client = None

# Fallback models in case of high-demand 503s
GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]


def _get_gemini_client() -> genai.Client:
    """Lazily initialise the Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Add it to your .env file."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ── Layer 1: Sensory Extraction (Gemini Flash) ───────────────────────

SENSORY_PROMPT = (
    "Transcribe the audio speech and lyrics exactly (including Hindi, Hinglish, English). "
    "Describe what is happening visually in detail. "
    "List all visible on-screen text and graphics. "
    "Be completely objective and factual. Do not evaluate, summarize bias, or score the video."
)


def extract_sensory_data(video_path: str) -> str:
    """Upload video to Gemini and extract a neutral sensory description.

    Parameters
    ----------
    video_path : str
        Path to the video file (.webm or .mp4).

    Returns
    -------
    str
        Raw sensory description (transcript + visual + on-screen text).
    """
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return "No video media available for analysis."

    client = _get_gemini_client()
    uploaded_file = None

    try:
        # Upload the video file to Gemini Files API
        uploaded_file = client.files.upload(file=video_path)
        size_kb = os.path.getsize(video_path) / 1024
        print(f"  ☁️  Uploaded to Gemini ({size_kb:.0f} KB)")

        # Wait for file to become ACTIVE
        poll_count = 0
        while getattr(uploaded_file, "state", None) and getattr(uploaded_file.state, "name", "") == "PROCESSING":
            time.sleep(1.0)
            uploaded_file = client.files.get(name=uploaded_file.name)
            poll_count += 1
            if poll_count > 15:
                break

        # Try models with automatic fallback on 503 / rate limits
        last_err = None
        for model in GEMINI_MODELS:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=[uploaded_file, SENSORY_PROMPT],
                    )
                    result = response.text.strip()
                    print(f"  👁  Sensory ({model}): {result[:120]}…")
                    return result
                except Exception as exc:
                    last_err = exc
                    if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                        time.sleep(1.5)
                        continue
                    break

        error_msg = f"Gemini extraction failed across models: {last_err}"
        print(f"  ⚠️  {error_msg}")
        return error_msg

    except Exception as exc:
        error_msg = f"Gemini file upload/processing error: {exc}"
        print(f"  ⚠️  {error_msg}")
        return error_msg

    finally:
        # Clean up cloud file
        if uploaded_file and hasattr(uploaded_file, "name"):
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


# ── Layer 2: Alignment Scoring (Local LM Studio) ─────────────────────

SYSTEM_PROMPT = (
    "You are a JSON-only evaluator. You receive a sensory description of "
    "a short video, its caption, and a target bias. "
    "You must return ONLY a valid JSON object with exactly two keys:\n"
    '  • "alignment_score": a float between 0.0 (completely unrelated) '
    "and 1.0 (perfectly aligned with the target bias).\n"
    '  • "reasoning": a brief one-sentence explanation of the score.\n'
    "Do not include any other text, markdown, or commentary."
)


def evaluate_alignment(
    sensory_summary: str,
    caption: str,
    target_bias: str = TARGET_BIAS,
) -> dict:
    """Score content alignment using local LM Studio.

    Parameters
    ----------
    sensory_summary : str
        Neutral sensory description from Gemini (Layer 1).
    caption : str
        Reel caption / description text.
    target_bias : str
        The topic / bias to score against.

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
                    f"Sensory Description:\n{sensory_summary}\n\n"
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
        response = requests.post(
            LM_STUDIO_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()

        raw = response.json()["choices"][0]["message"]["content"]
        result = json.loads(raw)

        return {
            "alignment_score": float(result.get("alignment_score", 0.0)),
            "reasoning": str(result.get("reasoning", "No reasoning provided.")),
        }

    except json.JSONDecodeError:
        try:
            print(f"  ⚠️  Raw LLM response (not JSON): {response.text[:300]}")
        except Exception:
            pass
        return {
            "alignment_score": 0.0,
            "reasoning": "Parse Error — failed to decode LLM response as JSON.",
        }
    except requests.RequestException as exc:
        return {
            "alignment_score": 0.0,
            "reasoning": f"LM Studio request failed: {exc}",
        }
    except (KeyError, IndexError, TypeError) as exc:
        return {
            "alignment_score": 0.0,
            "reasoning": f"Unexpected response structure: {exc}",
        }
