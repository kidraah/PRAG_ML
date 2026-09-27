"""
PRAG — Perception Integration Test

Calls evaluate_alignment() with mock transcript / OCR data to verify
that the LM Studio endpoint is reachable and returns a valid score.

Usage:
    python -m tests.test_perception
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.evaluator import evaluate_alignment


def test_evaluate_alignment() -> None:
    """Send mock data to LM Studio and print the alignment result."""

    mock_transcript = (
        "Today we're going to look at the latest breakthroughs in "
        "artificial intelligence and how transformer architectures are "
        "revolutionising natural language processing."
    )

    mock_ocr_text = "AI  Deep Learning  GPT  Neural Networks"

    target_bias = "Artificial Intelligence"

    print("=" * 60)
    print("  PRAG — Perception Integration Test")
    print("=" * 60)
    print()
    print(f"  Transcript : {mock_transcript[:60]}…")
    print(f"  OCR Text   : {mock_ocr_text}")
    print(f"  Target Bias: {target_bias}")
    print()
    print("  Calling LM Studio …")
    print()

    result = evaluate_alignment(mock_transcript, mock_ocr_text, target_bias)

    score = result["alignment_score"]
    reasoning = result["reasoning"]

    print(f"  Alignment Score : {score}")
    print(f"  Reasoning       : {reasoning}")
    print()

    if score > 0.0:
        print("  ✅  LM Studio is reachable and returned a valid score.")
    else:
        print("  ⚠️   Score is 0.0 — check that LM Studio is running on port 1234.")
        print(f"      Detail: {reasoning}")

    print()


if __name__ == "__main__":
    test_evaluate_alignment()
