"""
FeedLab — Audio Transcriber

Uses mlx-whisper (Apple-Silicon-optimised Whisper) to convert an audio
file into a transcript string.
"""

import mlx_whisper


def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file and return the transcript text.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to an audio file (e.g. .wav, .mp3, .m4a).

    Returns
    -------
    str
        The full transcript produced by Whisper.
    """
    result = mlx_whisper.transcribe(file_path)
    return result.get("text", "").strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m perception.transcriber <audio_file>")
        sys.exit(1)

    transcript = transcribe_audio(sys.argv[1])
    print(transcript)
