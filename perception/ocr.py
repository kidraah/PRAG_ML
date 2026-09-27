"""
PRAG — Cover Text OCR (EasyOCR)

Reads the text burned into a reel's cover image. News reels usually put
the headline there, and for reels set to a music track it is often the
only real information besides the caption.

EasyOCR (torch) runs on the CPU; callers keep it on its own thread.
"""

import warnings

from config import OCR_LANGUAGES, OCR_MIN_CONFIDENCE

_reader = None


def _get_reader():
    """Lazily initialise and cache the EasyOCR reader (downloads models on first run)."""
    global _reader
    if _reader is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import easyocr

            _reader = easyocr.Reader(OCR_LANGUAGES, gpu=False, verbose=False)
    return _reader


def extract_text_from_frame(image_path: str) -> str:
    """Run OCR on an image and return the confidently detected text as one string.

    Parameters
    ----------
    image_path : str
        Path to an image file (e.g. .png, .jpg, .bmp).

    Returns
    -------
    str
        Space-joined text (Hindi and English) detected in the image.
    """
    results = _get_reader().readtext(image_path, detail=1)
    return " ".join(text for _box, text, conf in results if conf >= OCR_MIN_CONFIDENCE).strip()


def warm_up() -> None:
    """Load the OCR models before the first reel needs them."""
    import numpy as np

    _get_reader().readtext(np.zeros((64, 64, 3), dtype=np.uint8))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m perception.ocr <image_file>")
        sys.exit(1)

    text = extract_text_from_frame(sys.argv[1])
    print(text)
