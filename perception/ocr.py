"""
FeedLab — OCR Text Extractor

Uses EasyOCR to pull English text from a single image / video frame.
"""

import easyocr

# Initialise the reader once (model download happens on first run)
_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    """Lazily initialise and cache the EasyOCR reader."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text_from_frame(image_path: str) -> str:
    """Run OCR on an image and return all detected text as a single string.

    Parameters
    ----------
    image_path : str
        Path to an image file (e.g. .png, .jpg, .bmp).

    Returns
    -------
    str
        Space-joined English text detected in the image.
    """
    reader = _get_reader()
    results = reader.readtext(image_path, detail=0)
    return " ".join(results).strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m perception.ocr <image_file>")
        sys.exit(1)

    text = extract_text_from_frame(sys.argv[1])
    print(text)
