import logging
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image

from ingestion.types import PageUnit

logger = logging.getLogger(__name__)


def load_image(file_path: Path) -> List[PageUnit]:
    logger.info("Loading image with OCR: %s", file_path)
    try:
        image = Image.open(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to open image {file_path}: {exc}") from exc

    text = (pytesseract.image_to_string(image) or "").strip()
    if not text:
        raise ValueError(
            f"No OCR text extracted from image: {file_path}. "
            "Ensure Tesseract OCR is installed and in PATH."
        )

    return [
        PageUnit(
            id=f"{file_path.name}:image:1",
            content=text,
            unit_type="page",
            index=1,
            source=file_path.name,
        )
    ]
