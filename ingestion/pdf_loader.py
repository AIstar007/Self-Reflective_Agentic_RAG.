import logging
from pathlib import Path
from typing import List

from pypdf import PdfReader

from ingestion.types import PageUnit

logger = logging.getLogger(__name__)


def load_pdf(file_path: Path) -> List[PageUnit]:
    logger.info("Loading PDF: %s", file_path)
    reader = PdfReader(str(file_path))
    units: List[PageUnit] = []

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        units.append(
            PageUnit(
                id=f"{file_path.name}:page:{i}",
                content=text,
                unit_type="page",
                index=i,
                source=file_path.name,
            )
        )

    if not units:
        raise ValueError(f"No text extracted from PDF: {file_path}")

    return units
