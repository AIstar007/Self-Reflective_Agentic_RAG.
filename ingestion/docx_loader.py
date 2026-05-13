import logging
from pathlib import Path
from typing import List

from docx import Document

from ingestion.types import PageUnit

logger = logging.getLogger(__name__)


def load_docx(file_path: Path) -> List[PageUnit]:
    logger.info("Loading DOCX: %s", file_path)
    document = Document(str(file_path))

    units: List[PageUnit] = []
    buffer: List[str] = []
    section_index = 1

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        style_name = paragraph.style.name.lower() if paragraph.style and paragraph.style.name else ""
        is_heading = style_name.startswith("heading")

        if is_heading and buffer:
            units.append(
                PageUnit(
                    id=f"{file_path.name}:section:{section_index}",
                    content="\n".join(buffer).strip(),
                    unit_type="section",
                    index=section_index,
                    source=file_path.name,
                )
            )
            buffer = []
            section_index += 1

        if text:
            buffer.append(text)

    if buffer:
        units.append(
            PageUnit(
                id=f"{file_path.name}:section:{section_index}",
                content="\n".join(buffer).strip(),
                unit_type="section",
                index=section_index,
                source=file_path.name,
            )
        )

    if not units:
        raise ValueError(f"No text extracted from DOCX: {file_path}")

    return units
