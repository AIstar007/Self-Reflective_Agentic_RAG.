import logging
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

from ingestion.types import PageUnit

logger = logging.getLogger(__name__)


def load_html(file_path: Path) -> List[PageUnit]:
    logger.info("Loading HTML: %s", file_path)
    html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    units: List[PageUnit] = []
    headings = soup.find_all(["h1", "h2", "h3"])

    if headings:
        for i, heading in enumerate(headings, start=1):
            section_parts: List[str] = [heading.get_text(" ", strip=True)]
            for sibling in heading.find_next_siblings():
                if sibling.name in {"h1", "h2", "h3"}:
                    break
                text = sibling.get_text(" ", strip=True)
                if text:
                    section_parts.append(text)
            content = "\n".join(section_parts).strip()
            if content:
                units.append(
                    PageUnit(
                        id=f"{file_path.name}:section:{i}",
                        content=content,
                        unit_type="section",
                        index=i,
                        source=file_path.name,
                    )
                )
    else:
        body_text = soup.get_text(" ", strip=True)
        if body_text:
            units.append(
                PageUnit(
                    id=f"{file_path.name}:section:1",
                    content=body_text,
                    unit_type="section",
                    index=1,
                    source=file_path.name,
                )
            )

    if not units:
        raise ValueError(f"No text extracted from HTML: {file_path}")

    return units
