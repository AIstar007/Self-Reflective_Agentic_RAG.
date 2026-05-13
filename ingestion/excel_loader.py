import logging
from pathlib import Path
from typing import List

import pandas as pd

from ingestion.types import PageUnit

logger = logging.getLogger(__name__)


def load_xlsx(file_path: Path) -> List[PageUnit]:
    logger.info("Loading XLSX: %s", file_path)
    workbook = pd.read_excel(file_path, sheet_name=None)

    units: List[PageUnit] = []
    for i, (sheet_name, df) in enumerate(workbook.items(), start=1):
        if df.empty:
            continue

        # Keep a compact but meaningful table representation per sheet.
        table_text = df.fillna("").to_csv(index=False)
        content = f"Sheet: {sheet_name}\n{table_text}".strip()

        units.append(
            PageUnit(
                id=f"{file_path.name}:sheet:{i}",
                content=content,
                unit_type="sheet",
                index=i,
                source=file_path.name,
            )
        )

    if not units:
        raise ValueError(f"No tabular content extracted from XLSX: {file_path}")

    return units
