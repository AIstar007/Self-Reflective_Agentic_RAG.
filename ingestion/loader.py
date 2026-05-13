import logging
from pathlib import Path
from typing import List

from ingestion.docx_loader import load_docx
from ingestion.excel_loader import load_xlsx
from ingestion.html_loader import load_html
from ingestion.image_loader import load_image
from ingestion.pdf_loader import load_pdf
from ingestion.ppt_loader import load_pptx
from ingestion.types import PageUnit

logger = logging.getLogger(__name__)


class DocumentLoader:
    def load(self, file_path: str) -> List[PageUnit]:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()
        logger.info("Routing document %s with extension %s", path.name, extension)

        if extension == ".pdf":
            return load_pdf(path)
        if extension == ".docx":
            return load_docx(path)
        if extension == ".pptx":
            return load_pptx(path)
        if extension in {".xlsx", ".xlsm"}:
            return load_xlsx(path)
        if extension in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
            return load_image(path)
        if extension in {".html", ".htm"}:
            return load_html(path)

        raise ValueError(f"Unsupported document type: {extension}")
