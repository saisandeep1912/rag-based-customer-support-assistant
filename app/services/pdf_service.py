from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class PDFService:
    def extract_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        reader = PdfReader(str(pdf_path))
        pages: list[tuple[int, str]] = []

        for index, page in enumerate(reader.pages, start=1):
            pages.append((index, page.extract_text() or ""))

        return pages
