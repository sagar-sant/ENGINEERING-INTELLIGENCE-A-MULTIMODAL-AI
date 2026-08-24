from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pymupdf


@dataclass
class PageContent:
    """Represents the extracted content and metadata for one PDF page."""

    page_number: int
    text: str
    character_count: int
    needs_ocr: bool


@dataclass
class DocumentContent:
    """Represents all extracted content and metadata for a PDF document."""

    file_name: str
    file_path: str
    page_count: int
    metadata: dict[str, Any]
    pages: list[PageContent]

    def to_dict(self) -> dict[str, Any]:
        """Convert the document object into a JSON-friendly dictionary."""
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "page_count": self.page_count,
            "metadata": self.metadata,
            "pages": [asdict(page) for page in self.pages],
        }


def load_pdf(
    pdf_path: str | Path,
    min_text_characters: int = 50,
) -> DocumentContent:
    """
    Extract text and metadata from a PDF.

    Pages containing very little extracted text are marked as OCR candidates.
    OCR itself is not performed by this function.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {pdf_path.suffix}")

    document = pymupdf.open(pdf_path)

    try:
        metadata = document.metadata or {}

        pages: list[PageContent] = []

        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()

            pages.append(
                PageContent(
                    page_number=page_index + 1,
                    text=text,
                    character_count=len(text),
                    needs_ocr=len(text) < min_text_characters,
                )
            )

        return DocumentContent(
            file_name=pdf_path.name,
            file_path=str(pdf_path.resolve()),
            page_count=len(document),
            metadata=metadata,
            pages=pages,
        )

    finally:
        document.close()