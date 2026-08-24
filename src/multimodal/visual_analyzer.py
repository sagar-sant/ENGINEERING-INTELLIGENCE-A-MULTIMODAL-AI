from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class PageVisualInfo:
    """Visual information detected on one PDF page."""

    page_number: int
    image_count: int
    has_images: bool
    page_width: float
    page_height: float

    @property
    def has_visual_content(self) -> bool:
        """Return whether the page contains visual content."""
        return self.has_images


def analyze_pdf_visuals(pdf_path: str | Path) -> list[PageVisualInfo]:
    """
    Analyze visual content across all pages of a PDF.

    Currently detects embedded images and records page dimensions.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    visual_pages: list[PageVisualInfo] = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            images = page.get_images(full=True)
            rectangle = page.rect

            visual_pages.append(
                PageVisualInfo(
                    page_number=page_number,
                    image_count=len(images),
                    has_images=len(images) > 0,
                    page_width=rectangle.width,
                    page_height=rectangle.height,
                )
            )

    return visual_pages