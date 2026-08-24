from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


@dataclass
class OCRResult:
    """Stores OCR output for one PDF page."""

    page_number: int
    text: str
    character_count: int
    confidence: float


def _calculate_average_confidence(data: dict) -> float:
    """Calculate average OCR confidence from Tesseract output."""

    confidences = []

    for confidence in data.get("conf", []):
        try:
            value = float(confidence)

            if value >= 0:
                confidences.append(value)

        except (ValueError, TypeError):
            continue

    if not confidences:
        return 0.0

    return sum(confidences) / len(confidences)


def ocr_page(
    pdf_path: str | Path,
    page_number: int,
    dpi: int = 200,
) -> OCRResult:
    """
    Render one PDF page as an image and perform OCR using Tesseract.

    page_number is one-based, matching the page numbers shown to users.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if page_number < 1:
        raise ValueError("page_number must be at least 1")

    document = pymupdf.open(pdf_path)

    try:
        if page_number > len(document):
            raise ValueError(
                f"Page {page_number} does not exist. "
                f"Document contains {len(document)} pages."
            )

        page = document[page_number - 1]

        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )

        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config="--psm 3",
            output_type=pytesseract.Output.DICT,
        )

        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 3",
        ).strip()

        confidence = _calculate_average_confidence(data)

        return OCRResult(
            page_number=page_number,
            text=text,
            character_count=len(text),
            confidence=round(confidence, 2),
        )

    finally:
        document.close()


def ocr_candidate_pages(
    pdf_path: str | Path,
    page_numbers: list[int],
    dpi: int = 200,
) -> list[OCRResult]:
    """
    Run OCR on a selected list of pages.

    This allows the system to perform selective OCR instead of
    unnecessarily processing every page.
    """

    results = []

    for page_number in page_numbers:
        result = ocr_page(
            pdf_path=pdf_path,
            page_number=page_number,
            dpi=dpi,
        )

        results.append(result)

    return results