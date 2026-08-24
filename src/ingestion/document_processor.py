from pathlib import Path

from src.extraction.ocr_processor import ocr_page
from src.ingestion.document_models import ProcessedDocument, ProcessedPage
from src.ingestion.pdf_loader import load_pdf
from src.multimodal.visual_metadata import build_visual_metadata


def process_document(pdf_path: str | Path) -> ProcessedDocument:
    """
    Process an engineering PDF into a unified document representation.

    Text is extracted using PyMuPDF when available, with OCR used for
    pages identified as OCR candidates.

    Visual metadata is collected for pages that contain meaningful
    visual-content indicators.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ---------------------------------------------------------
    # Step 1: Load the PDF and perform text/OCR processing
    # ---------------------------------------------------------
    loaded_document = load_pdf(pdf_path)

    processed_pages: list[ProcessedPage] = []

    for page in loaded_document.pages:

        if page.needs_ocr:
            ocr_result = ocr_page(pdf_path, page.page_number)

            processed_page = ProcessedPage(
                page_number=page.page_number,
                text=ocr_result.text,
                extraction_method="ocr",
                character_count=ocr_result.character_count,
                ocr_confidence=ocr_result.confidence,
            )

        else:
            processed_page = ProcessedPage(
                page_number=page.page_number,
                text=page.text,
                extraction_method="pymupdf",
                character_count=page.character_count,
                ocr_confidence=None,
            )

        processed_pages.append(processed_page)

    # ---------------------------------------------------------
    # Step 2: Build visual metadata
    # ---------------------------------------------------------
    visual_metadata = build_visual_metadata(
        pdf_path,
        render_candidates=False,
        extract_images=False,
    )

    # ---------------------------------------------------------
    # Step 3: Build unified document
    # ---------------------------------------------------------
    return ProcessedDocument(
        file_name=loaded_document.file_name,
        file_path=loaded_document.file_path,
        page_count=loaded_document.page_count,
        metadata=loaded_document.metadata,
        pages=processed_pages,
        visual_metadata=visual_metadata,
    )