from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.multimodal.visual_metadata import build_visual_metadata
from src.multimodal.visual_understanding import (
    VisualUnderstanding,
    build_visual_understanding,
)


@dataclass
class MultimodalPage:
    """Combined text and visual representation of a PDF page."""

    page_number: int

    text: str
    text_character_count: int

    visual_score: int
    is_visual_candidate: bool
    visual_reason: str

    drawing_count: int = 0

    rendered_page_path: Optional[Path] = None

    embedded_image_paths: List[Path] = field(
        default_factory=list
    )

    visual_understanding: Optional[VisualUnderstanding] = None


@dataclass
class MultimodalDocument:
    """Combined multimodal representation of a PDF document."""

    file_name: str
    page_count: int

    pages: List[MultimodalPage] = field(
        default_factory=list
    )


def build_multimodal_document(
    pdf_path: str | Path,
    render_candidates: bool = True,
    extract_images: bool = True,
) -> MultimodalDocument:
    """
    Build a combined multimodal representation of a PDF.

    Text comes from the existing document-processing pipeline.

    Visual metadata comes from the multimodal visual detector.

    Figure captions are extracted from the actual page text and
    attached to the visual understanding for visual pages.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # ---------------------------------------------------------
    # Existing text-processing pipeline
    # ---------------------------------------------------------

    from src.ingestion.document_processor import process_document

    document = process_document(pdf_path)

    # ---------------------------------------------------------
    # Visual metadata
    # ---------------------------------------------------------

    visual_metadata = build_visual_metadata(
        pdf_path,
        render_candidates=render_candidates,
        extract_images=extract_images,
    )

    visual_by_page = {
        page.page_number: page
        for page in visual_metadata
    }

    pages: List[MultimodalPage] = []

    # ---------------------------------------------------------
    # Combine text + visual information
    # ---------------------------------------------------------

    for page in document.pages:

        visual = visual_by_page.get(
            page.page_number
        )

        # -----------------------------------------------------
        # Normal text-only page
        # -----------------------------------------------------

        if visual is None:
            pages.append(
                MultimodalPage(
                    page_number=page.page_number,
                    text=page.text,
                    text_character_count=page.character_count,
                    visual_score=0,
                    is_visual_candidate=False,
                    visual_reason=(
                        "No visual metadata available."
                    ),
                    drawing_count=0,
                    rendered_page_path=None,
                    embedded_image_paths=[],
                    visual_understanding=None,
                )
            )

            continue

        # -----------------------------------------------------
        # Determine meaningful visual content
        # -----------------------------------------------------

        has_embedded_images = bool(
            visual.embedded_image_paths
        )

        is_meaningful_visual = (
            visual.is_visual_candidate
            or has_embedded_images
        )

        understanding = None

        if is_meaningful_visual:
            understanding = build_visual_understanding(
                page_number=page.page_number,
                image_paths=visual.embedded_image_paths,
                has_drawings=(
                    visual.drawing_count > 0
                ),
                text_character_count=(
                    page.character_count
                ),
                text=page.text,
            )

        pages.append(
            MultimodalPage(
                page_number=page.page_number,
                text=page.text,
                text_character_count=page.character_count,
                visual_score=visual.visual_score,
                is_visual_candidate=(
                    visual.is_visual_candidate
                ),
                visual_reason=visual.reason,
                drawing_count=visual.drawing_count,
                rendered_page_path=(
                    visual.rendered_page_path
                ),
                embedded_image_paths=list(
                    visual.embedded_image_paths
                ),
                visual_understanding=understanding,
            )
        )

    return MultimodalDocument(
        file_name=document.file_name,
        page_count=document.page_count,
        pages=pages,
    )