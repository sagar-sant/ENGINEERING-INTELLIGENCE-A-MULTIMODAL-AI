from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.multimodal.visual_content_detector import detect_visual_pages
from src.multimodal.page_renderer import render_page
from src.multimodal.image_extractor import extract_page_images


@dataclass
class VisualPageMetadata:
    """Structured multimodal metadata for a PDF page."""

    page_number: int

    width: float
    height: float

    text_character_count: int
    image_count: int
    drawing_count: int

    visual_score: int
    is_visual_candidate: bool
    reason: str

    rendered_page_path: Optional[Path] = None
    embedded_image_paths: List[Path] = field(default_factory=list)


def build_visual_metadata(
    pdf_path: str,
    render_candidates: bool = True,
    extract_images: bool = True,
) -> List[VisualPageMetadata]:
    """
    Build structured multimodal metadata for visually significant pages.

    Pages are first identified by the visual content detector.
    Candidate pages can then be rendered and their embedded images extracted.
    """

    candidates = detect_visual_pages(pdf_path)

    metadata = []

    for page in candidates:
        rendered_path = None
        image_paths = []

        if render_candidates:
            rendered_path = render_page(pdf_path, page.page_number)

        if extract_images and page.has_images:
            images = extract_page_images(pdf_path, page.page_number)
            image_paths = [image.file_path for image in images]

        metadata.append(
            VisualPageMetadata(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                text_character_count=page.text_character_count,
                image_count=page.image_count,
                drawing_count=page.drawing_count,
                visual_score=page.visual_score,
                is_visual_candidate=page.is_visual_candidate,
                reason=page.reason,
                rendered_page_path=rendered_path,
                embedded_image_paths=image_paths,
            )
        )

    return metadata