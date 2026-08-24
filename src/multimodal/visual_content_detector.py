from dataclasses import dataclass
from pathlib import Path
from typing import List

import pymupdf


@dataclass
class VisualPage:
    """
    Metadata describing potential visual content on a PDF page.

    This class identifies pages that are candidates for visual analysis.
    It does not claim to understand the semantic meaning of diagrams,
    drawings, or schematics.
    """

    page_number: int
    width: float
    height: float

    image_count: int
    drawing_count: int
    text_character_count: int

    visual_score: int

    has_images: bool
    has_drawings: bool
    is_visual_candidate: bool

    reason: str


def calculate_visual_score(
    image_count: int,
    drawing_count: int,
    text_character_count: int,
) -> int:
    """
    Calculate a simple heuristic visual-content score.

    Score components:
        Embedded images:       +5 each
        High vector density:   +3
        Moderate vector count: +2
        Low text + graphics:   +3
    """

    score = 0

    if image_count > 0:
        score += min(image_count * 5, 15)

    if drawing_count >= 1000:
        score += 3
    elif drawing_count >= 100:
        score += 2
    elif drawing_count >= 20:
        score += 1

    if text_character_count < 300 and drawing_count >= 10:
        score += 3

    return score


def detect_visual_pages(pdf_path: str | Path) -> List[VisualPage]:
    """
    Detect pages that may contain meaningful visual content.

    The detector uses PDF-level structural information:

    - embedded images
    - vector drawing objects
    - text density

    It is a candidate detector, not a semantic image classifier.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results: List[VisualPage] = []

    with pymupdf.open(pdf_path) as document:

        for page_index, page in enumerate(document):

            page_number = page_index + 1

            rect = page.rect
            width = rect.width
            height = rect.height

            text = page.get_text("text").strip()
            text_character_count = len(text)

            image_count = len(page.get_images(full=True))

            try:
                drawing_count = len(page.get_drawings())
            except Exception:
                drawing_count = 0

            has_images = image_count > 0
            has_drawings = drawing_count > 0

            visual_score = calculate_visual_score(
                image_count=image_count,
                drawing_count=drawing_count,
                text_character_count=text_character_count,
            )

            reasons = []

            if image_count > 0:
                reasons.append(
                    f"{image_count} embedded image(s)"
                )

            if drawing_count >= 1000:
                reasons.append(
                    f"very high vector complexity ({drawing_count})"
                )
            elif drawing_count >= 100:
                reasons.append(
                    f"high vector complexity ({drawing_count})"
                )
            elif drawing_count >= 20:
                reasons.append(
                    f"moderate vector content ({drawing_count})"
                )

            if text_character_count < 300 and drawing_count >= 10:
                reasons.append("low-text visual page")

            # A score of 5+ indicates a strong enough signal
            # to consider the page for visual processing.
            is_visual_candidate = visual_score >= 5

            if not reasons:
                reason = "No strong visual indicators"
            else:
                reason = "; ".join(reasons)

            results.append(
                VisualPage(
                    page_number=page_number,
                    width=width,
                    height=height,
                    image_count=image_count,
                    drawing_count=drawing_count,
                    text_character_count=text_character_count,
                    visual_score=visual_score,
                    has_images=has_images,
                    has_drawings=has_drawings,
                    is_visual_candidate=is_visual_candidate,
                    reason=reason,
                )
            )

    return results


def get_visual_candidate_pages(
    pdf_path: str | Path,
) -> List[VisualPage]:
    """
    Return only pages identified as visual-content candidates.
    """

    pages = detect_visual_pages(pdf_path)

    return [
        page
        for page in pages
        if page.is_visual_candidate
    ]