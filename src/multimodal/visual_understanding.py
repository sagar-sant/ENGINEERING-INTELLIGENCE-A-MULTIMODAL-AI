from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re


@dataclass
class VisualElement:
    """Represents a meaningful visual element identified on a PDF page."""

    element_type: str
    description: str
    page_number: int
    image_path: Optional[Path] = None


@dataclass
class VisualUnderstanding:
    """Structured visual understanding for a visually significant PDF page."""

    page_number: int
    elements: List[VisualElement]
    summary: str
    captions: List[str]


def extract_figure_captions(text: str) -> List[str]:
    """
    Extract likely figure/diagram captions from page text.

    Examples:
        Figure 5-7. SUPER P4DLR Layout
        Figure 6-1: Chassis Front and Rear Views
        Fig. 2-5 - Accessing the Inside of the SuperServer
    """

    if not text:
        return []

    captions: List[str] = []

    patterns = [
        r"(?im)^\s*(Figure\s+\d+(?:-\d+)?\s*[\.:—-]\s*.+)$",
        r"(?im)^\s*(Fig\.\s*\d+(?:-\d+)?\s*[\.:—-]\s*.+)$",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for match in matches:
            caption = match.strip()

            if caption not in captions:
                captions.append(caption)

    return captions


def create_visual_understanding(
    page_number: int,
    image_paths: Optional[List[Path]] = None,
    has_drawings: bool = False,
    text_character_count: int = 0,
    text: str = "",
) -> VisualUnderstanding:
    """
    Create a structured representation of meaningful visual content.

    This module performs structural interpretation only.

    It does not claim to semantically understand the contents
    of an image, diagram, chart, or figure.
    """

    elements: List[VisualElement] = []

    image_paths = image_paths or []

    captions = extract_figure_captions(text)

    # ---------------------------------------------------------
    # Embedded images
    # ---------------------------------------------------------

    for index, image_path in enumerate(image_paths, start=1):
        elements.append(
            VisualElement(
                element_type="embedded_image",
                description=(
                    f"Embedded image {index} extracted from "
                    f"page {page_number}."
                ),
                page_number=page_number,
                image_path=image_path,
            )
        )

    # ---------------------------------------------------------
    # Vector graphics
    # ---------------------------------------------------------

    if has_drawings:
        elements.append(
            VisualElement(
                element_type="vector_graphics",
                description=(
                    f"Page {page_number} contains "
                    "vector drawing objects associated "
                    "with visual content."
                ),
                page_number=page_number,
            )
        )

    # ---------------------------------------------------------
    # Figure captions
    # ---------------------------------------------------------

    for caption in captions:
        elements.append(
            VisualElement(
                element_type="figure_caption",
                description=caption,
                page_number=page_number,
            )
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary_parts: List[str] = []

    if image_paths:
        image_count = len(image_paths)

        if image_count == 1:
            summary_parts.append("1 embedded image")
        else:
            summary_parts.append(
                f"{image_count} embedded images"
            )

    if has_drawings:
        summary_parts.append("vector graphics")

    if captions:
        summary_parts.append(
            f"{len(captions)} figure caption(s)"
        )

    if not summary_parts:
        summary = "No structured visual elements identified."
    else:
        summary = (
            f"Page {page_number} contains "
            + " and ".join(summary_parts)
            + "."
        )

    return VisualUnderstanding(
        page_number=page_number,
        elements=elements,
        summary=summary,
        captions=captions,
    )


def build_visual_understanding(
    page_number: int,
    image_paths: Optional[List[Path]] = None,
    has_drawings: bool = False,
    text_character_count: int = 0,
    text: str = "",
) -> Optional[VisualUnderstanding]:
    """
    Build structured visual understanding for a PDF page.

    Visual understanding is created only when meaningful visual
    signals are supplied.
    """

    image_paths = image_paths or []

    if not image_paths and not has_drawings:
        return None

    return create_visual_understanding(
        page_number=page_number,
        image_paths=image_paths,
        has_drawings=has_drawings,
        text_character_count=text_character_count,
        text=text,
    )