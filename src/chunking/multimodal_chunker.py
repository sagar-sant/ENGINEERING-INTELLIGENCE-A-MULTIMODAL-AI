from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.multimodal.multimodal_processor import (
    MultimodalDocument,
    MultimodalPage,
)


@dataclass
class MultimodalChunk:
    """A retrievable chunk containing text and optional visual context."""

    chunk_id: str
    page_number: int

    text: str

    start_character: int
    end_character: int

    has_visual_context: bool

    rendered_page_path: Optional[Path] = None

    embedded_image_paths: List[Path] = field(
        default_factory=list
    )

    visual_summary: Optional[str] = None


def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[tuple[str, int, int]]:
    """
    Split text into overlapping character-based chunks.

    Returns:
        List of:
            (chunk_text, start_character, end_character)
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks: List[tuple[str, int, int]] = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                (
                    chunk_text,
                    start,
                    end,
                )
            )

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def chunk_page(
    page: MultimodalPage,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[MultimodalChunk]:
    """Convert one multimodal page into retrievable chunks."""

    # ---------------------------------------------------------
    # Text chunks
    # ---------------------------------------------------------

    text_chunks = split_text(
        page.text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # ---------------------------------------------------------
    # Visual context
    # ---------------------------------------------------------

    visual_summary = None

    if page.visual_understanding is not None:
        visual_summary = (
            page.visual_understanding.summary
        )

    # A page has visual context only when the visual detector
    # identified it as meaningful or embedded images exist.
    has_visual_context = (
        page.is_visual_candidate
        or bool(page.embedded_image_paths)
    )

    chunks: List[MultimodalChunk] = []

    # ---------------------------------------------------------
    # Normal text chunks
    # ---------------------------------------------------------

    for index, (
        chunk_text,
        start,
        end,
    ) in enumerate(
        text_chunks,
        start=1,
    ):

        chunks.append(
            MultimodalChunk(
                chunk_id=(
                    f"page-{page.page_number}"
                    f"-chunk-{index}"
                ),
                page_number=page.page_number,
                text=chunk_text,
                start_character=start,
                end_character=end,
                has_visual_context=has_visual_context,
                rendered_page_path=(
                    page.rendered_page_path
                ),
                embedded_image_paths=list(
                    page.embedded_image_paths
                ),
                visual_summary=visual_summary,
            )
        )

    # ---------------------------------------------------------
    # Visual-only pages
    # ---------------------------------------------------------

    if not chunks and has_visual_context:

        chunks.append(
            MultimodalChunk(
                chunk_id=(
                    f"page-{page.page_number}"
                    "-visual"
                ),
                page_number=page.page_number,
                text="",
                start_character=0,
                end_character=0,
                has_visual_context=True,
                rendered_page_path=(
                    page.rendered_page_path
                ),
                embedded_image_paths=list(
                    page.embedded_image_paths
                ),
                visual_summary=visual_summary,
            )
        )

    return chunks


def chunk_multimodal_document(
    document: MultimodalDocument,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[MultimodalChunk]:
    """Convert a multimodal document into retrievable chunks."""

    chunks: List[MultimodalChunk] = []

    for page in document.pages:

        page_chunks = chunk_page(
            page,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks.extend(page_chunks)

    return chunks