import re

from src.chunking.chunk_models import DocumentChunk
from src.chunking.chunk_quality import analyze_chunk_quality
from src.ingestion.document_models import ProcessedDocument


HEADING_PATTERNS = [
    re.compile(r"^\d+(?:\.\d+)*\s+.+$"),
    re.compile(r"^[A-Z][A-Z0-9\s\-]{4,}$"),
    re.compile(r"^(Chapter|Section|Appendix)\s+.+$", re.IGNORECASE),
]


def is_likely_heading(line: str) -> bool:
    """Return True when a line looks like an engineering-document heading."""

    line = line.strip()

    if not line:
        return False

    return any(pattern.match(line) for pattern in HEADING_PATTERNS)


def split_into_sections(text: str) -> list[str]:
    """
    Split page text at likely section headings.

    The function is intentionally heuristic and does not claim
    to understand the semantic structure of the document.
    """

    lines = text.splitlines()

    sections: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        if is_likely_heading(line) and current_lines:
            section = "\n".join(current_lines).strip()

            if section:
                sections.append(section)

            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        section = "\n".join(current_lines).strip()

        if section:
            sections.append(section)

    return sections


def chunk_sections(
    document: ProcessedDocument,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Create chunks while attempting to preserve likely section boundaries.

    Large sections are split using the same overlapping character
    strategy as the standard text chunker.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []

    for page in document.pages:
        text = page.text.strip()

        if not text:
            continue

        sections = split_into_sections(text)

        chunk_index = 0
        page_offset = 0

        for section in sections:
            section_length = len(section)
            start = 0

            while start < section_length:
                end = min(start + chunk_size, section_length)

                chunk_text = section[start:end].strip()

                if chunk_text:
                    quality = analyze_chunk_quality(chunk_text)

                    chunk_id = (
                        f"{document.file_name.rsplit('.', 1)[0]}"
                        f"_page_{page.page_number:03d}"
                        f"_chunk_{chunk_index:03d}"
                    )

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_name=document.file_name,
                            page_number=page.page_number,
                            text=chunk_text,
                            character_count=len(chunk_text),
                            start_character=page_offset + start,
                            end_character=page_offset + end,
                            metadata={
                                "extraction_method": page.extraction_method,
                                "ocr_confidence": page.ocr_confidence,
                                "section_aware": True,
                                "quality": {
                                    "character_count": quality.character_count,
                                    "word_count": quality.word_count,
                                    "sentence_count": quality.sentence_count,
                                    "has_heading": quality.has_heading,
                                    "has_numbered_list": quality.has_numbered_list,
                                    "has_warning": quality.has_warning,
                                    "has_table_like_content": (
                                        quality.has_table_like_content
                                    ),
                                    "quality_score": quality.quality_score,
                                },
                            },
                        )
                    )

                    chunk_index += 1

                if end >= section_length:
                    break

                start = end - chunk_overlap

            page_offset += section_length + 1

    return chunks