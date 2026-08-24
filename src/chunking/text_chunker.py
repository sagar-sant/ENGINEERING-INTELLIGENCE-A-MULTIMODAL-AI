from pathlib import Path

from src.chunking.chunk_models import DocumentChunk
from src.chunking.chunk_quality import analyze_chunk_quality
from src.ingestion.document_models import ProcessedDocument


def chunk_document(
    document: ProcessedDocument,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Split a processed document into overlapping text chunks.

    Chunking is performed independently for each page so that
    page-level provenance is preserved.

    Each chunk also receives basic structural quality metadata.
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

        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:

            end = min(start + chunk_size, text_length)

            chunk_text = text[start:end].strip()

            if chunk_text:
                quality = analyze_chunk_quality(chunk_text)

                chunk_id = (
                    f"{Path(document.file_name).stem}"
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
                        start_character=start,
                        end_character=end,
                        metadata={
                            "extraction_method": page.extraction_method,
                            "ocr_confidence": page.ocr_confidence,
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

            if end >= text_length:
                break

            start = end - chunk_overlap
            chunk_index += 1

    return chunks