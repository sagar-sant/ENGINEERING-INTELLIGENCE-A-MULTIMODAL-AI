from pathlib import Path

from src.chunking.multimodal_chunker import chunk_multimodal_document
from src.chunking.text_chunker import chunk_document
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.ingestion.document_processor import process_document
from src.multimodal.multimodal_processor import build_multimodal_document
from src.vectorstore.indexer import VectorIndexer


class DocumentIndexingPipeline:
    """
    End-to-end pipeline for processing and indexing an engineering PDF.
    """

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        vector_indexer: VectorIndexer,
    ) -> None:
        self.embedding_pipeline = embedding_pipeline
        self.vector_indexer = vector_indexer

    def index_document(
        self,
        pdf_path: str | Path,
        use_multimodal: bool = False,
    ) -> dict:
        """
        Process, chunk, embed, and index a PDF document.

        Args:
            pdf_path: Path to the PDF document.
            use_multimodal: Whether to include multimodal page context.

        Returns:
            Summary of the indexing operation.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if use_multimodal:
            multimodal_document = build_multimodal_document(
                pdf_path,
                render_candidates=True,
                extract_images=True,
            )

            multimodal_chunks = chunk_multimodal_document(
                multimodal_document
            )

            indexed_count = self.vector_indexer.index_chunks(
                multimodal_chunks
            )

            return {
                "file_name": multimodal_document.file_name,
                "file_path": str(pdf_path),
                "page_count": multimodal_document.page_count,
                "chunk_count": len(multimodal_chunks),
                "indexed_count": indexed_count,
                "multimodal": True,
                "visual_context_chunks": sum(
                    chunk.has_visual_context
                    for chunk in multimodal_chunks
                ),
            }

        document = process_document(pdf_path)

        chunks = chunk_document(document)

        indexed_count = self.vector_indexer.index_chunks(
            chunks
        )

        return {
            "file_name": document.file_name,
            "file_path": document.file_path,
            "page_count": document.page_count,
            "chunk_count": len(chunks),
            "indexed_count": indexed_count,
            "multimodal": False,
        }