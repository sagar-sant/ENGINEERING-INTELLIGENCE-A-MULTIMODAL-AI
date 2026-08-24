from typing import Iterable

from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstore.vector_store import VectorStore
from src.vectorstore.vector_store_models import VectorRecord


class VectorIndexer:
    """
    Converts document chunks into embeddings and stores them
    in the configured vector store.
    """

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_pipeline = embedding_pipeline
        self.vector_store = vector_store

    def index_chunks(
        self,
        chunks: Iterable,
    ) -> int:
        """
        Embed and index document chunks.

        Supports both standard DocumentChunk objects and
        multimodal chunks that expose the same core fields.

        Returns:
            Number of chunks indexed.
        """
        chunks = list(chunks)

        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_pipeline.embed_texts(texts)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        records = []

        for chunk, embedding in zip(chunks, embeddings):
            metadata = {
                "page_number": chunk.page_number,
            }

            if hasattr(chunk, "document_name"):
                metadata["document_name"] = chunk.document_name

            if hasattr(chunk, "character_count"):
                metadata["character_count"] = chunk.character_count

            if hasattr(chunk, "has_visual_context"):
                metadata["has_visual_context"] = (
                    chunk.has_visual_context
                )

            if hasattr(chunk, "rendered_page_path"):
                metadata["rendered_page_path"] = (
                    str(chunk.rendered_page_path)
                    if chunk.rendered_page_path is not None
                    else None
                )

            if hasattr(chunk, "embedded_image_paths"):
                metadata["embedded_image_paths"] = [
                    str(path)
                    for path in chunk.embedded_image_paths
                ]

            if hasattr(chunk, "visual_summary"):
                metadata["visual_summary"] = (
                    chunk.visual_summary
                )

            if hasattr(chunk, "metadata"):
                metadata.update(chunk.metadata)

            records.append(
                VectorRecord(
                    record_id=chunk.chunk_id,
                    vector=embedding,
                    text=chunk.text,
                    metadata=metadata,
                )
            )

        self.vector_store.add_many(records)

        return len(records)