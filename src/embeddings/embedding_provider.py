from abc import ABC, abstractmethod

from src.chunking.chunk_models import DocumentChunk
from src.embeddings.embedding_models import ChunkEmbedding


class EmbeddingProvider(ABC):
    """Abstract interface for document embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the embedding model name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding vector dimensions."""
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""
        raise NotImplementedError

    def embed_chunk(self, chunk: DocumentChunk) -> ChunkEmbedding:
        """Generate an embedding for a document chunk."""

        vector = self.embed_text(chunk.text)

        if len(vector) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected "
                f"{self.dimensions}, got {len(vector)}"
            )

        return ChunkEmbedding(
            chunk_id=chunk.chunk_id,
            model_name=self.model_name,
            dimensions=self.dimensions,
            vector=vector,
            metadata={
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
            },
        )