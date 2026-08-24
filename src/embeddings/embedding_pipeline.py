from src.chunking.chunk_models import DocumentChunk
from src.embeddings.embedding_models import ChunkEmbedding
from src.embeddings.embedding_provider import EmbeddingProvider


class EmbeddingPipeline:
    """
    Coordinates embedding generation for document chunks and text queries.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[ChunkEmbedding]:
        """
        Generate embeddings for a collection of document chunks.
        """
        embeddings: list[ChunkEmbedding] = []

        for chunk in chunks:
            embedding = self.provider.embed_chunk(chunk)
            embeddings.append(embedding)

        return embeddings

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for plain text queries.

        The configured provider must expose an embed_text method.
        """
        embeddings: list[list[float]] = []

        for text in texts:
            if not text.strip():
                embeddings.append([])
                continue

            if not hasattr(self.provider, "embed_text"):
                raise AttributeError(
                    "The configured embedding provider must "
                    "implement embed_text()."
                )

            embedding = self.provider.embed_text(text)
            embeddings.append(embedding)

        return embeddings


def embed_chunks(
    chunks: list[DocumentChunk],
    provider: EmbeddingProvider,
) -> list[ChunkEmbedding]:
    """
    Backward-compatible helper for embedding document chunks.
    """
    pipeline = EmbeddingPipeline(provider)
    return pipeline.embed_chunks(chunks)