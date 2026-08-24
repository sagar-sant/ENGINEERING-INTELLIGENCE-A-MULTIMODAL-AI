import hashlib

from src.embeddings.embedding_provider import EmbeddingProvider


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic local embedding provider for development and testing.

    This is NOT a semantic embedding model. It provides stable vectors
    so the rest of the embedding and retrieval pipeline can be developed
    without requiring an external embedding API.
    """

    def __init__(self, dimensions: int = 384):
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")

        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        """Return the provider name."""
        return "local-hash-embedding"

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self._dimensions

    def embed_text(self, text: str) -> list[float]:
        """
        Generate a deterministic vector from text.

        The same input text always produces the same vector.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        vector: list[float] = []

        counter = 0

        while len(vector) < self._dimensions:
            digest = hashlib.sha256(
                f"{counter}:{text}".encode("utf-8")
            ).digest()

            for byte in digest:
                value = (byte / 255.0) * 2.0 - 1.0
                vector.append(value)

                if len(vector) >= self._dimensions:
                    break

            counter += 1

        return vector