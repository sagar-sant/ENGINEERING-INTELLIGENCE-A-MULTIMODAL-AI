import math
from typing import Iterable

from src.vectorstore.vector_store_models import (
    VectorRecord,
    VectorSearchResult,
)


class InMemoryVectorStore:
    """
    Lightweight in-memory vector store.

    Stores vectors and performs cosine-similarity search.
    This provides a local vector-store implementation before
    introducing an external vector database.
    """

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def add(self, record: VectorRecord) -> None:
        """Add or replace a vector record."""
        if not record.record_id:
            raise ValueError("record_id cannot be empty.")

        if not record.vector:
            raise ValueError("vector cannot be empty.")

        self._records[record.record_id] = record

    def add_many(self, records: Iterable[VectorRecord]) -> None:
        """Add multiple vector records."""
        for record in records:
            self.add(record)

    def get(self, record_id: str) -> VectorRecord | None:
        """Return a record by ID."""
        return self._records.get(record_id)

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Vectors must have the same dimensionality."
            )

        if not vector_a:
            return 0.0

        dot_product = sum(
            a * b for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """
        Return the top-k records ranked by cosine similarity.
        """
        if not query_vector:
            return []

        if top_k <= 0:
            return []

        results = []

        for record in self._records.values():
            score = self.cosine_similarity(
                query_vector,
                record.vector,
            )

            results.append(
                VectorSearchResult(
                    record_id=record.record_id,
                    score=score,
                    text=record.text,
                    metadata=record.metadata,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

    def delete(self, record_id: str) -> bool:
        """Delete a record and return whether it existed."""
        if record_id not in self._records:
            return False

        del self._records[record_id]
        return True

    def clear(self) -> None:
        """Remove all stored records."""
        self._records.clear()