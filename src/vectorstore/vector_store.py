from typing import Iterable

from src.vectorstore.in_memory_store import InMemoryVectorStore
from src.vectorstore.vector_store_models import (
    VectorRecord,
    VectorSearchResult,
)


class VectorStore:
    """
    High-level vector-store interface.

    Uses the in-memory implementation internally so the rest of
    the application does not depend directly on a specific vector
    database.
    """

    def __init__(self) -> None:
        self._store = InMemoryVectorStore()

    def add(
        self,
        record: VectorRecord,
    ) -> None:
        """Add a single vector record."""
        self._store.add(record)

    def add_many(
        self,
        records: Iterable[VectorRecord],
    ) -> None:
        """Add multiple vector records."""
        self._store.add_many(records)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Search for the most similar stored vectors."""
        return self._store.search(
            query_vector=query_vector,
            top_k=top_k,
        )

    def get(
        self,
        record_id: str,
    ) -> VectorRecord | None:
        """Retrieve a stored record by ID."""
        return self._store.get(record_id)

    def count(self) -> int:
        """Return the number of stored vectors."""
        return self._store.count()

    def delete(
        self,
        record_id: str,
    ) -> bool:
        """Delete a vector record."""
        return self._store.delete(record_id)

    def clear(self) -> None:
        """Remove all stored vectors."""
        self._store.clear()