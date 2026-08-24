from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorRecord:
    """Represents one chunk stored in a vector store."""

    record_id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record into a dictionary."""
        return {
            "record_id": self.record_id,
            "vector": self.vector,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class VectorSearchResult:
    """Represents one result returned from vector similarity search."""

    record_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into a dictionary."""
        return {
            "record_id": self.record_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }