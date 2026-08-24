from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """Represents one retrieved document chunk."""

    chunk_id: str
    text: str
    score: float
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the retrieval result to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResponse:
    """Represents the complete result of a retrieval operation."""

    query: str
    results: list[RetrievalResult]
    retrieval_method: str
    total_results: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the response to a dictionary."""
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "retrieval_method": self.retrieval_method,
            "total_results": self.total_results,
        }