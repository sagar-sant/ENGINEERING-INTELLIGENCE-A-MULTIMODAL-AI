from dataclasses import dataclass, field
from typing import Any

from src.retrieval.retrieval_models import RetrievalResult


@dataclass
class RAGResponse:
    """Represents the result of a retrieval-augmented generation request."""

    query: str
    answer: str
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> list[dict[str, Any]]:
        """Return lightweight source information for the response."""
        return [
            {
                "chunk_id": result.chunk_id,
                "page_number": result.page_number,
                "score": result.score,
            }
            for result in self.retrieval_results
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert the response into a dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieval_results": [
                result.to_dict()
                for result in self.retrieval_results
            ],
            "sources": self.sources,
            "metadata": self.metadata,
        }