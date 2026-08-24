from pathlib import Path
from typing import Iterable

from src.chunking.chunk_models import DocumentChunk
from src.retrieval.retrieval_models import RetrievalResponse, RetrievalResult


class SimpleRetriever:
    """
    Lightweight keyword-based retriever.

    This provides a deterministic retrieval layer before a vector
    database is introduced. It scores chunks based on query-term
    frequency and returns the highest-scoring chunks.
    """

    def __init__(self, chunks: Iterable[DocumentChunk]):
        self.chunks = list(chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Convert text into simple normalized tokens."""
        return [
            token.strip(".,;:!?()[]{}\"'")
            for token in text.lower().split()
            if token.strip(".,;:!?()[]{}\"'")
        ]

    @classmethod
    def _score(cls, query: str, text: str) -> float:
        """Calculate a simple keyword-overlap score."""
        query_tokens = set(cls._tokenize(query))
        text_tokens = cls._tokenize(text)

        if not query_tokens or not text_tokens:
            return 0.0

        matches = sum(1 for token in text_tokens if token in query_tokens)

        return matches / len(query_tokens)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve the most relevant chunks for a query.
        """
        if not query.strip():
            return RetrievalResponse(
                query=query,
                results=[],
                retrieval_method="keyword",
                total_results=0,
            )

        scored_chunks = []

        for chunk in self.chunks:
            score = self._score(query, chunk.text)

            if score > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = scored_chunks[:top_k]

        results = [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
                page_number=chunk.page_number,
                metadata=chunk.metadata,
            )
            for score, chunk in selected
        ]

        return RetrievalResponse(
            query=query,
            results=results,
            retrieval_method="keyword",
            total_results=len(results),
        )