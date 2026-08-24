from typing import Iterable

from src.chunking.chunk_models import DocumentChunk
from src.retrieval.retrieval_models import RetrievalResponse, RetrievalResult
from src.retrieval.retriever import SimpleRetriever


class HybridRetriever:
    """
    Combines keyword retrieval with optional embedding similarity.

    Keyword retrieval provides deterministic lexical matching.
    Embedding scores can be supplied when an embedding pipeline
    is available.
    """

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        embedding_pipeline=None,
        keyword_weight: float = 0.5,
        embedding_weight: float = 0.5,
    ):
        self.chunks = list(chunks)
        self.keyword_retriever = SimpleRetriever(self.chunks)
        self.embedding_pipeline = embedding_pipeline

        total_weight = keyword_weight + embedding_weight

        if total_weight <= 0:
            raise ValueError("Retrieval weights must sum to a positive value.")

        self.keyword_weight = keyword_weight / total_weight
        self.embedding_weight = embedding_weight / total_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve relevant chunks using lexical and embedding signals.
        """
        if not query.strip():
            return RetrievalResponse(
                query=query,
                results=[],
                retrieval_method="hybrid",
                total_results=0,
            )

        keyword_response = self.keyword_retriever.retrieve(
            query=query,
            top_k=len(self.chunks),
        )

        keyword_scores = {
            result.chunk_id: result.score
            for result in keyword_response.results
        }

        embedding_scores = {}

        if self.embedding_pipeline is not None:
            embedding_scores = self._calculate_embedding_scores(query)

        combined_results = []

        for chunk in self.chunks:
            keyword_score = keyword_scores.get(chunk.chunk_id, 0.0)
            embedding_score = embedding_scores.get(chunk.chunk_id, 0.0)

            combined_score = (
                self.keyword_weight * keyword_score
                + self.embedding_weight * embedding_score
            )

            if combined_score > 0:
                combined_results.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=combined_score,
                        page_number=chunk.page_number,
                        metadata=chunk.metadata,
                    )
                )

        combined_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        results = combined_results[:top_k]

        return RetrievalResponse(
            query=query,
            results=results,
            retrieval_method="hybrid",
            total_results=len(results),
        )

    def _calculate_embedding_scores(
        self,
        query: str,
    ) -> dict[str, float]:
        """
        Calculate embedding similarity scores when an embedding
        pipeline is available.

        The method supports pipelines exposing an `similarity_search`
        method that returns chunk IDs and scores.
        """
        if not hasattr(self.embedding_pipeline, "similarity_search"):
            return {}

        matches = self.embedding_pipeline.similarity_search(
            query,
            top_k=len(self.chunks),
        )

        scores = {}

        for match in matches:
            if isinstance(match, RetrievalResult):
                scores[match.chunk_id] = match.score
                continue

            if isinstance(match, tuple) and len(match) >= 2:
                chunk_id, score = match[0], match[1]
                scores[str(chunk_id)] = float(score)
                continue

            if isinstance(match, dict):
                chunk_id = match.get("chunk_id")
                score = match.get("score")

                if chunk_id is not None and score is not None:
                    scores[str(chunk_id)] = float(score)

        return scores