from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstore.vector_store import VectorStore


@dataclass
class RetrievalResult:
    """
    Normalized retrieval result exposed to the application.
    """

    chunk_id: str
    text: str
    score: float
    page_number: int | None
    metadata: dict[str, Any]


@dataclass
class RetrievalResponse:
    """
    Collection of retrieved document results.
    """

    results: list[RetrievalResult]
    retrieval_method: str
    total_results: int


class VectorRetriever:
    """
    Vector-based document retriever.

    Responsibilities:
        - embed the user's query
        - search the vector store
        - normalize vector-store results
        - optionally prioritize visual-context chunks
    """

    VISUAL_QUERY_TERMS = {
        "figure",
        "figures",
        "diagram",
        "diagrams",
        "illustration",
        "illustrations",
        "image",
        "images",
        "picture",
        "pictures",
        "visual",
        "visuals",
        "graphic",
        "graphics",
        "schematic",
        "schematics",
        "layout",
        "layouts",
        "label",
        "labels",
        "connector",
        "connectors",
        "component",
        "components",
        "table",
        "tables",
        "chart",
        "charts",
        "drawing",
        "drawings",
        "port",
        "ports",
        "motherboard",
        "chassis",
        "arrow",
        "arrows",
        "visible",
        "shown",
        "shows",
        "pictured",
        "depicted",
    }

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        vector_store: VectorStore,
        visual_boost: float = 0.50,
    ) -> None:
        self.embedding_pipeline = embedding_pipeline
        self.vector_store = vector_store
        self.visual_boost = visual_boost

    # =========================================================
    # QUERY EMBEDDING
    # =========================================================

    def _embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Convert the user's query into an embedding vector.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embeddings = self.embedding_pipeline.embed_texts(
            [query]
        )

        if not embeddings:
            raise RuntimeError(
                "Embedding pipeline returned no query embedding."
            )

        vector = embeddings[0]

        if not vector:
            raise RuntimeError(
                "Query embedding is empty."
            )

        return vector

    # =========================================================
    # VISUAL QUERY DETECTION
    # =========================================================

    @classmethod
    def _is_visual_query(
        cls,
        query: str,
    ) -> bool:
        """
        Determine whether the query concerns visual content.
        """

        words = {
            word.strip(
                ".,!?;:()[]{}\"'"
            ).lower()
            for word in query.split()
        }

        return bool(
            words & cls.VISUAL_QUERY_TERMS
        )

    # =========================================================
    # VISUAL RESULT DETECTION
    # =========================================================

    @staticmethod
    def _is_visual_result(
        result: Any,
    ) -> bool:
        """
        Determine whether a vector result contains visual context.
        """

        metadata = getattr(
            result,
            "metadata",
            None,
        ) or {}

        return (
            metadata.get(
                "has_visual_context"
            )
            is True
        )

    # =========================================================
    # PAGE NUMBER
    # =========================================================

    @staticmethod
    def _page_number(
        result: Any,
    ) -> int | None:
        """
        Safely extract the page number.
        """

        metadata = getattr(
            result,
            "metadata",
            None,
        ) or {}

        page_number = metadata.get(
            "page_number"
        )

        if page_number is None:
            return None

        try:
            return int(page_number)
        except (
            TypeError,
            ValueError,
        ):
            return None

    # =========================================================
    # RESULT NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_result(
        result: Any,
    ) -> RetrievalResult:
        """
        Convert a VectorSearchResult into RetrievalResult.
        """

        metadata = dict(
            getattr(
                result,
                "metadata",
                None,
            )
            or {}
        )

        record_id = getattr(
            result,
            "record_id",
            None,
        )

        text = getattr(
            result,
            "text",
            "",
        )

        score = getattr(
            result,
            "score",
            0.0,
        )

        return RetrievalResult(
            chunk_id=str(
                record_id
            ),
            text=str(
                text
            ),
            score=float(
                score
            ),
            page_number=VectorRetriever._page_number(
                result
            ),
            metadata=metadata,
        )

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResponse:
        """
        Retrieve the most relevant document chunks.

        Normal queries:
            cosine-similarity retrieval.

        Visual queries:
            larger candidate pool + visual-context boost.
        """

        if not query or not query.strip():
            return RetrievalResponse(
                results=[],
                retrieval_method="vector",
                total_results=0,
            )

        if top_k <= 0:
            return RetrievalResponse(
                results=[],
                retrieval_method="vector",
                total_results=0,
            )

        # -----------------------------------------------------
        # Query embedding
        # -----------------------------------------------------

        query_vector = self._embed_query(
            query
        )

        # -----------------------------------------------------
        # Validate query dimensionality
        # -----------------------------------------------------

        query_dimensions = len(
            query_vector
        )

        if query_dimensions <= 0:
            return RetrievalResponse(
                results=[],
                retrieval_method="vector",
                total_results=0,
            )

        # -----------------------------------------------------
        # Visual detection
        # -----------------------------------------------------

        visual_query = self._is_visual_query(
            query
        )

        # -----------------------------------------------------
        # Candidate count
        # -----------------------------------------------------

        if visual_query:
            candidate_k = max(
                top_k * 10,
                50,
            )
        else:
            candidate_k = top_k

        # -----------------------------------------------------
        # Search vector store
        # -----------------------------------------------------

        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_k,
        )

        # -----------------------------------------------------
        # No results
        # -----------------------------------------------------

        if not raw_results:
            return RetrievalResponse(
                results=[],
                retrieval_method="vector",
                total_results=0,
            )

        # -----------------------------------------------------
        # Score and rerank
        # -----------------------------------------------------

        scored_results = []

        for result in raw_results:

            base_score = float(
                getattr(
                    result,
                    "score",
                    0.0,
                )
            )

            visual_result = (
                self._is_visual_result(
                    result
                )
            )

            adjusted_score = base_score

            if (
                visual_query
                and visual_result
            ):
                adjusted_score += (
                    self.visual_boost
                )

            scored_results.append(
                (
                    adjusted_score,
                    base_score,
                    visual_result,
                    result,
                )
            )

        # -----------------------------------------------------
        # Sort by adjusted similarity
        # -----------------------------------------------------

        scored_results.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        # -----------------------------------------------------
        # Normalize
        # -----------------------------------------------------

        normalized_results = []

        for (
            adjusted_score,
            original_score,
            visual_result,
            result,
        ) in scored_results[:top_k]:

            normalized = (
                self._normalize_result(
                    result
                )
            )

            normalized.metadata[
                "original_similarity_score"
            ] = original_score

            normalized.metadata[
                "reranked_score"
            ] = adjusted_score

            normalized.metadata[
                "visual_query"
            ] = visual_query

            normalized.metadata[
                "visual_result"
            ] = visual_result

            normalized.score = (
                adjusted_score
            )

            normalized_results.append(
                normalized
            )

        # -----------------------------------------------------
        # Response
        # -----------------------------------------------------

        return RetrievalResponse(
            results=normalized_results,
            retrieval_method=(
                "vector_visual_rerank"
                if visual_query
                else "vector"
            ),
            total_results=len(
                normalized_results
            ),
        )