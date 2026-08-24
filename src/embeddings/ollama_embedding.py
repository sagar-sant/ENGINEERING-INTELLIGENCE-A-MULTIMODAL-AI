from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from src.chunking.chunk_models import DocumentChunk
from src.embeddings.embedding_models import ChunkEmbedding
from src.embeddings.embedding_provider import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama-based embedding provider.

    Default model:
        nomic-embed-text

    Used for:
        - document chunk embeddings
        - query embeddings
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 120,
    ) -> None:
        self._model_name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # nomic-embed-text produces 768-dimensional vectors.
        self._dimensions = 768

    @property
    def model_name(self) -> str:
        """
        Name of the embedding model.
        """
        return self._model_name

    @property
    def dimensions(self) -> int:
        """
        Expected embedding vector dimension.
        """
        return self._dimensions

    def _embed(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding using Ollama.
        """

        if not text or not text.strip():
            return []

        payload = {
            "model": self._model_name,
            "input": text,
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.base_url}. "
                "Make sure Ollama is running."
            ) from exc

        embeddings = response_data.get(
            "embeddings",
            [],
        )

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embedding."
            )

        vector = [
            float(value)
            for value in embeddings[0]
        ]

        # Discover the actual dimension from Ollama
        # on the first successful request.
        self._dimensions = len(vector)

        return vector

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a text query.
        """

        return self._embed(text)

    def embed_chunk(
        self,
        chunk: DocumentChunk,
    ) -> ChunkEmbedding:
        """
        Generate an embedding for a document chunk.
        """

        vector = self._embed(
            chunk.text
        )

        return ChunkEmbedding(
            chunk_id=chunk.chunk_id,
            model_name=self.model_name,
            dimensions=len(vector),
            vector=vector,
            metadata=dict(
                getattr(
                    chunk,
                    "metadata",
                    {},
                )
                or {}
            ),
        )