from dataclasses import dataclass
from typing import Iterable


@dataclass
class ContextItem:
    """One retrieved item used to construct RAG context."""

    chunk_id: str
    text: str
    score: float | None = None
    page_number: int | None = None
    metadata: dict | None = None


class ContextBuilder:
    """
    Builds bounded textual context from retrieved document chunks.

    Retrieval metadata such as chunk ID, score, and page number
    is preserved. Multimodal metadata is included when available.
    """

    def __init__(
        self,
        max_characters: int = 6000,
    ) -> None:
        if max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than zero."
            )

        self.max_characters = max_characters

    def build(
        self,
        results: Iterable,
    ) -> str:
        """
        Build a bounded context string from retrieval results.

        Results are processed in retrieval order until the
        maximum character limit is reached.
        """

        results = list(results)

        if not results:
            return ""

        context_parts: list[str] = []
        total_characters = 0

        for index, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = getattr(
                result,
                "chunk_id",
                None,
            )

            text = getattr(
                result,
                "text",
                "",
            ) or ""

            score = getattr(
                result,
                "score",
                None,
            )

            page_number = getattr(
                result,
                "page_number",
                None,
            )

            metadata = getattr(
                result,
                "metadata",
                {},
            ) or {}

            if page_number is None:
                page_number = metadata.get(
                    "page_number"
                )

            if chunk_id is None:
                chunk_id = metadata.get(
                    "chunk_id"
                )

            visual_summary = metadata.get(
                "visual_summary"
            )

            has_visual_context = (
                metadata.get(
                    "has_visual_context"
                )
                is True
            )

            parts: list[str] = [
                f"[Retrieved Context {index}]"
            ]

            if chunk_id is not None:
                parts.append(
                    f"Chunk ID: {chunk_id}"
                )

            if score is not None:
                parts.append(
                    f"Score: {score}"
                )

            if page_number is not None:
                parts.append(
                    f"Page {page_number}"
                )

            if text.strip():
                parts.append(
                    f"Text:\n{text.strip()}"
                )

            if (
                has_visual_context
                and visual_summary
            ):
                parts.append(
                    "Visual Context:\n"
                    f"{visual_summary}"
                )

            item_text = "\n".join(parts)

            remaining = (
                self.max_characters
                - total_characters
            )

            if remaining <= 0:
                break

            if len(item_text) > remaining:
                item_text = item_text[:remaining]

            if item_text:
                context_parts.append(
                    item_text
                )

                total_characters += len(
                    item_text
                )

            if total_characters >= self.max_characters:
                break

        return "\n\n".join(context_parts)