from __future__ import annotations

from pathlib import Path

from src.rag.context_builder import ContextBuilder
from src.rag.prompt_builder import PromptBuilder
from src.rag.rag_models import RAGResponse


class RAGPipeline:
    """
    Coordinates retrieval, context construction, prompt creation,
    and language-model generation.

    Supports:
    - text-only generation
    - multimodal generation
    - selecting a specific LLM per request
    - explicitly controlling whether images are sent
    """

    def __init__(
        self,
        retriever,
        llm=None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:

        self.retriever = retriever
        self.llm = llm

        self.context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder()
        )

        self.prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else PromptBuilder()
        )

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):
        """Retrieve relevant document chunks."""

        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

    # =========================================================
    # PROMPT
    # =========================================================

    def build_prompt(
        self,
        query: str,
        top_k: int = 5,
    ):
        """Retrieve context and build an LLM prompt."""

        retrieval_response = self.retrieve(
            query=query,
            top_k=top_k,
        )

        context = self.context_builder.build(
            retrieval_response.results
        )

        prompt = self.prompt_builder.build(
            query=query,
            context=context,
        )

        return retrieval_response, prompt

    # =========================================================
    # VISUAL PATHS
    # =========================================================

    @staticmethod
    def _get_visual_paths(
        results,
    ) -> list[str]:
        """
        Extract valid rendered page images from retrieved results.

        Only chunks explicitly marked as visual are considered.
        """

        paths: list[str] = []
        seen: set[str] = set()

        for result in results:

            metadata = getattr(
                result,
                "metadata",
                None,
            ) or {}

            if metadata.get(
                "has_visual_context"
            ) is not True:
                continue

            rendered_path = metadata.get(
                "rendered_page_path"
            )

            if not rendered_path:
                continue

            path = Path(
                str(rendered_path)
            )

            if not path.exists():
                continue

            normalized_path = str(
                path.resolve()
            )

            if normalized_path in seen:
                continue

            seen.add(
                normalized_path
            )

            paths.append(
                normalized_path
            )

        return paths

    # =========================================================
    # GENERATE
    # =========================================================

    def generate(
        self,
        query: str,
        top_k: int = 5,
        llm=None,
        use_images: bool = False,
    ) -> RAGResponse:
        """
        Run retrieval and generate an answer.

        Parameters
        ----------
        query:
            User's question.

        top_k:
            Number of document chunks to retrieve.

        llm:
            Optional LLM override. This allows the application
            router to select the correct model for each question.

        use_images:
            If True, retrieved page images are sent to the
            selected multimodal model.

            IMPORTANT:
            Text models must use use_images=False.
        """

        # -----------------------------------------------------
        # Retrieve context and construct prompt.
        # -----------------------------------------------------

        retrieval_response, prompt = (
            self.build_prompt(
                query=query,
                top_k=top_k,
            )
        )

        # -----------------------------------------------------
        # Select LLM.
        # -----------------------------------------------------

        selected_llm = (
            llm
            if llm is not None
            else self.llm
        )

        # -----------------------------------------------------
        # Determine visual images.
        #
        # IMPORTANT:
        # Only collect/send images when explicitly requested.
        # This prevents text-only models from receiving
        # multimodal payloads.
        # -----------------------------------------------------

        visual_paths: list[str] = []

        if use_images:
            visual_paths = (
                self._get_visual_paths(
                    retrieval_response.results
                )
            )

        # -----------------------------------------------------
        # No LLM.
        # -----------------------------------------------------

        if selected_llm is None:

            answer = (
                "No language model is configured. "
                "Retrieved document context is available, "
                "but an answer could not be generated."
            )

        # -----------------------------------------------------
        # Multimodal generation.
        # -----------------------------------------------------

        elif (
            use_images
            and visual_paths
            and hasattr(
                selected_llm,
                "generate_with_images",
            )
        ):

            answer = (
                selected_llm.generate_with_images(
                    prompt.to_messages(),
                    visual_paths,
                )
            )

        # -----------------------------------------------------
        # Normal text generation.
        # -----------------------------------------------------

        elif hasattr(
            selected_llm,
            "generate",
        ):

            answer = selected_llm.generate(
                prompt.to_messages()
            )

        # -----------------------------------------------------
        # Alternative LLM interface.
        # -----------------------------------------------------

        elif hasattr(
            selected_llm,
            "invoke",
        ):

            answer = selected_llm.invoke(
                prompt.to_messages()
            )

        else:

            raise TypeError(
                "The configured LLM must provide "
                "generate(), invoke(), or "
                "generate_with_images()."
            )

        # -----------------------------------------------------
        # Normalize response.
        # -----------------------------------------------------

        if not isinstance(
            answer,
            str,
        ):

            if hasattr(
                answer,
                "content",
            ):

                answer = answer.content

            else:

                answer = str(answer)

        answer = answer.strip()

        # -----------------------------------------------------
        # Return response.
        # -----------------------------------------------------

        return RAGResponse(
            query=query,
            answer=answer,
            retrieval_results=(
                retrieval_response.results
            ),
            metadata={
                "retrieval_method": (
                    retrieval_response.retrieval_method
                ),
                "retrieved_count": (
                    retrieval_response.total_results
                ),
                "multimodal": bool(
                    use_images and visual_paths
                ),
                "visual_image_count": (
                    len(visual_paths)
                    if use_images
                    else 0
                ),
                "visual_image_paths": (
                    visual_paths
                    if use_images
                    else []
                ),
                "model": getattr(
                    selected_llm,
                    "model",
                    None,
                ),
            },
        )