from dataclasses import dataclass


@dataclass
class RAGPrompt:
    """Represents a prompt prepared for a retrieval-augmented LLM."""

    system_prompt: str
    user_prompt: str

    def to_messages(self) -> list[dict[str, str]]:
        """Return the prompt in chat-message format."""
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self.user_prompt,
            },
        ]


class PromptBuilder:
    """
    Builds prompts that constrain an LLM to retrieved document context.

    The prompt distinguishes between:
    - extracted document text
    - visual information contained in supplied page images
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an engineering document assistant. "
        "Answer the user's question using only the supplied "
        "engineering document context and supplied document images. "
        "Do not invent specifications, procedures, values, labels, "
        "components, or other engineering details. "
        "When a page image is supplied, inspect the image directly "
        "when answering questions about visual information. "
        "For visual questions, treat what is actually visible in "
        "the image as the primary evidence. "
        "Distinguish clearly between information visible in the "
        "image and information stated only in the extracted text. "
        "If the supplied context and images do not contain enough "
        "information to answer the question, say that the "
        "information is not available in the provided document "
        "context."
    )

    def __init__(
        self,
        system_prompt: str | None = None,
    ) -> None:
        self.system_prompt = (
            system_prompt.strip()
            if system_prompt and system_prompt.strip()
            else self.DEFAULT_SYSTEM_PROMPT
        )

    def build(
        self,
        query: str,
        context: str,
    ) -> RAGPrompt:
        """
        Build a retrieval-augmented prompt from a query and context.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        formatted_context = (
            context.strip()
            if context.strip()
            else "[No relevant document context was retrieved.]"
        )

        user_prompt = (
            "Use the following engineering document context "
            "to answer the question.\n\n"
            "DOCUMENT TEXT CONTEXT:\n"
            "----------------------\n"
            f"{formatted_context}\n"
            "----------------------\n\n"
            "VISUAL INSPECTION INSTRUCTIONS:\n"
            "-------------------------------\n"
            "If one or more document page images are supplied with "
            "this request, inspect those images directly.\n\n"
            "For questions about figures, diagrams, tables, labels, "
            "connectors, components, ports, symbols, arrows, "
            "callouts, captions, layout, or other visual information, "
            "use the actual image as the primary evidence.\n\n"
            "Do not assume that an item is visible merely because "
            "it is mentioned in the extracted document text.\n\n"
            "If the image and extracted text disagree about what is "
            "visually present, describe only what can actually be "
            "verified from the supplied image.\n"
            "-------------------------------\n\n"
            f"QUESTION:\n{query.strip()}"
        )

        return RAGPrompt(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )