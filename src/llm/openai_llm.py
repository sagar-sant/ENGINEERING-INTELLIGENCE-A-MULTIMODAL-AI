import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


class OpenAILLM:
    """
    OpenAI-based language model adapter.

    Provides the interface expected by RAGPipeline:
        generate(messages)

    Messages should be in the standard chat format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ]
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        load_dotenv()

        self.model = model
        self.temperature = temperature

        resolved_api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Set it in the environment or in a .env file."
            )

        self.client = OpenAI(
            api_key=resolved_api_key
        )

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Generate a text response from chat messages.
        """

        if not messages:
            raise ValueError(
                "messages cannot be empty."
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        content = response.choices[0].message.content

        if content is None:
            return ""

        return content.strip()