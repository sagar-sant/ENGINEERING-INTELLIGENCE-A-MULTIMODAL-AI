from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from PIL import Image

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama


class OllamaLLM:
    """
    LangChain-backed Ollama adapter.

    This class intentionally preserves the application's existing
    OllamaLLM interface so DocumentService does not need to be rewritten.

    Supports:
        - text generation
        - conversation history
        - system prompts
        - multimodal/image generation
        - Ollama context control
        - model-specific output limits
        - DeepSeek/Qwen thinking control
        - configurable timeout
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        timeout: int = 180,
        max_image_dimension: int = 1400,
        jpeg_quality: int = 80,
        num_predict: int = 256,
        num_ctx: int = 8192,
        think: bool = False,
        keep_alive: str = "5m",
    ) -> None:

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality

        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.think = think
        self.keep_alive = keep_alive

        # LangChain ChatOllama
        #
        # client_kwargs controls the HTTP timeout used by the
        # underlying Ollama client.
        self.llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.num_predict,
            num_ctx=self.num_ctx,
            think=self.think,
            keep_alive=self.keep_alive,
            client_kwargs={
                "timeout": self.timeout,
            },
        )

    # =========================================================
    # MESSAGE CONVERSION
    # =========================================================

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> list[Any]:

        converted: list[Any] = []

        for message in messages:
            role = str(
                message.get("role", "user")
            ).lower()

            content = message.get(
                "content",
                "",
            )

            if role == "system":
                converted.append(
                    SystemMessage(
                        content=content
                    )
                )

            elif role == "assistant":
                converted.append(
                    AIMessage(
                        content=content
                    )
                )

            else:
                converted.append(
                    HumanMessage(
                        content=content
                    )
                )

        return converted

    # =========================================================
    # RESPONSE EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_content(response: Any) -> str:
        """
        Convert LangChain's AIMessage content into plain text.

        Some multimodal/model combinations can return content as
        a list of blocks instead of a simple string.
        """

        content = getattr(
            response,
            "content",
            response,
        )

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue

                if isinstance(block, dict):
                    text = block.get("text")

                    if text:
                        parts.append(
                            str(text)
                        )

            return "\n".join(parts).strip()

        return str(content).strip()

    # =========================================================
    # THINKING CLEANUP
    # =========================================================

    @staticmethod
    def _remove_thinking(
        text: str,
    ) -> str:

        if not text:
            return ""

        result = text.strip()

        # Remove complete <think>...</think> blocks.
        while True:

            lower = result.lower()

            start = lower.find(
                "<think>"
            )

            if start == -1:
                break

            end = lower.find(
                "</think>",
                start + len("<think>"),
            )

            if end == -1:
                # Remove unmatched leading think tag.
                result = result[
                    start + len("<think>"):
                ]
                break

            end += len("</think>")

            result = (
                result[:start]
                + result[end:]
            )

        return result.strip()

    # =========================================================
    # INTERNAL INVOCATION
    # =========================================================

    def _invoke(
        self,
        messages: list[dict[str, Any]],
    ) -> str:

        if not messages:
            raise ValueError(
                "messages cannot be empty."
            )

        converted = self._convert_messages(
            messages
        )

        try:
            response = self.llm.invoke(
                converted
            )

        except Exception as exc:
            raise RuntimeError(
                f"Ollama/LangChain generation failed "
                f"for model={self.model}: {exc}"
            ) from exc

        answer = self._extract_content(
            response
        )

        answer = self._remove_thinking(
            answer
        )

        # DeepSeek/Qwen can occasionally consume their
        # generation budget in reasoning. Give a useful
        # diagnostic instead of silently returning nothing.
        if not answer:

            raise RuntimeError(
                "Ollama generated no final answer. "
                f"Model={self.model}. "
                "The model may have exhausted its generation "
                "budget during reasoning."
            )

        return answer

    # =========================================================
    # TEXT GENERATION
    # =========================================================

    def generate(
        self,
        messages: list[dict[str, Any]],
    ) -> str:

        return self._invoke(
            messages
        )

    # =========================================================
    # IMAGE PREPARATION
    # =========================================================

    def _prepare_image(
        self,
        image_path: str,
    ) -> tuple[Path, str]:

        source_path = Path(
            image_path
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        try:
            image = Image.open(
                source_path
            )

            image.load()

        except Exception as exc:
            raise RuntimeError(
                f"Could not open image: {image_path}"
            ) from exc

        try:
            if image.mode != "RGB":
                image = image.convert(
                    "RGB"
                )

            width, height = image.size

            largest_dimension = max(
                width,
                height,
            )

            if (
                largest_dimension
                > self.max_image_dimension
            ):

                scale = (
                    self.max_image_dimension
                    / largest_dimension
                )

                new_width = max(
                    1,
                    int(width * scale),
                )

                new_height = max(
                    1,
                    int(height * scale),
                )

                image = image.resize(
                    (
                        new_width,
                        new_height,
                    ),
                    Image.Resampling.LANCZOS,
                )

            temporary_path = (
                Path.cwd()
                / (
                    ".ollama_tmp_"
                    + source_path.stem
                    + ".jpg"
                )
            )

            image.save(
                temporary_path,
                format="JPEG",
                quality=self.jpeg_quality,
                optimize=True,
            )

            image_bytes = (
                temporary_path.read_bytes()
            )

            encoded = base64.b64encode(
                image_bytes
            ).decode("utf-8")

            return temporary_path, encoded

        finally:
            try:
                image.close()
            except Exception:
                pass

    # =========================================================
    # MULTIMODAL GENERATION
    # =========================================================

    def generate_with_images(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[str],
    ) -> str:

        if not messages:
            raise ValueError(
                "messages cannot be empty."
            )

        if not image_paths:
            return self.generate(
                messages
            )

        temporary_files: list[Path] = []

        try:
            encoded_images: list[str] = []

            for image_path in image_paths:

                temporary_path, encoded = (
                    self._prepare_image(
                        image_path
                    )
                )

                temporary_files.append(
                    temporary_path
                )

                encoded_images.append(
                    encoded
                )

            converted: list[Any] = []

            for message in messages:

                role = str(
                    message.get(
                        "role",
                        "user",
                    )
                ).lower()

                content = message.get(
                    "content",
                    "",
                )

                # Images belong on the user message.
                if (
                    role == "user"
                    and encoded_images
                ):

                    content_blocks: list[
                        dict[str, Any]
                    ] = [
                        {
                            "type": "text",
                            "text": str(
                                content
                            ),
                        }
                    ]

                    for encoded in encoded_images:

                        content_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": (
                                    "data:image/jpeg;base64,"
                                    + encoded
                                ),
                            }
                        )

                    converted.append(
                        HumanMessage(
                            content=content_blocks
                        )
                    )

                elif role == "system":

                    converted.append(
                        SystemMessage(
                            content=content
                        )
                    )

                elif role == "assistant":

                    converted.append(
                        AIMessage(
                            content=content
                        )
                    )

                else:

                    converted.append(
                        HumanMessage(
                            content=content
                        )
                    )

            try:

                response = self.llm.invoke(
                    converted
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Ollama/LangChain vision generation "
                    f"failed for model={self.model}: {exc}"
                ) from exc

            answer = self._extract_content(
                response
            )

            answer = self._remove_thinking(
                answer
            )

            if not answer:
                raise RuntimeError(
                    "Ollama generated no final visual answer. "
                    f"Model={self.model}."
                )

            return answer

        finally:

            for temporary_path in (
                temporary_files
            ):

                try:
                    temporary_path.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

    # =========================================================
    # CONVENIENCE METHODS
    # =========================================================

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:

        messages: list[
            dict[str, Any]
        ] = []

        if system_prompt:

            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return self.generate(
            messages
        )

    def generate_vision(
        self,
        prompt: str,
        image_paths: list[str],
        system_prompt: str | None = None,
    ) -> str:

        messages: list[
            dict[str, Any]
        ] = []

        if system_prompt:

            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return self.generate_with_images(
            messages,
            image_paths,
        )

    # =========================================================
    # REPRESENTATION
    # =========================================================

    def __repr__(self) -> str:

        return (
            "OllamaLLM("
            f"model={self.model!r}, "
            f"num_ctx={self.num_ctx}, "
            f"num_predict={self.num_predict}, "
            f"think={self.think}, "
            "backend='langchain-ollama'"
            ")"
        )