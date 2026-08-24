from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class ModelDecision:
    """Result returned by the model router."""

    model: str
    mode: str
    reason: str


class ModelRouter:
    """
    Route user requests to the appropriate local Ollama model.

    Routing priorities:
        1. Actual image input -> vision
        2. Natural conversation -> Gemma 3
        3. Code generation -> Llama 3.1
        4. Quantitative/calculation problems -> DeepSeek-R1
        5. Complex technical reasoning -> DeepSeek-R1
        6. Current/web-research requests -> Qwen 3
        7. Simple factual/general questions -> Gemma 3
        8. General/technical text -> Qwen 3

    The router does not answer questions and does not calculate results.
    Deterministic mathematics is handled by the universal calculation engine.
    """

    # ------------------------------------------------------------------
    # Natural conversation
    # ------------------------------------------------------------------
    CONVERSATION_PATTERNS = (
        r"^(hi|hello|hey|hiya|howdy)[!.]?$",
        r"^(good morning|good afternoon|good evening|good night)[!.]?$",
        r"\bhow are you\b",
        r"\bhow's your day\b",
        r"\bwhat do you do\b",
        r"\bwhat is your job\b",
        r"\bwhat are your duties\b",
        r"\bwho are you\b",
        r"\bwhat is your name\b",
        r"\bwhat can you do\b",
        r"\bwhat do you like\b",
        r"\bthank(s| you)\b",
        r"^(bye|goodbye)[!.]?$",
        r"\bcan we talk\b",
        r"\bjust chat\b",
        r"\blet's chat\b",
        r"\bi['’]?m feeling\b",
        r"\bi am feeling\b",
        r"\bi feel\b",
        r"\bi want to\b",
        r"\bi like to\b",
        r"\bi enjoy\b",
        r"\bi['’]?m tired\b",
        r"\bi am tired\b",
        r"\bi['’]?m bored\b",
        r"\bi am bored\b",
        r"\btell me a short story\b",
        r"\btell me a story\b",
        r"\bmake me laugh\b",
        r"\bwhat should i do\b",
    )

    # ------------------------------------------------------------------
    # Programming / code generation
    # ------------------------------------------------------------------
    CODE_PATTERNS = (
        r"\bwrite (?:a|an|the)?\s*(?:python|java|c\+\+|c#|javascript|typescript|sql|r|matlab|bash|shell)\s+(?:program|script|function|code)\b",
        r"\bwrite (?:a|an|the)?\s*program\b",
        r"\bwrite (?:a|an|the)?\s*script\b",
        r"\bwrite (?:a|an|the)?\s*function\b",
        r"\bwrite (?:some )?code\b",
        r"\bcreate (?:a|an)?\s*(?:python|java|c\+\+|c#|javascript|typescript|sql|r|matlab|bash|shell)\b",
        r"\bimplement\b.*\b(?:function|class|program|algorithm|code)\b",
        r"\bdebug\b.*\b(?:code|program|script|function)\b",
        r"\bfix\b.*\b(?:code|program|script|function)\b",
        r"\brefactor\b.*\b(?:code|program|script)\b",
        r"\bgenerate\b.*\b(?:code|program|script|function)\b",
        r"\bexample of\b.*\b(?:code|python|program)\b",
        r"\bpython\s+(?:code|program|function|script)\b",
        r"\bsource code\b",
    )

    # ------------------------------------------------------------------
    # Explicit web/current-information requests
    # ------------------------------------------------------------------
    WEB_PATTERNS = (
        r"\bcurrent\s+(?:prime minister|president|price|status|events?|news|weather|exchange rate)\b",
        r"\btoday\b",
        r"\btonight\b",
        r"\blatest\b",
        r"\brecent\b",
        r"\bthis week\b",
        r"\bthis month\b",
        r"\bnews\b",
        r"\bwho is the current\b",
        r"\bwhat is the current\s+(?:prime minister|president|price|status|rate|weather)\b",
        r"\bwhat are the latest\b",
        r"\blook up\b",
        r"\bsearch the web\b",
        r"\bonline\b",
        r"\baccording to\b.*\b(?:latest|current|today)\b",
    )

    # ------------------------------------------------------------------
    # Calculation / mathematics
    # ------------------------------------------------------------------
    CALCULATION_PATTERNS = (
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bevaluate\b",
        r"\bsolve\b",
        r"\bfind\b.*\b(?:value|answer|result|root|roots)\b",
        r"\bdetermine\b.*\b(?:value|answer|result)\b",
        r"\bderive\b",
        r"\bintegrat(?:e|ion)\b",
        r"\bdifferentiat(?:e|ion)\b",
        r"\bderivative\b",
        r"\bantiderivative\b",
        r"\beigenvalue\b",
        r"\beigenvector\b",
        r"\bdeterminant\b",
        r"\bmatrix\b.*\b(?:multiply|inverse|solve|determinant)\b",
        r"\bmean\b",
        r"\baverage\b",
        r"\bmedian\b",
        r"\bvariance\b",
        r"\bstandard deviation\b",
        r"\bstandard error\b",
        r"\bregression\b.*\b(?:calculate|fit|equation|slope)\b",
        r"\bprobability\b.*\b(?:calculate|find|determine)\b",
        r"\bwhat is\b.*\b(?:square|cube|root|factorial|power)\b",
        r"\braise\b.*\bto the\b",
        r"\bhow much\b.*\b(?:cost|energy|power|current|voltage|force|stress)\b",
    )

    # Pure arithmetic expressions. These are useful as a router signal,
    # although DocumentService may intercept them with its local calculator.
    PURE_ARITHMETIC_PATTERNS = (
        r"^\s*[-+]?\d[\d,\s]*(?:[+\-*/%^().]\s*[-+]?\d[\d,\s]*)+\s*[?=]?\s*$",
        r"^\s*what\s+is\s+[-+]?\d[\d,\s]*(?:[+\-*/%^().]\s*[-+]?\d[\d,\s]*)+\s*\??\s*$",
        r"^\s*calculate\s+[-+]?\d[\d,\s]*(?:[+\-*/%^().]\s*[-+]?\d[\d,\s]*)+\s*\??\s*$",
    )

    # ------------------------------------------------------------------
    # Complex technical reasoning
    # ------------------------------------------------------------------
    COMPLEX_PATTERNS = (
        r"\banaly[sz]e\b",
        r"\bcompare\b",
        r"\bexplain\b",
        r"\bwhy\b",
        r"\bhow does\b",
        r"\bhow do\b",
        r"\brelationship\b",
        r"\bderive\b",
        r"\bdesign\b",
        r"\boptimi[sz]e\b",
        r"\btroubleshoot\b",
        r"\bdiagnos",
        r"\broot cause\b",
        r"\bstep[- ]by[- ]step\b",
        r"\btrade[- ]?off\b",
        r"\bjustify\b",
        r"\bassess\b",
        r"\bevaluate\b",
        r"\bdiscuss\b",
        r"\bexplain why\b",
    )

    # ------------------------------------------------------------------
    # Simple factual questions
    # ------------------------------------------------------------------
    SIMPLE_PATTERNS = (
        r"^what is\b",
        r"^what are\b",
        r"^who is\b",
        r"^where is\b",
        r"^when is\b",
        r"^define\b",
        r"^name\b",
        r"^list\b",
    )

    # ------------------------------------------------------------------
    # Visual-language signals are NOT sufficient to select vision.
    # Actual image input is required. These are retained only to improve
    # the reason string when an image really is attached.
    # ------------------------------------------------------------------
    VISUAL_PATTERNS = (
        r"\bimage\b",
        r"\bpicture\b",
        r"\bdiagram\b",
        r"\bschematic\b",
        r"\bdrawing\b",
        r"\bfigure\b",
        r"\bphoto\b",
        r"\bchart\b",
        r"\bplot\b",
        r"\bshown\b",
        r"\bvisible\b",
    )

    def __init__(
        self,
        text_model: str = "phi3:mini",
        advanced_text_model: str = "qwen3:8b",
        reasoning_model: str = "deepseek-r1:7b",
        gemma3_model: str = "gemma3:4b",
        llama_model: str = "llama3.1:8b",
        mistral_model: str = "mistral:7b",
        gemma2_model: str = "gemma2:9b",
        vision_model: str = "qwen2.5vl:7b",
        advanced_vision_model: str = "qwen3-vl:8b",
    ) -> None:
        self.text_model = text_model
        self.advanced_text_model = advanced_text_model
        self.reasoning_model = reasoning_model
        self.gemma3_model = gemma3_model
        self.llama_model = llama_model
        self.mistral_model = mistral_model
        self.gemma2_model = gemma2_model
        self.vision_model = vision_model
        self.advanced_vision_model = advanced_vision_model

    @staticmethod
    def _matches(question: str, patterns: tuple[str, ...]) -> bool:
        return any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in patterns
        )

    @staticmethod
    def _is_pure_arithmetic(question: str) -> bool:
        return any(
            re.fullmatch(pattern, question, re.IGNORECASE)
            for pattern in ModelRouter.PURE_ARITHMETIC_PATTERNS
        )

    @staticmethod
    def _looks_like_code(question: str) -> bool:
        return ModelRouter._matches(question, ModelRouter.CODE_PATTERNS)

    @staticmethod
    def _looks_like_calculation(question: str) -> bool:
        return (
            ModelRouter._is_pure_arithmetic(question)
            or ModelRouter._matches(
                question,
                ModelRouter.CALCULATION_PATTERNS,
            )
        )

    @staticmethod
    def _looks_like_web_request(question: str) -> bool:
        return ModelRouter._matches(question, ModelRouter.WEB_PATTERNS)

    @staticmethod
    def _looks_like_conversation(question: str) -> bool:
        return ModelRouter._matches(
            question,
            ModelRouter.CONVERSATION_PATTERNS,
        )

    @staticmethod
    def _looks_complex(question: str) -> bool:
        return ModelRouter._matches(
            question,
            ModelRouter.COMPLEX_PATTERNS,
        )

    def route(
        self,
        question: str,
        has_images: bool = False,
    ) -> ModelDecision:
        q = str(question or "").strip()

        # --------------------------------------------------------------
        # Empty input
        # --------------------------------------------------------------
        if not q:
            return ModelDecision(
                self.gemma3_model,
                "conversation",
                "Empty/neutral request; Gemma 3 is suitable for a natural response.",
            )

        # --------------------------------------------------------------
        # 1. REAL IMAGE INPUT HAS HIGHEST PRIORITY.
        #
        # A text question containing words such as "schematic" or "image"
        # must NOT activate vision by itself. The caller must actually
        # provide an image.
        # --------------------------------------------------------------
        if has_images:
            if self._looks_complex(q) or self._looks_like_calculation(q):
                return ModelDecision(
                    self.advanced_vision_model,
                    "vision_advanced",
                    "An image is attached and the request requires advanced visual/technical reasoning.",
                )

            return ModelDecision(
                self.vision_model,
                "vision",
                "An actual image is attached; visual evidence is required.",
            )

        # --------------------------------------------------------------
        # 2. NATURAL CONVERSATION.
        #
        # Keep this before generic "why/how/explain" and before technical
        # routing so greetings and casual discussion remain natural.
        # --------------------------------------------------------------
        if self._looks_like_conversation(q):
            return ModelDecision(
                self.gemma3_model,
                "conversation",
                "Natural conversational request; Gemma 3 provides the preferred conversational behavior.",
            )

        # --------------------------------------------------------------
        # 3. CODE GENERATION.
        #
        # Code requests must not be classified as calculations merely
        # because they contain words such as average, mean, calculate, etc.
        # --------------------------------------------------------------
        if self._looks_like_code(q):
            return ModelDecision(
                self.llama_model,
                "code",
                "Programming/code-generation request; Llama 3.1 is selected for code generation.",
            )

        # --------------------------------------------------------------
        # 4. WEB / CURRENT INFORMATION.
        #
        # Do this before ordinary technical reasoning. The actual web
        # research implementation can decide whether retrieval is needed.
        # --------------------------------------------------------------
        if self._looks_like_calculation(q):
            return ModelDecision(
                self.reasoning_model,
                "calculation",
                "Quantitative problem; deterministic calculation should be attempted, with DeepSeek-R1 available for planning/reasoning.",
            )

        if self._looks_like_web_request(q):
            return ModelDecision(
                self.advanced_text_model,
                "web_research",
                "The request asks for current, recent, or web-based information.",
            )

        # --------------------------------------------------------------
        # 5. COMPLEX TECHNICAL REASONING.
        #
        # All quantitative questions are sent through the calculation
        # pipeline. DocumentService/universal calculator may intercept
        # deterministic cases before an LLM is invoked.
        # --------------------------------------------------------------
        if self._looks_like_calculation(q):
            return ModelDecision(
                self.reasoning_model,
                "calculation",
                "Quantitative problem; deterministic calculation should be attempted, with DeepSeek-R1 available for planning/reasoning.",
            )

        # --------------------------------------------------------------
        # 6. COMPLEX TECHNICAL REASONING.
        # --------------------------------------------------------------
        if self._looks_complex(q):
            return ModelDecision(
                self.reasoning_model,
                "reasoning",
                "Complex technical reasoning request; DeepSeek-R1 is selected.",
            )

        # --------------------------------------------------------------
        # 6. SIMPLE FACTUAL QUESTIONS.
        #
        # Gemma 3 is the preferred lightweight natural answer model.
        # --------------------------------------------------------------
        if self._matches(q, self.SIMPLE_PATTERNS):
            return ModelDecision(
                self.gemma3_model,
                "text_compact",
                "Simple factual/general question; Gemma 3 is suitable.",
            )

        # --------------------------------------------------------------
        # 7. DEFAULT TECHNICAL/GENERAL TEXT.
        # --------------------------------------------------------------
        return ModelDecision(
            self.advanced_text_model,
            "text_advanced",
            "General or technical request requiring stronger text generation.",
        )


__all__ = ["ModelDecision", "ModelRouter"]