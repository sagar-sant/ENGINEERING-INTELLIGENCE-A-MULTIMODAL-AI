"""
Engineering Intelligence - LangGraph orchestration layer.

This module provides a safe orchestration layer around the existing
application components.

IMPORTANT:
    This module does not replace DocumentService, ModelRouter, calculator.py,
    CalculationPlanner, retrieval, vision, or web research.

    It coordinates them.

Architecture:

    USER
      |
      v
    classify
      |
      +--> conversation
      |
      +--> calculation
      |
      +--> document
      |
      +--> visual
      |
      +--> web
      |
      +--> engineering
              |
              v
           response

The existing application remains the source of truth for actual execution.
"""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


RouteName = Literal[
    "conversation",
    "calculation",
    "document",
    "visual",
    "web",
    "engineering",
]


class EngineeringState(TypedDict, total=False):
    """State carried through the engineering workflow."""

    question: str

    route: RouteName | str | None
    route_reason: str | None

    answer: str | None

    metadata: dict[str, Any]
    sources: list[dict[str, Any]]

    calculation: dict[str, Any] | None

    document_context: bool
    visual_context: bool
    web_research_used: bool

    error: str | None


class EngineeringGraph:
    """
    LangGraph orchestration wrapper.

    The graph is intentionally lightweight. Heavy operations remain inside
    the existing application so that introducing LangGraph does not alter
    proven behavior.
    """

    def __init__(
        self,
        document_service: Any | None = None,
    ) -> None:
        self.document_service = document_service
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # GRAPH CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_graph(self):
        workflow = StateGraph(EngineeringState)

        workflow.add_node("classify", self._classify)
        workflow.add_node("conversation", self._conversation)
        workflow.add_node("calculation", self._calculation)
        workflow.add_node("document", self._document)
        workflow.add_node("visual", self._visual)
        workflow.add_node("web", self._web)
        workflow.add_node("engineering", self._engineering)
        workflow.add_node("finalize", self._finalize)

        workflow.add_edge(START, "classify")

        workflow.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {
                "conversation": "conversation",
                "calculation": "calculation",
                "document": "document",
                "visual": "visual",
                "web": "web",
                "engineering": "engineering",
            },
        )

        for node in (
            "conversation",
            "calculation",
            "document",
            "visual",
            "web",
            "engineering",
        ):
            workflow.add_edge(node, "finalize")

        workflow.add_edge("finalize", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------

    def _classify(
        self,
        state: EngineeringState,
    ) -> dict[str, Any]:
        question = str(state.get("question", "")).strip()

        if not question:
            return {
                "route": "conversation",
                "route_reason": "Empty or conversational input.",
            }

        service = self.document_service

        # Prefer the application's existing routing logic whenever possible.
        if service is not None:
            try:
                if hasattr(service, "_is_calculation_question"):
                    if service._is_calculation_question(question):
                        return {
                            "route": "calculation",
                            "route_reason": (
                                "Existing deterministic calculation "
                                "detector identified a calculation."
                            ),
                        }

                if hasattr(service, "_is_visual_question"):
                    if service._is_visual_question(question):
                        return {
                            "route": "visual",
                            "route_reason": (
                                "Existing visual-question detector "
                                "identified an image-related request."
                            ),
                        }

                if hasattr(service, "_is_document_question"):
                    if service._is_document_question(question):
                        return {
                            "route": "document",
                            "route_reason": (
                                "Existing document-question detector "
                                "identified a document request."
                            ),
                        }

                if hasattr(service, "_needs_web"):
                    if service._needs_web(question):
                        return {
                            "route": "web",
                            "route_reason": (
                                "Existing web-research detector "
                                "identified a current-information request."
                            ),
                        }
            except Exception:
                # Classification must never break the application.
                pass

        q = question.lower()

        # Visual language.
        visual_patterns = (
            r"\bimage\b",
            r"\bschematic\b",
            r"\bdrawing\b",
            r"\bdiagram\b",
            r"\bfigure\b",
            r"\bcircuit\b",
            r"\bphoto\b",
            r"\bpicture\b",
            r"\bvisible\b",
            r"\bshown\b",
            r"\bshown in\b",
        )

        if any(re.search(pattern, q) for pattern in visual_patterns):
            return {
                "route": "visual",
                "route_reason": "Visual/engineering-image terminology detected.",
            }

        # Document language.
        document_patterns = (
            r"\bdocument\b",
            r"\bmanual\b",
            r"\bdatasheet\b",
            r"\breport\b",
            r"\bpdf\b",
            r"\bpage\b",
            r"\bsection\b",
            r"\bchapter\b",
            r"\bdocumentation\b",
        )

        if any(re.search(pattern, q) for pattern in document_patterns):
            return {
                "route": "document",
                "route_reason": "Document-related terminology detected.",
            }

        # Current / web-dependent information.
        web_patterns = (
            r"\bcurrent\b",
            r"\blatest\b",
            r"\btoday\b",
            r"\bnow\b",
            r"\brecent\b",
            r"\b2026\b",
            r"\bnews\b",
            r"\bupdated\b",
            r"\bnewest\b",
        )

        if any(re.search(pattern, q) for pattern in web_patterns):
            return {
                "route": "web",
                "route_reason": "Potentially time-sensitive information detected.",
            }

        # Mathematical / engineering calculation language.
        calculation_patterns = (
            r"\bcalculate\b",
            r"\bcompute\b",
            r"\bsolve\b",
            r"\bfind\b",
            r"\bevaluate\b",
            r"\bhow much\b",
            r"\bhow many\b",
            r"\bsquare\b",
            r"\bsquared\b",
            r"\bcube\b",
            r"\bcubed\b",
            r"\bpower\b",
            r"\braised\b",
            r"\bsquare root\b",
            r"\bderivative\b",
            r"\bintegral\b",
            r"\bintegrate\b",
            r"\bdifferentiate\b",
        )

        if any(re.search(pattern, q) for pattern in calculation_patterns):
            return {
                "route": "calculation",
                "route_reason": "Calculation terminology detected.",
            }

        # Very simple conversational requests should remain lightweight.
        conversation_patterns = (
            r"^hello\b",
            r"^hi\b",
            r"^hey\b",
            r"\bhow are you\b",
            r"\bwhat do you do\b",
            r"\bwhat can you help\b",
            r"\bwho are you\b",
            r"\bwhat is your job\b",
            r"\btell me a story\b",
            r"\btell me a short story\b",
        )

        if any(re.search(pattern, q) for pattern in conversation_patterns):
            return {
                "route": "conversation",
                "route_reason": "Natural conversational request detected.",
            }

        return {
            "route": "engineering",
            "route_reason": (
                "General engineering/science/technical request "
                "requiring existing application routing."
            ),
        }

    # ------------------------------------------------------------------
    # CONDITIONAL ROUTING
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_classification(
        state: EngineeringState,
    ) -> RouteName:
        route = state.get("route")

        valid_routes = {
            "conversation",
            "calculation",
            "document",
            "visual",
            "web",
            "engineering",
        }

        if route in valid_routes:
            return route  # type: ignore[return-value]

        return "engineering"

    # ------------------------------------------------------------------
    # EXECUTION NODES
    # ------------------------------------------------------------------

    def _conversation(
        self,
        state: EngineeringState,
    ) -> dict[str, Any]:
        return self._delegate_to_service(state)

    def _calculation(
        self,
        state: EngineeringState,
    ) -> dict[str, Any]:
        service = self.document_service

        if service is None:
            return self._delegate_to_service(state)

        try:
            calculation = None

            if hasattr(service, "_try_universal_calculation"):
                calculation = service._try_universal_calculation(
                    state["question"]
                )

            # Let DocumentService perform its normal verified explanation
            # path. This prevents duplicate calculation logic.
            result = service.ask(state["question"])

            return {
                "answer": result.answer,
                "sources": result.sources,
                "metadata": result.metadata,
                "calculation": calculation,
            }

        except Exception as exc:
            return {
                "error": str(exc),
                "route_reason": (
                    "Calculation node failed; existing service should "
                    "remain available as fallback."
                ),
            }

    def _document(
        self,
        state: EngineeringState,
    ) -> dict[str, Any]:
        return self._delegate_to_service(state)

    def _visual(
        self,
        state: EngineeringState,
    ) -> dict[str, Any]:
        return self._delegate_to_service(state)

    def _web(
        self,
        state: EngineeringState,
    ) -> dict[str, Any]:
        return self._delegate_to_service(state)

    def _engineering(
        self,
        state: EngineeringState,
    ) -> dict[str, Any]:
        return self._delegate_to_service(state)

    # ------------------------------------------------------------------
    # SERVICE DELEGATION
    # ------------------------------------------------------------------

    def _delegate_to_service(
        self,
        state: EngineeringState,
    ) -> dict[str, Any]:
        service = self.document_service

        if service is None:
            return {
                "answer": (
                    "Engineering graph is initialized, but no "
                    "DocumentService is connected yet."
                ),
                "metadata": {
                    "model": None,
                    "mode": "graph_test",
                    "graph": True,
                },
                "sources": [],
            }

        try:
            result = service.ask(state["question"])

            metadata = dict(result.metadata or {})
            metadata["langgraph"] = True
            metadata["graph_route"] = state.get("route")

            return {
                "answer": result.answer,
                "sources": result.sources,
                "metadata": metadata,
            }

        except Exception as exc:
            return {
                "error": str(exc),
                "answer": None,
                "sources": [],
                "metadata": {
                    "langgraph": True,
                    "graph_route": state.get("route"),
                },
            }

    # ------------------------------------------------------------------
    # FINALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize(
        state: EngineeringState,
    ) -> dict[str, Any]:
        metadata = dict(state.get("metadata") or {})

        metadata["langgraph"] = True
        metadata["graph_route"] = state.get("route")
        metadata["graph_route_reason"] = state.get("route_reason")

        if state.get("error"):
            metadata["graph_error"] = state["error"]

        return {
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def invoke(
        self,
        question: str,
    ) -> EngineeringState:
        """
        Execute the LangGraph workflow.

        The returned state contains the final answer, metadata, sources,
        routing information, and calculation information where applicable.
        """
        question = str(question).strip()

        return self.graph.invoke(
            {
                "question": question,
                "route": None,
                "route_reason": None,
                "answer": None,
                "metadata": {},
                "sources": [],
                "calculation": None,
                "document_context": False,
                "visual_context": False,
                "web_research_used": False,
                "error": None,
            }
        )

    def get_graph(self):
        """Return the compiled LangGraph application."""
        return self.graph


def create_engineering_graph(
    document_service: Any | None = None,
) -> EngineeringGraph:
    """Create the Engineering Intelligence orchestration graph."""
    return EngineeringGraph(
        document_service=document_service,
    )