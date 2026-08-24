from __future__ import annotations

"""
Universal Calculation Planner

Converts a natural-language quantitative problem into a compact, structured
mathematical plan using a local Ollama model. The plan is then executed by the
deterministic universal calculator.

The planner is NOT the calculator:
    LLM -> equations/variables/operation types
    SymPy/NumPy/SciPy -> actual numerical/symbolic result

The planner never asks the model for hidden chain-of-thought.
"""

import json
import re
from typing import Any

from src.tools.calculator import execute_plan


class CalculationPlanner:
    def __init__(
        self,
        llm,
        max_problem_chars: int = 12000,
    ) -> None:
        self.llm = llm
        self.max_problem_chars = max_problem_chars

    @staticmethod
    def _clean_json(text: str) -> str:
        text = str(text or "").strip()

        # Remove common markdown fences.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

        # Prefer the first complete JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start:end + 1]

        return text

    @staticmethod
    def _validate_plan(plan: dict[str, Any]) -> None:
        if not isinstance(plan, dict):
            raise ValueError("Planner did not return a JSON object.")

        plan_type = str(plan.get("type", "")).strip().lower()

        allowed = {
            "expression",
            "equation",
            "system",
            "differentiate",
            "integrate",
            "limit",
            "series",
            "matrix",
            "linear_system",
            "statistics",
            "numpy_matrix",
            "scipy_integral",
            "scipy_root",
            "unit",
        }

        if plan_type not in allowed:
            raise ValueError(
                f"Unsupported calculation plan type: {plan_type}"
            )

        # Required fields by operation.
        required = {
            "expression": ("expression",),
            "equation": ("equation", "variable"),
            "system": ("equations", "variables"),
            "differentiate": ("expression", "variable"),
            "integrate": ("expression", "variable"),
            "limit": ("expression", "variable", "point"),
            "series": ("expression", "variable"),
            "matrix": ("matrix", "operation"),
            "linear_system": ("matrix", "rhs"),
            "statistics": ("values",),
            "numpy_matrix": ("matrix", "operation"),
            "scipy_integral": (
                "function",
                "variable",
                "lower",
                "upper",
            ),
            "scipy_root": (
                "function",
                "variable",
                "initial_guess",
            ),
            "unit": (
                "value",
                "from_unit",
                "to_unit",
            ),
        }

        for key in required[plan_type]:
            if key not in plan:
                raise ValueError(
                    f"Calculation plan missing required field: {key}"
                )

        # Guard against absurdly large generated plans.
        if len(json.dumps(plan, default=str)) > 50000:
            raise ValueError("Calculation plan is too large.")

    def _planner_prompt(self, question: str) -> str:
        problem = question[: self.max_problem_chars]

        return f"""
You are a mathematical problem-to-equation planner for an engineering AI.

Convert the user's quantitative problem into ONE deterministic calculation
plan that can be executed by SymPy/NumPy/SciPy.

You are NOT being asked to solve the numerical values yourself.
Do NOT provide chain-of-thought or hidden reasoning.

Your job:
1. Extract known variables and units.
2. Choose equations/operations needed to solve the problem.
3. Express the mathematics in a machine-executable form.
4. Include all dependent equations needed for multi-step problems.
5. Prefer exact symbolic expressions where possible.
6. Preserve units by converting them explicitly to compatible base units.
7. Never invent missing numerical values.
8. If a required value is genuinely missing, return:
   {{"type":"unsupported","reason":"missing value: ..."}}

Supported plan types:
- expression
- equation
- system
- differentiate
- integrate
- limit
- series
- matrix
- linear_system
- statistics
- numpy_matrix
- scipy_integral
- scipy_root
- unit

For a multi-step engineering problem, use "system" only when the equations
form a simultaneous system. Otherwise use "expression" with a single final
expression or use a sequence only if your application supports dependencies.
The safest general approach for multiple dependent calculations is a symbolic
expression containing all supplied variables.

For expressions, variables are supplied separately:
{{
  "type": "expression",
  "expression": "formula using variables",
  "variables": {{"x": 2, "y": 3}}
}}

For an equation:
{{
  "type": "equation",
  "equation": "x^2 - 5*x + 6 = 0",
  "variable": "x"
}}

For a system:
{{
  "type": "system",
  "equations": ["x+y=10", "x-y=2"],
  "variables": ["x","y"]
}}

Return ONLY valid JSON.

USER PROBLEM:
{problem}
""".strip()

    def plan(self, question: str) -> dict[str, Any]:
        if not str(question or "").strip():
            raise ValueError("Question cannot be empty.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise mathematical planner. "
                    "Return only valid JSON. Never reveal hidden reasoning."
                ),
            },
            {
                "role": "user",
                "content": self._planner_prompt(question),
            },
        ]

        raw = self.llm.generate(messages)
        cleaned = self._clean_json(raw)

        plan = json.loads(cleaned)

        if str(plan.get("type", "")).lower() == "unsupported":
            raise ValueError(
                str(plan.get("reason", "Problem is unsupported."))
            )

        self._validate_plan(plan)
        return plan

    def solve(self, question: str) -> dict[str, Any]:
        plan = self.plan(question)
        result = execute_plan(plan)

        return {
            "plan": plan,
            "calculation": result,
            "planner_model": getattr(
                self.llm,
                "model",
                None,
            ),
        }
