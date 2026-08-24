from __future__ import annotations

"""
Universal Engineering Mathematics Engine
========================================

Deterministic computation layer for an Engineering AI application.

The LLM is used as a planner/explainer by the application, but numerical
results should be computed here whenever a structured expression/problem can
be supplied.

Capabilities:
- Safe arithmetic (no eval)
- SymPy symbolic algebra
- Equation solving and systems of equations
- Symbolic differentiation/integration/limits/series
- Matrices, determinants, inverse, eigenvalues/eigenvectors
- Numerical evaluation
- NumPy statistics, arrays and linear algebra
- Optional SciPy numerical methods when installed
- Basic engineering unit conversions
- Structured calculation plans supplied by the LLM
- Dependency-aware multi-expression evaluation

The public entry point is `calculate()`.
It returns a dictionary compatible with DocumentService.
"""

import ast
import math
import operator
import re
from typing import Any

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import scipy
    from scipy import integrate as scipy_integrate
    from scipy import optimize as scipy_optimize
except ImportError:  # pragma: no cover
    scipy = None
    scipy_integrate = None
    scipy_optimize = None


# ============================================================================
# Core requirements / formatting
# ============================================================================

def _require_sympy() -> None:
    if sp is None:
        raise RuntimeError("SymPy is not installed.")


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is not installed.")


def _safe_float(value: Any) -> float:
    return float(value)


def _format_number(value: Any, digits: int = 12) -> str:
    """Safely format Python, NumPy and SymPy numeric values."""
    if sp is not None:
        try:
            if isinstance(value, sp.Integer) or getattr(value, "is_Integer", False):
                return f"{int(value):,}"
            if isinstance(value, sp.Rational) and not getattr(value, "is_Integer", False):
                return str(value)
            if isinstance(value, sp.Float):
                value = float(value)
        except Exception:
            pass

    if isinstance(value, (int, np.integer if np is not None else int)):
        return f"{int(value):,}"

    if isinstance(value, (float, np.floating if np is not None else float)):
        value = float(value)
        if math.isfinite(value):
            return f"{value:.{digits}g}"
        return str(value)

    return str(value)


def _result(
    expression: str,
    result: Any,
    tool: str,
    *,
    details: dict[str, Any] | None = None,
    formatted_result: str | None = None,
) -> dict[str, Any]:
    return {
        "expression": expression,
        "result": result,
        "formatted_result": (
            formatted_result
            if formatted_result is not None
            else _format_number(result)
        ),
        "tool": tool,
        **({"details": details} if details else {}),
    }


# ============================================================================
# Safe arithmetic
# ============================================================================

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_arithmetic(expression: str) -> Any:
    """
    Evaluate arithmetic without eval().

    Permitted:
      numeric constants, parentheses, + - * / % ** and ^

    Names, attributes, function calls, imports and indexing are rejected.
    """
    _require_sympy()

    text = str(expression).strip().replace(",", "").replace("^", "**")
    tree = ast.parse(text, mode="eval")

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError("Only numeric constants are allowed.")
            return (
                sp.Integer(node.value)
                if isinstance(node.value, int)
                else sp.Float(node.value)
            )

        if isinstance(node, ast.UnaryOp):
            fn = _UNARYOPS.get(type(node.op))
            if fn is None:
                raise ValueError("Unsupported unary operator.")
            return fn(visit(node.operand))

        if isinstance(node, ast.BinOp):
            fn = _BINOPS.get(type(node.op))
            if fn is None:
                raise ValueError("Unsupported arithmetic operator.")
            return fn(visit(node.left), visit(node.right))

        raise ValueError("Unsupported expression.")

    return sp.simplify(visit(tree))


# ============================================================================
# Symbol / expression environment
# ============================================================================

def _sympy_namespace() -> dict[str, Any]:
    _require_sympy()

    namespace = {
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "I": sp.I,
        "oo": sp.oo,
        "inf": sp.oo,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "floor": sp.floor,
        "ceiling": sp.ceiling,
        "factorial": sp.factorial,
        "binomial": sp.binomial,
        "Matrix": sp.Matrix,
        "Rational": sp.Rational,
    }

    return namespace


def parse_expression(expression: str, variables: dict[str, Any] | None = None) -> Any:
    """
    Parse a mathematical expression with SymPy.

    This is deliberately restricted to a known mathematical namespace.
    """
    _require_sympy()

    namespace = _sympy_namespace()

    for name, value in (variables or {}).items():
        if not re.fullmatch(r"[A-Za-z_]\w*", str(name)):
            raise ValueError(f"Invalid variable name: {name}")
        namespace[str(name)] = sp.sympify(value)

    # SymPy's parser is used only against a controlled local dictionary.
    return sp.sympify(expression.replace("^", "**"), locals=namespace)


# ============================================================================
# Universal symbolic mathematics
# ============================================================================


def calculate_power(value: float | int, exponent: int) -> dict[str, Any]:
    """Deterministically calculate an arbitrary integer power with SymPy."""
    _require_sympy()
    if isinstance(value, float) and value.is_integer():
        x = sp.Integer(int(value))
    else:
        x = sp.sympify(value)
    result = sp.expand(x ** int(exponent))
    return _result(
        f"{x}^{int(exponent)}",
        result,
        "sympy",
    )

def evaluate_expression(
    expression: str,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = parse_expression(expression, variables)

    return _result(
        expression,
        result,
        "sympy",
        details={"variables": variables or {}},
    )


def solve_equation(
    equation: str,
    variable: str,
) -> dict[str, Any]:
    _require_sympy()

    x = sp.Symbol(variable)
    text = equation.replace("^", "**")

    if "=" in text:
        left, right = text.split("=", 1)
        expr = sp.Eq(
            parse_expression(left),
            parse_expression(right),
        )
    else:
        expr = parse_expression(text)

    solutions = sp.solve(expr, x)

    return _result(
        f"{equation} for {variable}",
        solutions,
        "sympy",
        details={
            "equation": equation,
            "variable": variable,
            "solutions": solutions,
        },
        formatted_result=", ".join(map(str, solutions)) or "No solution",
    )


def solve_system(
    equations: list[str],
    variables: list[str],
) -> dict[str, Any]:
    _require_sympy()

    symbols = sp.symbols(" ".join(variables))
    if len(variables) == 1:
        symbols = (symbols,)

    parsed = []
    for equation in equations:
        text = equation.replace("^", "**")
        if "=" in text:
            left, right = text.split("=", 1)
            parsed.append(
                sp.Eq(parse_expression(left), parse_expression(right))
            )
        else:
            parsed.append(parse_expression(text))

    solution = sp.solve(parsed, symbols, dict=True)

    return _result(
        f"Solve system: {equations}",
        solution,
        "sympy",
        details={
            "equations": equations,
            "variables": variables,
            "solution": solution,
        },
    )


def differentiate(
    expression: str,
    variable: str,
    order: int = 1,
) -> dict[str, Any]:
    _require_sympy()

    x = sp.Symbol(variable)
    expr = parse_expression(expression)
    result = sp.diff(expr, x, order)

    return _result(
        f"d^{order}/d{variable}^{order} ({expression})",
        result,
        "sympy",
        details={
            "expression": expression,
            "variable": variable,
            "order": order,
        },
    )


def integrate(
    expression: str,
    variable: str,
    lower: Any | None = None,
    upper: Any | None = None,
) -> dict[str, Any]:
    _require_sympy()

    x = sp.Symbol(variable)
    expr = parse_expression(expression)

    if lower is None or upper is None:
        result = sp.integrate(expr, x)
        description = f"∫ {expression} d{variable}"
    else:
        lo = parse_expression(str(lower))
        hi = parse_expression(str(upper))
        result = sp.integrate(expr, (x, lo, hi))
        description = f"∫[{lower},{upper}] {expression} d{variable}"

    return _result(
        description,
        result,
        "sympy",
        details={
            "expression": expression,
            "variable": variable,
            "lower": lower,
            "upper": upper,
        },
    )


def limit(
    expression: str,
    variable: str,
    point: Any,
    direction: str = "+-",
) -> dict[str, Any]:
    _require_sympy()

    x = sp.Symbol(variable)
    expr = parse_expression(expression)
    p = parse_expression(str(point))

    result = sp.limit(expr, x, p, dir=direction)

    return _result(
        f"lim({expression}) as {variable}->{point}",
        result,
        "sympy",
    )


def series(
    expression: str,
    variable: str,
    point: Any = 0,
    order: int = 6,
) -> dict[str, Any]:
    _require_sympy()

    x = sp.Symbol(variable)
    expr = parse_expression(expression)
    p = parse_expression(str(point))

    result = sp.series(expr, x, p, order)

    return _result(
        f"Series of {expression} around {variable}={point}",
        result,
        "sympy",
    )


# ============================================================================
# Linear algebra
# ============================================================================

def matrix_operation(
    matrix: list[list[Any]],
    operation: str,
    matrix_b: list[list[Any]] | None = None,
) -> dict[str, Any]:
    _require_sympy()

    A = sp.Matrix(matrix)
    op = operation.lower().strip()

    if op in {"det", "determinant"}:
        result = A.det()
    elif op in {"inverse", "inv"}:
        result = A.inv()
    elif op in {"transpose", "transposed", "t"}:
        result = A.T
    elif op in {"rank"}:
        result = A.rank()
    elif op in {"trace"}:
        result = A.trace()
    elif op in {"eigenvalues", "eigenvalue"}:
        result = A.eigenvals()
    elif op in {"eigenvectors", "eigenvector"}:
        result = A.eigenvects()
    elif op in {"norm"}:
        result = A.norm()
    elif op in {"multiply", "matmul"}:
        if matrix_b is None:
            raise ValueError("matrix_b is required for multiplication.")
        result = A * sp.Matrix(matrix_b)
    elif op in {"add"}:
        if matrix_b is None:
            raise ValueError("matrix_b is required for addition.")
        result = A + sp.Matrix(matrix_b)
    elif op in {"subtract", "sub"}:
        if matrix_b is None:
            raise ValueError("matrix_b is required for subtraction.")
        result = A - sp.Matrix(matrix_b)
    else:
        raise ValueError(f"Unsupported matrix operation: {operation}")

    return _result(
        f"Matrix operation: {operation}",
        result,
        "sympy",
        details={
            "matrix": matrix,
            "operation": operation,
            "matrix_b": matrix_b,
        },
    )


def solve_linear_system(
    matrix: list[list[Any]],
    rhs: list[Any],
) -> dict[str, Any]:
    _require_sympy()

    A = sp.Matrix(matrix)
    b = sp.Matrix(rhs)

    solution = sp.linsolve((A, b))

    return _result(
        "Solve Ax = b",
        solution,
        "sympy",
        details={
            "A": matrix,
            "b": rhs,
            "solution": solution,
        },
    )


# ============================================================================
# NumPy numerical/statistical engine
# ============================================================================

def numpy_statistics(values: list[float]) -> dict[str, Any]:
    _require_numpy()

    if not values:
        raise ValueError("At least one value is required.")

    arr = np.asarray(values, dtype=float)

    details = {
        "count": int(arr.size),
        "sum": float(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "variance_population": float(np.var(arr)),
        "std_population": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

    if arr.size > 1:
        details["variance_sample"] = float(np.var(arr, ddof=1))
        details["std_sample"] = float(np.std(arr, ddof=1))

    return _result(
        f"statistics({values})",
        details["mean"],
        "numpy",
        details=details,
        formatted_result=(
            f"mean={details['mean']:.8g}, "
            f"std={details['std_population']:.8g}, "
            f"n={details['count']}"
        ),
    )


def numpy_matrix_operation(
    matrix: list[list[float]],
    operation: str,
) -> dict[str, Any]:
    _require_numpy()

    A = np.asarray(matrix, dtype=float)
    op = operation.lower().strip()

    if op in {"det", "determinant"}:
        result = float(np.linalg.det(A))
    elif op in {"inverse", "inv"}:
        result = np.linalg.inv(A)
    elif op in {"eigenvalues", "eigvals"}:
        result = np.linalg.eigvals(A)
    elif op in {"rank"}:
        result = int(np.linalg.matrix_rank(A))
    elif op in {"norm"}:
        result = float(np.linalg.norm(A))
    elif op in {"transpose", "t"}:
        result = A.T
    else:
        raise ValueError(f"Unsupported NumPy matrix operation: {operation}")

    return _result(
        f"NumPy matrix {operation}",
        result,
        "numpy",
    )


# ============================================================================
# Optional SciPy numerical engine
# ============================================================================

def scipy_integrate_function(
    function: str,
    variable: str,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    if scipy_integrate is None:
        raise RuntimeError(
            "SciPy is not installed. Install it with: pip install scipy"
        )

    # SymPy parses the expression; lambdify creates the numerical function.
    _require_sympy()

    x = sp.Symbol(variable)
    expr = parse_expression(function)
    fn = sp.lambdify(x, expr, modules=["numpy"])

    value, error = scipy_integrate.quad(fn, lower, upper)

    return _result(
        f"Numerical integral of {function} from {lower} to {upper}",
        float(value),
        "scipy",
        details={"estimated_error": float(error)},
    )


def scipy_root(
    function: str,
    variable: str,
    initial_guess: float,
) -> dict[str, Any]:
    if scipy_optimize is None:
        raise RuntimeError(
            "SciPy is not installed. Install it with: pip install scipy"
        )

    _require_sympy()

    x = sp.Symbol(variable)
    expr = parse_expression(function)
    fn = sp.lambdify(x, expr, modules=["numpy"])

    root = scipy_optimize.newton(fn, initial_guess)

    return _result(
        f"Root of {function}",
        float(root),
        "scipy",
    )


# ============================================================================
# Unit conversion
# ============================================================================

_UNIT_FACTORS = {
    # length
    "m": ("length", 1.0),
    "mm": ("length", 1e-3),
    "cm": ("length", 1e-2),
    "km": ("length", 1e3),
    "in": ("length", 0.0254),
    "ft": ("length", 0.3048),

    # force
    "n": ("force", 1.0),
    "kn": ("force", 1e3),
    "mn": ("force", 1e6),
    "lbf": ("force", 4.4482216152605),

    # pressure/stress
    "pa": ("pressure", 1.0),
    "kpa": ("pressure", 1e3),
    "mpa": ("pressure", 1e6),
    "gpa": ("pressure", 1e9),
    "psi": ("pressure", 6894.757293168),

    # energy
    "j": ("energy", 1.0),
    "kj": ("energy", 1e3),
    "mj": ("energy", 1e6),
    "wh": ("energy", 3600.0),
    "kwh": ("energy", 3.6e6),

    # power
    "w": ("power", 1.0),
    "kw": ("power", 1e3),
    "mw": ("power", 1e6),

    # time
    "s": ("time", 1.0),
    "ms": ("time", 1e-3),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),

    # mass
    "kg": ("mass", 1.0),
    "g": ("mass", 1e-3),
    "mg": ("mass", 1e-6),
    "lb": ("mass", 0.45359237),

    # volume
    "l": ("volume", 1e-3),
    "ml": ("volume", 1e-6),
    "m3": ("volume", 1.0),
    "cm3": ("volume", 1e-6),
}


def convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    fu = from_unit.strip().lower()
    tu = to_unit.strip().lower()

    if fu not in _UNIT_FACTORS:
        raise ValueError(f"Unsupported unit: {from_unit}")
    if tu not in _UNIT_FACTORS:
        raise ValueError(f"Unsupported unit: {to_unit}")

    fdim, ffactor = _UNIT_FACTORS[fu]
    tdim, tfactor = _UNIT_FACTORS[tu]

    if fdim != tdim:
        raise ValueError(
            f"Cannot convert {from_unit} to {to_unit}: incompatible dimensions."
        )

    result = float(value) * ffactor / tfactor

    return _result(
        f"{value} {from_unit} -> {to_unit}",
        result,
        "unit-conversion",
        formatted_result=f"{result:.12g} {to_unit}",
    )


# ============================================================================
# Structured universal calculation plan
# ============================================================================

def execute_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a structured plan.

    Example:
    {
      "type": "system",
      "equations": ["V = I*R1", "V = I*R2"],
      "variables": ["V", "I"]
    }

    Or:
    {
      "type": "expression",
      "expression": "sqrt(3)*230"
    }

    The LLM may produce this structure, but this function performs the
    actual calculation deterministically.
    """
    if not isinstance(plan, dict):
        raise ValueError("Calculation plan must be a dictionary.")

    kind = str(plan.get("type", "")).lower().strip()

    if kind in {"expression", "evaluate", "arithmetic"}:
        return evaluate_expression(
            str(plan["expression"]),
            plan.get("variables"),
        )

    if kind in {"equation", "solve"}:
        return solve_equation(
            str(plan["equation"]),
            str(plan["variable"]),
        )

    if kind in {"system", "equation_system"}:
        return solve_system(
            list(plan["equations"]),
            list(plan["variables"]),
        )

    if kind in {"differentiate", "derivative"}:
        return differentiate(
            str(plan["expression"]),
            str(plan["variable"]),
            int(plan.get("order", 1)),
        )

    if kind in {"integrate", "integral"}:
        return integrate(
            str(plan["expression"]),
            str(plan["variable"]),
            plan.get("lower"),
            plan.get("upper"),
        )

    if kind == "limit":
        return limit(
            str(plan["expression"]),
            str(plan["variable"]),
            plan["point"],
            str(plan.get("direction", "+-")),
        )

    if kind == "series":
        return series(
            str(plan["expression"]),
            str(plan["variable"]),
            plan.get("point", 0),
            int(plan.get("order", 6)),
        )

    if kind in {"matrix", "matrix_operation"}:
        return matrix_operation(
            plan["matrix"],
            str(plan["operation"]),
            plan.get("matrix_b"),
        )

    if kind in {"linear_system", "linsolve"}:
        return solve_linear_system(
            plan["matrix"],
            plan["rhs"],
        )

    if kind in {"statistics", "stats"}:
        return numpy_statistics(list(plan["values"]))

    if kind in {"numpy_matrix", "numpy_matrix_operation"}:
        return numpy_matrix_operation(
            plan["matrix"],
            str(plan["operation"]),
        )

    if kind in {"scipy_integral", "numerical_integral"}:
        return scipy_integrate_function(
            str(plan["function"]),
            str(plan["variable"]),
            float(plan["lower"]),
            float(plan["upper"]),
        )

    if kind in {"scipy_root", "root"}:
        return scipy_root(
            str(plan["function"]),
            str(plan["variable"]),
            float(plan["initial_guess"]),
        )

    if kind in {"unit", "unit_conversion", "convert"}:
        return convert_units(
            float(plan["value"]),
            str(plan["from_unit"]),
            str(plan["to_unit"]),
        )

    raise ValueError(f"Unsupported calculation plan type: {kind}")


# ============================================================================
# Natural-language dispatcher
# ============================================================================

def calculate(question: str) -> dict[str, Any] | None:
    """
    Conservative automatic calculation detector.

    This is intentionally limited to *high-confidence* generic patterns.
    For arbitrary engineering word problems, DocumentService should ask the
    LLM to produce a structured calculation plan and then call execute_plan().
    """
    q = str(question or "").strip()
    ql = q.lower()

    if not q:
        return None

    # Explicit arithmetic.
    if re.fullmatch(r"\s*[0-9()\s+\-*/%^.,]+\s*", q):
        try:
            return evaluate_expression(q)
        except Exception:
            pass

    # Cube / square.
    m = re.search(
        r"\b(?:cube|third\s+power)\s+of\s+(-?\d+(?:\.\d+)?)\b",
        q,
        re.IGNORECASE,
    )
    if m:
        return calculate_power(float(m.group(1)), 3)

    m = re.search(
        r"\b(?:square|second\s+power)\s+of\s+(-?\d+(?:\.\d+)?)\b",
        q,
        re.IGNORECASE,
    )
    if m:
        return calculate_power(float(m.group(1)), 2)

    # Mean / average with an explicit list.
    values = re.findall(r"-?\d+(?:\.\d+)?", q)
    if (
        len(values) >= 2
        and re.search(r"\b(?:mean|average)\b", ql)
        and (
            "[" in q
            or "]" in q
            or re.search(r"\b(?:numbers|values|data)\b", ql)
        )
    ):
        try:
            return numpy_statistics([float(v) for v in values])
        except Exception:
            pass

    return None


# Backward compatibility for existing DocumentService imports.
try_engineering_calculation = calculate


__all__ = [
    "safe_arithmetic",
    "parse_expression",
    "calculate_power",
    "evaluate_expression",
    "solve_equation",
    "solve_system",
    "differentiate",
    "integrate",
    "limit",
    "series",
    "matrix_operation",
    "solve_linear_system",
    "numpy_statistics",
    "numpy_matrix_operation",
    "scipy_integrate_function",
    "scipy_root",
    "convert_units",
    "execute_plan",
    "calculate",
    "try_engineering_calculation",
]
