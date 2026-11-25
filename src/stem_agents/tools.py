"""Verification tools. Principle: the LLM proposes, the tools dispose.

Every claim is checked by deterministic math, never by another LLM call:
  1. symbolic proof (sympy) when the expression is small enough,
  2. high-precision numerical testing (mpmath, 30 digits) at random points,
  3. Monte Carlo cross-validation for stochastic-model claims.
LLM-written expressions are parsed through an AST whitelist (no eval of arbitrary code).
"""
from __future__ import annotations

import ast
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import mpmath as mp
import numpy as np
import sympy as sp

from .models import Claim, ClaimResult

SYMBOLIC_OPS_LIMIT = 40
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
                  ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


class UnsafeExpression(ValueError):
    pass


def _namespace(symbols: dict[str, sp.Symbol]) -> dict[str, Any]:
    x = sp.Symbol("x")
    ns: dict[str, Any] = {
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "erf": sp.erf, "sin": sp.sin,
        "cos": sp.cos, "tan": sp.tan, "Abs": sp.Abs, "diff": sp.diff, "pi": sp.pi, "E": sp.E,
        "Phi": sp.Lambda(x, (1 + sp.erf(x / sp.sqrt(2))) / 2),  # standard normal CDF
    }
    ns.update(symbols)
    return ns


def safe_eval(expr: str, ns: dict[str, Any]) -> sp.Expr:
    """Evaluate a math expression with an AST whitelist (names, numbers, arithmetic, calls)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"syntax error in {expr!r}: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpression(f"disallowed syntax {type(node).__name__} in {expr!r}")
        if isinstance(node, ast.Name) and node.id not in ns:
            raise UnsafeExpression(f"unknown name {node.id!r} in {expr!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise UnsafeExpression(f"disallowed constant {node.value!r}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.keywords):
            raise UnsafeExpression("only plain function calls are allowed")
    value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, dict(ns))  # noqa: S307
    return sp.sympify(value)


def _build(claim: Claim) -> tuple[sp.Expr, sp.Expr, list[sp.Symbol], dict[str, sp.Symbol]]:
    names = [v.name for v in claim.variables] + list(claim.params)
    symbols = {n: sp.Symbol(n) for n in names}
    ns = _namespace(symbols)
    for name, expr in claim.definitions.items():
        ns[name] = safe_eval(expr, ns)
    lhs, rhs = safe_eval(claim.lhs, ns), safe_eval(claim.rhs, ns)
    return lhs, rhs, [symbols[v.name] for v in claim.variables], symbols


def check_identity(claim: Claim, n_points: int = 25, seed: int = 0, tol: float = 1e-9) -> ClaimResult:
    lhs, rhs, syms, _ = _build(claim)
    residual = lhs - rhs
    if sp.count_ops(residual) <= SYMBOLIC_OPS_LIMIT and sp.simplify(residual) == 0:
        return ClaimResult(claim_id=claim.id, status="verified_symbolic", detail="simplify(lhs - rhs) == 0")

    mp.mp.dps = 30
    f_l = sp.lambdify(syms, lhs, modules="mpmath")
    f_r = sp.lambdify(syms, rhs, modules="mpmath")
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(n_points):
        point = {v.name: rng.uniform(v.low, v.high) for v in claim.variables}
        args = [mp.mpf(point[s.name]) for s in syms]
        a, b = f_l(*args), f_r(*args)
        rel = float(abs(a - b) / (1 + abs(a) + abs(b)))
        worst = max(worst, rel)
        if rel > tol:
            return ClaimResult(claim_id=claim.id, status="refuted",
                               detail=f"lhs={float(a):.6g} != rhs={float(b):.6g} (relative gap {rel:.2e})",
                               counterexample={k: round(v, 6) for k, v in point.items()})
    return ClaimResult(claim_id=claim.id, status="verified_numeric",
                       detail=f"{n_points} random points, worst relative gap {worst:.1e} (30-digit arithmetic)")


def mc_european_call(S: float, K: float, r: float, sigma: float, T: float,
                     n: int = 400_000, seed: int = 0) -> tuple[float, float]:
    """Risk-neutral Monte Carlo price and standard error under geometric Brownian motion."""
    z = np.random.default_rng(seed).standard_normal(n)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    pv = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    return float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(n))


def check_monte_carlo(claim: Claim, max_sigma: float = 4.0) -> ClaimResult:
    """Cross-validate a closed-form call price (``lhs``) against Monte Carlo at ``params``."""
    lhs, _, _, symbols = _build(claim)
    p = claim.params
    missing = {"S", "K", "r", "sigma", "T"} - set(p)
    if missing:
        return ClaimResult(claim_id=claim.id, status="error", detail=f"params missing {sorted(missing)}")
    closed = float(sp.N(lhs.subs({symbols[k]: v for k, v in p.items()})))
    tau = p["T"] - p.get("t", 0.0)
    mc, se = mc_european_call(p["S"], p["K"], p["r"], p["sigma"], tau)
    z = abs(mc - closed) / se
    detail = f"closed-form={closed:.4f}, MC={mc:.4f} +/- {se:.4f} ({z:.1f} sigma)"
    return ClaimResult(claim_id=claim.id, status="verified_numeric" if z < max_sigma else "refuted",
                       detail=detail)


# ---- Tool registry --------------------------------------------------------------------
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]


@dataclass
class ToolResult:
    ok: bool
    value: Any = None
    error: str | None = None
    seconds: float = 0.0


class ToolRegistry:
    """Uniform, fail-safe tool calling: exceptions become structured errors, never crashes."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def describe(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        try:
            value = self._tools[name].fn(**kwargs)
            return ToolResult(True, value, seconds=time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - tools must never crash the graph
            return ToolResult(False, error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - t0)


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("check_identity", "Prove or refute lhs == rhs symbolically, then numerically.", check_identity))
    reg.register(Tool("check_monte_carlo", "Cross-check a closed-form option price against Monte Carlo.", check_monte_carlo))
    return reg
