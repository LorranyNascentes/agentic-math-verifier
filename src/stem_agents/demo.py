"""Offline demo scenario: verify properties of the Black-Scholes call price.

The scripted "model" makes a realistic mistake on its first attempt (wrong sign on the
drift term of the PDE). The validator refutes it with a counterexample, the critic feeds
that back, and the second attempt is proven correct -- exercising the full revise loop
with no API key.
"""
import json

PROBLEM = ("Let C(S,t) = S*Phi(d1) - K*exp(-r(T-t))*Phi(d2) be the Black-Scholes European call price. "
           "Verify: (a) C satisfies the Black-Scholes PDE, (b) delta = dC/dS = Phi(d1), "
           "(c) put-call parity holds, (d) the closed form agrees with Monte Carlo under GBM.")

_DEFS = {
    "tau": "T - t",
    "d1": "(log(S/K) + (r + sigma**2/2)*tau)/(sigma*sqrt(tau))",
    "d2": "d1 - sigma*sqrt(tau)",
    "C": "S*Phi(d1) - K*exp(-r*tau)*Phi(d2)",
    "P": "K*exp(-r*tau)*Phi(-d2) - S*Phi(-d1)",
}
_VARS = [{"name": "S", "low": 50, "high": 150}, {"name": "K", "low": 80, "high": 120},
         {"name": "r", "low": 0.01, "high": 0.10}, {"name": "sigma", "low": 0.1, "high": 0.5},
         {"name": "t", "low": 0.0, "high": 0.5}, {"name": "T", "low": 0.6, "high": 2.0}]


def _claims(pde_lhs: str) -> str:
    return json.dumps({"claims": [
        {"id": "bs_pde", "description": "C satisfies the Black-Scholes PDE", "kind": "identity",
         "lhs": pde_lhs, "rhs": "0", "definitions": _DEFS, "variables": _VARS},
        {"id": "delta", "description": "delta equals Phi(d1)", "kind": "identity",
         "lhs": "diff(C, S)", "rhs": "Phi(d1)", "definitions": _DEFS, "variables": _VARS},
        {"id": "put_call_parity", "description": "C - P = S - K*exp(-r*tau)", "kind": "identity",
         "lhs": "C - P", "rhs": "S - K*exp(-r*tau)", "definitions": _DEFS, "variables": _VARS},
        {"id": "monte_carlo", "description": "closed form matches GBM Monte Carlo", "kind": "monte_carlo",
         "lhs": "C", "definitions": _DEFS,
         "params": {"S": 100, "K": 100, "r": 0.05, "sigma": 0.2, "t": 0.0, "T": 1.0}},
    ]})


_HYP_1 = {"hypotheses": [{
    "statement": "C solves dC/dt + 1/2 sigma^2 S^2 C_SS - r S C_S - r C = 0",
    "rationale": "Standard risk-neutral pricing PDE (sign of the drift term recalled from memory).",
    "confidence": 0.7}]}
_HYP_2 = {"hypotheses": [{
    "statement": "C solves dC/dt + 1/2 sigma^2 S^2 C_SS + r S C_S - r C = 0; delta = Phi(d1); parity holds.",
    "rationale": "Drift term is +r S C_S under the risk-neutral measure (Ito + delta hedging).",
    "confidence": 0.95}]}

SCRIPT = {
    "planner": [json.dumps({"steps": [
        "State the Black-Scholes PDE and closed-form call price",
        "Formalize each property as a checkable identity or Monte Carlo test",
        "Validate with symbolic + high-precision numerics; revise on refutation"]})],
    "hypothesizer": [json.dumps(_HYP_1), json.dumps(_HYP_2)],
    "formalizer": [
        _claims("diff(C, t) + sigma**2*S**2/2*diff(C, S, 2) - r*S*diff(C, S) - r*C"),  # flawed
        "Here is the corrected formalization:\n```json\n"
        + _claims("diff(C, t) + sigma**2*S**2/2*diff(C, S, 2) + r*S*diff(C, S) - r*C") + "\n```",
    ],
    "critic": [
        json.dumps({"verdict": "revise", "failed_claim_ids": ["bs_pde"], "feedback":
                    "bs_pde was refuted by a concrete counterexample. The residual is nonzero because the "
                    "drift term enters with the wrong sign: under the risk-neutral measure it is +r*S*C_S."}),
        json.dumps({"verdict": "accept", "failed_claim_ids": [], "feedback":
                    "All four claims verified independently (PDE, delta, parity, Monte Carlo)."}),
    ],
    "reporter": [json.dumps({
        "title": "Black-Scholes call price: verification report",
        "summary": "The closed-form call price satisfies the PDE, its delta is Phi(d1), put-call parity "
                   "holds, and the price agrees with Monte Carlo. The first PDE formalization was refuted "
                   "(sign error) and repaired after review.",
        "findings": ["PDE residual vanishes to ~1e-30 at 25 random points after the sign fix",
                     "delta = Phi(d1) and put-call parity verified", "Monte Carlo agrees within 4 sigma"],
        "limitations": ["Numerical verification is evidence, not a formal proof, for the non-symbolic claims",
                        "Assumes constant volatility and rates, no dividends"]})],
}
