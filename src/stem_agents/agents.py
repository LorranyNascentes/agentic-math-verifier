"""The research workflow: plan -> hypothesize -> formalize -> validate -> critique -> (loop) -> report.

Design rules
  * Agents (LLM) generate; tools (math) verify; the critic's verdict is derived from tool
    evidence, so a persuasive-but-wrong LLM cannot talk its way to "accepted".
  * Every LLM output is schema-validated with automatic repair (structured.generate).
  * State is immutable and JSON-serializable -> checkpoint/resume/audit for free.
"""
from __future__ import annotations

import json
from .graph import END, StateGraph
from .llm import LLM
from .models import (AgentState, ClaimResult, Critique, Formalization, HypothesisSet, Plan, Report)
from .structured import generate
from .tools import ToolRegistry, default_registry

SYSTEM = {
    "planner": "You are a research planner for mathematical/quantitative problems. Output short, concrete steps.",
    "hypothesizer": ("You are a quantitative researcher. Propose precise, falsifiable hypotheses. "
                     "If feedback from a failed validation is given, fix the specific error it identifies."),
    "formalizer": ("You translate hypotheses into machine-checkable claims. Expression DSL: + - * / **, "
                   "exp log sqrt erf sin cos tan Abs diff(expr, var[, n]) pi E, and Phi (normal CDF). "
                   "Give sampling ranges for every free variable so identities can be tested."),
    "critic": "You are a skeptical reviewer. Explain precisely why claims failed and how to repair them.",
    "reporter": "You write concise, honest technical reports. State limitations plainly.",
}


def build_graph(llm: LLM, tools: ToolRegistry | None = None) -> StateGraph:
    tools = tools or default_registry()

    def planner(s: AgentState) -> dict:
        plan = generate(llm, "planner", Plan, SYSTEM["planner"], f"Problem:\n{s.problem}")
        return {"plan": plan.steps}

    def hypothesizer(s: AgentState) -> dict:
        user = f"Problem:\n{s.problem}\n\nPlan:\n" + "\n".join(f"- {p}" for p in s.plan)
        if s.feedback_log:
            user += f"\n\nReviewer feedback from the previous attempt (address it):\n{s.feedback_log[-1]}"
        out = generate(llm, "hypothesizer", HypothesisSet, SYSTEM["hypothesizer"], user)
        return {"hypotheses": out.hypotheses, "iteration": s.iteration + 1}

    def formalizer(s: AgentState) -> dict:
        user = "Hypotheses:\n" + "\n".join(f"{i + 1}. {h.statement}" for i, h in enumerate(s.hypotheses))
        if s.feedback_log:
            user += f"\n\nPrevious formalization failed review:\n{s.feedback_log[-1]}"
        out = generate(llm, "formalizer", Formalization, SYSTEM["formalizer"], user)
        return {"claims": out.claims}

    def validator(s: AgentState) -> dict:
        results: list[ClaimResult] = []
        for claim in s.claims:
            tool = "check_identity" if claim.kind == "identity" else "check_monte_carlo"
            res = tools.call(tool, claim=claim)
            results.append(res.value if res.ok else ClaimResult(claim_id=claim.id, status="error", detail=res.error or "tool failed"))
        return {"results": results}

    def critic(s: AgentState) -> dict:
        failed = [r for r in s.results if not r.verified]
                evidence = json.dumps([r.model_dump(mode="json") for r in s.results], indent=2)
        claims_json = json.dumps([c.model_dump(mode="json") for c in s.claims])
        review = generate(llm, "critic", Critique, SYSTEM["critic"],
                          f"Validation evidence:\n{evidence}\n\nClaims:\n{claims_json}")
        # Tool evidence is authoritative; the LLM only contributes the explanation.
        verdict = "revise" if failed else "accept"
        note = "" if review.verdict == verdict else " [verdict overridden by tool evidence]"
        critique = Critique(verdict=verdict, feedback=review.feedback + note,
                            failed_claim_ids=[r.claim_id for r in failed])
        status = "accepted" if verdict == "accept" else (
            "exhausted" if s.iteration >= s.max_iterations else "running")
        return {"critique": critique, "status": status,
                "feedback_log": s.feedback_log + [f"[iteration {s.iteration}] {critique.feedback}"]}

    def reporter(s: AgentState) -> dict:
        evidence = json.dumps([r.model_dump(mode="json") for r in s.results])
        rep = generate(llm, "reporter", Report, SYSTEM["reporter"],
                       f"Problem: {s.problem}\nFinal status: {s.status} after {s.iteration} iteration(s).\n"
                       f"Evidence: {evidence}\nReview history: {s.feedback_log}")
        rows = "\n".join(f"| {r.claim_id} | {r.status} | {r.detail} |" for r in s.results)
        history = "\n".join(f"- {line}" for line in s.feedback_log)
        md = (f"# {rep.title}\n\n**Status:** {s.status} ({s.iteration} iteration(s))\n\n{rep.summary}\n\n"
              "## Findings\n" + "\n".join(f"- {x}" for x in rep.findings) +
              f"\n\n## Verification evidence\n| claim | status | detail |\n|---|---|---|\n{rows}\n\n"
              f"## Review history\n{history}\n\n## Limitations\n" + "\n".join(f"- {x}" for x in rep.limitations) + "\n")
        return {"report_md": md}

    g = StateGraph()
    for name, fn in [("planner", planner), ("hypothesizer", hypothesizer), ("formalizer", formalizer),
                     ("validator", validator), ("critic", critic), ("reporter", reporter)]:
        g.add_node(name, fn)
    g.set_entry_point("planner")
    g.add_edge("planner", "hypothesizer").add_edge("hypothesizer", "formalizer")
    g.add_edge("formalizer", "validator").add_edge("validator", "critic").add_edge("reporter", END)
    g.add_conditional_edges("critic", lambda s: "hypothesizer" if s.status == "running" else "reporter",
                            ["hypothesizer", "reporter"])
    return g
