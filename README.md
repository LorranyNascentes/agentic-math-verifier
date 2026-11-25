# stem-research-agents

A **verifiable multi-step agentic system** for quantitative / mathematical reasoning.
An LLM-driven workflow proposes hypotheses, translates them into machine-checkable claims,
and **proves or refutes them with deterministic math tools** (SymPy, 30-digit MPMath, Monte Carlo).
Failed validations produce concrete counterexamples that are fed back into the next iteration.

> **LLM proposes, tools dispose.** The critic's verdict is derived from tool evidence, so a
> confident-but-wrong model cannot talk its way to "accepted".

```
            ┌──────────────────────── revise (counterexample + feedback) ───────────────────────┐
            ▼                                                                                    │
 planner → hypothesizer → formalizer → validator ──► critic ──accept / exhausted──► reporter → END
                           (LLM → typed       (SymPy · MPMath ·    (verdict forced by
                            Claim schema)      Monte Carlo)         tool evidence)
```

## Quickstart (no API key needed)

```bash
pip install -e .                 # deps: pydantic, sympy, numpy
python -m stem_agents            # offline demo with a scripted LLM (built-in orchestrator)

pip install -e ".[langgraph]"    # optional
python -m stem_agents --engine langgraph   # same workflow, executed by LangGraph
python -m unittest discover -s tests -v
```

The demo verifies four properties of the Black-Scholes call price. The scripted model makes a
realistic **sign error** in its first PDE formalization; the validator refutes it with a
counterexample, the critic explains the failure, and iteration 2 is proven correct:

```
step  4  validator   193 ms -> critic          # bs_pde REFUTED  -> revise
step  9  critic        0 ms -> reporter        # all claims verified -> accept
| bs_pde          | verified_numeric | 25 random points, worst relative gap 1.4e-29 (30-digit arithmetic) |
| monte_carlo     | verified_numeric | closed-form=10.4506, MC=10.4685 +/- 0.0233 (0.8 sigma)             |
```

Run against a real model:

```bash
pip install anthropic && export ANTHROPIC_API_KEY=...
python -m stem_agents --provider anthropic --problem "Show that GBM discounted price is a martingale ..."
```

## What this demonstrates

| Capability | Where |
|---|---|
| Production-style multi-step agent workflow with loops, routing, termination guarantees | `graph.py`, `agents.py` |
| Custom orchestration (nodes, conditional edges, tracing, checkpoint/resume, step limits) | `graph.py` |
| **LangGraph adapter**: the same nodes/edges run on `langgraph` unchanged (`--engine langgraph`) | `langgraph_adapter.py` |
| Structured output generation: **pydantic v2** models (`extra="forbid"`), JSON Schema prompts, **error-feedback self-repair loop** | `models.py`, `structured.py` |
| Tool integration with fail-safe registry (exceptions become structured results, never crashes) | `tools.py` |
| Safe execution of LLM-written math: AST whitelist, no arbitrary `eval` | `tools.safe_eval` + tests |
| State management: frozen pydantic state, JSON-serializable; per-step checkpoints; crash recovery | `models.AgentState`, `graph.resume` |
| Reasoning workflow: hypothesis generation → formalization → technical validation → critique | `agents.py` |
| Translating math/finance concepts into modular software (Itô/PDE, greeks, put-call parity, risk-neutral MC) | `demo.py`, `tools.py` |
| Provider-agnostic LLM layer + deterministic `ScriptedLLM` so the whole system is testable in CI | `llm.py` |
| Test suite: models, repair loop, sandbox, graph routing, resume, full refute→repair run, engine parity | `tests/` |

## Design decisions

- **Verification over persuasion.** Three evidence levels: `verified_symbolic` (SymPy proof) →
  `verified_numeric` (random testing in 30-digit arithmetic) → Monte Carlo cross-check for
  stochastic claims. Refutations carry a reproducible counterexample.
- **Honest labelling.** Numerical agreement is reported as *numerical evidence*, not proof; the
  report's limitations section says so.
- **Pydantic for the contract with the model.** Every LLM output is a pydantic model; validation
  errors are formatted as `claims.1.kind: ...` and fed back to the model for repair.
- **Engine-agnostic nodes.** Nodes are plain `AgentState -> dict` functions; the built-in
  `StateGraph` and the LangGraph adapter run the identical workflow (covered by a parity test).
- **Determinism.** Seeded sampling and a scripted LLM make runs reproducible and CI-friendly.
- **Bounded loops.** `max_iterations` (workflow) and `max_steps` (graph) prevent runaway agents;
  exhaustion is a first-class status that still produces a report.

## Extending

- New claim type: add a checker in `tools.py`, register it in `default_registry()`, extend `Claim.kind`.
- New domain (physics, e.g. energy conservation of an integrator; CS, e.g. algorithm invariants):
  add a checker + a scripted scenario in `demo.py`.
- Other orchestrators: nodes are plain `state -> dict` functions; `langgraph_adapter.py` is ~40 lines
  and shows the pattern for AutoGen/CrewAI.

## Project layout

```
src/stem_agents/
  models.py      pydantic schemas for LLM output, ClaimResult, frozen AgentState
  structured.py  generate(): pydantic validation + repair loop
  llm.py         LLM protocol, ScriptedLLM, AnthropicLLM
  tools.py       safe_eval, symbolic/numeric/Monte Carlo checkers, ToolRegistry
  graph.py       StateGraph: routing, tracing, checkpoint/resume
  langgraph_adapter.py  run the same graph on LangGraph
  agents.py      the six workflow nodes + graph wiring
  demo.py        Black-Scholes scenario with a built-in mistake to repair
tests/           unit + end-to-end + engine-parity tests (stdlib unittest)
```

## Roadmap

- [ ] Physics scenario (symplectic integrator energy drift) and CS scenario (algorithm invariants)
- [x] LangGraph adapter exposing the same nodes
- [ ] Lean/SMT backend for formal proofs where numerics are only evidence
- [ ] Parallel claim validation and per-claim timeouts

## Author

Lorrany Araujo Nascentes — nascenteslorrany@gmail.com

MIT licensed.
