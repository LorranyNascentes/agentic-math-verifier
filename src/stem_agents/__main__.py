"""CLI:  python -m stem_agents --demo   |   python -m stem_agents --provider anthropic --problem "..." """
import argparse
import time
from pathlib import Path

from .agents import build_graph
from .demo import PROBLEM, SCRIPT
from .llm import AnthropicLLM, ScriptedLLM
from .models import AgentState


def main() -> None:
    ap = argparse.ArgumentParser(prog="stem_agents")
    ap.add_argument("--provider", choices=["scripted", "anthropic"], default="scripted")
    ap.add_argument("--problem", default=PROBLEM)
    ap.add_argument("--max-iterations", type=int, default=3)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--engine", choices=["native", "langgraph"], default="native",
                    help="orchestrator: built-in StateGraph (traced + checkpointed) or LangGraph")
    args = ap.parse_args()

    llm = ScriptedLLM(SCRIPT) if args.provider == "scripted" else AnthropicLLM()
    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    graph = build_graph(llm)
    state0 = AgentState(problem=args.problem, max_iterations=args.max_iterations)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.engine == "langgraph":
        from .langgraph_adapter import run_on_langgraph

        final = run_on_langgraph(graph, state0)
    else:
        result = graph.run(state0, checkpoint_dir=run_dir / "checkpoints")
        for ev in result.trace:
            print(f"step {ev.step:>2}  {ev.node:<13} {ev.seconds * 1000:7.0f} ms  -> {ev.next}")
        final = result.state
    print(f"\nstatus={final.status}  iterations={final.iteration}  artifacts={run_dir}\n")
    print(final.report_md)


if __name__ == "__main__":
    main()
