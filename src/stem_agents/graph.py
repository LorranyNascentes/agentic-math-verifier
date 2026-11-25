"""A small explicit state-machine orchestrator (LangGraph-style, zero dependencies).

Nodes are pure functions ``state -> partial update``. Edges can be static or routed by
a function of state (this is how the critic loops back to the hypothesizer). Every step
is traced and checkpointed to JSON so runs are inspectable and resumable.
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

END = "__end__"
Node = Callable[[Any], dict]
Router = Callable[[Any], str]


class GraphError(RuntimeError):
    pass


def apply_update(state: Any, update: dict) -> Any:
    """Return a new state with ``update`` applied (pydantic models or frozen dataclasses)."""
    if hasattr(state, "model_copy"):
        return state.model_copy(update=update)
    return dataclasses.replace(state, **update)


@dataclass
class TraceEvent:
    step: int
    node: str
    seconds: float
    next: str


@dataclass
class RunResult:
    state: Any
    trace: list[TraceEvent]


class StateGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, str] = {}
        self._routers: dict[str, tuple[Router, set[str]]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: Node) -> "StateGraph":
        self._nodes[name] = fn
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self._entry = name
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self._edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: Router, targets: list[str]) -> "StateGraph":
        self._routers[src] = (router, set(targets))
        return self

    # Read-only topology, used by adapters (e.g. langgraph_adapter).
    @property
    def nodes(self) -> dict[str, Node]:
        return dict(self._nodes)

    @property
    def edges(self) -> dict[str, str]:
        return dict(self._edges)

    @property
    def routers(self) -> dict[str, tuple[Router, set[str]]]:
        return dict(self._routers)

    @property
    def entry(self) -> str | None:
        return self._entry

    def _next(self, node: str, state: Any) -> str:
        if node in self._routers:
            router, targets = self._routers[node]
            dst = router(state)
            if dst not in targets:
                raise GraphError(f"router at {node!r} returned {dst!r}, expected one of {sorted(targets)}")
            return dst
        return self._edges.get(node, END)

    def run(self, state: Any, *, start_at: str | None = None, step_offset: int = 0,
            max_steps: int = 40, checkpoint_dir: str | Path | None = None) -> RunResult:
        node = start_at or self._entry
        if node is None:
            raise GraphError("no entry point set")
        ckpt = Path(checkpoint_dir) if checkpoint_dir else None
        if ckpt:
            ckpt.mkdir(parents=True, exist_ok=True)
        trace: list[TraceEvent] = []
        step = step_offset
        while node != END:
            if len(trace) >= max_steps:
                raise GraphError(f"exceeded max_steps={max_steps} (possible infinite loop at {node!r})")
            if node not in self._nodes:
                raise GraphError(f"unknown node {node!r}")
            step += 1
            t0 = time.perf_counter()
            state = apply_update(state, self._nodes[node](state))
            nxt = self._next(node, state)
            trace.append(TraceEvent(step, node, time.perf_counter() - t0, nxt))
            if ckpt:
                payload = {"step": step, "node": node, "next": nxt, "state": state.to_dict()}
                (ckpt / f"{step:02d}_{node}.json").write_text(json.dumps(payload, indent=2))
            node = nxt
        return RunResult(state, trace)

    def resume(self, checkpoint_file: str | Path, state_cls: type, **kwargs: Any) -> RunResult:
        """Continue a run from a checkpoint written by ``run``."""
        data = json.loads(Path(checkpoint_file).read_text())
        state = state_cls.from_dict(data["state"])
        if data["next"] == END:
            return RunResult(state, [])
        return self.run(state, start_at=data["next"], step_offset=data["step"], **kwargs)
