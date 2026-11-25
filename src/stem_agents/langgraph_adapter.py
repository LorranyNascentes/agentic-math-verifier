"""Run the exact same workflow nodes on LangGraph.

Nodes are plain ``AgentState -> dict`` functions, so the adapter only (1) wraps each node to
validate LangGraph's dict state into ``AgentState`` and dump updates back to JSON-safe dicts,
and (2) mirrors our edges/routers onto ``langgraph.graph.StateGraph``.

    pip install "stem-research-agents[langgraph]"
"""
from typing import Any, Callable, TypedDict

from pydantic import BaseModel

from .graph import END, StateGraph
from .models import AgentState


def _dump(value: Any) -> Any:
    """Convert pydantic objects (possibly nested in lists/dicts) to JSON-safe primitives."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    return value


def _wrap(fn: Callable[[Any], dict], state_cls: type[BaseModel]) -> Callable[[dict], dict]:
    def node(state):  # unannotated on purpose: LangGraph infers the input schema from annotations
        return _dump(fn(state_cls.model_validate(state)))

    node.__name__ = getattr(fn, "__name__", "node")
    return node


def to_langgraph(graph: StateGraph, state_cls: type[BaseModel] = AgentState):
    """Compile ``graph`` (built by ``build_graph``) into a LangGraph runnable."""
    from langgraph.graph import END as LG_END  # lazy: optional dependency
    from langgraph.graph import StateGraph as LGStateGraph

    target = lambda name: LG_END if name == END else name  # noqa: E731
    schema = TypedDict("GraphState", {name: Any for name in state_cls.model_fields}, total=False)

    lg = LGStateGraph(schema)
    for name, fn in graph.nodes.items():
        lg.add_node(name, _wrap(fn, state_cls))
    lg.set_entry_point(graph.entry)
    for src, dst in graph.edges.items():
        lg.add_edge(src, target(dst))
    for src, (router, targets) in graph.routers.items():
        lg.add_conditional_edges(
            src,
            lambda state, router=router: router(state_cls.model_validate(state)),
            {t: target(t) for t in targets},
        )
    return lg.compile()


def run_on_langgraph(graph: StateGraph, state: AgentState, recursion_limit: int = 50) -> AgentState:
    """Execute ``graph`` on LangGraph and return the final, validated ``AgentState``."""
    compiled = to_langgraph(graph, type(state))
    final = compiled.invoke(state.model_dump(mode="json"), config={"recursion_limit": recursion_limit})
    return type(state).model_validate(final)
