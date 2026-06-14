"""LangGraph-based pipeline runner.

Graph shape:

    ingest ──► fan_out ──► stt:<engine_id>  ┐
                       ──► stt:<engine_id>  ├──► judge ──► output ──► END
                       ──► stt:<engine_id>  ┘

`fan_out` uses LangGraph's `Send` API to dispatch the same segment to
every engine in parallel. Each `stt:<id>` node appends its hypothesis
to the shared state via an additive reducer; `judge` aggregates them.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from .types import (
    AudioSegment,
    AudioSource,
    Context,
    Decision,
    Hypothesis,
    OutputSink,
    STTEngine,
)


class GraphState(TypedDict, total=False):
    segment: AudioSegment
    ctx: Context
    # Hypotheses accumulate via the additive reducer — each parallel
    # STT node returns a single-element list which gets concatenated.
    hypotheses: Annotated[list[Hypothesis], operator.add]
    decision: Decision


def _make_stt_node(engine: STTEngine):
    """Build a node that runs one engine on the current segment."""

    def node(state: GraphState) -> dict[str, Any]:
        h = engine.transcribe(state["segment"], state["ctx"])
        return {"hypotheses": [h]}

    node.__name__ = f"stt_{engine.id}"
    return node


def _fan_out(state: GraphState) -> dict[str, Any]:
    # Reset the accumulator. Downstream `Send`s will refill it in parallel.
    return {"hypotheses": []}


def _route_to_engines(state: GraphState, engine_ids: list[str]) -> list[Send]:
    # `Send` carries the per-branch payload; we forward the same state to
    # every engine node.
    payload = {"segment": state["segment"], "ctx": state["ctx"]}
    return [Send(f"stt_{eid}", payload) for eid in engine_ids]


def build_graph(
    engines: list[STTEngine],
    judge,
    sink: OutputSink,
):
    """Compile a LangGraph that drives one segment through the pipeline."""
    if not engines:
        raise ValueError("build_graph requires at least one engine")

    def judge_node(state: GraphState) -> dict[str, Any]:
        decision = judge.decide(state["hypotheses"], state["ctx"])
        return {"decision": decision}

    def output_node(state: GraphState) -> dict[str, Any]:
        sink.emit(state["decision"], state["ctx"])
        return {}

    g = StateGraph(GraphState)
    g.add_node("fan_out", _fan_out)
    for e in engines:
        g.add_node(f"stt_{e.id}", _make_stt_node(e))
    g.add_node("judge", judge_node)
    g.add_node("output", output_node)

    engine_ids = [e.id for e in engines]
    g.add_edge(START, "fan_out")
    g.add_conditional_edges(
        "fan_out",
        lambda s: _route_to_engines(s, engine_ids),
        [f"stt_{eid}" for eid in engine_ids],
    )
    for eid in engine_ids:
        g.add_edge(f"stt_{eid}", "judge")
    g.add_edge("judge", "output")
    g.add_edge("output", END)

    return g.compile()


def run_source(
    compiled_graph,
    source: AudioSource,
    language_hint: str | None = None,
) -> None:
    for seg in source.stream():
        ctx = Context(trace_id=uuid.uuid4().hex[:8], language_hint=language_hint)
        compiled_graph.invoke({"segment": seg, "ctx": ctx, "hypotheses": []})
