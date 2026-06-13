"""Output sinks. M0/M1 ship console output only."""

from __future__ import annotations

from typing import Any

from .types import Context, Decision


class ConsoleSink:
    name = "console"

    def __init__(self, show_hypotheses: bool = True) -> None:
        self.show_hypotheses = show_hypotheses

    def emit(self, decision: Decision, ctx: Context) -> None:
        print(f"\n=== segment (trace={ctx.trace_id}) ===")
        if self.show_hypotheses:
            for h in decision.hypotheses:
                marker = "★" if h.stt_id == decision.chosen_stt else " "
                print(
                    f" {marker} [{h.stt_id:<14}] "
                    f"lp={h.avg_logprob if h.avg_logprob is None else f'{h.avg_logprob:+.3f}':>7} "
                    f"ns={h.no_speech_prob if h.no_speech_prob is None else f'{h.no_speech_prob:.2f}':>5} "
                    f"lat={h.latency_ms:>5}ms  {h.text!r}"
                )
        print(f"--> chosen: {decision.chosen_stt}  (conf~{decision.confidence:.3f})")
        print(f"    text: {decision.text}")
        print(f"    why : {decision.rationale}")


_REGISTRY: dict[str, Any] = {
    "console": ConsoleSink,
}


def build_sink(spec: dict[str, Any]):
    spec = dict(spec)
    kind = spec.pop("sink", "console")
    if kind not in _REGISTRY:
        raise ValueError(f"unknown output sink: {kind!r}")
    return _REGISTRY[kind](**spec)
