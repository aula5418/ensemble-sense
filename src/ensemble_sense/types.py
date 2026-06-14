"""Core data types shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

STTId = str


@dataclass
class AudioSegment:
    """A chunk of mono PCM audio with timing metadata."""

    segment_id: str
    pcm: np.ndarray         # float32, mono, normalized [-1, 1]
    sample_rate: int
    start_ts: float = 0.0   # seconds since stream start
    end_ts: float = 0.0
    source_id: str = "default"

    @property
    def duration(self) -> float:
        return float(self.pcm.shape[0]) / float(self.sample_rate)


@dataclass
class Token:
    text: str
    logprob: float | None = None
    start: float | None = None
    end: float | None = None


@dataclass
class Hypothesis:
    """A single STT engine's output for one segment."""

    stt_id: STTId
    text: str
    tokens: list[Token] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    language: str | None = None
    latency_ms: int = 0
    raw: Any = None         # engine-native payload for debugging


@dataclass
class Decision:
    """Judge output for one segment."""

    text: str
    chosen_stt: STTId | Literal["merged"]
    confidence: float
    rationale: str
    hypotheses: list[Hypothesis] = field(default_factory=list)


@dataclass
class Context:
    """Per-request context passed through the pipeline."""

    trace_id: str
    language_hint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


@runtime_checkable
class AudioSource(Protocol):
    def stream(self) -> "Iterable[AudioSegment]": ...


@runtime_checkable
class STTEngine(Protocol):
    id: STTId

    def transcribe(self, seg: AudioSegment, ctx: Context) -> Hypothesis: ...


@runtime_checkable
class Judge(Protocol):
    def decide(self, hyps: list[Hypothesis], ctx: Context) -> Decision: ...


@runtime_checkable
class OutputSink(Protocol):
    def emit(self, decision: Decision, ctx: Context) -> None: ...


# Late import for type hint above without runtime cost.
from collections.abc import Iterable  # noqa: E402
