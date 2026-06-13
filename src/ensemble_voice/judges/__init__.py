"""Judge strategies — pick one transcript from N hypotheses."""

from __future__ import annotations

from typing import Any, Callable

from .confidence import ConfidenceJudge


def _confidence_factory(**kw):
    return ConfidenceJudge(**kw)


def _llm_factory(**kw):
    from .llm_judge import LLMJudge
    return LLMJudge(**kw)


_REGISTRY: dict[str, Callable] = {
    "confidence": _confidence_factory,
    "llm": _llm_factory,
}


def build_judge(spec: dict[str, Any]):
    spec = dict(spec)
    strategy = spec.pop("strategy", "confidence")
    if strategy not in _REGISTRY:
        raise ValueError(f"unknown judge strategy: {strategy!r}")
    return _REGISTRY[strategy](**spec)


__all__ = ["build_judge", "ConfidenceJudge"]
