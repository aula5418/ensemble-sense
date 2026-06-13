"""STT engine implementations and registry."""

from __future__ import annotations

from typing import Any, Callable

from ..types import STTEngine


def _faster_whisper_factory(**kw):
    from .faster_whisper_engine import FasterWhisperEngine
    return FasterWhisperEngine(**kw)


def _sensevoice_factory(**kw):
    from .sensevoice_engine import SenseVoiceEngine
    return SenseVoiceEngine(**kw)


def _reazon_nemo_factory(**kw):
    from .reazon_engine import ReazonNemoEngine
    return ReazonNemoEngine(**kw)


_REGISTRY: dict[str, Callable[..., STTEngine]] = {
    "faster_whisper": _faster_whisper_factory,
    "sensevoice": _sensevoice_factory,
    "reazon_nemo": _reazon_nemo_factory,
}


def build_engine(spec: dict[str, Any]) -> STTEngine:
    """Build an STT engine from a config dict.

    Required keys: ``id``, ``engine``. Remaining keys are passed to the
    engine constructor.
    """
    spec = dict(spec)
    engine_id = spec.pop("id")
    engine_kind = spec.pop("engine")
    if engine_kind not in _REGISTRY:
        raise ValueError(f"unknown engine kind: {engine_kind!r}")
    return _REGISTRY[engine_kind](id=engine_id, **spec)


__all__ = ["build_engine"]
