"""STT pool: run multiple engines on the same segment concurrently."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .types import AudioSegment, Context, Hypothesis, STTEngine


class STTPool:
    def __init__(self, engines: list[STTEngine]) -> None:
        if not engines:
            raise ValueError("STTPool requires at least one engine")
        self._engines = engines
        # One worker per engine: faster-whisper releases the GIL during
        # inference, so threads give real parallelism on CPU.
        self._pool = ThreadPoolExecutor(max_workers=len(engines))

    @property
    def engine_ids(self) -> list[str]:
        return [e.id for e in self._engines]

    def transcribe_all(self, seg: AudioSegment, ctx: Context) -> list[Hypothesis]:
        futures = [self._pool.submit(e.transcribe, seg, ctx) for e in self._engines]
        return [f.result() for f in futures]

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
