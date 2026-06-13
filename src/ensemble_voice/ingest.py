"""Audio ingest sources. M0 ships a file-based source only."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from .types import AudioSegment


TARGET_SR = 16_000


def _to_mono(pcm: np.ndarray) -> np.ndarray:
    if pcm.ndim == 1:
        return pcm
    return pcm.mean(axis=1)


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return pcm
    duration = pcm.shape[0] / src_sr
    dst_n = int(round(duration * dst_sr))
    src_x = np.linspace(0.0, duration, num=pcm.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_x, src_x, pcm).astype(np.float32)


class FileSource:
    """Load a single audio file as one AudioSegment.

    No VAD in M0: the whole file is treated as a single utterance.
    """

    def __init__(self, path: str | Path, source_id: str | None = None) -> None:
        self.path = Path(path)
        self.source_id = source_id or self.path.stem

    def stream(self) -> Iterable[AudioSegment]:
        pcm, sr = sf.read(str(self.path), dtype="float32", always_2d=False)
        pcm = _to_mono(pcm).astype(np.float32, copy=False)
        pcm = _resample_linear(pcm, sr, TARGET_SR)
        yield AudioSegment(
            segment_id=uuid.uuid4().hex[:8],
            pcm=pcm,
            sample_rate=TARGET_SR,
            start_ts=0.0,
            end_ts=pcm.shape[0] / TARGET_SR,
            source_id=self.source_id,
        )
