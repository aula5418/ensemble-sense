"""ReazonSpeech (NeMo flavor) wrapper."""

from __future__ import annotations

import time

import numpy as np

from ..types import AudioSegment, Context, Hypothesis


class ReazonNemoEngine:
    def __init__(self, id: str, device: str = "cpu") -> None:
        from reazonspeech.nemo.asr import load_model  # noqa: F401  (heavy)

        self.id = id
        self._device = device
        self._model = load_model(device=device)

    def transcribe(self, seg: AudioSegment, ctx: Context) -> Hypothesis:
        from reazonspeech.nemo.asr import transcribe
        from reazonspeech.nemo.asr.audio import PaddedAudio

        t0 = time.perf_counter()
        audio = PaddedAudio(
            waveform=seg.pcm.astype(np.float32, copy=False),
            samplerate=seg.sample_rate,
        )
        ret = transcribe(self._model, audio)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        return Hypothesis(
            stt_id=self.id,
            text=(ret.text or "").strip(),
            avg_logprob=None,        # NeMo CTC doesn't surface per-utterance logprob here
            no_speech_prob=None,
            language="ja",
            latency_ms=latency_ms,
            raw=ret,
        )
