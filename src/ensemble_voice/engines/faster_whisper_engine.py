"""faster-whisper wrapper conforming to the STTEngine protocol."""

from __future__ import annotations

import time

from faster_whisper import WhisperModel

from ..types import AudioSegment, Context, Hypothesis, Token


class FasterWhisperEngine:
    def __init__(
        self,
        id: str,
        model: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 1,
    ) -> None:
        self.id = id
        self._language = language
        self._beam_size = beam_size
        self._model = WhisperModel(
            model_size_or_path=model,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, seg: AudioSegment, ctx: Context) -> Hypothesis:
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            seg.pcm,
            language=self._language or ctx.language_hint,
            beam_size=self._beam_size,
            word_timestamps=False,
        )
        # Drain the generator while collecting token-level stats.
        texts: list[str] = []
        tokens: list[Token] = []
        logprob_sum = 0.0
        logprob_n = 0
        no_speech_sum = 0.0
        no_speech_n = 0
        for s in segments:
            texts.append(s.text)
            tokens.append(Token(text=s.text, logprob=s.avg_logprob, start=s.start, end=s.end))
            if s.avg_logprob is not None:
                logprob_sum += s.avg_logprob
                logprob_n += 1
            if s.no_speech_prob is not None:
                no_speech_sum += s.no_speech_prob
                no_speech_n += 1

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return Hypothesis(
            stt_id=self.id,
            text="".join(texts).strip(),
            tokens=tokens,
            avg_logprob=(logprob_sum / logprob_n) if logprob_n else None,
            no_speech_prob=(no_speech_sum / no_speech_n) if no_speech_n else None,
            language=info.language,
            latency_ms=latency_ms,
            raw={"language_probability": info.language_probability},
        )
