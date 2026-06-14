"""SenseVoice (FunAudioLLM) wrapper via funasr's AutoModel."""

from __future__ import annotations

import time

from ..types import AudioSegment, Context, Hypothesis


class SenseVoiceEngine:
    def __init__(
        self,
        id: str,
        model: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        language: str = "auto",
        use_itn: bool = True,
        vad_model: str | None = None,
    ) -> None:
        # Import lazily so users without funasr can still load the package.
        from funasr import AutoModel

        self.id = id
        self._language = language
        self._use_itn = use_itn
        kwargs: dict = {
            "model": model,
            "trust_remote_code": True,
            "device": device,
            "disable_update": True,
        }
        if vad_model:
            kwargs["vad_model"] = vad_model
        self._model = AutoModel(**kwargs)

    def transcribe(self, seg: AudioSegment, ctx: Context) -> Hypothesis:
        t0 = time.perf_counter()
        result = self._model.generate(
            input=seg.pcm,
            cache={},
            language=ctx.language_hint or self._language,
            use_itn=self._use_itn,
            batch_size_s=60,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # funasr returns a list of dicts with "text" — SenseVoice prefixes
        # tags like <|ja|><|NEUTRAL|><|Speech|><|withitn|> that we strip.
        text = ""
        if result:
            raw_text = result[0].get("text", "")
            text = _strip_sensevoice_tags(raw_text).strip()

        return Hypothesis(
            stt_id=self.id,
            text=text,
            avg_logprob=None,        # SenseVoice doesn't surface logprobs
            no_speech_prob=None,
            language=ctx.language_hint or self._language,
            latency_ms=latency_ms,
            raw=result,
        )


def _strip_sensevoice_tags(s: str) -> str:
    """Remove the <|...|> prefix tags SenseVoice emits."""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "<" and s.startswith("<|", i):
            close = s.find("|>", i)
            if close != -1:
                i = close + 2
                continue
        out.append(s[i])
        i += 1
    return "".join(out)
