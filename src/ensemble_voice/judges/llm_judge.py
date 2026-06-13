"""LLM-based Judges.

Two backends ship out of the box:

- ``AnthropicAPIBackend`` — uses the Anthropic SDK (needs ANTHROPIC_API_KEY).
- ``ClaudeCodeCLIBackend`` — shells out to the ``claude`` CLI in print
  mode, reusing the user's existing Claude Code session. No API key needed.

The :class:`LLMJudge` orchestrates the prompt and parsing; backends only
implement ``run(prompt) -> str``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Protocol

from ..types import Context, Decision, Hypothesis


SYSTEM_PROMPT = """\
You arbitrate between several speech-to-text (STT) transcripts of the SAME
audio clip. Each transcript is produced by a different engine and may
contain different errors.

Pick the transcript that is most likely to be the accurate transcription
of what was said. Prefer transcripts that:
- are linguistically natural (correct grammar / collocations),
- are internally consistent (no garbled fragments),
- use plausible proper nouns / numbers / loanwords,
- match the dominant language of the clip (if a language hint is given).

If two transcripts are equally plausible, prefer the longer one only if
the extra content is itself plausible — short fragments often mean the
engine cut off early.

Reply with ONLY a JSON object:
{
  "chosen_stt": "<engine_id>",
  "text": "<the transcript you chose, verbatim from that engine>",
  "rationale": "<one sentence on why>"
}
Do not include any text outside the JSON."""


class LLMBackend(Protocol):
    def run(self, system: str, user: str) -> str: ...


class AnthropicAPIBackend:
    """Talks to api.anthropic.com using the official SDK."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 600,
        api_key: str | None = None,
    ) -> None:
        import anthropic

        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def run(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )


class ClaudeCodeCLIBackend:
    """Shells out to the ``claude`` CLI in non-interactive print mode."""

    def __init__(
        self,
        binary: str | None = None,
        timeout_s: float = 60.0,
        extra_args: list[str] | None = None,
    ) -> None:
        bin_path = binary or shutil.which("claude")
        if not bin_path:
            raise RuntimeError(
                "claude CLI not found on PATH — install Claude Code or use "
                "the anthropic_api backend"
            )
        self._binary = bin_path
        self._timeout_s = timeout_s
        self._extra_args = list(extra_args or [])

    def run(self, system: str, user: str) -> str:
        # Claude Code's -p / --print mode reads the user message from stdin
        # (or argv) and prints the assistant reply to stdout. We pass the
        # system prompt via --append-system-prompt so we don't override the
        # CLI's own. The user message comes via stdin to avoid shell escaping.
        cmd = [self._binary, "-p", *self._extra_args]
        if system:
            cmd += ["--append-system-prompt", system]
        proc = subprocess.run(
            cmd,
            input=user,
            text=True,
            capture_output=True,
            timeout=self._timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed (rc={proc.returncode}): {proc.stderr.strip()}"
            )
        return proc.stdout


class LLMJudge:
    """Pick the best transcript using an LLM backend."""

    name = "llm"

    def __init__(
        self,
        backend: str = "auto",
        model: str | None = None,
        max_tokens: int = 600,
        timeout_s: float = 60.0,
    ) -> None:
        self._backend = _build_backend(
            backend=backend,
            model=model,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        self._backend_name = backend

    def decide(self, hyps: list[Hypothesis], ctx: Context) -> Decision:
        if not hyps:
            raise ValueError("LLMJudge received no hypotheses")
        if len(hyps) == 1:
            h = hyps[0]
            return Decision(
                text=h.text,
                chosen_stt=h.stt_id,
                confidence=1.0,
                rationale="llm-judge — single hypothesis, no arbitration needed",
                hypotheses=hyps,
            )

        user_msg = _format_user_message(hyps, ctx)
        try:
            raw = self._backend.run(SYSTEM_PROMPT, user_msg)
        except Exception as e:
            return Decision(
                text=hyps[0].text,
                chosen_stt=hyps[0].stt_id,
                confidence=0.5,
                rationale=f"llm-judge backend error ({e!r}); defaulted to first",
                hypotheses=hyps,
            )
        chosen_id, chosen_text, rationale = _parse_response(raw, hyps)
        return Decision(
            text=chosen_text,
            chosen_stt=chosen_id,
            confidence=0.9,
            rationale=f"llm-judge ({self._backend_name}) — {rationale}",
            hypotheses=hyps,
        )


def _build_backend(
    backend: str, model: str | None, max_tokens: int, timeout_s: float
) -> LLMBackend:
    if backend == "auto":
        if shutil.which("claude"):
            backend = "claude_code"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            backend = "anthropic_api"
        else:
            raise RuntimeError(
                "LLMJudge auto-detect: neither `claude` CLI nor "
                "ANTHROPIC_API_KEY available"
            )
    if backend == "claude_code":
        return ClaudeCodeCLIBackend(timeout_s=timeout_s)
    if backend == "anthropic_api":
        return AnthropicAPIBackend(
            model=model or "claude-haiku-4-5", max_tokens=max_tokens
        )
    raise ValueError(f"unknown LLM backend: {backend!r}")


def _format_user_message(hyps: list[Hypothesis], ctx: Context) -> str:
    lines = []
    if ctx.language_hint:
        lines.append(f"Language hint: {ctx.language_hint}")
    lines.append("Transcripts:")
    for h in hyps:
        lines.append(f"- engine_id: {h.stt_id}")
        lines.append(f"  text: {h.text!r}")
        meta = []
        if h.avg_logprob is not None:
            meta.append(f"avg_logprob={h.avg_logprob:.3f}")
        if h.no_speech_prob is not None:
            meta.append(f"no_speech_prob={h.no_speech_prob:.3f}")
        meta.append(f"latency_ms={h.latency_ms}")
        lines.append(f"  meta: {', '.join(meta)}")
    return "\n".join(lines)


def _parse_response(raw: str, hyps: list[Hypothesis]) -> tuple[str, str, str]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return hyps[0].stt_id, hyps[0].text, f"LLM did not return JSON; raw={raw[:200]!r}"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return hyps[0].stt_id, hyps[0].text, f"LLM JSON parse failed: {e}"

    chosen_id = obj.get("chosen_stt", "")
    matched = next((h for h in hyps if h.stt_id == chosen_id), None)
    if matched is None:
        return hyps[0].stt_id, hyps[0].text, (
            f"LLM picked unknown engine_id {chosen_id!r}; defaulting to first"
        )
    # Trust the engine's verbatim text over any LLM rewording.
    chosen_text = matched.text
    return chosen_id, chosen_text, obj.get("rationale", "")
