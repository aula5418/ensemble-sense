"""Confidence-based Judge — picks the hypothesis with the best score."""

from __future__ import annotations

import math

from ..types import Context, Decision, Hypothesis


def _score(h: Hypothesis) -> float:
    if h.avg_logprob is not None:
        score = h.avg_logprob
    else:
        score = -1.0 if h.text else -10.0
    if h.no_speech_prob is not None:
        score -= 2.0 * h.no_speech_prob
    return score


class ConfidenceJudge:
    name = "confidence"

    def decide(self, hyps: list[Hypothesis], ctx: Context) -> Decision:
        if not hyps:
            raise ValueError("ConfidenceJudge received no hypotheses")
        scored = [(h, _score(h)) for h in hyps]
        scored.sort(key=lambda p: p[1], reverse=True)
        winner, top_score = scored[0]

        rationale = "confidence-judge — " + " | ".join(
            f"{h.stt_id}: lp={h.avg_logprob}, ns={h.no_speech_prob}, "
            f"score={s:.3f}, lat={h.latency_ms}ms"
            for h, s in scored
        )
        confidence = math.exp(top_score) if top_score < 0 else 1.0
        return Decision(
            text=winner.text,
            chosen_stt=winner.stt_id,
            confidence=confidence,
            rationale=rationale,
            hypotheses=hyps,
        )
