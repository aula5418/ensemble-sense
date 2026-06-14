# ensemble-sense

**Multimodal ensemble pipeline with a first-class Judge layer.**

Run several models in parallel on the same input — speech-to-text engines,
LLMs, vision/OCR models, anything — then let a configurable **Judge**
(confidence, voting, or LLM arbitration) decide which output to trust.

The Judge layer is the differentiator. Unlike single-model stacks
(WhisperX, Willow, LocalAI), ensemble-sense treats *model selection*
as an explicit pipeline stage.

See [`DESIGN.md`](./DESIGN.md) for the full architecture.

## What's in v0.0.2

- **3 STT engines** out of the box (the modality we shipped first):
  - `faster_whisper` (any model size)
  - `sensevoice` (FunAudioLLM SenseVoiceSmall, via funasr)
  - `reazon_nemo` (ReazonSpeech NeMo flavor, Japanese-focused)
- **2 Judges**:
  - `confidence` — picks the highest scaled `avg_logprob` − `no_speech_prob` penalty
  - `llm` — asks an LLM to arbitrate; two backends:
    - `claude_code` — shells out to the local `claude` CLI (no API key needed)
    - `anthropic_api` — uses ANTHROPIC_API_KEY via the official SDK
- **LangGraph-orchestrated**: parallel fan-out, additive reducer for hypotheses,
  conditional edges — the same graph extends to vision/LLM/TTS pools.
- File ingest, console output sink, YAML config.

## Architecture (6 layers)

```
[1] Sense Ingest  →  [2] Router  →  [3] Model Pool  →
[4] Judge / Arbiter  →  [5] Intent / Dispatcher  →  [6] Output Control
```

Orchestrated as a **LangGraph** state graph:

```
START ──► fan_out ──┬─► model_<id-1> ─┐
                    ├─► model_<id-2> ─┤
                    └─► model_<id-N> ─┴─► judge ─► output ─► END
```

`fan_out` dispatches the same input to every model via LangGraph's
`Send` API; outputs accumulate into a list via an additive reducer; the
`judge` node aggregates and decides. The same graph drives any
modality — just swap the engines in the pool.

## Quickstart

```bash
git clone https://github.com/aula5418/ensemble-sense
cd ensemble-sense

# Bootstrap a venv (system pip not required)
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install -e .

# Minimal: 2 Whisper sizes + confidence Judge (no heavy deps)
.venv/bin/ensemble-sense run \
  --config configs/whisper_only.yaml \
  path/to/audio.wav

# Full: 4 engines + LLM Judge backed by your local Claude Code
.venv/bin/pip install funasr torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install "git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/nemo-asr"
.venv/bin/ensemble-sense run \
  --config configs/example.yaml \
  --language ja \
  path/to/audio_ja.wav
```

## Plug in your own

Engines, Judges, and Output sinks are Protocols — drop in your own
without touching the graph:

```python
from ensemble_sense.types import Judge, Decision, Context, Hypothesis

class MyJudge:
    def decide(self, hyps: list[Hypothesis], ctx: Context) -> Decision:
        ...
```

## Configuration

`configs/example.yaml`:

```yaml
stt:
  - { id: whisper-tiny, engine: faster_whisper, model: tiny,  device: cpu, compute_type: int8 }
  - { id: whisper-base, engine: faster_whisper, model: base,  device: cpu, compute_type: int8 }
  - { id: sensevoice,   engine: sensevoice,    model: iic/SenseVoiceSmall, device: cpu, language: auto }
  - { id: reazon,       engine: reazon_nemo,   device: cpu }
judge:
  strategy: llm
  backend: auto       # auto → claude_code if installed, else anthropic_api
output:
  sink: console
```

## Roadmap

- [x] **M0** — file ingest → faster-whisper → console out
- [x] **M1** — Model Pool (2+ engines) + Confidence Judge
- [x] **M1.5** — SenseVoice + ReazonSpeech engines, LLM Judge (Claude Code + Anthropic API backends)
- [x] **M2** — LangGraph orchestration, rename to `ensemble-sense`, multimodal framing
- [ ] **M3** — Router + selective dispatch + observability UI
- [ ] **M4** — Vision / OCR pool (multimodal: same graph, different engines)
- [ ] **M5** — Intent/Dispatcher with MCP tool calls
- [ ] **M6** — TTS output + feedback loop + plugin entry points
- [ ] **M7** — Docker reference deployment + docs

## License

Apache-2.0
