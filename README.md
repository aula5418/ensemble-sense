# ensemble-voice

Pluggable **multi-STT ensemble** pipeline. Run several speech-to-text
engines in parallel on the same audio, then let a configurable **Judge**
layer (confidence, voting, or LLM arbitration) decide which transcript
to trust.

Unlike single-STT stacks (WhisperX, Willow, LocalAI), ensemble-voice
treats *STT selection* as a first-class layer.

See [`DESIGN.md`](./DESIGN.md) for the full architecture.

## What's in v0.0.1

- **4 engines** out of the box:
  - `faster_whisper` (any model size)
  - `sensevoice` (FunAudioLLM SenseVoiceSmall, via funasr)
  - `reazon_nemo` (ReazonSpeech NeMo flavor, Japanese-focused)
  - …plus a Protocol so adding more is a single file
- **2 Judges**:
  - `confidence` — picks the highest scaled `avg_logprob` − `no_speech_prob` penalty
  - `llm` — asks an LLM to arbitrate; two backends:
    - `claude_code` — shells out to the local `claude` CLI (no API key needed)
    - `anthropic_api` — uses ANTHROPIC_API_KEY via the official SDK
- File ingest, console output sink, YAML config.

## Quickstart

```bash
git clone https://github.com/aula5418/ensemble-voice
cd ensemble-voice

# Bootstrap a venv (system pip not required)
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install -e .

# Minimal: 2 Whisper sizes + confidence Judge (no heavy deps)
.venv/bin/ensemble-voice run \
  --config configs/whisper_only.yaml \
  path/to/audio.wav

# Full: 4 engines + LLM Judge backed by your local Claude Code
.venv/bin/pip install funasr torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install "git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/nemo-asr"
.venv/bin/ensemble-voice run \
  --config configs/example.yaml \
  --language ja \
  path/to/audio_ja.wav
```

## Architecture (6 layers)

```
[1] Audio Ingest  →  [2] Router  →  [3] STT Pool  →
[4] Judge / Arbiter  →  [5] Intent / Dispatcher  →  [6] Output Control
```

Orchestrated as a **LangGraph** state graph:

```
START ──► fan_out ──┬─► stt_<id-1> ─┐
                    ├─► stt_<id-2> ─┤
                    └─► stt_<id-N> ─┴─► judge ─► output ─► END
```

`fan_out` dispatches the same segment to every engine via LangGraph's
`Send` API; hypotheses accumulate into a single list via an additive
reducer; the `judge` node aggregates and decides.

The Judge layer is the differentiator. Plug your own with:

```python
# my_judge.py
from ensemble_voice.types import Judge, Decision, Context, Hypothesis

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
- [x] **M1** — STT Pool (2+ engines) + Confidence Judge
- [x] **M1.5** — SenseVoice + ReazonSpeech engines, LLM Judge (Anthropic + Claude Code backends)
- [ ] **M2** — Router + selective dispatch + observability UI
- [ ] **M3** — Intent/Dispatcher with MCP tool calls
- [ ] **M4** — TTS output + feedback loop + plugin entry points
- [ ] **M5** — Docker reference deployment + docs

## License

Apache-2.0
