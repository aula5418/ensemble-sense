# ensemble-sense — Design Doc

A pluggable, **multimodal** ensemble pipeline. Run multiple models of
the same modality (STT engines today; LLMs, vision/OCR, TTS later) in
parallel on the same input, then let a configurable **Judge** layer
decide which output to trust.

Unlike single-model stacks (WhisperX, Willow, LocalAI), ensemble-sense
treats **model selection / arbitration as a first-class layer** — the
same graph drives any modality by swapping the engines in the pool.

---

## Goals

- Run multiple STT engines in parallel (or selectively) on the same audio.
- Decide which transcript to trust via a dedicated **Judge** layer
  (confidence, voting, LLM arbitration, or rules).
- Route the chosen transcript into an Intent layer that drives actions
  and produces controlled output (TTS, side-effects, logs).
- Be OSS, plugin-friendly, and language/engine-agnostic.

## Non-Goals (v1)

- Training new STT models.
- Full-duplex barge-in / interruption handling.
- Cloud-only deployment. (Local-first is the default.)

---

## Pipeline Overview

```
[1] Audio Ingest
    ├ mic / file / stream (WebRTC, RTP)
    └ VAD (Silero-VAD) — utterance segmentation
         ↓
[2] Preprocess / Router
    ├ Denoise (RNNoise)
    ├ Language / quality estimation
    └ Decide which STT(s) receive the segment
         ↓
[3] STT Pool (parallel)
    ├ Whisper (faster-whisper)
    ├ Reazon-NeMo
    ├ Vosk (low-latency lane)
    └ Plugin interface for additional engines
         ↓
[4] Judge / Arbiter            ← core differentiator
    ├ a. Confidence-based (logprob, no_speech_prob)
    ├ b. Voting / ROVER
    ├ c. LLM semantic arbitration
    └ d. Rules (short → Vosk, long → Whisper, etc.)
         ↓
[5] Intent / Dispatcher
    ├ Intent extraction (LLM or rules)
    ├ Command parsing
    └ Tool invocation (MCP, function calling)
         ↓
[6] Output Control
    ├ TTS (VOICEVOX / Style-Bert-VITS2)
    ├ Action execution (shell, API)
    ├ Logs / captions
    └ Feedback loop on misrecognition
```

---

## Layers

### 1. Audio Ingest

Responsibilities:

- Accept audio from mic, file, or network stream (WebRTC / RTP / WebSocket).
- Normalize sample rate, channels, encoding.
- Run **VAD** (Silero-VAD recommended) to cut continuous audio into
  utterance-sized segments. Each segment carries:
  - `segment_id`, `start_ts`, `end_ts`, `pcm`, `source_id`.

Interface:

```python
class AudioSource(Protocol):
    def stream(self) -> Iterator[AudioSegment]: ...
```

### 2. Preprocess / Router

Responsibilities:

- Optional denoise (RNNoise, DeepFilterNet).
- Lightweight **feature extraction**: estimated language, SNR, length,
  speaker hint (from upstream diarization if present).
- **Routing decision**: which STT engines should receive this segment.
  - "Fan-out-all" mode (always send to every engine) for accuracy.
  - "Selective" mode for cost / latency.

Interface:

```python
class Router(Protocol):
    def route(self, seg: AudioSegment) -> list[STTId]: ...
```

### 3. STT Pool

Responsibilities:

- Hold a pool of STT engine workers, each behind a uniform interface.
- Run inference concurrently (asyncio + process/thread pool, or a
  shared GPU batch queue via `faster-whisper` batched inference).
- Emit `Hypothesis` objects with confidence metadata.

Interface:

```python
class STTEngine(Protocol):
    id: STTId
    capabilities: STTCaps          # languages, streaming, max_len
    def transcribe(self, seg: AudioSegment) -> Hypothesis: ...

@dataclass
class Hypothesis:
    stt_id: STTId
    text: str
    tokens: list[Token]            # with logprobs if available
    avg_logprob: float | None
    no_speech_prob: float | None
    language: str | None
    latency_ms: int
```

Bundled engines (v1):

- `faster-whisper` (large-v3, distil, tiny)
- `vosk` (low-latency lane)
- `reazonspeech-nemo` (Japanese)
- Plugin spec for adding more.

### 4. Judge / Arbiter

The core differentiator. Takes the set of hypotheses for one segment and
returns a single chosen transcript plus a rationale.

Strategies (composable, configurable):

| Strategy        | When to use                                     |
| --------------- | ----------------------------------------------- |
| `confidence`    | Pick highest `avg_logprob` above threshold      |
| `vote` (ROVER)  | Token-level alignment + majority vote           |
| `llm_judge`     | LLM picks the most semantically coherent text   |
| `rule`          | Length / language / latency budget heuristics   |
| `cascade`       | Try cheap engine first; escalate on low conf.   |

Interface:

```python
class Judge(Protocol):
    def decide(self, hyps: list[Hypothesis], ctx: Context) -> Decision: ...

@dataclass
class Decision:
    text: str
    chosen_stt: STTId | Literal["merged"]
    confidence: float
    rationale: str                 # for logs / debugging
```

The Judge layer is the place where users will most want to plug in
custom logic; it MUST be swappable without touching other layers.

### 5. Intent / Dispatcher

Responsibilities:

- Convert decided text into a structured **Intent**.
- Dispatch to a tool / action handler.

Two backends:

- **Rule-based**: regex / keyword tables for fixed command sets.
- **LLM-based**: function-calling / MCP tools for open-ended assistants.

Interface:

```python
class IntentResolver(Protocol):
    def resolve(self, text: str, ctx: Context) -> Intent: ...

class Dispatcher(Protocol):
    def dispatch(self, intent: Intent) -> ActionResult: ...
```

### 6. Output Control

Responsibilities:

- Render results back to the user (TTS, captions, UI events).
- Persist transcripts and decisions for audit / retraining.
- Provide a **feedback loop**: if the action handler reports
  "low confidence" or the user issues a correction, re-run the Judge
  with adjusted weights or fall back to the next-best hypothesis.

Output sinks:

- TTS: VOICEVOX, Style-Bert-VITS2, OpenAI TTS (pluggable).
- Action: shell, HTTP, MCP tool call.
- Log: JSONL transcript + decision trace.
- UI: WebSocket events for captions / debug console.

---

## Cross-Cutting Concerns

### Configuration

Single YAML (or TOML) describes the full pipeline:

```yaml
ingest:
  source: mic
  vad: silero
router:
  mode: fan_out_all
stt:
  - id: whisper-large
    engine: faster_whisper
    model: large-v3
    device: cuda
  - id: vosk-ja
    engine: vosk
    model: vosk-model-ja-0.22
judge:
  strategy: cascade
  steps:
    - { type: confidence, threshold: -0.4 }
    - { type: llm_judge, model: claude-haiku-4-5 }
intent:
  backend: llm
  tools: [mcp://localhost/tools]
output:
  tts: voicevox
  log: ./logs/transcript.jsonl
```

### Observability

- Every segment carries a trace ID through all 6 layers.
- Each layer emits structured events (start, end, latency, decision).
- A debug UI replays a session and shows per-engine hypotheses and the
  Judge rationale side-by-side.

### Concurrency Model

- Each layer is an async stage connected by bounded queues.
- STT Pool uses a worker-per-engine + optional GPU batch coordinator.
- Backpressure: if downstream stalls, Router can switch to selective
  mode to shed load.

### Plugin Interface

Engines, Routers, Judges, IntentResolvers, and Output sinks all follow
the same pattern:

- Implement a Protocol.
- Register via entry point (`ensemble_sense.plugins`).
- Declare capabilities + config schema.

---

## Open Decisions

These need to be resolved before implementation starts:

1. **Primary Judge strategy** — confidence, voting, or LLM arbitration as
   the default?
2. **Parallel execution model** — always-on fan-out (high cost,
   maximum accuracy) vs. router-gated selective dispatch (low cost)?
3. **Target use case** — assistant / meeting transcription / live
   captions / general-purpose library? This changes latency targets.
4. **Implementation language** — pure Python, or Rust/Go for the
   ingest + routing layers with Python STT workers?
5. **Distribution form** — library, Docker server, or desktop app?

---

## Milestones (proposed)

- **M0** — Repo skeleton, Protocols, config loader, single Whisper engine
  end-to-end (no Judge, single output).
- **M1** — STT Pool with 2+ engines + Confidence Judge.
- **M2** — Router + selective dispatch + observability UI.
- **M3** — LLM Judge + Intent/Dispatcher with MCP tool calls.
- **M4** — TTS output + feedback loop + plugin entry points.
- **M5** — Docker reference deployment, docs, examples.

---

## References / Prior Art

- WhisperX — Whisper + diarization + alignment (single-STT).
- Willow — local voice assistant (ESP32 + Whisper).
- LocalAI — local OpenAI-compatible server (single-STT per request).
- ROVER — classic multi-system transcript combination algorithm.
- NVIDIA Riva / Triton — multi-model serving infra (closed/proprietary).
