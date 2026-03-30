# Usage Guide

This guide covers the three interfaces to VoxID: the Python library, the CLI, and the REST API. All three share the same core — identity registry, style router, and engine adapters.

## Core Concepts

**Identity** — a named entity (person, brand, character) that owns one or more voice styles. Stored as `identity.toml` + `consent.json` under `~/.voxid/identities/{id}/`.

**Style** — a named voice register within an identity (e.g., `conversational`, `technical`). Defined by reference audio + transcript. Engine-specific prompts are derived from these and cached under `prompts/`.

**Routing** — automatic style selection based on text content. The router classifies text using a two-tier system (rule-based + centroid classifier) and returns a style decision with confidence score.

**Prompt-as-cache** — engine-specific speaker embeddings (`prompts/*.safetensors`) are derived artifacts. Switching an identity to a different engine rebuilds the cache from reference audio — no re-enrollment needed.

---

## Python Library

### Initialization

```python
from voxid import VoxID
from voxid.config import VoxIDConfig
from pathlib import Path

# Default config (~/.voxid/)
vox = VoxID()

# Custom config
config = VoxIDConfig(
    store_path=Path("/data/voices"),
    default_engine="qwen3-tts",
    router_confidence_threshold=0.8,
    cache_ttl_seconds=3600,
)
vox = VoxID(config=config)
```

### Identity Management

```python
# Create
identity = vox.create_identity(
    id="alice",
    name="Alice Chen",
    description="AI engineering lead",
    default_style="conversational",
    metadata={"team": "platform"},
)

# List
identity_ids = vox.list_identities()  # ["alice", "bob"]

# Delete (via store)
vox._store.delete_identity("alice")
```

### Style Management

```python
# Add a style
style = vox.add_style(
    identity_id="alice",
    id="conversational",
    label="Conversational",
    description="Warm, relaxed, natural pacing for casual content",
    ref_audio="samples/alice_casual.wav",
    ref_text="This is how I normally speak in conversation.",
    engine="qwen3-tts",  # optional, defaults to config
    language="en-US",
)

# List styles
style_ids = vox.list_styles("alice")  # ["conversational", "technical"]

# Rebuild prompt cache for a different engine
vox._store.invalidate_prompt_cache("alice", "conversational", "fish-speech")
vox._ensure_prompt("alice", "conversational", "fish-speech")
```

### Enrollment

```python
from voxid import VoxID
from voxid.enrollment import EnrollmentPipeline

vox = VoxID()

# Create an enrollment session with phonetically optimized prompts
pipeline = EnrollmentPipeline(vox)
session = pipeline.create_session(
    identity_id="alice",
    styles=["conversational", "technical"],
    prompts_per_style=5,
)

# Record a sample (audio from microphone or file)
import numpy as np
import soundfile as sf
audio, sr = sf.read("recording.wav")
sample, report = pipeline.record_sample(session, np.asarray(audio), sr)
# report.passed, report.snr_db, report.rejection_reasons

# Finalize — selects best sample per style, registers via add_style()
styles = pipeline.finalize(session)

# Convenience method
session = vox.enroll("alice", ["conversational", "technical"])

# Check enrollment health (age + drift)
health = vox.check_enrollment_health("alice")
# health.re_enrollment_recommended, health.reasons
```

### Generation

```python
# Auto-routed generation
audio_path, sample_rate = vox.generate(
    text="Let me explain how the caching layer works.",
    identity_id="alice",
)

# Explicit style
audio_path, sr = vox.generate(
    text="Breaking: system outage detected.",
    identity_id="alice",
    style="emphatic",
)

# Explicit engine override
audio_path, sr = vox.generate(
    text="Hello world.",
    identity_id="alice",
    engine="fish-speech",
)
```

### Routing (Dry Run)

```python
decision = vox.route(
    text="The transformer attention mechanism computes Q, K, V projections.",
    identity_id="alice",
)
# {
#     "style": "technical",
#     "confidence": 0.94,
#     "tier": "rule-based",
#     "scores": {"technical": 0.94, "conversational": 0.42, ...}
# }
```

### Segment Generation

For long-form text, VoxID splits at prosodic boundaries, routes each segment independently, smooths transitions, generates audio per segment, and optionally stitches into a single file.

```python
result = vox.generate_segments(
    text=open("script.txt").read(),
    identity_id="alice",
    engine="qwen3-tts",
    stitch=True,
    export_plan_path=Path("plan.json"),  # optional
)

# result.segments — list of SegmentResult (index, text, style, audio_path, duration_ms)
# result.stitched_path — Path to stitched WAV (if stitch=True)
# result.total_duration_ms — total audio duration
# result.plan — list of SegmentPlanItem (routing decisions per segment)
```

### Manifest-Based Generation

SceneManifest is the contract for video pipeline integration. Each scene maps to one narration unit with optional style override.

```python
from voxid.schemas import SceneManifest, SceneNarration

manifest = SceneManifest(
    identity_id="alice",
    engine="qwen3-tts",
    scenes=[
        SceneNarration(scene_id="intro", text="Welcome to the demo."),
        SceneNarration(scene_id="tech", text="The API uses REST over HTTP/2.", style="technical"),
        SceneNarration(scene_id="outro", text="Thanks for watching!"),
    ],
    metadata={"project": "demo-video"},
)

# Generate all scenes
result = vox.generate_from_manifest(manifest, stitch=True)

# Dry-run (routing only, no audio)
plan = vox.plan_from_manifest(manifest)
```

### Export and Import

```python
from pathlib import Path

# Export with HMAC signing
vox.export_identity(
    identity_id="alice",
    output_path=Path("alice.voxid"),
    signing_key=b"my-secret-key",
)

# Import with verification
identity = vox.import_identity(
    archive_path=Path("alice.voxid"),
    signing_key=b"my-secret-key",
)
```

### Video Integration

#### Manim

```python
from voxid.video import build_manim_config, build_scene_timings

# After generating from manifest
result = vox.generate_from_manifest(manifest, stitch=True)

# Get timing dict for Manim self.wait() calls
timings = build_scene_timings(result)
# {"intro": 2.4, "tech": 3.1, "outro": 1.8}

# Full Manim config (timings + audio paths + styles)
config = build_manim_config(result)
```

#### Remotion

```python
from voxid.video import build_remotion_props, export_remotion_props

# Build Remotion composition props
props = build_remotion_props(result, fps=30)
# {"fps": 30, "totalDurationInFrames": 219, "scenes": [...]}

# Export to JSON for Remotion to consume
export_remotion_props(result, Path("remotion-props.json"), fps=30)
```

#### FFmpeg Compositing

```python
from voxid.video import check_ffmpeg, composite_video_audio

if check_ffmpeg():
    composite_video_audio(
        video_path=Path("animation.mp4"),
        audio_path=Path("output/stitched.wav"),
        output_path=Path("final.mp4"),
    )
```

---

## CLI

### Identity Commands

```bash
# Create a voice identity
voxid identity create alice --name "Alice Chen" --description "AI engineer"

# Create with custom default style
voxid identity create bob --name "Bob" --default-style narration

# List all identities
voxid identity list
```

### Style Commands

```bash
# Add a style
voxid style add alice conversational \
    --audio samples/alice_casual.wav \
    --transcript "This is how I normally speak in conversation." \
    --description "Warm, relaxed, natural pacing" \
    --language en-US

# Add with explicit engine
voxid style add alice technical \
    --audio samples/alice_technical.wav \
    --transcript "The system processes requests in parallel." \
    --description "Precise, measured, neutral affect" \
    --engine fish-speech

# List styles
voxid style list alice

# Rebuild prompt cache for a different engine
voxid style rebuild alice conversational --engine cosyvoice2
```

### Enrollment Commands

```bash
# Interactive enrollment with guided recording
voxid enroll alice --styles conversational,technical --prompts-per-style 5

# Import pre-recorded audio (non-interactive)
voxid enroll alice --styles conversational --import-audio ./recordings/

# Resume an interrupted session
voxid enroll alice --styles conversational --resume abc12345

# Skip consent for re-enrollment
voxid enroll alice --styles conversational --skip-consent

# Specify audio input device
voxid enroll alice --styles conversational --device "Built-in Microphone"
```

The enrollment flow:

1. Records consent audio (first time only)
2. Displays phonetically balanced prompts one at a time
3. Records audio, validates against quality gates (SNR, duration, speech ratio, RMS, peak, sample rate)
4. Tracks phoneme coverage across recordings
5. Selects best sample per style by SNR
6. Registers styles via `add_style()`

Import mode (`--import-audio`):

- Discovers WAV/MP3 files, matches to styles by filename stem
- Validates each file against the same quality gates
- Reads transcript from `.txt` sidecar files (e.g., `conversational.txt`)

### Generation Commands

```bash
# Auto-routed
voxid generate "Here's what happened today." --identity alice

# Explicit style
voxid generate "Alert: production incident." --identity alice --style emphatic

# Output to specific path
voxid generate "Hello." --identity alice -o output.wav

# From file
voxid generate --file script.txt --identity alice

# Segment generation (long-form)
voxid generate --file article.txt --identity alice --segments

# Segment generation with plan export
voxid generate --file article.txt --identity alice --segments --plan plan.json

# From SceneManifest JSON
voxid generate --manifest scenes.json --identity alice
```

### Routing

```bash
# Dry-run routing decision
voxid route "The gradient descent converged after 50 epochs." --identity alice
# Output:
#   style:      technical
#   confidence: 0.94
#   tier:       rule-based
#   scores:
#     technical            0.94 ██████████████████
#     conversational       0.42 ████████
```

### Export and Import

```bash
# Export with signing
voxid export alice alice_backup.voxid --key my-secret-key

# Import with verification
voxid import alice_backup.voxid --key my-secret-key
```

### API Server

```bash
# Default (0.0.0.0:8765)
voxid serve

# Custom host/port
voxid serve --host 127.0.0.1 --port 9000

# With auto-reload for development
voxid serve --reload
```

---

## REST API

### Endpoints

| Method   | Path                                 | Description                |
| -------- | ------------------------------------ | -------------------------- |
| `POST`   | `/api/identities`                    | Create identity            |
| `GET`    | `/api/identities`                    | List identities            |
| `GET`    | `/api/identities/{id}`               | Get identity               |
| `DELETE` | `/api/identities/{id}`               | Delete identity            |
| `POST`   | `/api/identities/{id}/styles`        | Add style                  |
| `GET`    | `/api/identities/{id}/styles`        | List styles                |
| `POST`   | `/api/generate`                      | Single-shot generation     |
| `POST`   | `/api/generate/segments`             | Segment generation         |
| `POST`   | `/api/generate/manifest`             | Manifest-driven generation |
| `POST`   | `/api/generate/stream`               | SSE streaming generation   |
| `POST`   | `/api/route`                         | Route without generating   |
| `GET`    | `/api/health`                        | Health check               |
| `POST`   | `/api/enroll/sessions`               | Create enrollment session  |
| `GET`    | `/api/enroll/sessions/{id}`          | Get session status         |
| `POST`   | `/api/enroll/sessions/{id}/samples`  | Upload audio sample        |
| `POST`   | `/api/enroll/sessions/{id}/complete` | Finalize enrollment        |
| `DELETE` | `/api/enroll/sessions/{id}`          | Abandon session            |
| `GET`    | `/api/enroll/prompts`                | Get prompts for style      |
| `GET`    | `/api/enroll/prompts/next`           | Adaptive prompt            |
| `GET`    | `/api/v1/serving/health`             | Multi-GPU dispatch status  |

### Authentication

Set the `VOXID_API_KEY` environment variable to enable API key authentication. When set, all endpoints except `/health`, `/docs`, and `/openapi.json` require one of:

- Header: `X-API-Key: <key>`
- Query parameter: `?api_key=<key>`

When `VOXID_API_KEY` is unset, authentication is disabled (open access).

### Rate Limiting

Rate limiting applies to `POST /generate*` endpoints. Configure via environment variables:

- `VOXID_RATE_LIMIT` — max requests per window (default: 60)
- `VOXID_RATE_WINDOW` — window in seconds (default: 60)

Exceeding the limit returns `429 Too Many Requests` with a `Retry-After` header.

Rate limiting keys on the `X-API-Key` header value, or falls back to client IP.

### Example Requests

**Create identity:**

```bash
curl -X POST http://localhost:8765/api/identities \
  -H "Content-Type: application/json" \
  -d '{
    "id": "alice",
    "name": "Alice Chen",
    "description": "AI engineering lead",
    "default_style": "conversational"
  }'
```

**Add style:**

```bash
curl -X POST http://localhost:8765/api/identities/alice/styles \
  -H "Content-Type: application/json" \
  -d '{
    "id": "technical",
    "label": "Technical",
    "description": "Precise, measured delivery",
    "ref_audio_path": "/path/to/ref_audio.wav",
    "ref_text": "The system processes requests in parallel.",
    "language": "en-US"
  }'
```

**Generate audio:**

```bash
curl -X POST http://localhost:8765/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The cache hit ratio improved by 15%.",
    "identity_id": "alice"
  }'
```

**Segment generation:**

```bash
curl -X POST http://localhost:8765/api/generate/segments \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long form text here...",
    "identity_id": "alice",
    "stitch": true
  }'
```

**Route (dry run):**

```bash
curl -X POST http://localhost:8765/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking: production database is down.",
    "identity_id": "alice"
  }'
```

**Create enrollment session:**

```bash
curl -X POST http://localhost:8765/api/enroll/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "identity_id": "alice",
    "styles": ["conversational", "technical"],
    "prompts_per_style": 5
  }'
```

**Upload audio sample:**

```bash
curl -X POST http://localhost:8765/api/enroll/sessions/{session_id}/samples \
  -F "file=@recording.wav"
```

**Complete enrollment:**

```bash
curl -X POST http://localhost:8765/api/enroll/sessions/{session_id}/complete
```

**SSE streaming:**

```bash
curl -N -X POST http://localhost:8765/api/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long form text...",
    "identity_id": "alice",
    "stitch": true
  }'
```

Events emitted: `segment` (per-segment progress), `complete` (final summary), `error` (on failure).

### OpenAPI Documentation

When the server is running, interactive API docs are available at:

- Swagger UI: `http://localhost:8765/docs`
- ReDoc: `http://localhost:8765/redoc`
- OpenAPI JSON: `http://localhost:8765/openapi.json`

---

## Synthesis Detection

Detect AI-generated or deepfake audio using the anti-spoofing ensemble.

```python
from voxid.security.spoofing import SynthesisDetector
from voxid.security.spoofing.config import SpoofingConfig

detector = SynthesisDetector(SpoofingConfig())
decision = detector.detect(audio_array, sr)
# decision.label: SpoofLabel.GENUINE | SYNTHETIC | UNCERTAIN
# decision.score: float [0, 1] (spoofing probability)
# decision.artifact_type: ArtifactType (VOCODER, DIFFUSION, etc.)
# decision.confidence: float [0, 1]
# decision.model_scores: {"aasist": 0.1, "rawnet2": 0.15, "lcnn": 0.08}
```

Diffusion artifact analysis detects synthesis-specific spectral patterns:

```python
from voxid.security.spoofing import DiffusionArtifactAnalyzer

analyzer = DiffusionArtifactAnalyzer()
analysis = analyzer.analyze(audio_array, sr)
# analysis.spectral_smoothness, analysis.temporal_discontinuity
# analysis.harmonic_regularity, analysis.suspicious: bool
```

---

## Unified Tokenizer

Engine-agnostic speaker representation combining acoustic and semantic tokens.

```python
from voxid.tokenizer import UnifiedTokenizer
from voxid.tokenizer.config import TokenizerConfig

tokenizer = UnifiedTokenizer(TokenizerConfig())

# Tokenize a speaker's audio
speaker = tokenizer.tokenize(
    audio_path=Path("ref_audio.wav"),
    identity_id="alice",
    style_id="conversational",
)
# speaker.unified_embedding: NDArray — combined acoustic + semantic
# speaker.acoustic.codes: (n_codebooks, T) at 40 Hz
# speaker.semantic.codes: (T,) at 50 Hz

# Compare speakers
similarity = tokenizer.speaker_similarity(Path("a.wav"), Path("b.wav"))

# Save/load tokenized speaker
tokenizer.save_tokenized(speaker, Path("speaker.safetensors"))
loaded = tokenizer.load_tokenized(Path("speaker.safetensors"))
```

Project unified embeddings to engine-specific space:

```python
from voxid.tokenizer import EngineProjector

projector = EngineProjector()
projector.fit("qwen3-tts", unified_embeddings, engine_embeddings)
engine_emb = projector.project("qwen3-tts", speaker.unified_embedding)
```

---

## Context-Aware Generation

Rolling-window context tracking for prosodic continuity across long documents.

```python
from voxid.context import ContextManager, ContextConditioner
from voxid.context.conditioning import ConditioningConfig

# Track context across segments
ctx_mgr = ContextManager(window_size=5)
ctx_mgr.set_total_segments(10)

# After generating each segment, record its prosodic features
ctx_mgr.record(SegmentHistory(
    text="First paragraph...",
    style="conversational",
    duration_ms=3200,
    final_f0=180.0,
    final_energy=0.05,
    speaking_rate=3.2,
))

# Build context for next segment
context = ctx_mgr.build_context(segment_index=1)
# context.doc_position: 0.1 (10% through document)
# context.history: [SegmentHistory, ...]

# Condition generation based on context
conditioner = ContextConditioner(ConditioningConfig(strength=0.5))
result = conditioner.condition(context, boundary_type="sentence")
# result.ssml_prefix: '<prosody rate="102%" pitch="+1st">'
# result.context_params: {"speed": 1.02, "pitch_hz": 183}
# result.stitch.pause_ms: 200, result.stitch.crossfade_ms: 20
```

---

## Multi-GPU Serving

Async GPU dispatcher for high-throughput TTS serving with vLLM integration.

### Configuration

Create `serving.toml`:

```toml
dispatch_strategy = "round_robin"  # or "least_loaded"
health_check_interval_s = 30.0

[[workers]]
engine = "qwen3-tts"
device = "cuda:0"
max_batch_size = 4
max_queue_depth = 16

[[workers]]
engine = "fish-speech"
device = "cuda:1"
max_batch_size = 2
max_queue_depth = 8
```

### CLI

```bash
# Start server with multi-GPU dispatch
voxid serve --port 8765 --config serving.toml
```

### REST API

```bash
# Check worker health
curl http://localhost:8765/api/v1/serving/health
# {"enabled": true, "total_workers": 2, "healthy_workers": 2, "workers": [...]}
```

### Python API

```python
from voxid.serving import GPUDispatcher, load_serving_config

config = load_serving_config(Path("serving.toml"))
dispatcher = GPUDispatcher(config)
await dispatcher.start()

result = await dispatcher.dispatch(GenerationRequest(
    request_id="req-1",
    text="Hello world",
    prompt_path=Path("prompt.safetensors"),
    engine="qwen3-tts",
))
# result.waveform: NDArray, result.sample_rate: int

health = dispatcher.health()
# health.total_workers, health.healthy_workers, health.workers

await dispatcher.stop()
```

---

## Cross-Lingual Enrollment

Enroll and generate in multiple languages while maintaining speaker identity.

### CLI

```bash
# Enroll with Chinese prompts
voxid enroll alice --styles conversational --language zh

# Enroll with Japanese prompts
voxid enroll alice --styles narration --language ja
```

### Supported Languages by Engine

| Engine      | Languages                              |
| ----------- | -------------------------------------- |
| Qwen3-TTS   | en, zh, ja, ko, de, fr, ru, pt, es, it |
| Fish Speech | en, zh, ja, ko, es, pt, ar, ru, fr, de |
| CosyVoice2  | en, zh, ja, ko, de, fr, ru, pt, es     |
| IndexTTS-2  | en, zh                                 |
| Chatterbox  | 22 languages                           |

The dispatcher automatically routes requests to engines that support the requested language.

---

## Web Enrollment UI

The web UI is served automatically when running `voxid serve`:

- **Root:** http://localhost:8765/ — main application
- **Enrollment:** http://localhost:8765/enrollment — guided enrollment with waveform visualization
- **Dashboard:** http://localhost:8765/dashboard — identity overview

Features: real-time waveform + spectrogram visualization, quality meters (SNR, loudness, spectral balance), annotation overlay, session persistence across page refreshes.

---

## SceneManifest Format

The SceneManifest JSON format for video pipeline integration:

```json
{
  "identity_id": "alice",
  "engine": "qwen3-tts",
  "scenes": [
    {
      "scene_id": "intro",
      "text": "Welcome to the demo.",
      "style": null,
      "duration_hint": null,
      "language": null
    },
    {
      "scene_id": "explanation",
      "text": "The API uses token-based authentication.",
      "style": "technical",
      "duration_hint": 5.0,
      "language": "en-US"
    }
  ],
  "metadata": {
    "id": "demo-video-001",
    "project": "onboarding"
  }
}
```

| Field                    | Required | Description                                          |
| ------------------------ | -------- | ---------------------------------------------------- |
| `identity_id`            | Yes      | Target voice identity                                |
| `engine`                 | No       | Engine override for all scenes (default: per-style)  |
| `scenes`                 | Yes      | Ordered list of narration units                      |
| `scenes[].scene_id`      | Yes      | Unique ID within manifest                            |
| `scenes[].text`          | Yes      | Text to synthesize                                   |
| `scenes[].style`         | No       | Explicit style; `null` enables auto-routing          |
| `scenes[].duration_hint` | No       | Advisory target duration in seconds (not enforced)   |
| `scenes[].language`      | No       | BCP-47 override (default: style's language)          |
| `metadata`               | No       | Pass-through dict; `metadata.id` used as manifest ID |

---

## Docker Deployment

```bash
# Build
docker build -t voxid .

# Run with persistent storage
docker run -d \
  -p 8765:8765 \
  -v ~/.voxid:/data/voxid \
  -e VOXID_API_KEY=your-key-here \
  -e VOXID_RATE_LIMIT=100 \
  voxid
```

The container stores identity data at `/data/voxid` (mapped via `VOXID_STORE_PATH`). Mount a host volume to persist data across container restarts.
