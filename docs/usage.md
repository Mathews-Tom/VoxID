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

| Method   | Path                      | Description                |
| -------- | ------------------------- | -------------------------- |
| `POST`   | `/identities`             | Create identity            |
| `GET`    | `/identities`             | List identities            |
| `GET`    | `/identities/{id}`        | Get identity               |
| `DELETE` | `/identities/{id}`        | Delete identity            |
| `POST`   | `/identities/{id}/styles` | Add style                  |
| `GET`    | `/identities/{id}/styles` | List styles                |
| `POST`   | `/generate`               | Single-shot generation     |
| `POST`   | `/generate/segments`      | Segment generation         |
| `POST`   | `/generate/manifest`      | Manifest-driven generation |
| `POST`   | `/generate/stream`        | SSE streaming generation   |
| `POST`   | `/route`                  | Route without generating   |
| `GET`    | `/health`                 | Health check               |
| `POST`   | `/enroll/sessions`                | Create enrollment session  |
| `GET`    | `/enroll/sessions/{id}`           | Get session status         |
| `POST`   | `/enroll/sessions/{id}/samples`   | Upload audio sample        |
| `POST`   | `/enroll/sessions/{id}/complete`  | Finalize enrollment        |
| `DELETE` | `/enroll/sessions/{id}`           | Abandon session            |
| `GET`    | `/enroll/prompts`                 | Get prompts for style      |
| `GET`    | `/enroll/prompts/next`            | Adaptive prompt            |

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
curl -X POST http://localhost:8765/identities \
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
curl -X POST http://localhost:8765/identities/alice/styles \
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
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The cache hit ratio improved by 15%.",
    "identity_id": "alice"
  }'
```

**Segment generation:**

```bash
curl -X POST http://localhost:8765/generate/segments \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long form text here...",
    "identity_id": "alice",
    "stitch": true
  }'
```

**Route (dry run):**

```bash
curl -X POST http://localhost:8765/route \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking: production database is down.",
    "identity_id": "alice"
  }'
```

**Create enrollment session:**

```bash
curl -X POST http://localhost:8765/enroll/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "identity_id": "alice",
    "styles": ["conversational", "technical"],
    "prompts_per_style": 5
  }'
```

**Upload audio sample:**

```bash
curl -X POST http://localhost:8765/enroll/sessions/{session_id}/samples \
  -F "file=@recording.wav"
```

**Complete enrollment:**

```bash
curl -X POST http://localhost:8765/enroll/sessions/{session_id}/complete
```

**SSE streaming:**

```bash
curl -N -X POST http://localhost:8765/generate/stream \
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
