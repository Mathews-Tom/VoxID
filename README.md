# VoxID

Voice Identity Management Platform — a local-first Python library, CLI, and REST API for managing persistent voice identities across multiple TTS engines.

VoxID sits between your application and TTS engines. It introduces **voice identities** — named entities that own multiple voice styles, each backed by precomputed speaker embeddings, versioned on disk, and automatically selected based on text content.

## Features

- **Multi-style voice identities** — named entities with multiple registers (conversational, technical, narration, emphatic), persisted as TOML + SafeTensors
- **Content-aware style routing** — two-tier classifier analyzes input text and selects the appropriate voice register automatically
- **Engine-agnostic generation** — single API across Qwen3-TTS, Fish Speech, CosyVoice2, IndexTTS-2, and Chatterbox
- **Segment-level routing** — long-form text is split at prosodic boundaries, each segment routed independently with smoothing to prevent style thrashing
- **Portable `.voxid` archives** — HMAC-signed archives with consent records for identity transfer and backup
- **AudioSeal watermarking** — provenance tracking embedded in generated audio (optional, requires `audioseal`)
- **Scripted voice enrollment** — guided recording with phonetically balanced prompts, real-time quality feedback, adaptive phoneme coverage tracking, and multi-sample fusion
- **Voice drift detection** — cosine similarity monitoring against enrollment baseline with re-enrollment recommendations
- **Re-enrollment health checks** — age-based and drift-based triggers for enrollment refresh
- **Video pipeline integration** — SceneManifest contract for Manim and Remotion with word-level timing
- **Prompt-as-cache architecture** — engine-specific prompts are a derived cache; switching engines rebuilds the cache, not the enrollment

## Supported Engines

| Engine      | Slug          | Streaming | Emotion Control     | Languages                                   |
| ----------- | ------------- | --------- | ------------------- | ------------------------------------------- |
| Qwen3-TTS   | `qwen3-tts`   | —         | —                   | 10 (en, zh, ja, ko, de, fr, ru, pt, es, it) |
| Fish Speech | `fish-speech` | Yes       | —                   | 10 (en, zh, ja, ko, es, pt, ar, ru, fr, de) |
| CosyVoice2  | `cosyvoice2`  | Yes       | —                   | 9 (en, zh, ja, ko, de, fr, ru, pt, es)      |
| IndexTTS-2  | `indextts2`   | Yes       | Yes (disentangled)  | 2 (en, zh)                                  |
| Chatterbox  | `chatterbox`  | Yes       | Paralinguistic tags | 22                                          |
| Stub        | `stub`        | Yes       | —                   | 3 (en, zh, ja) — sine wave, no model needed |

Engines are optional dependencies. Install only what you need:

```bash
uv add voxid[qwen3-tts]       # CUDA/MPS
uv add voxid[qwen3-tts-mlx]   # Apple Silicon via mlx-audio
```

## Installation

Requires Python 3.12+.

```bash
# Core library (includes stub adapter for testing)
uv add voxid

# With Qwen3-TTS on Apple Silicon
uv add voxid[qwen3-tts-mlx]

# Development
git clone https://github.com/Mathews-Tom/VoxID.git
cd VoxID
uv sync --all-extras --group dev
```

## Quickstart

### Python Library

```python
from voxid import VoxID

vox = VoxID()

# Create an identity
vox.create_identity(id="alice", name="Alice")

# Add a voice style with reference audio
vox.add_style(
    identity_id="alice",
    id="conversational",
    label="Conversational",
    description="Warm, relaxed, natural pacing",
    ref_audio="samples/alice_casual.wav",
    ref_text="This is how I normally speak in conversation.",
)

# Or enroll with guided prompts (creates session + generates prompts)
session = vox.enroll("alice", ["conversational", "technical"])

# Generate — style is auto-routed from text content
audio_path, sr = vox.generate(
    text="Let me walk you through how this works.",
    identity_id="alice",
)

# Dry-run routing
decision = vox.route(text="The p99 latency increased after the migration.", identity_id="alice")
# {'style': 'technical', 'confidence': 0.92, 'tier': 'rule-based', 'scores': {...}}
```

### CLI

```bash
# Create identity and add a style
voxid identity create alice --name "Alice"
voxid style add alice conversational \
    --audio samples/alice_casual.wav \
    --transcript "This is how I normally speak." \
    --description "Warm, relaxed, natural pacing"

# Enroll with guided recording (interactive)
voxid enroll alice --styles conversational,technical

# Enroll from pre-recorded audio (non-interactive)
voxid enroll alice --styles conversational --import-audio ./recordings/

# Generate audio
voxid generate "Hello, welcome to the demo." --identity alice

# Generate with explicit style
voxid generate "The API returns a 429 status code." --identity alice --style technical

# Long-form segment generation
voxid generate --file script.txt --identity alice --segments

# Generate from a scene manifest
voxid generate --manifest scenes.json --identity alice

# Check routing decision without generating
voxid route "Breaking news from the lab." --identity alice

# Export/import identities
voxid export alice alice_backup.voxid --key my-signing-key
voxid import alice_backup.voxid --key my-signing-key

# Start the REST API server
voxid serve --port 8765
```

### REST API

```bash
# Start the server
voxid serve

# Create identity
curl -X POST http://localhost:8765/identities \
  -H "Content-Type: application/json" \
  -d '{"id": "alice", "name": "Alice"}'

# Generate audio
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world.", "identity_id": "alice"}'

# Route without generating
curl -X POST http://localhost:8765/route \
  -H "Content-Type: application/json" \
  -d '{"text": "The gradient exploded during training.", "identity_id": "alice"}'

# Create enrollment session
curl -X POST http://localhost:8765/enroll/sessions \
  -H "Content-Type: application/json" \
  -d '{"identity_id": "alice", "styles": ["conversational"], "prompts_per_style": 5}'

# Upload audio sample
curl -X POST http://localhost:8765/enroll/sessions/{id}/samples \
  -F "file=@recording.wav"
```

Set `VOXID_API_KEY` to enable API key authentication. Set `VOXID_RATE_LIMIT` and `VOXID_RATE_WINDOW` to configure rate limiting on generation endpoints.

### Docker

```bash
docker build -t voxid .
docker run -p 8765:8765 -v ~/.voxid:/data/voxid voxid
```

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Consumer Layer                       │
│   Python Library  │  REST API  │  CLI  │  VoiceBox      │
└────────┬──────────┴─────┬──────┴───┬───┴───────┬────────┘
         │                │          │           │
┌────────▼────────────────▼──────────▼───────────▼────────┐
│                     VoxID Core                          │
│  ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐  │
│  │  Identity    │ │   Style     │ │   Generation     │  │
│  │  Registry    │ │   Router    │ │   Dispatcher     │  │
│  └──────┬───────┘ └──────┬──────┘ └────────┬─────────┘  │
│         │                │                 │            │
│  ┌──────▼──────────┐ ┌───▼─────────────────▼─────────┐  │
│  │   Enrollment    │ │  Voice Prompt Store           │  │
│  │   Pipeline      │ │  (TOML + SafeTensors)         │  │
│  └─────────────────┘ └───────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   Engine Adapters                       │
│  Qwen3-TTS │ Fish Speech │ CosyVoice2 │ IndexTTS-2  │ … │
└─────────────────────────────────────────────────────────┘
```

**Storage layout:**

```text
~/.voxid/
├── config.toml
├── identities/
│   └── alice/
│       ├── identity.toml
│       ├── consent.json
│       ├── consent_audio.wav              # recorded consent (enrollment)
│       └── styles/
│           └── conversational/
│               ├── style.toml
│               ├── ref_audio.wav          # source of truth
│               ├── ref_text.txt           # source of truth
│               └── prompts/               # derived cache
│                   ├── qwen3-tts.safetensors
│                   └── fish-speech.safetensors
├── enrollment_sessions/                   # resumable enrollment state
│   └── {session_id}.json
├── cache/
│   └── router/
│       └── router_cache.db
└── output/
```

## Configuration

VoxID reads `~/.voxid/config.toml`:

```toml
store_path = "~/.voxid"
default_engine = "qwen3-tts"
router_confidence_threshold = 0.8
cache_ttl_seconds = 3600
max_embedding_versions = 3
```

### Environment Variables

| Variable            | Description                                           | Default |
| ------------------- | ----------------------------------------------------- | ------- |
| `VOXID_API_KEY`     | API key for REST authentication (unset = open access) | —       |
| `VOXID_RATE_LIMIT`  | Max requests per window on `/generate*` endpoints     | `60`    |
| `VOXID_RATE_WINDOW` | Rate limit window in seconds                          | `60`    |
| `VOXID_STORE_PATH`  | Override store path (used by Docker)                  | —       |

## Documentation

| Document                               | Description                                                           |
| -------------------------------------- | --------------------------------------------------------------------- |
| [Usage Guide](docs/usage.md)           | CLI, Python library, REST API, segments, manifests, video integration |
| [Developer Guide](docs/development.md) | Setup, project structure, testing, writing adapters, contributing     |
| [System Design](docs/system-design.md) | Architecture, data model, router algorithms, security                 |
| [Overview](docs/overview.md)           | Product overview, market analysis, technology landscape               |

## License

MIT
