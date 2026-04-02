# Developer Guide

Setup, project structure, testing, and contributing guide for VoxID.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- ffmpeg (optional, for video compositing)

## Setup

```bash
git clone https://github.com/Mathews-Tom/VoxID.git
cd VoxID

# Install all dependencies including dev tools
uv sync --all-extras --group dev

# Verify installation
uv run pytest --tb=short -q
uv run ruff check src/
uv run mypy src/voxid/
```

## Project Structure

```text
VoxID/
├── src/voxid/
│   ├── __init__.py              # Public API exports
│   ├── core.py                  # VoxID orchestrator class
│   ├── models.py                # Domain models (Identity, Style, ConsentRecord)
│   ├── schemas.py               # Pydantic schemas (SceneManifest, GenerationResult)
│   ├── config.py                # VoxIDConfig + config file loading
│   ├── store.py                 # VoicePromptStore — filesystem persistence
│   ├── serialization.py         # SafeTensors read/write + HMAC
│   ├── versioning.py            # Embedding version tracking
│   ├── archive.py               # .voxid archive export/import
│   ├── cli.py                   # Click CLI
│   │
│   ├── enrollment/              # Scripted enrollment pipeline
│   │   ├── __init__.py          # EnrollmentPipeline orchestrator + exports
│   │   ├── phoneme_tracker.py   # CMUdict phoneme lookup + coverage tracking
│   │   ├── script_generator.py  # Greedy phoneme-coverage prompt selection
│   │   ├── quality_gate.py      # 6-gate audio quality validation
│   │   ├── preprocessor.py      # Audio preprocessing (mono, resample, trim, LUFS)
│   │   ├── session.py           # Enrollment session state machine + persistence
│   │   ├── recorder.py          # Callback-based audio recorder + energy VAD
│   │   ├── cli_ui.py            # Terminal display helpers
│   │   ├── consent.py           # Consent recording, SHA-256, legal compliance
│   │   ├── vad.py               # Multi-backend VAD (Silero/WebRTC/energy)
│   │   ├── health.py            # Re-enrollment health check (age + drift)
│   │   ├── multilingual/        # Cross-lingual enrollment
│   │   │   ├── __init__.py      # MultilingualScriptGenerator + exports
│   │   │   ├── language_config.py # Per-language phoneme sets + prompts
│   │   │   ├── phoneme_universal.py # Universal phoneme mapping
│   │   │   └── script_generator.py  # Cross-lingual prompt selection
│   │   └── prompts/             # Bundled JSON corpora (5 styles, 310 sentences)
│   │
│   ├── adapters/                # TTS engine adapters
│   │   ├── __init__.py          # Registry, discovery, @register_adapter
│   │   ├── protocol.py          # TTSEngineAdapter protocol + EngineCapabilities
│   │   ├── stub.py              # Sine wave adapter (no model needed)
│   │   ├── qwen3_tts.py         # Qwen3-TTS (dual backend: qwen-tts + mlx-audio)
│   │   ├── fish_speech.py       # Fish Speech S2 Pro
│   │   ├── cosyvoice2.py        # CosyVoice2
│   │   ├── indextts2.py         # IndexTTS-2 (emotion control)
│   │   └── chatterbox.py        # Chatterbox (paralinguistic tags)
│   │
│   ├── router/                  # Style routing (three-tier)
│   │   ├── __init__.py          # StyleRouter orchestrator
│   │   ├── classifiers.py       # RuleBasedClassifier + CentroidClassifier
│   │   ├── semantic_classifier.py # SemanticStyleClassifier (MLP + n-gram)
│   │   └── cache.py             # SQLite LRU cache for routing decisions
│   │
│   ├── segments/                # Long-form text processing
│   │   ├── __init__.py          # build_segment_plan, export_plan
│   │   ├── segmenter.py         # TextSegmenter — prosodic boundary splitting
│   │   ├── smoother.py          # StyleSmoother — transition smoothing
│   │   └── stitcher.py          # AudioStitcher — adaptive pause concatenation
│   │
│   ├── context/                 # Context-aware generation
│   │   ├── __init__.py          # ContextManager + ContextConditioner exports
│   │   ├── manager.py           # Rolling-window segment history tracking
│   │   └── conditioning.py      # SSML/parameter/stitch conditioning signals
│   │
│   ├── tokenizer/               # Unified speaker tokenization
│   │   ├── __init__.py          # UnifiedTokenizer + EngineProjector exports
│   │   ├── unified.py           # Orchestrates acoustic + semantic tokenization
│   │   ├── acoustic.py          # WavTokenizer-based acoustic encoding (40 Hz)
│   │   ├── semantic.py          # HuBERT-based semantic encoding (50 Hz)
│   │   ├── projection.py        # Linear projection to engine-specific embeddings
│   │   ├── config.py            # Tokenizer configuration
│   │   └── types.py             # TokenizedSpeaker, AcousticTokens, SemanticTokens
│   │
│   ├── serving/                 # Multi-GPU async serving
│   │   ├── __init__.py          # GPUDispatcher + config exports
│   │   ├── dispatcher.py        # Request routing to worker pool
│   │   ├── worker.py            # Single-GPU async worker with queue
│   │   ├── config.py            # ServingConfig, WorkerConfig, GenerationRequest
│   │   └── plugin.py            # vLLM plugin registration entry point
│   │
│   ├── security/                # Security subsystem
│   │   ├── __init__.py
│   │   ├── audit.py             # Source code audit scanner
│   │   ├── consent.py           # Consent validation and enforcement
│   │   ├── drift.py             # Voice drift detection (cosine similarity)
│   │   ├── watermark.py         # AudioSeal watermark embed/detect
│   │   └── spoofing/            # Anti-spoofing ensemble
│   │       ├── __init__.py      # SynthesisDetector + DiffusionArtifactAnalyzer
│   │       ├── detector.py      # Ensemble detector (AASIST + RawNet2 + LCNN)
│   │       ├── models.py        # Individual model wrappers (JIT-loaded)
│   │       ├── diffusion.py     # Diffusion-specific artifact detection
│   │       ├── features.py      # Mel, CQT, LFCC feature extraction
│   │       └── config.py        # SpoofingConfig (weights, thresholds)
│   │
│   ├── api/                     # REST API (FastAPI)
│   │   ├── __init__.py
│   │   ├── app.py               # create_app factory + lifespan (GPU dispatch)
│   │   ├── auth.py              # API key middleware
│   │   ├── rate_limit.py        # Sliding window rate limiter
│   │   ├── deps.py              # Dependency injection (VoxID + GPUDispatcher)
│   │   ├── models.py            # Request/response Pydantic models
│   │   ├── static/enrollment/   # Web enrollment UI (HTML/CSS/JS)
│   │   └── routes/
│   │       ├── __init__.py      # Router aggregation
│   │       ├── identities.py    # CRUD for identities + styles
│   │       ├── generate.py      # Generation endpoints + SSE streaming
│   │       ├── route.py         # Routing dry-run
│   │       ├── health.py        # Health check
│   │       ├── enroll.py        # Enrollment session endpoints
│   │       └── serving.py       # Multi-GPU dispatch health endpoint
│   │
│   ├── video/                   # Video pipeline integration
│   │   ├── __init__.py
│   │   ├── timing.py            # Word-level timing estimation
│   │   ├── manim.py             # Manim scene timing + config
│   │   ├── remotion.py          # Remotion props + scene mapping
│   │   └── ffmpeg.py            # Video/audio compositing
│   │
│   └── plugins/
│       └── voicebox/            # VoiceBox plugin
│           ├── __init__.py
│           ├── models.py        # VoiceBox data models
│           ├── backend.py       # VoxIDBackend adapter for VoiceBox
│           └── sync.py          # Bidirectional profile sync
│
├── tests/                       # Test suite (~825 tests, 80 files)
│   ├── conftest.py              # Shared fixtures
│   ├── test_core.py             # Core orchestrator
│   ├── test_adapters.py         # Engine adapter protocol
│   ├── test_router*.py          # Router, cache, classifiers, edge cases
│   ├── test_semantic_classifier.py # Semantic MLP classifier
│   ├── test_segmenter*.py       # Segmenter, smoother, stitcher, integration
│   ├── test_security_*.py       # Audit, consent, drift, watermark
│   ├── test_api*.py             # API, auth, rate limiting
│   ├── test_video_*.py          # FFmpeg, Manim, Remotion, timing
│   ├── test_voicebox_*.py       # VoiceBox backend, models, sync
│   ├── test_enrollment_*.py     # Full enrollment pipeline
│   ├── test_vad.py              # VAD backend abstraction
│   └── unit/                    # Unit tests for new modules
│       ├── context/             # Context manager + conditioning
│       ├── tokenizer/           # Unified tokenizer + projection
│       ├── serving/             # GPU dispatcher + worker
│       └── security/spoofing/   # Synthesis detector + features
│
├── docs/
│   ├── usage.md                 # User-facing usage guide
│   ├── development.md           # This file
│   ├── system-design.md         # Architecture and design decisions
│   └── overview.md              # Product overview and market analysis
│
├── pyproject.toml               # Project metadata + tool config
├── Dockerfile                   # Multi-stage Docker build
└── uv.lock                      # Locked dependencies
```

## Key Architectural Concepts

### Adapter Protocol

Every TTS engine implements the `TTSEngineAdapter` protocol defined in `adapters/protocol.py`:

```python
@runtime_checkable
class TTSEngineAdapter(Protocol):
    @property
    def engine_name(self) -> str: ...

    @property
    def capabilities(self) -> EngineCapabilities: ...

    def build_prompt(self, ref_audio: Path, ref_text: str, output_path: Path) -> Path: ...
    def generate(self, text: str, prompt_path: Path, language: str = "en", context_params: dict[str, float] | None = None) -> tuple[np.ndarray, int]: ...
    def generate_streaming(self, text: str, prompt_path: Path, language: str = "en") -> Iterator[np.ndarray]: ...
```

`EngineCapabilities` declares what the engine supports:

```python
@dataclass(frozen=True)
class EngineCapabilities:
    supports_streaming: bool = False
    supports_emotion_control: bool = False
    supports_paralinguistic_tags: bool = False
    max_ref_audio_seconds: float = 30.0
    supported_languages: tuple[str, ...] = ("en",)
    streaming_latency_ms: int = 0
    supports_word_timing: bool = False
```

The dispatcher uses these flags for automatic engine selection when the default engine doesn't meet requirements (e.g., language support, streaming).

### Prompt-as-Cache

Engine-specific prompts (speaker embeddings) are stored under `prompts/{engine_slug}.safetensors` inside each style directory. They are **derived** from `ref_audio.wav` + `ref_text.txt` via the adapter's `build_prompt()` method. Deleting a prompt file triggers re-extraction on next generation. This means:

- Switching engines = cache rebuild, not re-enrollment
- Reference audio is the single source of truth
- Prompts are lazily built on first generation (or eagerly via `voxid style rebuild`)

### Style Router

The router in `router/__init__.py` orchestrates three classification tiers:

1. **RuleBasedClassifier** (Tier 1, ~0ms) — keyword/pattern heuristics, returns if confidence ≥ 0.9
2. **SemanticStyleClassifier** (Tier 1.5, ~10ms) — MLP with character/word n-grams, trained with `fit()`, supports contextual blending with neighboring segments
3. **CentroidClassifier** (Tier 2, ~15ms) — TF-IDF + cosine similarity against style centroids, always returns a decision

Routing flow: single style available → return immediately → check SQLite cache → rule-based → semantic MLP → centroid → cache result. All decisions are cached in SQLite with configurable TTL.

### Context-Aware Generation

The `context/` module provides prosodic continuity for long-form generation:

- **ContextManager** — FIFO buffer tracking the last N segment histories (text, style, duration, final F0, energy, speaking rate)
- **ContextConditioner** — translates context into three conditioning signals:
  - **Text-level:** SSML prosody hints (`<prosody rate="..." pitch="...">`)
  - **Parameter-level:** numeric continuity params (speed, pitch_hz, energy)
  - **Stitch-level:** context-derived pause durations and crossfade settings

### Unified Tokenizer

The `tokenizer/` module creates engine-agnostic speaker representations:

- **AcousticTokenizer** — WavTokenizer for causal acoustic encoding at 40 Hz
- **SemanticTokenizer** — HuBERT-based semantic encoding at 50 Hz with k-means quantization
- **UnifiedTokenizer** — combines both into a single unified embedding
- **EngineProjector** — linear OLS projection from unified to engine-specific embedding space

### Synthesis Detection

The `security/spoofing/` module detects AI-generated audio:

- **SynthesisDetector** — weighted ensemble of AASIST (0.4), RawNet2 (0.35), LCNN (0.25) producing GENUINE/SYNTHETIC/UNCERTAIN
- **DiffusionArtifactAnalyzer** — detects diffusion-model-specific spectral artifacts (smoothness, discontinuity, regularity)

### Multi-GPU Serving

The `serving/` module provides async GPU dispatch:

- **GPUDispatcher** — routes requests to a pool of TTSWorkers by engine affinity, using round-robin or least-loaded strategy
- **TTSWorker** — single-GPU async worker with FIFO queue and backpressure (raises `QueueFullError` when full)
- **vLLM Plugin** — registration entry point for vLLM integration via `register_voxid_plugin()`

### Segment Pipeline

For long-form text (`generate_segments`):

1. **TextSegmenter** splits at paragraph and sentence boundaries
2. **StyleRouter** routes each segment independently (three-tier cascade)
3. **StyleSmoother** prevents rapid style switching (minimum segment length + confidence delta threshold)
4. **ContextConditioner** generates prosodic continuity signals from rolling history
5. Audio generated per segment with context-aware parameters
6. **AudioStitcher** concatenates with adaptive pauses (paragraph: 500ms, sentence: 200ms, clause: 100ms) and equal-power crossfades

## Writing a New Adapter

1. Create `src/voxid/adapters/my_engine.py`
2. Implement the `TTSEngineAdapter` protocol
3. Decorate with `@register_adapter`
4. Add the module to `discover_adapters()` in `adapters/__init__.py`

Minimal example:

```python
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..serialization import load_prompt, save_prompt
from . import register_adapter
from .protocol import EngineCapabilities

_SAMPLE_RATE = 24000


@register_adapter
class MyEngineAdapter:
    engine_name: str = "my-engine"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_streaming=False,
            supported_languages=("en", "zh"),
        )

    def _ensure_model(self):
        # Lazy-load the model
        ...

    def build_prompt(self, ref_audio: Path, ref_text: str, output_path: Path) -> Path:
        model = self._ensure_model()
        embedding = model.extract_speaker_embedding(str(ref_audio))
        tensors = {"ref_spk_embedding": np.asarray(embedding, dtype=np.float32)}
        metadata = {"engine": "my-engine", "ref_text": ref_text}
        save_prompt(tensors, output_path, metadata=metadata)
        return output_path

    def generate(self, text: str, prompt_path: Path, language: str = "en") -> tuple[np.ndarray, int]:
        model = self._ensure_model()
        tensors, metadata = load_prompt(prompt_path)
        audio = model.generate(text=text, speaker_embedding=tensors["ref_spk_embedding"])
        return np.asarray(audio, dtype=np.float32), _SAMPLE_RATE

    def generate_streaming(self, text: str, prompt_path: Path, language: str = "en") -> Iterator[np.ndarray]:
        raise NotImplementedError("Streaming not supported")
```

Then add to discovery:

```python
# In adapters/__init__.py, discover_adapters()
for module_name in [
    ...
    "voxid.adapters.my_engine",
]:
```

## Testing

### Running Tests

```bash
# Full suite
uv run pytest

# Verbose with short tracebacks
uv run pytest --tb=short -v

# Specific test file
uv run pytest tests/test_router.py

# Specific test
uv run pytest tests/test_router.py::TestStyleRouter::test_route_single_style

# With coverage
uv run pytest --cov=voxid --cov-report=term-missing
```

### Test Organization

Tests mirror the source structure. Each module has a corresponding `test_*.py` file. Key conventions:

- **Fixtures** in `conftest.py` provide temporary directories, pre-built identities, and store instances
- **The stub adapter** (`engine_name="stub"`) generates sine waves and requires no model downloads — all tests use it by default
- **Integration tests** (`test_segments_integration.py`, `test_qwen3_integration.py`) test multi-component flows

### Writing Tests

Follow `test_<unit>_<scenario>_<expected_outcome>` naming:

```python
def test_router_single_style_returns_default():
    ...

def test_smoother_short_segment_inherits_previous_style():
    ...

def test_archive_import_invalid_hmac_raises_value_error():
    ...
```

Use Arrange-Act-Assert structure. One behavior per test. Mock at boundaries (external APIs, filesystem in unit tests), use real implementations in integration tests.

### Linting and Type Checking

```bash
# Ruff linter
uv run ruff check src/ tests/

# Ruff formatter
uv run ruff format src/ tests/

# mypy strict mode
uv run mypy src/voxid/
```

Configuration is in `pyproject.toml`:

- **ruff**: Python 3.12 target, rules `E`, `F`, `I`, `N`, `W`, `UP`
- **mypy**: strict mode, Python 3.12

## Dependencies

### Core (always installed)

| Package             | Purpose                                       |
| ------------------- | --------------------------------------------- |
| `pydantic`          | Schema validation (API models, SceneManifest) |
| `safetensors`       | Secure tensor serialization for prompts       |
| `soundfile`         | WAV read/write                                |
| `pydub`             | Audio manipulation                            |
| `click`             | CLI framework                                 |
| `tomli` / `tomli-w` | TOML read/write for config and metadata       |
| `numpy`             | Audio array operations                        |
| `fastapi`           | REST API framework                            |
| `uvicorn`           | ASGI server                                   |
| `sse-starlette`     | Server-Sent Events for streaming              |
| `cmudict`           | CMU Pronouncing Dictionary for phoneme lookup |
| `pyloudnorm`        | ITU-R BS.1770-4 LUFS loudness normalization   |
| `sounddevice`       | Audio input capture for enrollment recording  |
| `python-multipart`  | Multipart file upload for enrollment API      |

### Optional (per engine)

| Extra           | Packages    | Engine                    |
| --------------- | ----------- | ------------------------- |
| `qwen3-tts`     | `qwen-tts`  | Qwen3-TTS (CUDA/MPS)      |
| `qwen3-tts-mlx` | `mlx-audio` | Qwen3-TTS (Apple Silicon) |

These extras are mutually exclusive (configured in `pyproject.toml` `[tool.uv.conflicts]`).

### Dev

| Package          | Purpose                        |
| ---------------- | ------------------------------ |
| `pytest`         | Test runner                    |
| `pytest-cov`     | Coverage reporting             |
| `pytest-asyncio` | Async test support             |
| `mypy`           | Static type checking           |
| `ruff`           | Linting + formatting           |
| `httpx`          | Test HTTP client for API tests |

## Common Development Tasks

### Adding a new API endpoint

1. Add request/response models to `api/models.py`
2. Create or extend a route in `api/routes/`
3. Register the router in `api/routes/__init__.py` (if new file)
4. Add tests in `tests/test_api.py`

### Adding a new security check

1. Implement in the appropriate `security/` module
2. Export from `security/__init__.py`
3. Add tests in `tests/test_security_*.py`

### Modifying the storage format

1. Update models in `models.py`
2. Update `to_toml()` / `from_toml()` serialization methods
3. Update `VoicePromptStore` in `store.py`
4. Update `ArchiveExporter` / `ArchiveImporter` in `archive.py`
5. Verify all roundtrip tests pass

### Enrollment Pipeline

The enrollment subsystem (`src/voxid/enrollment/`) provides guided voice enrollment with phonetically balanced prompts, real-time quality validation, and multi-sample fusion. Key components:

- **PhonemeTracker** — tracks ARPAbet phoneme coverage across recordings using CMUdict
- **ScriptGenerator** — greedy weighted selection of prompts maximizing phoneme coverage (nasals/affricates 1.5x, vowels 1.2x)
- **QualityGate** — 6-gate validation: duration (3-60s), SNR (≥25 dB), speech ratio (≥60%), RMS (-40 to -3 dBFS), peak (≤-1 dBFS), sample rate (≥24 kHz)
- **AudioPreprocessor** — mono → resample → trim silence → LUFS normalize (-16 LUFS). No noise suppression by design
- **EnrollmentSession** — cursor-based state machine over styles × prompts with JSON persistence
- **EnrollmentPipeline** — facade orchestrating all components, used by CLI, REST API, and `VoxID.enroll()`

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): add new feature
fix(scope): fix bug description
refactor(scope): structural change without behavior change
test: add or update tests
docs: documentation changes
```

Required scopes for `feat` and `fix`: `core`, `api`, `adapters`, `router`, `segments`, `security`, `store`, `cli`, `video`, `enrollment`, `tokenizer`, `context`, `serving`, `spoofing`, `ui`.
