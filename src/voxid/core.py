from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .adapters import TTSEngineAdapter, _registry
from .adapters.protocol import EngineCapabilities
from .config import VoxIDConfig, load_config
from .context import (
    ConditioningConfig,
    ContextConditioner,
    ContextManager,
    SegmentHistory,
)
from .context.types import StitchParams
from .models import ConsentRecord, Identity, Style
from .router import StyleRouter
from .schemas import GeneratedScene, GenerationResult, SceneManifest
from .segments import (
    AudioStitcher,
    SegmentGenerationResult,
    SegmentResult,
    build_segment_plan,
    export_plan,
)
from .store import VoicePromptStore
from .video.timing import estimate_word_timings, timings_to_tuples


def _extract_segment_history(
    text: str,
    style: str,
    waveform: np.ndarray,
    sample_rate: int,
    duration_ms: int,
    trailing_window_ms: int = 200,
) -> SegmentHistory:
    """Extract prosodic features from a generated waveform's trailing window.

    Computes:
    - final_f0: estimated fundamental frequency from trailing autocorrelation
    - final_energy: RMS energy of trailing window
    - speaking_rate: words per second
    """
    word_count = len(text.split())
    duration_s = duration_ms / 1000.0
    speaking_rate = word_count / duration_s if duration_s > 0 else 0.0

    # Trailing window
    trailing_samples = int(sample_rate * trailing_window_ms / 1000)
    trailing_samples = min(trailing_samples, len(waveform))
    tail = waveform[-trailing_samples:].astype(np.float64)

    # RMS energy
    final_energy = float(np.sqrt(np.mean(tail**2))) if len(tail) > 0 else 0.0

    # F0 estimation via autocorrelation (50-500 Hz range)
    final_f0 = _estimate_f0(tail, sample_rate)

    return SegmentHistory(
        text=text,
        style=style,
        duration_ms=duration_ms,
        final_f0=final_f0,
        final_energy=final_energy,
        speaking_rate=speaking_rate,
    )


def _estimate_f0(
    signal: np.ndarray,
    sample_rate: int,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
) -> float:
    """Estimate fundamental frequency via autocorrelation.

    Returns 0.0 if the signal is too short or silent.
    """
    if len(signal) < 2:
        return 0.0

    # Normalize
    sig = signal - np.mean(signal)
    rms = float(np.sqrt(np.mean(sig**2)))
    if rms < 1e-6:
        return 0.0

    # Autocorrelation
    min_lag = int(sample_rate / f0_max)
    max_lag = int(sample_rate / f0_min)
    max_lag = min(max_lag, len(sig) - 1)

    if min_lag >= max_lag:
        return 0.0

    corr = np.correlate(sig, sig, mode="full")
    corr = corr[len(sig) - 1 :]  # positive lags only
    corr = corr / corr[0]  # normalize

    search = corr[min_lag : max_lag + 1]
    if len(search) == 0:
        return 0.0

    peak_idx = int(np.argmax(search)) + min_lag
    if corr[peak_idx] < 0.2:
        return 0.0

    return float(sample_rate / peak_idx)


def _get_adapter_for_engine(engine: str) -> type[TTSEngineAdapter]:
    """Resolve adapter class by engine slug.

    The registry may key entries by class name (when engine_name is a property)
    or by the engine slug string (when engine_name is a plain class attribute).
    Try the slug first, then scan for a matching engine_name class attribute.
    """
    if engine in _registry:
        return _registry[engine]

    # Fallback: iterate registry and check class-level engine_name attributes
    for cls in _registry.values():
        attr = cls.__dict__.get("engine_name")
        if isinstance(attr, str) and attr == engine:
            return cls
        # Property case: instantiating may have side effects, so we skip it.
        # Adapters registered under their class name should use a class-level str.

    raise KeyError(
        f"No adapter registered for engine {engine!r}. "
        f"Available registry keys: {list(_registry)}"
    )


class VoxID:
    def __init__(self, config: VoxIDConfig | None = None) -> None:
        from .adapters import discover_adapters

        discover_adapters()
        self._config = config or load_config()
        self._store = VoicePromptStore(self._config.store_path)
        self._router = StyleRouter(
            cache_dir=self._config.store_path / "cache" / "router",
            confidence_threshold=self._config.router_confidence_threshold,
            cache_ttl=self._config.cache_ttl_seconds,
        )

    def _available_styles(self, identity_id: str) -> list[str]:
        """Return registered styles, falling back to the identity's default."""
        identity = self._store.get_identity(identity_id)
        styles = self._store.list_styles(identity_id)
        return styles if styles else [identity.default_style]

    def _manifest_id(self, manifest: SceneManifest) -> str:
        """Extract or compute a stable manifest identifier."""
        return str(
            manifest.metadata.get(
                "id",
                hashlib.sha256(manifest.model_dump_json().encode()).hexdigest()[:16],
            )
        )

    def _resolve_scene_style(
        self,
        scene_text: str,
        scene_style: str | None,
        available_styles: list[str],
        default_style: str,
    ) -> str:
        """Return explicit scene style or route via StyleRouter."""
        if scene_style is not None:
            return scene_style
        decision = self._router.route(scene_text, available_styles, default_style)
        return decision.style

    def _select_engine(
        self,
        language: str,
        need_streaming: bool = False,
        need_emotion: bool = False,
    ) -> str:
        """Select best engine using capability flags.

        Preference order:
        1. Default engine from config
        2. Engine matching language requirement
        3. Engine matching streaming/emotion need
        """
        lang_code = language.split("-")[0]
        default = self._config.default_engine

        def _matches(caps: EngineCapabilities) -> bool:
            if lang_code not in caps.supported_languages:
                return False
            if need_streaming and not caps.supports_streaming:
                return False
            if need_emotion and not caps.supports_emotion_control:
                return False
            return True

        # Check if default supports requirements
        if default in _registry:
            caps = _registry[default]().capabilities
            if _matches(caps):
                return default

        # Scan all registered adapters for best match
        for name, cls in _registry.items():
            if _matches(cls().capabilities):
                return name

        return default

    def create_identity(
        self,
        id: str,
        name: str,
        description: str | None = None,
        default_style: str = "conversational",
        metadata: dict[str, Any] | None = None,
        consent: ConsentRecord | None = None,
    ) -> Identity:
        if consent is None:
            consent = ConsentRecord(
                timestamp=datetime.now(UTC).isoformat(),
                scope="personal",
                jurisdiction="US",
                transferable=False,
                document_hash="none",
            )
        identity = Identity(
            id=id,
            name=name,
            description=description,
            default_style=default_style,
            created_at=datetime.now(UTC).isoformat(),
            metadata=metadata or {},
            consent_record=consent,
        )
        self._store.create_identity(identity)
        return identity

    def add_style(
        self,
        identity_id: str,
        id: str,
        label: str,
        description: str,
        ref_audio: str | Path,
        ref_text: str,
        engine: str | None = None,
        language: str = "en-US",
        metadata: dict[str, Any] | None = None,
    ) -> Style:
        eng = engine or self._config.default_engine
        style = Style(
            id=id,
            identity_id=identity_id,
            label=label,
            description=description,
            default_engine=eng,
            ref_audio_path=str(ref_audio),
            ref_text=ref_text,
            language=language,
            metadata=metadata or {},
        )
        self._store.add_style(style, Path(ref_audio))

        # Eager prompt cache build for the default engine
        self._ensure_prompt(identity_id, id, eng)
        return style

    def _ensure_prompt(
        self,
        identity_id: str,
        style_id: str,
        engine: str,
    ) -> Path:
        cached = self._store.get_prompt_path(identity_id, style_id, engine)
        if cached is not None:
            return cached

        style = self._store.get_style(identity_id, style_id)
        adapter_cls = _get_adapter_for_engine(engine)
        adapter = adapter_cls()

        # Write directly to the canonical prompts location
        prompts_dir = self._store._prompts_dir(identity_id, style_id)
        prompts_dir.mkdir(parents=True, exist_ok=True)
        output_path = prompts_dir / f"{engine}.safetensors"

        adapter.build_prompt(
            Path(style.ref_audio_path),
            style.ref_text,
            output_path,
        )
        return output_path

    def _synthesize(
        self,
        text: str,
        identity_id: str,
        style_id: str,
        engine: str | None = None,
        language: str | None = None,
        context_params: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, int, str]:
        """Core synthesis: resolve engine, ensure prompt, generate waveform.

        Returns (waveform, sample_rate, engine_used).
        """
        style_obj = self._store.get_style(identity_id, style_id)
        eng = engine or style_obj.default_engine
        lang = language or style_obj.language

        prompt_path = self._ensure_prompt(identity_id, style_id, eng)
        adapter_cls = _get_adapter_for_engine(eng)
        adapter = adapter_cls()

        waveform, sr = adapter.generate(
            text,
            prompt_path,
            language=lang,
            context_params=context_params,
        )
        return waveform, sr, eng

    def generate(
        self,
        text: str,
        identity_id: str,
        style: str | None = None,
        engine: str | None = None,
    ) -> tuple[Path, int]:
        identity = self._store.get_identity(identity_id)
        available = self._available_styles(identity_id)
        style_id = self._resolve_scene_style(
            text,
            style,
            available,
            identity.default_style,
        )

        waveform, sr, _ = self._synthesize(text, identity_id, style_id, engine)

        output_dir = self._config.store_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
        output_path = output_dir / f"{identity_id}_{style_id}_{text_hash}.wav"
        sf.write(str(output_path), waveform, sr)
        return output_path, sr

    def generate_segments(
        self,
        text: str,
        identity_id: str,
        engine: str | None = None,
        output_dir: Path | None = None,
        stitch: bool = True,
        export_plan_path: Path | None = None,
        conditioning: ConditioningConfig | None = None,
    ) -> SegmentGenerationResult:
        """Generate audio for long-form text with per-segment routing.

        1. Segment the text (TextSegmenter)
        2. Route each segment (StyleRouter)
        3. Smooth style transitions (StyleSmoother)
        4. Build generation context and conditioning signals
        5. Generate audio per segment with context params
        6. Optionally stitch with context-derived pauses
        7. Optionally export the generation plan to JSON

        Args:
            conditioning: optional ConditioningConfig to enable
                context-aware generation. When None, the pipeline
                behaves identically to the pre-Phase-14 path.

        Returns SegmentGenerationResult with per-segment audio paths,
        stitched path, and the generation plan.
        """
        identity = self._store.get_identity(identity_id)
        available_styles = self._available_styles(identity_id)

        _, plan = build_segment_plan(
            text=text,
            router=self._router,
            available_styles=available_styles,
            default_style=identity.default_style,
        )

        if not plan:
            return SegmentGenerationResult(
                segments=[],
                stitched_path=None,
                total_duration_ms=0,
                plan=[],
            )

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if output_dir is None:
            output_dir = (
                self._config.store_path
                / "output"
                / "segments"
                / f"{identity_id}_{text_hash[:8]}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Context management (Phase 14)
        ctx_mgr = ContextManager()
        conditioner = ContextConditioner(conditioning) if conditioning else None
        ctx_mgr.set_total_segments(len(plan))
        ctx_mgr.set_style_sequence([item.style for item in plan])

        segment_results: list[SegmentResult] = []
        audio_segments: list[tuple[np.ndarray, int]] = []
        boundary_types: list[str] = []
        stitch_params_list: list[StitchParams] = []

        for item in plan:
            # Build conditioning for this segment
            gen_context = ctx_mgr.build_context(item.index)
            cond_result = (
                conditioner.condition(gen_context, item.boundary_type)
                if conditioner
                else None
            )

            # Apply text-level conditioning (SSML wrapping)
            gen_text = item.text
            context_params: dict[str, float] | None = None
            if cond_result:
                if cond_result.ssml_prefix or cond_result.ssml_suffix:
                    gen_text = (
                        cond_result.ssml_prefix + item.text + cond_result.ssml_suffix
                    )
                context_params = cond_result.context_params or None

            waveform, sr, _ = self._synthesize(
                gen_text,
                identity_id,
                item.style,
                engine,
                context_params=context_params,
            )

            seg_path = output_dir / f"segment_{item.index:04d}.wav"
            sf.write(str(seg_path), waveform, sr)
            duration_ms = int(len(waveform) / sr * 1000)

            ctx_mgr.record(
                _extract_segment_history(
                    text=item.text,
                    style=item.style,
                    waveform=waveform,
                    sample_rate=sr,
                    duration_ms=duration_ms,
                )
            )

            segment_results.append(
                SegmentResult(
                    index=item.index,
                    text=item.text,
                    style=item.style,
                    audio_path=seg_path,
                    duration_ms=duration_ms,
                    sample_rate=sr,
                    boundary_type=item.boundary_type,
                )
            )
            audio_segments.append((waveform, sr))
            boundary_types.append(item.boundary_type)
            stitch_params_list.append(
                cond_result.stitch if cond_result else StitchParams(pause_ms=200)
            )

        stitched_path: Path | None = None
        if stitch and audio_segments:
            stitcher = AudioStitcher()
            stitched_output = output_dir / "stitched.wav"
            stitched_path, _, _ = stitcher.stitch(
                audio_segments=audio_segments,
                boundary_types=boundary_types,
                output_path=stitched_output,
                stitch_params=stitch_params_list if conditioner else None,
            )

        if export_plan_path is not None:
            export_plan(plan, export_plan_path)

        total_duration_ms = sum(r.duration_ms for r in segment_results)

        return SegmentGenerationResult(
            segments=segment_results,
            stitched_path=stitched_path,
            total_duration_ms=total_duration_ms,
            plan=plan,
        )

    def generate_from_manifest(
        self,
        manifest: SceneManifest,
        output_dir: Path | None = None,
        stitch: bool = True,
    ) -> GenerationResult:
        """Generate audio for all scenes in a SceneManifest.

        For each scene:
        1. Route style (or use explicit style override from scene)
        2. Generate audio via adapter
        3. Estimate word-level timing
        4. Save per-scene WAV

        If stitch=True, concatenate all scene audio into one file.
        Returns GenerationResult with per-scene details.
        """
        identity = self._store.get_identity(manifest.identity_id)
        available_styles = self._available_styles(manifest.identity_id)
        mid = self._manifest_id(manifest)

        if output_dir is None:
            output_dir = self._config.store_path / "output" / "manifest" / mid
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_scenes: list[GeneratedScene] = []
        audio_segments: list[tuple[np.ndarray, int]] = []

        for scene in manifest.scenes:
            style_id = self._resolve_scene_style(
                scene.text,
                scene.style,
                available_styles,
                identity.default_style,
            )

            waveform, sr, eng = self._synthesize(
                scene.text,
                manifest.identity_id,
                style_id,
                manifest.engine,
            )

            duration_ms = len(waveform) * 1000 // sr
            scene_path = output_dir / f"{scene.scene_id}.wav"
            sf.write(str(scene_path), waveform, sr)

            word_timings = timings_to_tuples(
                estimate_word_timings(scene.text, duration_ms)
            )

            generated_scenes.append(
                GeneratedScene(
                    scene_id=scene.scene_id,
                    audio_path=str(scene_path),
                    duration_ms=duration_ms,
                    word_timings=word_timings,
                    style_used=style_id,
                    engine_used=eng,
                )
            )
            audio_segments.append((waveform, sr))

        if stitch and audio_segments:
            stitcher = AudioStitcher()
            stitched_output = output_dir / "stitched.wav"
            stitcher.stitch(
                audio_segments=audio_segments,
                boundary_types=["sentence"] * len(audio_segments),
                output_path=stitched_output,
            )

        total_duration_ms = sum(s.duration_ms for s in generated_scenes)

        return GenerationResult(
            manifest_id=mid,
            scenes=generated_scenes,
            total_duration_ms=total_duration_ms,
        )

    def plan_from_manifest(
        self,
        manifest: SceneManifest,
    ) -> GenerationResult:
        """Dry-run: route all scenes without generating audio.

        Returns GenerationResult with style/engine decisions
        but audio_path="" and duration_ms=0.
        """
        identity = self._store.get_identity(manifest.identity_id)
        available_styles = self._available_styles(manifest.identity_id)
        mid = self._manifest_id(manifest)

        planned_scenes: list[GeneratedScene] = []

        for scene in manifest.scenes:
            style_id = self._resolve_scene_style(
                scene.text,
                scene.style,
                available_styles,
                identity.default_style,
            )
            style_obj = self._store.get_style(manifest.identity_id, style_id)
            eng = manifest.engine or style_obj.default_engine

            planned_scenes.append(
                GeneratedScene(
                    scene_id=scene.scene_id,
                    audio_path="",
                    duration_ms=0,
                    word_timings=[],
                    style_used=style_id,
                    engine_used=eng,
                )
            )

        return GenerationResult(
            manifest_id=mid,
            scenes=planned_scenes,
            total_duration_ms=0,
        )

    def export_identity(
        self,
        identity_id: str,
        output_path: Path,
        signing_key: bytes | None = None,
    ) -> Path:
        """Export identity to a .voxid archive."""
        from .archive import ArchiveExporter

        exporter = ArchiveExporter(self._store)
        return exporter.export(
            identity_id,
            output_path,
            signing_key,
        )

    def import_identity(
        self,
        archive_path: Path,
        signing_key: bytes | None = None,
    ) -> Identity:
        """Import identity from a .voxid archive."""
        from .archive import ArchiveImporter

        importer = ArchiveImporter(self._store)
        return importer.import_archive(
            archive_path,
            signing_key,
        )

    def delete_identity(self, identity_id: str) -> None:
        self._store.delete_identity(identity_id)

    def list_identities(self) -> list[str]:
        return self._store.list_identities()

    def list_styles(self, identity_id: str) -> list[str]:
        return self._store.list_styles(identity_id)

    def route(
        self,
        text: str,
        identity_id: str,
    ) -> dict[str, Any]:
        """Dry-run style classification using the StyleRouter."""
        identity = self._store.get_identity(identity_id)
        available = self._available_styles(identity_id)
        decision = self._router.route(
            text,
            available,
            identity.default_style,
        )
        return {
            "style": decision.style,
            "confidence": decision.confidence,
            "tier": decision.tier,
            "scores": decision.scores,
        }

    def enroll(
        self,
        identity_id: str,
        styles: list[str],
        prompts_per_style: int = 5,
    ) -> Any:
        """Create an enrollment session. Used by CLI and API."""
        from voxid.enrollment import EnrollmentPipeline

        pipeline = EnrollmentPipeline(self)
        return pipeline.create_session(
            identity_id,
            styles,
            prompts_per_style,
        )

    def check_enrollment_health(
        self,
        identity_id: str,
    ) -> Any:
        """Assess whether an identity's enrollment should be refreshed."""
        from voxid.enrollment.health import check_enrollment_health

        return check_enrollment_health(self, identity_id)
