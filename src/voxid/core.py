from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from .adapters import TTSEngineAdapter, _registry
from .config import VoxIDConfig, load_config
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
        from .adapters import _registry, discover_adapters

        discover_adapters()

        default = self._config.default_engine
        # Check if default supports the language
        if default in _registry:
            adapter_cls = _registry[default]
            instance = adapter_cls()
            caps = instance.capabilities
            lang_code = language.split("-")[0]
            if lang_code in caps.supported_languages:
                if not need_streaming or caps.supports_streaming:
                    if not need_emotion or caps.supports_emotion_control:
                        return default

        # Scan all registered adapters for best match
        for name, cls in _registry.items():
            instance = cls()
            caps = instance.capabilities
            lang_code = language.split("-")[0]
            if lang_code not in caps.supported_languages:
                continue
            if need_streaming and not caps.supports_streaming:
                continue
            if need_emotion and not caps.supports_emotion_control:
                continue
            return name

        # Fallback to default
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

    def generate(
        self,
        text: str,
        identity_id: str,
        style: str | None = None,
        engine: str | None = None,
    ) -> tuple[Path, int]:
        identity = self._store.get_identity(identity_id)

        if style is not None:
            style_id = style
        else:
            available = self._store.list_styles(identity_id)
            if not available:
                style_id = identity.default_style
            else:
                decision = self._router.route(
                    text, available, identity.default_style,
                )
                style_id = decision.style

        style_obj = self._store.get_style(identity_id, style_id)
        eng = engine or style_obj.default_engine

        prompt_path = self._ensure_prompt(identity_id, style_id, eng)

        adapter_cls = _get_adapter_for_engine(eng)
        adapter = adapter_cls()

        waveform, sr = adapter.generate(
            text,
            prompt_path,
            language=style_obj.language,
        )

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
    ) -> SegmentGenerationResult:
        """Generate audio for long-form text with per-segment routing.

        1. Segment the text (TextSegmenter)
        2. Route each segment (StyleRouter)
        3. Smooth style transitions (StyleSmoother)
        4. Generate audio per segment
        5. Optionally stitch into a single file
        6. Optionally export the generation plan to JSON

        Returns SegmentGenerationResult with per-segment audio paths,
        stitched path, and the generation plan.
        """
        identity = self._store.get_identity(identity_id)
        available_styles = self._store.list_styles(identity_id)
        if not available_styles:
            available_styles = [identity.default_style]

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

        segment_results: list[SegmentResult] = []
        audio_segments: list[tuple[np.ndarray, int]] = []
        boundary_types: list[str] = []

        for item in plan:
            style_obj = self._store.get_style(identity_id, item.style)
            eng = engine or style_obj.default_engine

            prompt_path = self._ensure_prompt(identity_id, item.style, eng)

            adapter_cls = _get_adapter_for_engine(eng)
            adapter = adapter_cls()

            waveform, sr = adapter.generate(
                item.text,
                prompt_path,
                language=style_obj.language,
            )

            seg_filename = f"segment_{item.index:04d}.wav"
            seg_path = output_dir / seg_filename
            sf.write(str(seg_path), waveform, sr)

            duration_ms = int(len(waveform) / sr * 1000)

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

        stitched_path: Path | None = None
        if stitch and audio_segments:
            stitcher = AudioStitcher()
            stitched_output = output_dir / "stitched.wav"
            stitched_path, _, _ = stitcher.stitch(
                audio_segments=audio_segments,
                boundary_types=boundary_types,
                output_path=stitched_output,
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
        available_styles = self._store.list_styles(manifest.identity_id)
        if not available_styles:
            available_styles = [identity.default_style]

        manifest_id = manifest.metadata.get(
            "id",
            hashlib.sha256(
                manifest.model_dump_json().encode()
            ).hexdigest()[:16],
        )

        if output_dir is None:
            output_dir = (
                self._config.store_path / "output" / "manifest" / str(manifest_id)
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_scenes: list[GeneratedScene] = []
        audio_segments: list[tuple[np.ndarray, int]] = []

        for scene in manifest.scenes:
            if scene.style is not None:
                style_id = scene.style
            else:
                decision = self._router.route(
                    scene.text, available_styles, identity.default_style,
                )
                style_id = decision.style

            style_obj = self._store.get_style(manifest.identity_id, style_id)
            eng = manifest.engine or style_obj.default_engine

            prompt_path = self._ensure_prompt(manifest.identity_id, style_id, eng)

            adapter_cls = _get_adapter_for_engine(eng)
            adapter = adapter_cls()

            waveform, sr = adapter.generate(
                scene.text,
                prompt_path,
                language=style_obj.language,
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
            manifest_id=str(manifest_id),
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
        available_styles = self._store.list_styles(manifest.identity_id)
        if not available_styles:
            available_styles = [identity.default_style]

        manifest_id = manifest.metadata.get(
            "id",
            hashlib.sha256(
                manifest.model_dump_json().encode()
            ).hexdigest()[:16],
        )

        planned_scenes: list[GeneratedScene] = []

        for scene in manifest.scenes:
            if scene.style is not None:
                style_id = scene.style
            else:
                decision = self._router.route(
                    scene.text, available_styles, identity.default_style,
                )
                style_id = decision.style

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
            manifest_id=str(manifest_id),
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
            identity_id, output_path, signing_key,
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
            archive_path, signing_key,
        )

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
        available = self._store.list_styles(identity_id)
        if not available:
            available = [identity.default_style]
        decision = self._router.route(
            text, available, identity.default_style,
        )
        return {
            "style": decision.style,
            "confidence": decision.confidence,
            "tier": decision.tier,
            "scores": decision.scores,
        }
