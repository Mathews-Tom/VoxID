from __future__ import annotations

import datetime
import uuid
from pathlib import Path

import click
import numpy as np
import soundfile as sf

from .core import VoxID
from .enrollment.cli_ui import (
    display_coverage_bar,
    display_prompt,
    display_quality_result,
    display_session_header,
    display_session_summary,
)
from .enrollment.consent import ConsentManager
from .enrollment.preprocessor import AudioPreprocessor
from .enrollment.quality_gate import QualityGate
from .enrollment.recorder import AudioRecorder, play_audio, save_recording
from .enrollment.script_generator import ScriptGenerator
from .enrollment.session import (
    EnrollmentSample,
    EnrollmentSession,
    SessionStatus,
    SessionStore,
)


@click.group()
def cli() -> None:
    """VoxID — Voice Identity Management Platform"""


@cli.group()
def identity() -> None:
    """Manage voice identities."""


@identity.command("create")
@click.argument("identity_id")
@click.option("--name", required=True, help="Display name for the identity.")
@click.option("--description", default=None, help="Optional description.")
@click.option(
    "--default-style",
    default="conversational",
    show_default=True,
    help="Default style ID.",
)
def identity_create(
    identity_id: str,
    name: str,
    description: str | None,
    default_style: str,
) -> None:
    """Create a new voice identity."""
    vox = VoxID()
    ident = vox.create_identity(
        id=identity_id,
        name=name,
        description=description,
        default_style=default_style,
    )
    click.echo(
        click.style("Created identity: ", fg="green")
        + click.style(ident.id, bold=True)
    )
    click.echo(f"  name:          {ident.name}")
    click.echo(f"  default_style: {ident.default_style}")
    click.echo(f"  created_at:    {ident.created_at}")


@identity.command("list")
def identity_list() -> None:
    """List all voice identities."""
    vox = VoxID()
    ids = vox.list_identities()
    if not ids:
        click.echo("No identities found.")
        return
    for id_ in ids:
        click.echo(id_)


@identity.command("delete")
@click.argument("identity_id")
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def identity_delete(identity_id: str, force: bool) -> None:
    """Delete a voice identity and all its data."""
    vox = VoxID()
    # Verify identity exists before prompting
    ids = vox.list_identities()
    if identity_id not in ids:
        raise click.ClickException(f"Identity {identity_id!r} not found.")
    if not force:
        click.confirm(
            f"Delete identity {identity_id!r} and all associated styles, "
            "recordings, and consent data? This cannot be undone",
            abort=True,
        )
    vox.delete_identity(identity_id)
    click.echo(
        click.style("Deleted identity: ", fg="red")
        + click.style(identity_id, bold=True)
    )


@cli.group()
def style() -> None:
    """Manage voice styles."""


@style.command("add")
@click.argument("identity_id")
@click.argument("style_id")
@click.option(
    "--audio",
    required=True,
    type=click.Path(exists=True),
    help="Path to reference audio file.",
)
@click.option("--transcript", required=True, help="Transcription of reference audio.")
@click.option(
    "--label",
    default=None,
    help="Human-readable label (defaults to style_id).",
)
@click.option("--description", required=True, help="Style description.")
@click.option("--engine", default=None, help="TTS engine (default: from config).")
@click.option(
    "--language",
    default="en-US",
    show_default=True,
    help="BCP-47 language code.",
)
def style_add(
    identity_id: str,
    style_id: str,
    audio: str,
    transcript: str,
    label: str | None,
    description: str,
    engine: str | None,
    language: str,
) -> None:
    """Add a voice style to an identity."""
    vox = VoxID()
    s = vox.add_style(
        identity_id=identity_id,
        id=style_id,
        label=label or style_id,
        description=description,
        ref_audio=audio,
        ref_text=transcript,
        engine=engine,
        language=language,
    )
    click.echo(
        click.style("Added style: ", fg="green")
        + click.style(s.id, bold=True)
        + f" → identity {identity_id}"
    )
    click.echo(f"  engine:   {s.default_engine}")
    click.echo(f"  language: {s.language}")


@style.command("list")
@click.argument("identity_id")
def style_list(identity_id: str) -> None:
    """List styles for an identity."""
    vox = VoxID()
    styles = vox.list_styles(identity_id)
    if not styles:
        click.echo(f"No styles found for identity {identity_id!r}.")
        return
    for s in styles:
        click.echo(s)


@style.command("rebuild")
@click.argument("identity_id")
@click.argument("style_id")
@click.option("--engine", required=True, help="Engine to rebuild prompt cache for.")
def style_rebuild(identity_id: str, style_id: str, engine: str) -> None:
    """Rebuild prompt cache for a different engine."""
    vox = VoxID()
    # Invalidate existing cache so _ensure_prompt rebuilds
    vox._store.invalidate_prompt_cache(identity_id, style_id, engine)
    prompt_path = vox._ensure_prompt(identity_id, style_id, engine)
    click.echo(
        click.style("Rebuilt prompt cache: ", fg="green")
        + str(prompt_path)
    )


@cli.command()
@click.argument("text", required=False)
@click.option("--identity", "identity_id", required=True, help="Identity ID.")
@click.option("--style", "style_name", default=None, help="Style ID.")
@click.option("--engine", default=None, help="TTS engine override.")
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(),
    help="Output path (default: store output dir).",
)
@click.option(
    "--file",
    "file_path",
    default=None,
    type=click.Path(exists=True),
    help="Read text from file.",
)
@click.option(
    "--segments",
    is_flag=True,
    default=False,
    help="Enable segment-level generation.",
)
@click.option(
    "--plan",
    "plan_path",
    default=None,
    type=click.Path(),
    help="Export generation plan JSON.",
)
@click.option(
    "--manifest",
    "manifest_path",
    default=None,
    type=click.Path(exists=True),
    help="Generate from SceneManifest JSON file.",
)
def generate(
    text: str | None,
    identity_id: str,
    style_name: str | None,
    engine: str | None,
    output: str | None,
    file_path: str | None,
    segments: bool,
    plan_path: str | None,
    manifest_path: str | None,
) -> None:
    """Generate audio from text."""
    if manifest_path is not None:
        if text is not None or file_path is not None or segments:
            raise click.UsageError(
                "--manifest is mutually exclusive with TEXT, --file, and --segments."
            )
        import json

        from .schemas import SceneManifest

        manifest_data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest = SceneManifest.model_validate(manifest_data)

        vox = VoxID()
        manifest_result = vox.generate_from_manifest(
            manifest,
            output_dir=Path(output) if output is not None else None,
            stitch=True,
        )

        n_scenes = len(manifest_result.scenes)
        click.echo(
            click.style(
                f"Manifest {manifest_result.manifest_id} ({n_scenes} scenes):",
                fg="cyan",
            )
        )
        for scene in manifest_result.scenes:
            duration_s = scene.duration_ms / 1000
            click.echo(
                f"  [{scene.scene_id}] style={scene.style_used}"
                f"  engine={scene.engine_used}"
                f"  duration={duration_s:.1f}s"
            )

        stitched = (
            Path(manifest_result.scenes[0].audio_path).parent / "stitched.wav"
            if manifest_result.scenes
            else None
        )
        if stitched is not None and stitched.exists():
            total_s = manifest_result.total_duration_ms / 1000
            click.echo(
                click.style("Stitched: ", fg="green")
                + f"{stitched} ({total_s:.1f}s)"
            )
        return

    if file_path is not None:
        resolved_text = Path(file_path).read_text(encoding="utf-8")
    elif text is not None:
        resolved_text = text
    else:
        raise click.UsageError("Provide TEXT argument or --file PATH.")

    vox = VoxID()

    if segments:
        result = vox.generate_segments(
            text=resolved_text,
            identity_id=identity_id,
            engine=engine,
            output_dir=Path(output) if output is not None else None,
            stitch=True,
            export_plan_path=Path(plan_path) if plan_path is not None else None,
        )

        click.echo(
            click.style(
                f"Segment plan ({len(result.plan)} segments):", fg="cyan"
            )
        )
        for item in result.plan:
            smoothed_label = "smoothed" if item.was_smoothed else item.tier
            preview = item.text[:40].replace("\n", " ")
            click.echo(
                f"  [{item.index}] {item.style:<16s}"
                f" ({item.confidence:.2f}, {smoothed_label})"
                f' "{preview}..."'
            )

        seg_dir = (
            result.segments[0].audio_path.parent
            if result.segments
            else Path("output/segments")
        )
        click.echo(
            click.style(
                f"\nGenerated {len(result.segments)} segments", fg="green"
            )
            + f" → {seg_dir}"
        )

        if result.stitched_path is not None:
            total_s = result.total_duration_ms / 1000
            click.echo(
                click.style("Stitched: ", fg="green")
                + f"{result.stitched_path} ({total_s:.1f}s)"
            )
        return

    import shutil

    if style_name is None:
        routing = vox.route(text=resolved_text, identity_id=identity_id)
        click.echo(
            click.style("Auto-routed: ", fg="cyan")
            + f"style={routing['style']!r}"
            + f"  confidence={routing['confidence']:.2f}"
            + f"  tier={routing['tier']}"
        )

    audio_path, sr = vox.generate(
        text=resolved_text,
        identity_id=identity_id,
        style=style_name,
        engine=engine,
    )
    if output is not None:
        shutil.copy2(audio_path, output)
        audio_path = type(audio_path)(output)

    click.echo(
        click.style("Generated: ", fg="green")
        + str(audio_path)
    )
    click.echo(f"  sample_rate: {sr} Hz")


@cli.command()
@click.argument("text")
@click.option("--identity", "identity_id", required=True, help="Identity ID.")
def route(text: str, identity_id: str) -> None:
    """Show routing decision for text (dry run)."""
    vox = VoxID()
    result = vox.route(text=text, identity_id=identity_id)
    click.echo(click.style("Routing decision:", fg="cyan"))
    click.echo(f"  style:      {result['style']}")
    click.echo(f"  confidence: {result['confidence']:.2f}")
    click.echo(f"  tier:       {result['tier']}")
    click.echo("  scores:")
    for s, score in sorted(
        result["scores"].items(), key=lambda x: x[1], reverse=True,
    ):
        bar = "█" * int(score * 20)
        click.echo(f"    {s:20s} {score:.2f} {bar}")


@cli.command("export")
@click.argument("identity_id")
@click.argument("output", type=click.Path())
@click.option("--key", default=None, help="HMAC signing key.")
def export_cmd(identity_id: str, output: str, key: str | None) -> None:
    """Export identity to a .voxid archive."""
    vox = VoxID()
    signing_key = key.encode() if key else None
    path = vox.export_identity(
        identity_id, Path(output), signing_key,
    )
    click.echo(
        click.style("Exported: ", fg="green") + str(path)
    )


@cli.command("import")
@click.argument("archive", type=click.Path(exists=True))
@click.option("--key", default=None, help="HMAC verification key.")
def import_cmd(archive: str, key: str | None) -> None:
    """Import identity from a .voxid archive."""
    vox = VoxID()
    signing_key = key.encode() if key else None
    identity = vox.import_identity(
        Path(archive), signing_key,
    )
    click.echo(
        click.style("Imported: ", fg="green")
        + identity.id
    )


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--reload", is_flag=True, default=False)
def serve(host: str, port: int, reload: bool) -> None:
    """Start the VoxID REST API server."""
    import uvicorn
    click.echo(
        click.style("Starting VoxID API server", fg="green")
        + f" on {host}:{port}"
    )
    uvicorn.run(
        "voxid.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@cli.command()
@click.argument("identity_id")
@click.option(
    "--styles",
    required=True,
    help="Comma-separated style IDs.",
)
@click.option(
    "--prompts-per-style",
    default=5,
    type=int,
    show_default=True,
    help="Number of prompts per style.",
)
@click.option(
    "--resume",
    type=str,
    default=None,
    help="Resume an existing session by ID.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Audio input device name or index.",
)
@click.option(
    "--import-audio",
    type=click.Path(exists=True),
    default=None,
    help="Import pre-recorded audio instead of recording.",
)
@click.option(
    "--skip-consent",
    is_flag=True,
    default=False,
    help="Skip consent recording if valid consent already exists.",
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="ISO 639-1 language code (e.g. zh, ja, ko, es). Default: English.",
)
def enroll(
    identity_id: str,
    styles: str,
    prompts_per_style: int,
    resume: str | None,
    device: str | None,
    import_audio: str | None,
    skip_consent: bool,
    language: str | None,
) -> None:
    """Enroll a voice identity with guided recording or audio import."""
    vox = VoxID()
    store_path = vox._store._root
    style_list = [s.strip() for s in styles.split(",")]

    # Validate identity exists
    if identity_id not in vox.list_identities():
        raise click.ClickException(
            f"Identity '{identity_id}' not found. "
            f"Create it first: voxid identity create {identity_id} --name ..."
        )

    session_store = SessionStore(store_path)
    generator = ScriptGenerator()
    gate = QualityGate()
    preprocessor = AudioPreprocessor()
    consent_mgr = ConsentManager(store_path)

    if import_audio is not None:
        _run_import_mode(
            vox=vox,
            identity_id=identity_id,
            style_list=style_list,
            import_path=Path(import_audio),
            gate=gate,
            preprocessor=preprocessor,
        )
        return

    # Consent step (before recording)
    _handle_consent(
        consent_mgr=consent_mgr,
        identity_id=identity_id,
        identity_name=vox._store.get_identity(identity_id).name,
        gate=gate,
        device=device,
        skip_consent=skip_consent,
    )

    # Create or resume session
    is_multilingual = language is not None and language != "en"

    if resume is not None:
        session = session_store.load(resume)
        click.echo(
            click.style("Resumed session: ", fg="cyan") + resume,
        )
    else:
        if is_multilingual and language is not None:
            from .enrollment.multilingual import (
                MultilingualScriptGenerator,
                get_language_config,
            )
            from .enrollment.script_generator import EnrollmentPrompt

            try:
                get_language_config(language)
            except KeyError as exc:
                raise click.ClickException(str(exc)) from exc

            ml_gen = MultilingualScriptGenerator()
            prompts: dict[str, list[EnrollmentPrompt]] = {}
            for s in style_list:
                ml_prompts = ml_gen.select_prompts(
                    language, count=prompts_per_style,
                )
                prompts[s] = [
                    EnrollmentPrompt(
                        text=mp.text,
                        style=s,
                        phonemes=mp.phonemes,
                        unique_phoneme_count=mp.unique_phoneme_count,
                        nasal_count=0,
                        affricate_count=0,
                    )
                    for mp in ml_prompts
                ]
            click.echo(
                click.style("Language: ", fg="cyan") + language,
            )
        else:
            prompts = {
                s: generator.select_prompts(
                    s, count=prompts_per_style,
                )
                for s in style_list
            }

        session = EnrollmentSession(
            session_id=str(uuid.uuid4())[:8],
            identity_id=identity_id,
            styles=style_list,
            started_at=datetime.datetime.now(
                tz=datetime.UTC,
            ).isoformat(),
            status=SessionStatus.IN_PROGRESS,
            prompts_per_style=prompts_per_style,
            prompts=prompts,
            language=language,
        )

    display_session_header(identity_id, style_list, prompts_per_style)

    try:
        _run_recording_loop(
            session=session,
            session_store=session_store,
            gate=gate,
            preprocessor=preprocessor,
            store_path=store_path,
            device=device,
        )
    except KeyboardInterrupt:
        click.echo("\n")
        click.echo(click.style("Session interrupted.", fg="yellow"))
        session.abandon()
        session_store.save(session)
        click.echo(f"  Saved as: {session.session_id} (abandoned)")
        return

    # Register styles with VoxID
    session.complete()
    session_store.save(session)

    _register_styles(vox, session)
    display_session_summary(session)


def _handle_consent(
    consent_mgr: ConsentManager,
    identity_id: str,
    identity_name: str,
    gate: QualityGate,
    device: str | None,
    skip_consent: bool,
) -> None:
    """Record consent audio or verify existing consent."""
    if skip_consent:
        if consent_mgr.verify_consent_exists(identity_id):
            click.echo(
                click.style("Consent verified: ", fg="green")
                + "existing consent record found",
            )
            return
        raise click.ClickException(
            "No existing consent record. "
            "Record consent first (remove --skip-consent)."
        )

    statement = consent_mgr.generate_statement(identity_name)
    click.echo("")
    click.echo(click.style("Consent Required", bold=True, fg="cyan"))
    click.echo(f'  Please read aloud: "{statement}"')
    click.echo("")
    click.echo("  Press ENTER to record consent, Q to quit")

    key = click.getchar()
    if key in ("q", "Q"):
        raise click.ClickException("Enrollment cancelled.")

    recorder = AudioRecorder(device=device)
    click.echo("  Recording consent... (press ENTER to stop)")
    recorder.start()
    click.getchar()
    audio = recorder.stop()

    report = gate.validate(audio, recorder.sample_rate)
    display_quality_result(report)

    click.echo("  Press P to play back, any other key to continue")
    if click.getchar() in ("p", "P"):
        click.echo("  Playing...")
        play_audio(audio, recorder.sample_rate)

    if not report.passed:
        raise click.ClickException(
            "Consent recording failed quality check. "
            "Re-run enrollment to try again."
        )

    consent_mgr.record_consent(
        identity_id=identity_id,
        audio=audio,
        sr=recorder.sample_rate,
        scope="text-to-speech generation",
    )
    click.echo(click.style("  Consent recorded.", fg="green"))


def _run_recording_loop(
    session: EnrollmentSession,
    session_store: SessionStore,
    gate: QualityGate,
    preprocessor: AudioPreprocessor,
    store_path: Path,
    device: str | None,
) -> None:
    """Interactive recording loop for each prompt in the session."""
    recorder = AudioRecorder(device=device)
    total_prompts = sum(
        len(p) for p in session.prompts.values()
    )
    prompt_counter = 0

    while session.current_prompt() is not None:
        prompt = session.current_prompt()
        style = session.current_style()
        if prompt is None or style is None:
            break

        prompt_counter += 1
        display_prompt(
            text=prompt.text,
            prompt_num=prompt_counter,
            total=total_prompts,
            style=style,
        )

        click.echo("  Press ENTER to record, S to skip, Q to quit")
        key = click.getchar()
        if key in ("q", "Q"):
            break
        if key in ("s", "S"):
            session.skip_prompt()
            continue

        # Record
        click.echo("  Recording... (press ENTER to stop)")
        recorder.start()
        click.getchar()
        audio = recorder.stop()

        sr = recorder.sample_rate
        report = gate.validate(audio, sr)
        display_quality_result(report)

        if report.passed:
            click.echo("  Press P to play back, any other key to continue")
            if click.getchar() in ("p", "P"):
                click.echo("  Playing...")
                play_audio(audio, sr)

            processed, proc_sr = preprocessor.process(audio, sr)
            audio_dir = (
                store_path / "enrollment_sessions"
                / session.session_id / "samples"
            )
            audio_path = audio_dir / f"{style}_{prompt_counter}.wav"
            save_recording(processed, proc_sr, audio_path)

            sample = EnrollmentSample(
                prompt_index=session.current_prompt_index,
                prompt_text=prompt.text,
                style_id=style,
                attempt=1,
                audio_path=str(audio_path),
                duration_s=report.total_duration_s,
                quality_report=report,
                accepted=True,
                rejection_reason=None,
            )
            session.accept_sample(sample)
            session.advance()

            tracker = session.phoneme_trackers.get(style)
            if tracker:
                display_coverage_bar(tracker.coverage_percent())
        else:
            click.echo("  P to play back, R to retry, S to skip")
            retry_key = click.getchar()
            if retry_key in ("p", "P"):
                click.echo("  Playing...")
                play_audio(audio, sr)
                click.echo("  R to retry, S to skip")
                retry_key = click.getchar()
            if retry_key in ("s", "S"):
                session.reject_sample(
                    session.current_prompt_index,
                    "; ".join(report.rejection_reasons),
                )
                session.advance()
            # else retry (loop continues on same prompt)

        session_store.save(session)


def _run_import_mode(
    vox: VoxID,
    identity_id: str,
    style_list: list[str],
    import_path: Path,
    gate: QualityGate,
    preprocessor: AudioPreprocessor,
) -> None:
    """Non-interactive import of pre-recorded audio files."""
    audio_files = sorted(
        list(import_path.glob("*.wav"))
        + list(import_path.glob("*.mp3")),
    )
    if not audio_files:
        raise click.ClickException(
            f"No WAV/MP3 files found in {import_path}",
        )

    for style in style_list:
        # Match by filename stem: conversational.wav → style "conversational"
        matched = [
            f for f in audio_files if f.stem == style
        ]
        if not matched:
            # Fall back: assign files round-robin
            idx = style_list.index(style)
            if idx < len(audio_files):
                matched = [audio_files[idx]]

        if not matched:
            click.echo(
                click.style(f"  No audio file for style '{style}'", fg="red"),
            )
            continue

        best_path: Path | None = None
        best_snr = -999.0

        for audio_file in matched:
            audio_data, sr = sf.read(str(audio_file))
            audio_arr = np.asarray(audio_data, dtype=np.float64)
            report = gate.validate(audio_arr, sr)

            click.echo(f"  {audio_file.name} → {style}")
            display_quality_result(report)

            if report.passed and report.snr_db > best_snr:
                best_snr = report.snr_db
                best_path = audio_file

        if best_path is None:
            click.echo(
                click.style(
                    f"  No passing audio for style '{style}'", fg="red",
                ),
            )
            continue

        # Preprocess and register
        audio_data, sr = sf.read(str(best_path))
        audio_arr = np.asarray(audio_data, dtype=np.float64)
        processed, proc_sr = preprocessor.process(audio_arr, sr)
        out_path = best_path.parent / f"{style}_processed.wav"
        save_recording(processed, proc_sr, out_path)

        # Look for transcript sidecar
        transcript_path = best_path.with_suffix(".txt")
        if transcript_path.exists():
            transcript = transcript_path.read_text(encoding="utf-8").strip()
        else:
            transcript = f"Enrollment audio for {style}"
            click.echo(
                click.style(
                    f"  No transcript sidecar ({transcript_path.name}), "
                    f"using default",
                    fg="yellow",
                ),
            )

        vox.add_style(
            identity_id=identity_id,
            id=style,
            label=style.replace("_", " ").title(),
            description=f"Enrolled {style} style",
            ref_audio=str(out_path),
            ref_text=transcript,
        )
        click.echo(
            click.style("  Registered style: ", fg="green")
            + click.style(style, bold=True),
        )


def _register_styles(
    vox: VoxID,
    session: EnrollmentSession,
) -> None:
    """Register best samples as styles in VoxID."""
    for style in session.styles:
        best = session.best_sample_for_style(style)
        if best is None or best.audio_path is None:
            click.echo(
                click.style(
                    f"  No accepted sample for '{style}'", fg="yellow",
                ),
            )
            continue

        vox.add_style(
            identity_id=session.identity_id,
            id=style,
            label=style.replace("_", " ").title(),
            description=f"Enrolled {style} style",
            ref_audio=best.audio_path,
            ref_text=best.prompt_text,
        )
        click.echo(
            click.style("  Registered style: ", fg="green")
            + click.style(style, bold=True),
        )
