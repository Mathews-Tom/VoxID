from __future__ import annotations

import click

from .core import VoxID


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
@click.argument("text")
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
def generate(
    text: str,
    identity_id: str,
    style_name: str | None,
    engine: str | None,
    output: str | None,
) -> None:
    """Generate audio from text."""
    import shutil

    vox = VoxID()
    audio_path, sr = vox.generate(
        text=text,
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
    click.echo(f"  confidence: {result['confidence']}")
    click.echo(f"  tier:       {result['tier']}")
    click.echo(f"  scores:     {result['scores']}")
