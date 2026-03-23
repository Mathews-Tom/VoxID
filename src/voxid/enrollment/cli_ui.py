from __future__ import annotations

import time
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from .quality_gate import QualityReport
    from .session import EnrollmentSession

# VU meter thresholds (dBFS)
_GREEN_MAX = -18.0
_YELLOW_MAX = -9.0
_DBFS_MIN = -60.0
_DBFS_MAX = 0.0


def render_vu_meter(dbfs: float, width: int = 40) -> str:
    """Render a colored VU meter bar from dBFS level.

    Green: -60 to -18 dBFS (normal)
    Yellow: -18 to -9 dBFS (hot)
    Red: -9 to 0 dBFS (clipping risk)
    """
    clamped = max(_DBFS_MIN, min(_DBFS_MAX, dbfs))
    fraction = (clamped - _DBFS_MIN) / (_DBFS_MAX - _DBFS_MIN)
    filled = int(fraction * width)

    green_end = int((_GREEN_MAX - _DBFS_MIN) / (_DBFS_MAX - _DBFS_MIN) * width)
    yellow_end = int((_YELLOW_MAX - _DBFS_MIN) / (_DBFS_MAX - _DBFS_MIN) * width)

    parts: list[str] = []
    for i in range(width):
        char = "|" if i < filled else " "
        if i < green_end:
            parts.append(click.style(char, fg="green"))
        elif i < yellow_end:
            parts.append(click.style(char, fg="yellow"))
        else:
            parts.append(click.style(char, fg="red"))

    return "[" + "".join(parts) + f"] {dbfs:+.1f} dBFS"


def display_session_header(
    identity_id: str,
    styles: list[str],
    prompts_per_style: int,
) -> None:
    click.echo("")
    click.echo(
        click.style("Enrollment Session", bold=True, fg="cyan"),
    )
    click.echo(f"  Identity: {click.style(identity_id, bold=True)}")
    click.echo(f"  Styles:   {', '.join(styles)}")
    click.echo(f"  Prompts:  {prompts_per_style} per style")
    click.echo("")


def display_prompt(
    text: str,
    prompt_num: int,
    total: int,
    style: str,
    new_phonemes: list[str] | None = None,
) -> None:
    header = (
        click.style(f"[{prompt_num}/{total}]", fg="cyan")
        + f" Style: {click.style(style, bold=True)}"
    )
    click.echo(header)
    click.echo(f'  "{text}"')
    if new_phonemes:
        click.echo(
            click.style("  New phonemes: ", fg="yellow")
            + ", ".join(new_phonemes),
        )


def display_countdown(seconds: int = 3) -> None:
    for i in range(seconds, 0, -1):
        click.echo(f"  Recording in {i}...", nl=False)
        time.sleep(1)
        click.echo("\r", nl=False)
    click.echo("  Recording...         ")


def display_recording_status(
    elapsed_s: float,
    rms_dbfs: float,
    is_speech: bool,
) -> None:
    vu = render_vu_meter(rms_dbfs)
    speech_indicator = (
        click.style("SPEECH", fg="green")
        if is_speech
        else click.style("silence", fg="white")
    )
    click.echo(
        f"\r  {elapsed_s:5.1f}s {vu} {speech_indicator}",
        nl=False,
    )


def display_quality_result(report: QualityReport) -> None:
    click.echo("")
    if report.passed:
        click.echo(click.style("  PASSED", fg="green", bold=True))
    else:
        click.echo(click.style("  REJECTED", fg="red", bold=True))
        for reason in report.rejection_reasons:
            click.echo(f"    - {reason}")

    if report.warnings:
        for warning in report.warnings:
            click.echo(
                click.style(f"    ! {warning}", fg="yellow"),
            )

    click.echo(
        f"  SNR: {report.snr_db:.1f} dB"
        f"  RMS: {report.rms_dbfs:.1f} dBFS"
        f"  Speech: {report.speech_ratio:.0%}"
        f"  Duration: {report.total_duration_s:.1f}s",
    )


def display_coverage_bar(percent: float) -> None:
    width = 30
    filled = int(percent / 100.0 * width)
    bar = click.style("=" * filled, fg="green") + "-" * (width - filled)
    click.echo(f"  Coverage: [{bar}] {percent:.0f}%")


def display_session_summary(session: EnrollmentSession) -> None:
    click.echo("")
    click.echo(click.style("Session Summary", bold=True, fg="cyan"))
    summary = session.progress_summary()
    for style, info in summary.items():
        status = (
            click.style("complete", fg="green")
            if info["coverage_percent"] >= 100.0
            else click.style(f"{info['coverage_percent']:.0f}%", fg="yellow")
        )
        click.echo(
            f"  {click.style(style, bold=True)}: "
            f"{info['accepted']} accepted, "
            f"{info['rejected']} rejected "
            f"[{status}]",
        )
