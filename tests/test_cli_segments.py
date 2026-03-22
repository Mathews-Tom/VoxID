from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]
import tomli_w
from click.testing import CliRunner

import voxid.adapters.stub  # noqa: F401 — registers StubAdapter
from voxid.cli import cli

# ---------------------------------------------------------------------------
# Multi-paragraph test text (short enough to keep tests fast)
# ---------------------------------------------------------------------------

_MULTI_PARA_TEXT = (
    "The deployment pipeline runs on every merge to main. "
    "It executes three stages: build, test, and release.\n\n"
    "The build stage compiles the application and runs static analysis. "
    "Any lint error blocks progression to the next stage.\n\n"
    "The test stage runs the full suite of unit and integration tests. "
    "Coverage must stay above eighty percent.\n\n"
    "The release stage tags the Docker image and pushes it to the registry. "
    "A Slack notification confirms the deployment."
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ref_wav(tmp_path: Path) -> Path:
    """Minimal WAV file for use as reference audio."""
    audio = np.zeros(24000, dtype=np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, 24000)
    return path


@pytest.fixture
def cli_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ref_wav: Path,
) -> tuple[CliRunner, Path]:
    """Set up CLI environment with stub adapter and identity 'tom' with 4 styles."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    voxid_dir = tmp_path / ".voxid"
    voxid_dir.mkdir()

    config: dict[str, object] = {
        "store_path": str(voxid_dir),
        "default_engine": "stub",
    }
    (voxid_dir / "config.toml").write_bytes(tomli_w.dumps(config).encode())

    runner = CliRunner()

    # Create identity
    result = runner.invoke(
        cli,
        [
            "identity", "create", "tom",
            "--name", "Tom",
            "--default-style", "conversational",
        ],
    )
    assert result.exit_code == 0, f"identity create failed: {result.output}"

    # Add 4 styles
    for style_id, description in [
        ("conversational", "Relaxed peer-to-peer style"),
        ("technical", "Precise technical explanation"),
        ("narration", "Warm storytelling voice"),
        ("emphatic", "High-energy emphatic delivery"),
    ]:
        r = runner.invoke(
            cli,
            [
                "style", "add", "tom", style_id,
                "--audio", str(ref_wav),
                "--transcript", f"Reference transcript for {style_id}",
                "--description", description,
            ],
        )
        assert r.exit_code == 0, f"style add {style_id} failed: {r.output}"

    return runner, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cli_generate_segments_from_file(
    cli_env: tuple[CliRunner, Path],
    tmp_path: Path,
) -> None:
    """--file + --segments exits 0 and output contains 'segment' or 'Segment'."""
    runner, _ = cli_env
    text_file = tmp_path / "input.txt"
    text_file.write_text(_MULTI_PARA_TEXT, encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = runner.invoke(
        cli,
        [
            "generate",
            "--file", str(text_file),
            "--identity", "tom",
            "--segments",
            "-o", str(output_dir),
        ],
    )

    assert result.exit_code == 0, f"Command failed:\n{result.output}"
    assert "segment" in result.output.lower(), (
        f"Expected 'segment' in output, got:\n{result.output}"
    )


def test_cli_generate_segments_creates_per_segment_wavs(
    cli_env: tuple[CliRunner, Path],
    tmp_path: Path,
) -> None:
    """Per-segment WAV files exist in the output directory."""
    runner, _ = cli_env
    text_file = tmp_path / "input.txt"
    text_file.write_text(_MULTI_PARA_TEXT, encoding="utf-8")
    output_dir = tmp_path / "output_wavs"
    output_dir.mkdir()

    result = runner.invoke(
        cli,
        [
            "generate",
            "--file", str(text_file),
            "--identity", "tom",
            "--segments",
            "-o", str(output_dir),
        ],
    )

    assert result.exit_code == 0, f"Command failed:\n{result.output}"

    wav_files = list(output_dir.glob("segment_*.wav"))
    assert len(wav_files) >= 1, (
        f"No per-segment WAV files found in {output_dir}. "
        f"Contents: {list(output_dir.iterdir())}"
    )


def test_cli_generate_segments_creates_stitched(
    cli_env: tuple[CliRunner, Path],
    tmp_path: Path,
) -> None:
    """stitched.wav exists in the output directory."""
    runner, _ = cli_env
    text_file = tmp_path / "input.txt"
    text_file.write_text(_MULTI_PARA_TEXT, encoding="utf-8")
    output_dir = tmp_path / "output_stitch"
    output_dir.mkdir()

    result = runner.invoke(
        cli,
        [
            "generate",
            "--file", str(text_file),
            "--identity", "tom",
            "--segments",
            "-o", str(output_dir),
        ],
    )

    assert result.exit_code == 0, f"Command failed:\n{result.output}"

    stitched = output_dir / "stitched.wav"
    assert stitched.exists(), (
        f"stitched.wav not found in {output_dir}. "
        f"Contents: {list(output_dir.iterdir())}"
    )


def test_cli_generate_segments_with_plan_export(
    cli_env: tuple[CliRunner, Path],
    tmp_path: Path,
) -> None:
    """--plan exports valid JSON with all required fields."""
    runner, _ = cli_env
    text_file = tmp_path / "input.txt"
    text_file.write_text(_MULTI_PARA_TEXT, encoding="utf-8")
    output_dir = tmp_path / "output_plan"
    output_dir.mkdir()
    plan_file = tmp_path / "plan.json"

    result = runner.invoke(
        cli,
        [
            "generate",
            "--file", str(text_file),
            "--identity", "tom",
            "--segments",
            "-o", str(output_dir),
            "--plan", str(plan_file),
        ],
    )

    assert result.exit_code == 0, f"Command failed:\n{result.output}"
    assert plan_file.exists(), f"plan.json was not created at {plan_file}"

    plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
    assert isinstance(plan_data, list), "Plan JSON must be a list"
    assert len(plan_data) >= 1, "Plan must contain at least one segment"

    required_fields = {
        "index", "text", "style", "confidence",
        "tier", "boundary_type", "was_smoothed", "sentence_count",
    }
    for item in plan_data:
        missing = required_fields - item.keys()
        assert not missing, f"Plan item missing fields: {missing}"


def test_cli_generate_segments_shows_plan_table(
    cli_env: tuple[CliRunner, Path],
    tmp_path: Path,
) -> None:
    """Output contains segment plan table with style and confidence values."""
    runner, _ = cli_env
    text_file = tmp_path / "input.txt"
    text_file.write_text(_MULTI_PARA_TEXT, encoding="utf-8")
    output_dir = tmp_path / "output_table"
    output_dir.mkdir()

    result = runner.invoke(
        cli,
        [
            "generate",
            "--file", str(text_file),
            "--identity", "tom",
            "--segments",
            "-o", str(output_dir),
        ],
    )

    assert result.exit_code == 0, f"Command failed:\n{result.output}"
    assert "Segment plan" in result.output, (
        f"Expected 'Segment plan' in output, got:\n{result.output}"
    )
    # Each segment line contains a confidence float like "0.85"
    assert "0." in result.output, (
        f"Expected confidence float in output, got:\n{result.output}"
    )


def test_cli_generate_without_text_or_file_fails(
    cli_env: tuple[CliRunner, Path],
) -> None:
    """Omitting both TEXT argument and --file returns a non-zero exit code."""
    runner, _ = cli_env

    result = runner.invoke(
        cli,
        ["generate", "--identity", "tom", "--segments"],
    )

    assert result.exit_code != 0, (
        f"Expected non-zero exit code, got 0. Output:\n{result.output}"
    )
