from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]
import tomli_w
from click.testing import CliRunner

import voxid.adapters.stub  # noqa: F401 — registers StubAdapter
from voxid.cli import cli


@pytest.fixture
def ref_wav(tmp_path: Path) -> Path:
    """Minimal WAV file for use as reference audio in CLI tests."""
    audio = np.zeros(24000, dtype=np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, 24000)
    return path


@pytest.fixture
def cli_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """CliRunner with isolated store and stub engine config."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    voxid_dir = tmp_path / ".voxid"
    voxid_dir.mkdir()
    config: dict[str, object] = {
        "store_path": str(voxid_dir),
        "default_engine": "stub",
    }
    (voxid_dir / "config.toml").write_bytes(tomli_w.dumps(config).encode())
    return CliRunner()


# ── Identity tests ────────────────────────────────────────────────────────────


def test_cli_identity_create(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    result = cli_runner.invoke(
        cli, ["identity", "create", "testid", "--name", "Test User"]
    )

    assert result.exit_code == 0, result.output
    assert "Created identity" in result.output
    assert "testid" in result.output


def test_cli_identity_list_empty(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    result = cli_runner.invoke(cli, ["identity", "list"])

    assert result.exit_code == 0, result.output
    assert "No identities" in result.output


def test_cli_identity_list_after_create(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    cli_runner.invoke(
        cli, ["identity", "create", "listtest", "--name", "List Test"]
    )

    result = cli_runner.invoke(cli, ["identity", "list"])

    assert result.exit_code == 0, result.output
    assert "listtest" in result.output


# ── Style tests ───────────────────────────────────────────────────────────────


def test_cli_style_add(
    cli_runner: CliRunner, tmp_path: Path, ref_wav: Path
) -> None:
    cli_runner.invoke(
        cli, ["identity", "create", "styletest", "--name", "Style Test"]
    )

    result = cli_runner.invoke(
        cli,
        [
            "style",
            "add",
            "styletest",
            "conversational",
            "--audio",
            str(ref_wav),
            "--transcript",
            "Reference transcript text",
            "--description",
            "Relaxed conversational style",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Added style" in result.output


def test_cli_style_list(
    cli_runner: CliRunner, tmp_path: Path, ref_wav: Path
) -> None:
    cli_runner.invoke(
        cli, ["identity", "create", "slisttest", "--name", "Style List Test"]
    )
    cli_runner.invoke(
        cli,
        [
            "style",
            "add",
            "slisttest",
            "conversational",
            "--audio",
            str(ref_wav),
            "--transcript",
            "Reference text",
            "--description",
            "Relaxed style",
        ],
    )

    result = cli_runner.invoke(cli, ["style", "list", "slisttest"])

    assert result.exit_code == 0, result.output
    assert "conversational" in result.output


# ── Generate tests ────────────────────────────────────────────────────────────


def test_cli_generate_with_style(
    cli_runner: CliRunner, tmp_path: Path, ref_wav: Path
) -> None:
    cli_runner.invoke(
        cli, ["identity", "create", "gentest", "--name", "Gen Test"]
    )
    cli_runner.invoke(
        cli,
        [
            "style",
            "add",
            "gentest",
            "conversational",
            "--audio",
            str(ref_wav),
            "--transcript",
            "Hello there",
            "--description",
            "Relaxed style",
        ],
    )

    result = cli_runner.invoke(
        cli,
        [
            "generate",
            "Hello world",
            "--identity",
            "gentest",
            "--style",
            "conversational",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Generated" in result.output


def test_cli_generate_auto_route(
    cli_runner: CliRunner, tmp_path: Path, ref_wav: Path
) -> None:
    cli_runner.invoke(
        cli, ["identity", "create", "autoroutetest", "--name", "Auto Route Test"]
    )
    for style_id in ["conversational", "technical", "narration", "emphatic"]:
        cli_runner.invoke(
            cli,
            [
                "style",
                "add",
                "autoroutetest",
                style_id,
                "--audio",
                str(ref_wav),
                "--transcript",
                f"Reference for {style_id}",
                "--description",
                f"{style_id} style",
            ],
        )

    result = cli_runner.invoke(
        cli,
        [
            "generate",
            "Let me explain the technical API endpoints",
            "--identity",
            "autoroutetest",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Auto-routed" in result.output
    assert "Generated" in result.output


# ── Route tests ───────────────────────────────────────────────────────────────


def test_cli_route(
    cli_runner: CliRunner, tmp_path: Path, ref_wav: Path
) -> None:
    cli_runner.invoke(
        cli, ["identity", "create", "routetest", "--name", "Route Test"]
    )
    for style_id in ["conversational", "technical", "narration", "emphatic"]:
        cli_runner.invoke(
            cli,
            [
                "style",
                "add",
                "routetest",
                style_id,
                "--audio",
                str(ref_wav),
                "--transcript",
                f"Reference for {style_id}",
                "--description",
                f"{style_id} style",
            ],
        )

    result = cli_runner.invoke(
        cli,
        [
            "route",
            "The API uses OAuth2 with JWT tokens for authentication",
            "--identity",
            "routetest",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "style" in result.output
    assert "confidence" in result.output
