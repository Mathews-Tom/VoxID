from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import tomli_w
from click.testing import CliRunner

from voxid.cli import cli
from voxid.enrollment.cli_ui import render_vu_meter
from voxid.enrollment.consent import ConsentManager
from voxid.enrollment.quality_gate import QualityReport


def _setup_consent(tmp_path: Path) -> None:
    """Pre-create consent for an identity so --skip-consent works."""
    store_path = tmp_path / ".voxid"
    mgr = ConsentManager(store_path)
    audio = np.zeros(24000 * 3, dtype=np.float32)
    mgr.record_consent("alice", audio, 24000, scope="tts")


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


def _create_identity(runner: CliRunner) -> None:
    """Helper to create a test identity."""
    result = runner.invoke(
        cli, ["identity", "create", "alice", "--name", "Alice"],
    )
    assert result.exit_code == 0, result.output


def _make_ref_wav(path: Path, sr: int = 24000, duration_s: float = 5.0) -> Path:
    """Create a test WAV file with noise floor + sine (passes SNR gate)."""
    rng = np.random.default_rng(42)
    noise_floor = (0.001 * rng.standard_normal(int(sr * 0.5))).astype(np.float32)
    t = np.linspace(0, duration_s - 0.5, int(sr * (duration_s - 0.5)), endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio = np.concatenate([noise_floor, signal])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


# --- VU Meter ---


class TestRenderVuMeter:
    def test_quiet_all_green(self) -> None:
        bar = render_vu_meter(-50.0)
        assert "dBFS" in bar

    def test_loud_has_red(self) -> None:
        bar = render_vu_meter(-2.0)
        assert "dBFS" in bar

    def test_medium_has_yellow(self) -> None:
        bar = render_vu_meter(-15.0)
        assert "dBFS" in bar

    def test_clipping_full_red(self) -> None:
        bar = render_vu_meter(0.0)
        assert "+0.0 dBFS" in bar

    def test_below_minimum_clamped(self) -> None:
        bar = render_vu_meter(-100.0)
        # Bar position is clamped but display shows actual dBFS
        assert "-100.0 dBFS" in bar

    def test_custom_width(self) -> None:
        bar = render_vu_meter(-20.0, width=20)
        assert "dBFS" in bar


class TestDisplayCoverageBar:
    def test_0_percent(self, capsys: pytest.CaptureFixture[str]) -> None:
        from voxid.enrollment.cli_ui import display_coverage_bar
        display_coverage_bar(0.0)
        captured = capsys.readouterr()
        assert "0%" in captured.out
        assert "Coverage" in captured.out

    def test_100_percent(self, capsys: pytest.CaptureFixture[str]) -> None:
        from voxid.enrollment.cli_ui import display_coverage_bar
        display_coverage_bar(100.0)
        captured = capsys.readouterr()
        assert "100%" in captured.out


class TestDisplayQualityResult:
    def test_passed_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        from voxid.enrollment.cli_ui import display_quality_result
        report = QualityReport(
            passed=True, snr_db=40.0, rms_dbfs=-18.0, peak_dbfs=-4.0,
            speech_ratio=0.85, net_speech_duration_s=4.25,
            total_duration_s=5.0, sample_rate=24000,
        )
        display_quality_result(report)
        captured = capsys.readouterr()
        assert "PASSED" in captured.out

    def test_rejected_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        from voxid.enrollment.cli_ui import display_quality_result
        report = QualityReport(
            passed=False, snr_db=10.0, rms_dbfs=-50.0, peak_dbfs=-4.0,
            speech_ratio=0.3, net_speech_duration_s=1.5,
            total_duration_s=5.0, sample_rate=24000,
            rejection_reasons=["Too quiet", "Low speech"],
        )
        display_quality_result(report)
        captured = capsys.readouterr()
        assert "REJECTED" in captured.out
        assert "Too quiet" in captured.out


# --- Enroll command ---


class TestEnrollCommand:
    def test_enroll_command_exists(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli, ["enroll", "--help"])
        assert result.exit_code == 0
        assert "Enroll a voice identity" in result.output

    def test_enroll_requires_identity_id(
        self, cli_runner: CliRunner,
    ) -> None:
        result = cli_runner.invoke(cli, ["enroll"])
        assert result.exit_code != 0

    def test_enroll_requires_styles_option(
        self, cli_runner: CliRunner,
    ) -> None:
        _create_identity(cli_runner)
        result = cli_runner.invoke(cli, ["enroll", "alice"])
        assert result.exit_code != 0

    def test_enroll_nonexistent_identity_errors(
        self, cli_runner: CliRunner,
    ) -> None:
        result = cli_runner.invoke(
            cli, ["enroll", "nonexistent", "--styles", "conversational"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    @patch("voxid.cli.AudioRecorder")
    @patch("voxid.cli.click.getchar")
    def test_enroll_interactive_quit(
        self,
        mock_getchar: MagicMock,
        mock_recorder_cls: MagicMock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        _create_identity(cli_runner)
        _setup_consent(tmp_path)
        mock_getchar.return_value = "q"

        result = cli_runner.invoke(
            cli,
            ["enroll", "alice", "--styles", "phonetic", "--skip-consent"],
        )
        assert result.exit_code == 0

    @patch("voxid.cli.AudioRecorder")
    @patch("voxid.cli.click.getchar")
    def test_enroll_interactive_skip(
        self,
        mock_getchar: MagicMock,
        mock_recorder_cls: MagicMock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        _create_identity(cli_runner)
        _setup_consent(tmp_path)
        mock_getchar.side_effect = ["s", "s", "s", "s", "s", "q"]

        result = cli_runner.invoke(
            cli,
            ["enroll", "alice", "--styles", "phonetic", "--skip-consent"],
        )
        assert result.exit_code == 0

    @patch("voxid.cli.click.getchar")
    def test_enroll_consent_quit_cancels(
        self,
        mock_getchar: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        _create_identity(cli_runner)
        # Quit at consent prompt
        mock_getchar.return_value = "q"
        result = cli_runner.invoke(
            cli,
            ["enroll", "alice", "--styles", "phonetic"],
        )
        assert result.exit_code != 0
        assert "cancelled" in result.output


class TestImportMode:
    def test_import_audio_single_file_per_style(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        _create_identity(cli_runner)
        audio_dir = tmp_path / "audio_import"
        audio_dir.mkdir()

        # Create WAV file matching style name
        _make_ref_wav(audio_dir / "phonetic.wav")
        # Create transcript sidecar
        (audio_dir / "phonetic.txt").write_text(
            "The quick brown fox jumps over the lazy dog",
        )

        result = cli_runner.invoke(
            cli,
            [
                "enroll", "alice",
                "--styles", "phonetic",
                "--import-audio", str(audio_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Registered style" in result.output

    def test_import_audio_no_files_errors(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        _create_identity(cli_runner)
        empty_dir = tmp_path / "empty_import"
        empty_dir.mkdir()

        result = cli_runner.invoke(
            cli,
            [
                "enroll", "alice",
                "--styles", "phonetic",
                "--import-audio", str(empty_dir),
            ],
        )
        assert result.exit_code != 0
        assert "No WAV/MP3" in result.output

    def test_import_audio_missing_transcript_uses_default(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        _create_identity(cli_runner)
        audio_dir = tmp_path / "audio_import"
        audio_dir.mkdir()
        _make_ref_wav(audio_dir / "phonetic.wav")
        # No .txt sidecar

        result = cli_runner.invoke(
            cli,
            [
                "enroll", "alice",
                "--styles", "phonetic",
                "--import-audio", str(audio_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "No transcript sidecar" in result.output
        assert "Registered style" in result.output

    def test_import_audio_quality_gate_rejects_bad_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        _create_identity(cli_runner)
        audio_dir = tmp_path / "audio_import"
        audio_dir.mkdir()

        # Create a very short (0.5s) file that fails duration gate
        t = np.linspace(0, 0.5, int(24000 * 0.5), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(audio_dir / "phonetic.wav"), audio, 24000)

        result = cli_runner.invoke(
            cli,
            [
                "enroll", "alice",
                "--styles", "phonetic",
                "--import-audio", str(audio_dir),
            ],
        )
        assert result.exit_code == 0
        assert "REJECTED" in result.output
        assert "No passing audio" in result.output

    def test_import_audio_sidecar_txt_matched(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        _create_identity(cli_runner)
        audio_dir = tmp_path / "audio_import"
        audio_dir.mkdir()
        _make_ref_wav(audio_dir / "conversational.wav")
        (audio_dir / "conversational.txt").write_text(
            "Hey, so I was thinking about that project we discussed",
        )

        result = cli_runner.invoke(
            cli,
            [
                "enroll", "alice",
                "--styles", "conversational",
                "--import-audio", str(audio_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Registered style" in result.output

    def test_import_audio_multiple_styles(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        _create_identity(cli_runner)
        audio_dir = tmp_path / "audio_import"
        audio_dir.mkdir()
        _make_ref_wav(audio_dir / "conversational.wav")
        _make_ref_wav(audio_dir / "technical.wav")
        (audio_dir / "conversational.txt").write_text("Hey there")
        (audio_dir / "technical.txt").write_text("The algorithm processes")

        result = cli_runner.invoke(
            cli,
            [
                "enroll", "alice",
                "--styles", "conversational,technical",
                "--import-audio", str(audio_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert result.output.count("Registered style") == 2


class TestConsentIntegration:
    @patch("voxid.cli.AudioRecorder")
    @patch("voxid.cli.click.getchar")
    def test_enroll_skip_consent_when_exists(
        self,
        mock_getchar: MagicMock,
        mock_recorder_cls: MagicMock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        _create_identity(cli_runner)
        # Pre-create consent record
        store_path = tmp_path / ".voxid"
        consent_mgr = ConsentManager(store_path)
        consent_mgr.record_consent(
            "alice",
            _make_ref_wav(tmp_path / "c.wav").parent.joinpath("c.wav")
            and np.zeros(24000 * 3, dtype=np.float32),
            24000,
            scope="tts",
        )

        mock_getchar.return_value = "q"
        result = cli_runner.invoke(
            cli,
            [
                "enroll", "alice",
                "--styles", "phonetic",
                "--skip-consent",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Consent verified" in result.output

    @patch("voxid.cli.AudioRecorder")
    @patch("voxid.cli.click.getchar")
    def test_enroll_skip_consent_errors_when_no_existing(
        self,
        mock_getchar: MagicMock,
        mock_recorder_cls: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        _create_identity(cli_runner)
        result = cli_runner.invoke(
            cli,
            [
                "enroll", "alice",
                "--styles", "phonetic",
                "--skip-consent",
            ],
        )
        assert result.exit_code != 0
        assert "No existing consent" in result.output
