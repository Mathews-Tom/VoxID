from __future__ import annotations

from pathlib import Path

import tomli_w

from voxid.config import VoxIDConfig, default_store_path, load_config


def test_load_config_defaults_when_no_file(tmp_path: Path) -> None:
    # Arrange
    nonexistent = tmp_path / "no_config.toml"

    # Act
    config = load_config(nonexistent)

    # Assert
    assert isinstance(config, VoxIDConfig)
    assert config.default_engine == "qwen3-tts"
    assert config.router_confidence_threshold == 0.8
    assert config.cache_ttl_seconds == 3600
    assert config.max_embedding_versions == 3


def test_load_config_reads_toml_file(tmp_path: Path) -> None:
    # Arrange
    store_path = tmp_path / "custom_store"
    store_path.mkdir()
    raw = {
        "store_path": str(store_path),
        "default_engine": "custom-engine",
        "router_confidence_threshold": 0.95,
        "cache_ttl_seconds": 7200,
        "max_embedding_versions": 5,
    }
    config_file = tmp_path / "config.toml"
    config_file.write_bytes(tomli_w.dumps(raw).encode())

    # Act
    config = load_config(config_file)

    # Assert
    assert config.store_path == store_path
    assert config.default_engine == "custom-engine"
    assert config.router_confidence_threshold == 0.95
    assert config.cache_ttl_seconds == 7200
    assert config.max_embedding_versions == 5


def test_default_store_path_creates_directory(
    tmp_path: Path, monkeypatch: object
) -> None:
    # Arrange
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)  # type: ignore[attr-defined]

    # Act
    result = default_store_path()

    # Assert
    assert result == fake_home / ".voxid"
    assert result.is_dir()
