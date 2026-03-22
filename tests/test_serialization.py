from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxid.serialization import compute_hmac, load_prompt, save_prompt, verify_hmac


def test_save_load_prompt_roundtrip(tmp_path: Path) -> None:
    # Arrange
    tensors = {
        "embedding": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "features": np.zeros((4, 8), dtype=np.float32),
    }
    path = tmp_path / "prompt.safetensors"

    # Act
    save_prompt(tensors, path)
    loaded, _ = load_prompt(path)

    # Assert
    np.testing.assert_array_equal(loaded["embedding"], tensors["embedding"])
    np.testing.assert_array_equal(loaded["features"], tensors["features"])


def test_save_load_prompt_with_metadata(tmp_path: Path) -> None:
    # Arrange
    tensors = {"vec": np.ones(16, dtype=np.float32)}
    metadata = {"engine": "qwen3-tts", "language": "en-US", "version": "1"}
    path = tmp_path / "prompt_meta.safetensors"

    # Act
    save_prompt(tensors, path, metadata=metadata)
    _, loaded_meta = load_prompt(path)

    # Assert
    assert loaded_meta["engine"] == "qwen3-tts"
    assert loaded_meta["language"] == "en-US"
    assert loaded_meta["version"] == "1"


def test_compute_hmac_deterministic(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "data.bin"
    path.write_bytes(b"test content for hmac")
    key = b"secret-key-32-bytes-long-padding"

    # Act
    hmac1 = compute_hmac(path, key)
    hmac2 = compute_hmac(path, key)

    # Assert
    assert hmac1 == hmac2


def test_verify_hmac_correct_key_passes(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "verified.bin"
    path.write_bytes(b"content to verify")
    key = b"my-hmac-key"

    # Act
    expected = compute_hmac(path, key)
    result = verify_hmac(path, key, expected)

    # Assert
    assert result is True


def test_verify_hmac_wrong_key_raises(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "wrong_key.bin"
    path.write_bytes(b"content")
    key_a = b"key-a"
    key_b = b"key-b"

    # Act
    expected = compute_hmac(path, key_a)

    # Assert
    with pytest.raises(ValueError):
        verify_hmac(path, key_b, expected)


def test_verify_hmac_tampered_file_raises(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "tampered.bin"
    path.write_bytes(b"original content")
    key = b"integrity-key"
    expected = compute_hmac(path, key)

    # Act — tamper
    path.write_bytes(b"modified content")

    # Assert
    with pytest.raises(ValueError):
        verify_hmac(path, key, expected)


def test_no_pickle_in_serialization() -> None:
    # Arrange
    import voxid.serialization as mod

    source_path = Path(mod.__file__)

    # Act
    content = source_path.read_text(encoding="utf-8")

    # Assert
    assert "pickle" not in content
