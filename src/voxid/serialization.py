from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def save_prompt(
    tensors: dict[str, np.ndarray],
    path: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata or {})


def load_prompt(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    tensors = load_file(str(path))
    # safetensors stores metadata as dict[str, str] in the file header;
    # the loaded object exposes it via .metadata() on the internal handle,
    # but load_file returns a plain dict of tensors. Re-open to extract metadata.
    from safetensors import safe_open

    meta: dict[str, str] = {}
    with safe_open(str(path), framework="numpy") as f:  # type: ignore[no-untyped-call]
        raw = f.metadata()
        if raw is not None:
            meta = dict(raw)
    return tensors, meta


def compute_hmac(file_path: Path, key: bytes) -> str:
    h = hmac.new(key, digestmod=hashlib.sha256)
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hmac(file_path: Path, key: bytes, expected: str) -> bool:
    actual = compute_hmac(file_path, key)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            f"HMAC mismatch for {file_path}: expected {expected!r}, got {actual!r}"
        )
    return True
