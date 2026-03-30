"""vLLM plugin registration for VoxID TTS serving.

This module provides the entry point that vLLM calls to discover and
register VoxID's TTS workers as a plugin. Import and call
``register_voxid_plugin()`` to make VoxID available as a vLLM plugin.

Usage::

    # In a vLLM plugin config:
    from voxid.serving.plugin import register_voxid_plugin

    register_voxid_plugin(serving_config_path="/path/to/serving.toml")
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import ServingConfig, load_serving_config

logger = logging.getLogger(__name__)

# Module-level reference so the dispatcher can be retrieved after registration.
_registered_config: ServingConfig | None = None


def register_voxid_plugin(
    serving_config_path: str | Path | None = None,
    serving_config: ServingConfig | None = None,
) -> ServingConfig:
    """Register VoxID as a vLLM TTS plugin.

    Accepts either a path to a TOML config or a pre-built ``ServingConfig``.
    Exactly one must be provided.
    """
    global _registered_config  # noqa: PLW0603

    if serving_config is not None:
        _registered_config = serving_config
    elif serving_config_path is not None:
        _registered_config = load_serving_config(Path(serving_config_path))
    else:
        msg = "provide either serving_config_path or serving_config"
        raise ValueError(msg)

    logger.info(
        "voxid plugin registered: %d workers, strategy=%s",
        len(_registered_config.workers),
        _registered_config.dispatch_strategy.value,
    )
    return _registered_config


def get_registered_config() -> ServingConfig | None:
    """Return the config set by ``register_voxid_plugin()``, or None."""
    return _registered_config
