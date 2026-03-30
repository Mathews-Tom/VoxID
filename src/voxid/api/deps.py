from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from voxid.core import VoxID
from voxid.serving.dispatcher import GPUDispatcher


@lru_cache(maxsize=1)
def get_voxid() -> VoxID:
    """Singleton VoxID instance for the API."""
    return VoxID()


@lru_cache(maxsize=1)
def get_dispatcher() -> GPUDispatcher | None:
    """Return a GPUDispatcher if VOXID_SERVING_CONFIG is set, else None."""
    config_path = os.environ.get("VOXID_SERVING_CONFIG")
    if config_path is None:
        return None

    from voxid.serving.config import load_serving_config

    serving_config = load_serving_config(Path(config_path))
    return GPUDispatcher(serving_config)
