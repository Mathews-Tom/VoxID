from __future__ import annotations

from functools import lru_cache

from voxid.core import VoxID


@lru_cache(maxsize=1)
def get_voxid() -> VoxID:
    """Singleton VoxID instance for the API."""
    return VoxID()
