from __future__ import annotations

from .protocol import EngineCapabilities, TTSEngineAdapter

_registry: dict[str, type[TTSEngineAdapter]] = {}


def register_adapter(cls: type) -> type:
    """Class decorator to register a TTS engine adapter.

    Verifies the class satisfies TTSEngineAdapter at registration time.
    Raises TypeError if the class does not implement the protocol.
    """
    if not isinstance(cls, type):
        raise TypeError(f"register_adapter expects a class, got {type(cls)!r}")
    # Protocol runtime check requires an instance; instantiation may require args,
    # so we verify structural compliance by checking the protocol's members directly.
    required = {
        "engine_name", "capabilities",
        "build_prompt", "generate", "generate_streaming",
    }
    missing = required - set(dir(cls))
    if missing:
        raise TypeError(
            f"Adapter {cls.__name__!r} is missing required protocol members: {missing}"
        )
    engine_name: str = getattr(cls, "engine_name", None)  # type: ignore[assignment]
    if not isinstance(engine_name, str | property):
        # engine_name is a property on instances; class-level attr is a property object
        # Accept both str class-level constants and property descriptors
        if not isinstance(getattr(cls, "engine_name", None), property):
            raise TypeError(
                f"Adapter {cls.__name__!r}.engine_name must be a property or str, "
                f"got {type(engine_name)!r}"
            )
    # Derive the key from the class-level engine_name if it's a plain string,
    # otherwise defer to runtime. Store under the class name as fallback.
    key: str
    attr = cls.__dict__.get("engine_name")
    if isinstance(attr, property):
        # Cannot call property without an instance; use class name as temporary key.
        # The adapter must provide engine_name as a property returning a stable string.
        # We use a sentinel key derived from the class name and update on first get.
        key = cls.__name__
    else:
        key = str(attr)
    _registry[key] = cls
    return cls


def get_adapter(engine_name: str) -> type[TTSEngineAdapter]:
    """Return adapter class by engine name. Raises KeyError if not registered."""
    if engine_name not in _registry:
        raise KeyError(
            f"No adapter registered for engine {engine_name!r}. "
            f"Available: {list(_registry)}"
        )
    return _registry[engine_name]


def list_adapters() -> list[str]:
    return list(_registry)


__all__ = [
    "EngineCapabilities",
    "TTSEngineAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
