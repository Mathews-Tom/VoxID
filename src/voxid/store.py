from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import tomli
import tomli_w

from .models import ConsentRecord, Identity, Style


class VoicePromptStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._identities_dir = root / "identities"
        self._identities_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _identity_dir(self, identity_id: str) -> Path:
        return self._identities_dir / identity_id

    def _style_dir(self, identity_id: str, style_id: str) -> Path:
        return self._identity_dir(identity_id) / "styles" / style_id

    def _prompts_dir(self, identity_id: str, style_id: str) -> Path:
        return self._style_dir(identity_id, style_id) / "prompts"

    def _atomic_write(self, path: Path, content: str | bytes) -> None:
        """Write content to path atomically via a temp file in the same directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if isinstance(content, str) else "wb"
        kwargs = {"encoding": "utf-8"} if mode == "w" else {}
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        try:
            with open(fd, mode, **kwargs) as f:  # type: ignore[call-overload]
                f.write(content)
            Path(tmp).replace(path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ── Identity CRUD ─────────────────────────────────────────────────────────

    def create_identity(self, identity: Identity) -> Path:
        idir = self._identity_dir(identity.id)
        if idir.exists():
            raise ValueError(f"Identity {identity.id!r} already exists at {idir}")
        idir.mkdir(parents=True, exist_ok=False)

        toml_data = identity.to_toml()
        toml_bytes = tomli_w.dumps(toml_data).encode()
        self._atomic_write(idir / "identity.toml", toml_bytes)
        self._atomic_write(
            idir / "consent.json",
            json.dumps(identity.consent_record.to_dict(), indent=2),
        )
        return idir

    def get_identity(self, identity_id: str) -> Identity:
        idir = self._identity_dir(identity_id)
        toml_path = idir / "identity.toml"
        consent_path = idir / "consent.json"

        if not toml_path.exists():
            msg = f"Identity {identity_id!r} not found at {toml_path}"
            raise FileNotFoundError(msg)
        if not consent_path.exists():
            msg = f"Consent record missing for identity {identity_id!r}"
            raise FileNotFoundError(msg)

        with toml_path.open("rb") as f:
            toml_data = tomli.load(f)

        with consent_path.open("r", encoding="utf-8") as f:
            consent_data = json.load(f)

        consent = ConsentRecord.from_dict(consent_data)
        return Identity.from_toml(toml_data, consent)

    def list_identities(self) -> list[str]:
        if not self._identities_dir.exists():
            return []
        return sorted(
            d.name
            for d in self._identities_dir.iterdir()
            if d.is_dir() and (d / "identity.toml").exists()
        )

    def delete_identity(self, identity_id: str) -> None:
        idir = self._identity_dir(identity_id)
        if not idir.exists():
            raise FileNotFoundError(f"Identity {identity_id!r} not found")
        shutil.rmtree(idir)

    # ── Style CRUD ────────────────────────────────────────────────────────────

    def add_style(self, style: Style, ref_audio_src: Path) -> Path:
        if not self._identity_dir(style.identity_id).exists():
            raise FileNotFoundError(f"Identity {style.identity_id!r} does not exist")
        if not ref_audio_src.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_src}")

        sdir = self._style_dir(style.identity_id, style.id)
        if sdir.exists():
            raise ValueError(
                f"Style {style.id!r} already exists for identity {style.identity_id!r}"
            )
        sdir.mkdir(parents=True, exist_ok=False)

        # Copy ref audio into the style directory, preserving extension
        dest_audio = sdir / f"ref_audio{ref_audio_src.suffix}"
        shutil.copy2(ref_audio_src, dest_audio)

        # Persist the style with the canonical ref_audio_path pointing inside the store
        persisted_style = Style(
            id=style.id,
            identity_id=style.identity_id,
            label=style.label,
            description=style.description,
            default_engine=style.default_engine,
            ref_audio_path=str(dest_audio),
            ref_text=style.ref_text,
            language=style.language,
            metadata=style.metadata,
        )
        self._atomic_write(
            sdir / "style.toml", tomli_w.dumps(persisted_style.to_toml()).encode()
        )
        self._atomic_write(sdir / "ref_text.txt", style.ref_text)

        prompts_dir = sdir / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        # Run unified tokenizer if available (tokenizer extra installed)
        self._try_unified_tokenize(persisted_style, dest_audio)

        return sdir

    def get_style(self, identity_id: str, style_id: str) -> Style:
        sdir = self._style_dir(identity_id, style_id)
        toml_path = sdir / "style.toml"
        if not toml_path.exists():
            raise FileNotFoundError(
                f"Style {style_id!r} not found for identity {identity_id!r}"
            )
        with toml_path.open("rb") as f:
            data = tomli.load(f)
        return Style.from_toml(data)

    def list_styles(self, identity_id: str) -> list[str]:
        styles_dir = self._identity_dir(identity_id) / "styles"
        if not styles_dir.exists():
            return []
        return sorted(
            d.name
            for d in styles_dir.iterdir()
            if d.is_dir() and (d / "style.toml").exists()
        )

    def delete_style(self, identity_id: str, style_id: str) -> None:
        sdir = self._style_dir(identity_id, style_id)
        if not sdir.exists():
            raise FileNotFoundError(
                f"Style {style_id!r} not found for identity {identity_id!r}"
            )
        shutil.rmtree(sdir)

    # ── Unified tokenizer integration ──────────────────────────────────────────

    def _try_unified_tokenize(self, style: Style, audio_path: Path) -> None:
        """Run the unified tokenizer on a style's reference audio.

        Stores the result as ``unified.safetensors`` in the style directory.
        No-op if the tokenizer dependencies (wavtokenizer, transformers,
        scikit-learn) are not installed.
        """
        try:
            from voxid.tokenizer import TokenizerConfig, UnifiedTokenizer

            sdir = self._style_dir(style.identity_id, style.id)
            config = TokenizerConfig()
            tokenizer = UnifiedTokenizer(config)
            speaker = tokenizer.tokenize(
                audio_path,
                identity_id=style.identity_id,
                style_id=style.id,
            )
            tokenizer.save_tokenized(speaker, sdir / "unified.safetensors")
        except ImportError:
            return

    def get_unified_path(
        self, identity_id: str, style_id: str,
    ) -> Path | None:
        """Return path to unified.safetensors if it exists."""
        path = self._style_dir(identity_id, style_id) / "unified.safetensors"
        return path if path.exists() else None

    # ── Prompt cache ──────────────────────────────────────────────────────────

    def get_prompt_path(
        self, identity_id: str, style_id: str, engine: str,
    ) -> Path | None:
        path = self._prompts_dir(identity_id, style_id) / f"{engine}.safetensors"
        return path if path.exists() else None

    def set_prompt_path(
        self, identity_id: str, style_id: str, engine: str, prompt_path: Path
    ) -> None:
        dest = self._prompts_dir(identity_id, style_id) / f"{engine}.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if prompt_path.resolve() != dest.resolve():
            shutil.copy2(prompt_path, dest)

    def invalidate_prompt_cache(
        self, identity_id: str, style_id: str, engine: str | None = None
    ) -> None:
        prompts_dir = self._prompts_dir(identity_id, style_id)
        if not prompts_dir.exists():
            return
        if engine is not None:
            target = prompts_dir / f"{engine}.safetensors"
            target.unlink(missing_ok=True)
        else:
            for f in prompts_dir.glob("*.safetensors"):
                f.unlink()
