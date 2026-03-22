from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConsentRecord:
    timestamp: str
    scope: str
    jurisdiction: str
    transferable: bool
    document_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "scope": self.scope,
            "jurisdiction": self.jurisdiction,
            "transferable": self.transferable,
            "document_hash": self.document_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsentRecord:
        return cls(
            timestamp=data["timestamp"],
            scope=data["scope"],
            jurisdiction=data["jurisdiction"],
            transferable=data["transferable"],
            document_hash=data["document_hash"],
        )


@dataclass
class Identity:
    id: str
    name: str
    description: str | None
    default_style: str
    created_at: str
    metadata: dict[str, Any]
    consent_record: ConsentRecord

    def to_toml(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "default_style": self.default_style,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
        if self.description is not None:
            data["description"] = self.description
        return data

    @classmethod
    def from_toml(cls, data: dict[str, Any], consent: ConsentRecord) -> Identity:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            default_style=data["default_style"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
            consent_record=consent,
        )


@dataclass
class Style:
    id: str
    identity_id: str
    label: str
    description: str
    default_engine: str
    ref_audio_path: str
    ref_text: str
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_toml(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "identity_id": self.identity_id,
            "label": self.label,
            "description": self.description,
            "default_engine": self.default_engine,
            "ref_audio_path": self.ref_audio_path,
            "ref_text": self.ref_text,
            "language": self.language,
            "metadata": self.metadata,
        }

    @classmethod
    def from_toml(cls, data: dict[str, Any]) -> Style:
        return cls(
            id=data["id"],
            identity_id=data["identity_id"],
            label=data["label"],
            description=data["description"],
            default_engine=data["default_engine"],
            ref_audio_path=data["ref_audio_path"],
            ref_text=data["ref_text"],
            language=data["language"],
            metadata=data.get("metadata", {}),
        )
