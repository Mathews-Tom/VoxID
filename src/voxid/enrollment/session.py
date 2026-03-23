from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .phoneme_tracker import PhonemeTracker
from .quality_gate import QualityReport
from .script_generator import EnrollmentPrompt

logger = logging.getLogger(__name__)


class SessionStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    ABANDONED = "abandoned"


@dataclass
class EnrollmentSample:
    prompt_index: int
    prompt_text: str
    style_id: str
    attempt: int
    audio_path: str | None
    duration_s: float
    quality_report: QualityReport | None
    accepted: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_index": self.prompt_index,
            "prompt_text": self.prompt_text,
            "style_id": self.style_id,
            "attempt": self.attempt,
            "audio_path": self.audio_path,
            "duration_s": self.duration_s,
            "quality_report": (
                self.quality_report.to_dict()
                if self.quality_report is not None
                else None
            ),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnrollmentSample:
        qr = data.get("quality_report")
        return cls(
            prompt_index=data["prompt_index"],
            prompt_text=data["prompt_text"],
            style_id=data["style_id"],
            attempt=data["attempt"],
            audio_path=data.get("audio_path"),
            duration_s=data["duration_s"],
            quality_report=(
                QualityReport.from_dict(qr) if qr is not None else None
            ),
            accepted=data["accepted"],
            rejection_reason=data.get("rejection_reason"),
        )


@dataclass
class EnrollmentSession:
    session_id: str
    identity_id: str
    styles: list[str]
    started_at: str
    status: SessionStatus
    prompts_per_style: int
    prompts: dict[str, list[EnrollmentPrompt]]
    samples: list[EnrollmentSample] = field(default_factory=list)
    phoneme_trackers: dict[str, PhonemeTracker] = field(
        default_factory=dict,
    )
    current_style_index: int = 0
    current_prompt_index: int = 0

    def _require_in_progress(self) -> None:
        if self.status != SessionStatus.IN_PROGRESS:
            raise RuntimeError(
                f"Session is {self.status.value}, not in_progress"
            )

    def current_style(self) -> str | None:
        if self.current_style_index >= len(self.styles):
            return None
        return self.styles[self.current_style_index]

    def current_prompt(self) -> EnrollmentPrompt | None:
        style = self.current_style()
        if style is None:
            return None
        style_prompts = self.prompts.get(style, [])
        if self.current_prompt_index >= len(style_prompts):
            return None
        return style_prompts[self.current_prompt_index]

    def accept_sample(self, sample: EnrollmentSample) -> None:
        self._require_in_progress()
        self.samples.append(sample)
        style = sample.style_id
        if style not in self.phoneme_trackers:
            self.phoneme_trackers[style] = PhonemeTracker()
        self.phoneme_trackers[style].ingest(sample.prompt_text)

    def reject_sample(
        self, prompt_index: int, reason: str,
    ) -> None:
        self._require_in_progress()
        style = self.current_style()
        if style is None:
            return
        self.samples.append(
            EnrollmentSample(
                prompt_index=prompt_index,
                prompt_text=self._prompt_text_at(prompt_index),
                style_id=style,
                attempt=self._attempt_count(style, prompt_index) + 1,
                audio_path=None,
                duration_s=0.0,
                quality_report=None,
                accepted=False,
                rejection_reason=reason,
            ),
        )

    def skip_prompt(self) -> None:
        self._require_in_progress()
        self.advance()

    def advance(self) -> bool:
        """Move to the next prompt. Returns True if more prompts remain."""
        self._require_in_progress()
        style = self.current_style()
        if style is None:
            return False

        style_prompts = self.prompts.get(style, [])
        self.current_prompt_index += 1

        if self.current_prompt_index >= len(style_prompts):
            self.current_style_index += 1
            self.current_prompt_index = 0

        return self.current_style() is not None

    def complete(self) -> None:
        self.status = SessionStatus.COMPLETE

    def abandon(self) -> None:
        self.status = SessionStatus.ABANDONED

    def accepted_samples_for_style(
        self, style_id: str,
    ) -> list[EnrollmentSample]:
        return [
            s for s in self.samples
            if s.style_id == style_id and s.accepted
        ]

    def best_sample_for_style(
        self, style_id: str,
    ) -> EnrollmentSample | None:
        accepted = self.accepted_samples_for_style(style_id)
        if not accepted:
            return None
        return max(
            accepted,
            key=lambda s: (
                s.quality_report.snr_db
                if s.quality_report is not None
                else 0.0
            ),
        )

    def progress_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for style in self.styles:
            tracker = self.phoneme_trackers.get(style)
            style_samples = [
                s for s in self.samples if s.style_id == style
            ]
            accepted = [s for s in style_samples if s.accepted]
            rejected = [s for s in style_samples if not s.accepted]
            total_prompts = len(self.prompts.get(style, []))
            summary[style] = {
                "coverage_percent": (
                    tracker.coverage_percent() if tracker else 0.0
                ),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "total_prompts": total_prompts,
            }
        return summary

    def _prompt_text_at(self, prompt_index: int) -> str:
        style = self.current_style()
        if style is None:
            return ""
        prompts = self.prompts.get(style, [])
        if prompt_index < len(prompts):
            return prompts[prompt_index].text
        return ""

    def _attempt_count(self, style_id: str, prompt_index: int) -> int:
        return sum(
            1 for s in self.samples
            if s.style_id == style_id and s.prompt_index == prompt_index
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "identity_id": self.identity_id,
            "styles": list(self.styles),
            "started_at": self.started_at,
            "status": self.status.value,
            "prompts_per_style": self.prompts_per_style,
            "prompts": {
                style: [p.to_dict() for p in prompts]
                for style, prompts in self.prompts.items()
            },
            "samples": [s.to_dict() for s in self.samples],
            "phoneme_trackers": {
                style: t.to_dict()
                for style, t in self.phoneme_trackers.items()
            },
            "current_style_index": self.current_style_index,
            "current_prompt_index": self.current_prompt_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnrollmentSession:
        prompts: dict[str, list[EnrollmentPrompt]] = {}
        for style, prompt_list in data.get("prompts", {}).items():
            prompts[style] = [
                EnrollmentPrompt(**p) for p in prompt_list
            ]
        trackers: dict[str, PhonemeTracker] = {}
        for style, tracker_data in data.get(
            "phoneme_trackers", {},
        ).items():
            trackers[style] = PhonemeTracker.from_dict(tracker_data)

        return cls(
            session_id=data["session_id"],
            identity_id=data["identity_id"],
            styles=list(data["styles"]),
            started_at=data["started_at"],
            status=SessionStatus(data["status"]),
            prompts_per_style=data["prompts_per_style"],
            prompts=prompts,
            samples=[
                EnrollmentSample.from_dict(s)
                for s in data.get("samples", [])
            ],
            phoneme_trackers=trackers,
            current_style_index=data.get("current_style_index", 0),
            current_prompt_index=data.get("current_prompt_index", 0),
        )


class SessionStore:
    """Persists enrollment sessions as JSON files."""

    def __init__(self, base_path: Path) -> None:
        self._base = base_path / "enrollment_sessions"

    def _session_path(self, session_id: str) -> Path:
        return self._base / f"{session_id}.json"

    def save(self, session: EnrollmentSession) -> Path:
        """Atomically write session to JSON."""
        path = self._session_path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(session.to_dict(), indent=2)
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(content)
            Path(tmp).replace(path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def load(self, session_id: str) -> EnrollmentSession:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Session '{session_id}' not found at {path}"
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return EnrollmentSession.from_dict(data)

    def list_sessions(
        self, identity_id: str | None = None,
    ) -> list[str]:
        if not self._base.exists():
            return []
        sessions: list[str] = []
        for path in sorted(self._base.glob("*.json")):
            if identity_id is not None:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("identity_id") != identity_id:
                    continue
            sessions.append(path.stem)
        return sessions

    def delete(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Session '{session_id}' not found at {path}"
            )
        path.unlink()
