from __future__ import annotations

from dataclasses import dataclass

from .types import ConditioningResult, GenerationContext, StitchParams


@dataclass(frozen=True)
class ConditioningConfig:
    """Tuneable knobs for context conditioning.

    strength: master dial (0.0 = off, 1.0 = full conditioning).
    text_level: enable SSML prosody hints.
    parameter_level: enable speed/pitch/energy continuity params.
    stitch_level: enable context-derived pause durations.
    """

    strength: float = 0.5
    text_level: bool = True
    parameter_level: bool = True
    stitch_level: bool = True

    # Stitch defaults (ms) — overridden by context when stitch_level=True
    paragraph_pause_ms: int = 500
    sentence_pause_ms: int = 200
    clause_pause_ms: int = 100


# Boundary type → base pause (ms)
_BASE_PAUSE: dict[str, int] = {
    "paragraph": 500,
    "sentence": 200,
    "clause": 100,
}


class ContextConditioner:
    """Compute conditioning signals from generation context.

    Three independent strategies:
    1. Text-level: SSML prosody hints for rate/pitch continuity.
    2. Parameter-level: numeric context_params dict for adapter.
    3. Stitch-level: context-derived pause durations replacing fixed formulas.
    """

    def __init__(self, config: ConditioningConfig | None = None) -> None:
        self._config = config or ConditioningConfig()

    @property
    def config(self) -> ConditioningConfig:
        return self._config

    def condition(
        self,
        context: GenerationContext,
        boundary_type: str,
    ) -> ConditioningResult:
        """Produce conditioning signals for the next segment.

        Args:
            context: generation context built by ContextManager.
            boundary_type: "paragraph", "sentence", or "clause".

        Returns:
            ConditioningResult with SSML hints, adapter params, and stitch params.
        """
        s = self._config.strength

        ssml_prefix = ""
        ssml_suffix = ""
        context_params: dict[str, float] = {}
        stitch = self._default_stitch(boundary_type)

        if s <= 0.0 or not context.history:
            return ConditioningResult(
                ssml_prefix=ssml_prefix,
                ssml_suffix=ssml_suffix,
                context_params=context_params,
                stitch=stitch,
            )

        if self._config.text_level:
            ssml_prefix, ssml_suffix = self._text_conditioning(context, s)

        if self._config.parameter_level:
            context_params = self._parameter_conditioning(context, s)

        if self._config.stitch_level:
            stitch = self._stitch_conditioning(context, boundary_type, s)

        return ConditioningResult(
            ssml_prefix=ssml_prefix,
            ssml_suffix=ssml_suffix,
            context_params=context_params,
            stitch=stitch,
        )

    def _default_stitch(self, boundary_type: str) -> StitchParams:
        """Fixed-formula stitch params (current behavior)."""
        cfg = self._config
        pause_map: dict[str, int] = {
            "paragraph": cfg.paragraph_pause_ms,
            "sentence": cfg.sentence_pause_ms,
            "clause": cfg.clause_pause_ms,
        }
        return StitchParams(pause_ms=pause_map.get(boundary_type, cfg.clause_pause_ms))

    def _text_conditioning(
        self,
        context: GenerationContext,
        strength: float,
    ) -> tuple[str, str]:
        """Generate SSML prosody hints from trailing segment features.

        Maps the trailing segment's speaking rate to an SSML rate percentage,
        scaled by conditioning strength. This nudges the TTS engine toward
        rate continuity across segment boundaries.
        """
        last = context.history[-1]

        # Compute rate adjustment relative to a neutral baseline (3.0 wps)
        neutral_rate = 3.0
        rate_ratio = last.speaking_rate / neutral_rate if neutral_rate > 0 else 1.0
        # Blend toward neutral: strength=0 → no hint, strength=1 → full match
        blended_ratio = 1.0 + (rate_ratio - 1.0) * strength
        rate_pct = int(blended_ratio * 100)

        if rate_pct == 100:
            return "", ""

        prefix = f'<prosody rate="{rate_pct}%">'
        suffix = "</prosody>"
        return prefix, suffix

    def _parameter_conditioning(
        self,
        context: GenerationContext,
        strength: float,
    ) -> dict[str, float]:
        """Compute continuity parameters from trailing segment features.

        Returns a dict of adapter-consumable parameters:
        - speed: relative speed adjustment (1.0 = neutral)
        - pitch_hz: target pitch derived from trailing F0
        - energy: target energy level derived from trailing RMS
        """
        last = context.history[-1]

        neutral_rate = 3.0
        speed_ratio = last.speaking_rate / neutral_rate if neutral_rate > 0 else 1.0
        speed = 1.0 + (speed_ratio - 1.0) * strength

        pitch_hz = last.final_f0 * strength + 150.0 * (1.0 - strength)
        energy = last.final_energy * strength + 0.1 * (1.0 - strength)

        return {
            "speed": round(speed, 3),
            "pitch_hz": round(pitch_hz, 1),
            "energy": round(energy, 4),
        }

    def _stitch_conditioning(
        self,
        context: GenerationContext,
        boundary_type: str,
        strength: float,
    ) -> StitchParams:
        """Compute context-aware pause duration.

        Adjustments:
        - Style transition: +50% pause to signal voice change.
        - Fast speaking rate: shorter pauses for momentum.
        - Document ending (>80% position): slightly longer pauses for finality.
        - Strength=0 returns base pause; strength=1 returns full adjustment.
        """
        base_ms = _BASE_PAUSE.get(boundary_type, 100)

        if not context.history or not context.style_sequence:
            return StitchParams(pause_ms=base_ms)

        last = context.history[-1]
        adjustment = 0.0

        # Style transition: if previous style differs from the one being generated
        current_idx = len(context.history)
        if current_idx < len(context.style_sequence):
            upcoming_style = context.style_sequence[current_idx]
            if upcoming_style != last.style:
                adjustment += 0.5  # +50% for style change

        # Speaking rate: fast speech → compress pauses
        neutral_rate = 3.0
        if last.speaking_rate > neutral_rate * 1.2:
            adjustment -= 0.2  # -20% for fast speech
        elif last.speaking_rate < neutral_rate * 0.8:
            adjustment += 0.15  # +15% for slow speech

        # Document ending: last 20% gets longer pauses
        if context.doc_position > 0.8:
            adjustment += 0.1

        # Apply strength scaling
        scaled_adjustment = adjustment * strength
        adjusted_ms = int(base_ms * (1.0 + scaled_adjustment))

        # Clamp to reasonable range
        adjusted_ms = max(50, min(adjusted_ms, 1500))

        return StitchParams(pause_ms=adjusted_ms)
