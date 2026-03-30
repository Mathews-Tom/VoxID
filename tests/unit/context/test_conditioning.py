from __future__ import annotations

from voxid.context.conditioning import ConditioningConfig, ContextConditioner
from voxid.context.types import GenerationContext, SegmentHistory, StitchParams


def _make_history(
    text: str = "hello world",
    style: str = "conversational",
    duration_ms: int = 1000,
    final_f0: float = 150.0,
    final_energy: float = 0.1,
    speaking_rate: float = 3.0,
) -> SegmentHistory:
    return SegmentHistory(
        text=text,
        style=style,
        duration_ms=duration_ms,
        final_f0=final_f0,
        final_energy=final_energy,
        speaking_rate=speaking_rate,
    )


def _make_context(
    history: list[SegmentHistory] | None = None,
    doc_position: float = 0.5,
    total_segments: int = 10,
    style_sequence: list[str] | None = None,
) -> GenerationContext:
    return GenerationContext(
        history=history or [],
        doc_position=doc_position,
        total_segments=total_segments,
        style_sequence=style_sequence or ["conversational"] * total_segments,
    )


class TestConditionerDefaults:
    def test_default_config(self) -> None:
        c = ContextConditioner()
        assert c.config.strength == 0.5
        assert c.config.text_level is True
        assert c.config.parameter_level is True
        assert c.config.stitch_level is True

    def test_empty_history_returns_defaults(self) -> None:
        c = ContextConditioner()
        ctx = _make_context(history=[])
        result = c.condition(ctx, "sentence")
        assert result.ssml_prefix == ""
        assert result.ssml_suffix == ""
        assert result.context_params == {}

    def test_zero_strength_returns_defaults(self) -> None:
        cfg = ConditioningConfig(strength=0.0)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history()])
        result = c.condition(ctx, "sentence")
        assert result.ssml_prefix == ""
        assert result.context_params == {}


class TestTextConditioning:
    def test_neutral_rate_produces_no_ssml(self) -> None:
        cfg = ConditioningConfig(strength=1.0, parameter_level=False, stitch_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history(speaking_rate=3.0)])
        result = c.condition(ctx, "sentence")
        assert result.ssml_prefix == ""
        assert result.ssml_suffix == ""

    def test_fast_rate_produces_prosody_tag(self) -> None:
        cfg = ConditioningConfig(strength=1.0, parameter_level=False, stitch_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history(speaking_rate=4.5)])
        result = c.condition(ctx, "sentence")
        assert "prosody" in result.ssml_prefix
        assert "rate=" in result.ssml_prefix
        assert result.ssml_suffix == "</prosody>"

    def test_slow_rate_produces_slower_prosody(self) -> None:
        cfg = ConditioningConfig(strength=1.0, parameter_level=False, stitch_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history(speaking_rate=1.5)])
        result = c.condition(ctx, "sentence")
        assert 'rate="50%"' in result.ssml_prefix

    def test_text_level_disabled(self) -> None:
        cfg = ConditioningConfig(strength=1.0, text_level=False, parameter_level=False, stitch_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history(speaking_rate=4.5)])
        result = c.condition(ctx, "sentence")
        assert result.ssml_prefix == ""


class TestParameterConditioning:
    def test_produces_speed_pitch_energy(self) -> None:
        cfg = ConditioningConfig(strength=1.0, text_level=False, stitch_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history()])
        result = c.condition(ctx, "sentence")
        assert "speed" in result.context_params
        assert "pitch_hz" in result.context_params
        assert "energy" in result.context_params

    def test_neutral_rate_speed_is_one(self) -> None:
        cfg = ConditioningConfig(strength=1.0, text_level=False, stitch_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history(speaking_rate=3.0)])
        result = c.condition(ctx, "sentence")
        assert result.context_params["speed"] == 1.0

    def test_half_strength_blends_toward_neutral(self) -> None:
        cfg = ConditioningConfig(strength=0.5, text_level=False, stitch_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history(final_f0=200.0)])
        result = c.condition(ctx, "sentence")
        # pitch_hz = 200 * 0.5 + 150 * 0.5 = 175
        assert result.context_params["pitch_hz"] == 175.0

    def test_parameter_level_disabled(self) -> None:
        cfg = ConditioningConfig(strength=1.0, text_level=False, parameter_level=False, stitch_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history()])
        result = c.condition(ctx, "sentence")
        assert result.context_params == {}


class TestStitchConditioning:
    def test_default_stitch_paragraph(self) -> None:
        cfg = ConditioningConfig(strength=0.0)
        c = ContextConditioner(cfg)
        ctx = _make_context(history=[_make_history()])
        result = c.condition(ctx, "paragraph")
        assert result.stitch.pause_ms == 500

    def test_default_stitch_sentence(self) -> None:
        cfg = ConditioningConfig(strength=0.0)
        c = ContextConditioner(cfg)
        ctx = _make_context()
        result = c.condition(ctx, "sentence")
        assert result.stitch.pause_ms == 200

    def test_style_transition_increases_pause(self) -> None:
        cfg = ConditioningConfig(strength=1.0, text_level=False, parameter_level=False)
        c = ContextConditioner(cfg)
        # History has "conversational", but upcoming (index 1) is "technical"
        styles = ["conversational", "technical", "conversational"]
        ctx = _make_context(
            history=[_make_history(style="conversational")],
            style_sequence=styles,
            total_segments=3,
        )
        result = c.condition(ctx, "sentence")
        # Style transition adds +50%, so 200 * 1.5 = 300
        assert result.stitch.pause_ms > 200

    def test_fast_speech_compresses_pause(self) -> None:
        cfg = ConditioningConfig(strength=1.0, text_level=False, parameter_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(
            history=[_make_history(speaking_rate=5.0, style="conversational")],
            style_sequence=["conversational"] * 10,
        )
        result = c.condition(ctx, "sentence")
        # Fast speech -20%, so 200 * 0.8 = 160
        assert result.stitch.pause_ms < 200

    def test_document_ending_increases_pause(self) -> None:
        cfg = ConditioningConfig(strength=1.0, text_level=False, parameter_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(
            history=[_make_history(style="conversational")],
            doc_position=0.9,
            style_sequence=["conversational"] * 10,
        )
        result = c.condition(ctx, "sentence")
        assert result.stitch.pause_ms > 200

    def test_pause_clamped_to_range(self) -> None:
        cfg = ConditioningConfig(strength=1.0, text_level=False, parameter_level=False)
        c = ContextConditioner(cfg)
        # Extreme case: all adjustments stacked
        ctx = _make_context(
            history=[_make_history(speaking_rate=0.5, style="conversational")],
            doc_position=0.95,
            style_sequence=["conversational", "technical"],
            total_segments=2,
        )
        result = c.condition(ctx, "paragraph")
        assert 50 <= result.stitch.pause_ms <= 1500

    def test_stitch_level_disabled_returns_base(self) -> None:
        cfg = ConditioningConfig(strength=1.0, text_level=False, parameter_level=False, stitch_level=False)
        c = ContextConditioner(cfg)
        ctx = _make_context(
            history=[_make_history(speaking_rate=5.0)],
        )
        result = c.condition(ctx, "sentence")
        # No stitch conditioning → default formula
        assert result.stitch.pause_ms == 200


class TestStitchParams:
    def test_default_crossfade(self) -> None:
        sp = StitchParams(pause_ms=300)
        assert sp.crossfade_ms == 20

    def test_custom_crossfade(self) -> None:
        sp = StitchParams(pause_ms=300, crossfade_ms=50)
        assert sp.crossfade_ms == 50
