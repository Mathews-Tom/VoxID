#!/usr/bin/env python3
"""
VoxID Banner SVG Generator — Voice Persona Platform

Visual concept: a wide multi-register waveform spanning the full banner
width. The waveform transitions through four voice-style colors
(teal → coral → amber → emerald), each section with a distinct amplitude
character suggesting different speaking registers.

The bars are mirrored vertically (symmetric around center) for a polished
audio-visualizer aesthetic. Noise-driven heights create organic variation.
"""

import argparse
import math
from pathlib import Path


# ── Palette ──────────────────────────────────
BG = "#1A1A2E"
TEAL = "#1ABC9C"
CORAL = "#E8634A"
AMBER = "#F39C12"
EMERALD = "#2ECC71"

STYLE_COLORS = [TEAL, CORAL, AMBER, EMERALD]


# ── Noise ────────────────────────────────────

def smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)


def hash_noise(x: float, seed: int) -> float:
    return math.sin(x * 12.9898 + seed * 78.233) * 43758.5453 % 1.0


def coherent_noise(x: float, seed: int) -> float:
    i = math.floor(x)
    f = x - i
    return hash_noise(i, seed) + (hash_noise(i + 1, seed) - hash_noise(i, seed)) * smoothstep(f)


def fractal_noise(x: float, seed: int, octaves: int = 4) -> float:
    total = 0.0
    freq = 1.0
    amp = 1.0
    max_val = 0.0
    for o in range(octaves):
        total += coherent_noise(x * freq, seed + o * 17) * amp
        max_val += amp
        amp *= 0.5
        freq *= 2.0
    return total / max_val


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def lerp_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    )


# ── Waveform generation ─────────────────────

def register_envelope(t: float, register: int) -> float:
    """Shape the amplitude envelope per register for distinct character.

    t: position within the register section [0, 1]
    register: 0=conversational, 1=technical, 2=narration, 3=emphatic
    """
    if register == 0:
        # Conversational: smooth, moderate, gentle undulation
        return 0.5 + 0.3 * math.sin(t * math.pi * 2.5)
    elif register == 1:
        # Technical: precise, even, slightly clipped
        return 0.55 + 0.15 * math.sin(t * math.pi * 4)
    elif register == 2:
        # Narration: flowing, builds up, sustained
        return 0.4 + 0.45 * math.sin(t * math.pi)
    else:
        # Emphatic: dynamic, punchy peaks
        return 0.35 + 0.55 * abs(math.sin(t * math.pi * 3))


def generate_bars(
    width: int, height: int, bar_count: int,
    bar_width: float, max_height: float,
    seed: int,
) -> list[dict]:
    """Generate symmetric waveform bars spanning the full width.

    Returns list of {x, y, w, h, color} dicts. Bars are vertically
    centered (mirrored) for a polished visualizer look.
    """
    bars = []
    cx = width / 2.0
    cy = height / 2.0
    total_width = bar_count * (bar_width + 1.5)
    start_x = cx - total_width / 2

    for i in range(bar_count):
        # Position within full banner [0, 1]
        t_global = i / (bar_count - 1)

        # Determine which register section and local position
        section = min(int(t_global * 4), 3)
        t_local = (t_global * 4 - section)

        # Color: blend between adjacent style colors at transitions
        if t_global < 0.05:
            color = STYLE_COLORS[0]
        elif t_global > 0.95:
            color = STYLE_COLORS[3]
        else:
            section_f = t_global * 3  # 0..3 across 4 colors
            idx = min(int(section_f), 2)
            blend = section_f - idx
            # Smooth the blend zone
            if blend < 0.15 or blend > 0.85:
                color = STYLE_COLORS[idx] if blend < 0.5 else STYLE_COLORS[idx + 1]
            else:
                blend_t = (blend - 0.15) / 0.7
                color = lerp_hex(STYLE_COLORS[idx], STYLE_COLORS[idx + 1], blend_t)

        # Height: noise + register envelope
        noise = fractal_noise(i * 0.18, seed, octaves=5)
        envelope = register_envelope(t_local, section)
        h = max_height * envelope * (0.3 + 0.7 * noise)

        # Ensure minimum visibility
        h = max(h, max_height * 0.08)

        x = start_x + i * (bar_width + 1.5)
        y = cy - h / 2  # vertically centered

        # Slight opacity variation for depth
        opacity = 0.75 + 0.2 * noise

        bars.append({
            "x": x, "y": y, "w": bar_width, "h": h,
            "color": color, "opacity": opacity,
        })

    return bars


# ── SVG rendering ────────────────────────────

def render_svg(
    width: int, height: int, bars: list[dict],
    bg: str, transparent: bool, glow: bool,
) -> str:
    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )

    if not transparent:
        svg.append(f'<rect width="100%" height="100%" fill="{bg}"/>')

    if glow:
        svg.append("""
<defs>
<filter id="glow">
<feGaussianBlur stdDeviation="1.8" result="blur"/>
<feMerge>
<feMergeNode in="blur"/>
<feMergeNode in="SourceGraphic"/>
</feMerge>
</filter>
</defs>
""")

    glow_attr = 'filter="url(#glow)" ' if glow else ""

    for bar in bars:
        svg.append(
            f'<rect x="{bar["x"]:.2f}" y="{bar["y"]:.2f}" '
            f'width="{bar["w"]:.2f}" height="{bar["h"]:.2f}" '
            f'rx="1.5" ry="1.5" '
            f'fill="{bar["color"]}" opacity="{bar["opacity"]:.2f}" '
            f'{glow_attr}/>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser(description="Generate VoxID banner SVG")
    parser.add_argument("--output", type=Path, default=Path("banner.svg"))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--bg", default=BG)
    parser.add_argument("--transparent-bg", action="store_true")
    parser.add_argument("--no-glow", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=160)

    args = parser.parse_args()

    bars = generate_bars(
        width=args.width, height=args.height,
        bar_count=args.bars,
        bar_width=4.5,
        max_height=args.height * 0.75,
        seed=args.seed,
    )

    svg = render_svg(
        args.width, args.height, bars,
        bg=args.bg,
        transparent=args.transparent_bg,
        glow=not args.no_glow,
    )

    args.output.write_text(svg)
    print(f"Saved: {args.output} ({len(bars)} bars)")


if __name__ == "__main__":
    main()
