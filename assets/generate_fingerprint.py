#!/usr/bin/env python3
"""
High-Quality Fingerprint SVG Generator

Features:
- Deterministic procedural fingerprint geometry
- True normal-direction noise displacement
- Chaikin smoothing for clean ridges
- Multi-octave coherent noise (no external deps)
- Layered strokes for premium rendering
- Optional glow + transparent background
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np


Point = Tuple[float, float]


# ---------------------------
# Noise utilities
# ---------------------------

def smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def hash_noise(x: float, seed: int) -> float:
    return math.sin(x * 12.9898 + seed * 78.233) * 43758.5453 % 1.0


def coherent_noise(x: float, seed: int) -> float:
    i = math.floor(x)
    f = x - i

    a = hash_noise(i, seed)
    b = hash_noise(i + 1, seed)

    return lerp(a, b, smoothstep(f))


def fractal_noise(
    x: float,
    seed: int,
    octaves: int = 3,
    persistence: float = 0.5
) -> float:
    total = 0.0
    freq = 1.0
    amp = 1.0
    max_val = 0.0

    for o in range(octaves):
        total += coherent_noise(x * freq, seed + o * 17) * amp
        max_val += amp
        amp *= persistence
        freq *= 2.0

    return total / max_val


# ---------------------------
# Geometry utilities
# ---------------------------

def normal_vector(angle: float) -> Tuple[float, float]:
    return (-math.sin(angle), math.cos(angle))


def chaikin(points: List[Point], iterations: int = 2) -> List[Point]:
    pts = points

    for _ in range(iterations):
        new_pts = []

        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]

            q = (
                0.75 * p0[0] + 0.25 * p1[0],
                0.75 * p0[1] + 0.25 * p1[1],
            )
            r = (
                0.25 * p0[0] + 0.75 * p1[0],
                0.25 * p0[1] + 0.75 * p1[1],
            )

            new_pts.extend([q, r])

        pts = new_pts

    return pts


def points_to_path(points: List[Point]) -> str:
    if not points:
        return ""

    d = f"M {points[0][0]:.2f},{points[0][1]:.2f}"

    for p in points[1:]:
        d += f" L {p[0]:.2f},{p[1]:.2f}"

    return d


# ---------------------------
# Ridge generation
# ---------------------------

def generate_ridge(
    cx: float,
    cy: float,
    radius: float,
    angle_span: float,
    seed: int,
    noise_amp: float,
    points: int = 90,
) -> List[Point]:

    angles = np.linspace(
        -angle_span / 2,
        angle_span / 2,
        points,
    )

    pts: List[Point] = []

    for i, angle in enumerate(angles):

        nx, ny = normal_vector(angle)

        noise = fractal_noise(
            i * 0.15,
            seed,
            octaves=4,
        )

        displacement = (noise - 0.5) * noise_amp

        x = cx + radius * math.cos(angle) + nx * displacement
        y = cy + radius * math.sin(angle) + ny * displacement

        pts.append((x, y))

    return chaikin(pts, iterations=2)


def generate_fingerprint(
    cx: float,
    cy: float,
    ridges: int,
    core_ridges: int,
    spacing: float,
    scale: float,
    seed: int,
) -> List[List[Point]]:

    rng = np.random.default_rng(seed)

    all_paths: List[List[Point]] = []

    # Core whorl
    for r in range(core_ridges):

        radius = (r + 1) * spacing * scale

        pts = generate_ridge(
            cx,
            cy,
            radius,
            angle_span=2.6,
            seed=seed + r,
            noise_amp=spacing * 0.9,
        )

        all_paths.append(pts)

    # Outer loops
    for r in range(core_ridges, ridges):

        radius = (r + 1) * spacing * scale

        pts = generate_ridge(
            cx,
            cy,
            radius,
            angle_span=2.2,
            seed=seed + r * 3,
            noise_amp=spacing * 0.7,
        )

        all_paths.append(pts)

    return all_paths


# ---------------------------
# SVG rendering
# ---------------------------

def render_svg(
    paths: List[List[Point]],
    width: int,
    height: int,
    color: str,
    accent: str,
    bg: str,
    transparent: bool,
    glow: bool,
) -> str:

    svg = []

    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )

    if not transparent:
        svg.append(
            f'<rect width="100%" height="100%" fill="{bg}"/>'
        )

    if glow:
        svg.append(
            """
<defs>
<filter id="glow">
<feGaussianBlur stdDeviation="1.2" result="coloredBlur"/>
<feMerge>
<feMergeNode in="coloredBlur"/>
<feMergeNode in="SourceGraphic"/>
</feMerge>
</filter>
</defs>
"""
        )

    for i, pts in enumerate(paths):

        path_d = points_to_path(pts)

        stroke_width = max(1.0, 2.4 - i * 0.04)

        glow_attr = 'filter="url(#glow)" ' if glow else ""
        svg.append(
            f'<path d="{path_d}" '
            f'stroke="{color}" '
            f'stroke-width="{stroke_width:.2f}" '
            f'fill="none" '
            f'stroke-linecap="round" '
            f'stroke-linejoin="round" '
            f'opacity="0.95" '
            f'{glow_attr}/>'
        )

        svg.append(
            f'<path d="{path_d}" '
            f'stroke="{accent}" '
            f'stroke-width="{stroke_width * 0.35:.2f}" '
            f'fill="none" '
            f'opacity="0.45"/>'
        )

    svg.append("</svg>")

    return "\n".join(svg)


# ---------------------------
# CLI
# ---------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Generate high-quality fingerprint SVG"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fingerprint.svg"),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--color",
        default="#E8634A",
    )

    parser.add_argument(
        "--accent-color",
        default="#F39A84",
    )

    parser.add_argument(
        "--bg",
        default="#1A1A2E",
    )

    parser.add_argument(
        "--transparent-bg",
        action="store_true",
    )

    parser.add_argument(
        "--no-glow",
        action="store_true",
    )

    parser.add_argument(
        "--ridges",
        type=int,
        default=38,
    )

    parser.add_argument(
        "--core-ridges",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--ridge-spacing",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.ridges <= 0:
        raise ValueError("ridges must be > 0")

    if args.core_ridges >= args.ridges:
        raise ValueError(
            "core-ridges must be smaller than ridges"
        )

    cx = args.width / 2
    cy = args.height / 2

    paths = generate_fingerprint(
        cx,
        cy,
        ridges=args.ridges,
        core_ridges=args.core_ridges,
        spacing=args.ridge_spacing,
        scale=args.scale,
        seed=args.seed,
    )

    svg = render_svg(
        paths,
        args.width,
        args.height,
        color=args.color,
        accent=args.accent_color,
        bg=args.bg,
        transparent=args.transparent_bg,
        glow=not args.no_glow,
    )

    args.output.write_text(svg)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()