"""VoxID Explainer Video — 90-second animated overview.

Voice Persona Platform — persistent voice personas across any TTS engine.
"""

from manim import *
import numpy as np

# ── Palette ──────────────────────────────────
BG = "#1A1A2E"
SURFACE = "#16213E"
TEAL = "#1ABC9C"
CORAL = "#E8634A"
AMBER = "#F39C12"
EMERALD = "#2ECC71"
TXT = "#ECF0F1"
MUTED = "#95A5A6"
BLUE_ACC = "#5DADE2"


def fade_all(scene, run_time=0.6):
    if scene.mobjects:
        scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=run_time)
        scene.wait(0.2)


class VoxIDExplainer(Scene):
    def construct(self):
        self.camera.background_color = BG

        self.scene_hook()
        fade_all(self)
        self.scene_problem()
        fade_all(self)
        self.scene_enrollment()
        fade_all(self)
        self.scene_persona()
        fade_all(self)
        self.scene_generation()
        fade_all(self)
        self.scene_cta()

    # ══════════════════════════════════════════
    # Scene 1: Hook (0–8s)
    # ══════════════════════════════════════════
    def scene_hook(self):
        # Voice-wave pulse ripple — concentric sound arcs from center
        arcs = VGroup()
        for i in range(6):
            arc = Arc(
                radius=0.5 + i * 0.7,
                start_angle=PI / 4,
                angle=PI / 2,
                color=TEAL,
                stroke_width=3 - i * 0.35,
            )
            arcs.add(arc)
        # Mirror arcs on the left side
        arcs_left = VGroup()
        for i in range(6):
            arc = Arc(
                radius=0.5 + i * 0.7,
                start_angle=PI * 5 / 4,
                angle=PI / 2,
                color=TEAL,
                stroke_width=3 - i * 0.35,
            )
            arcs_left.add(arc)

        self.play(
            LaggedStart(
                *[Create(a) for a in arcs],
                lag_ratio=0.12,
            ),
            LaggedStart(
                *[Create(a) for a in arcs_left],
                lag_ratio=0.12,
            ),
            run_time=1.5,
        )
        self.play(
            *[a.animate.set_stroke(opacity=0.15) for a in arcs],
            *[a.animate.set_stroke(opacity=0.15) for a in arcs_left],
            run_time=0.8,
        )

        # Tagline
        tagline = Text(
            "One voice. Many styles. Every engine.",
            font_size=32, color=TXT,
        )
        tagline.move_to(UP * 0.5)
        self.play(FadeIn(tagline, shift=UP * 0.2), run_time=1)
        self.wait(1)

        # VoxID logo
        vox = Text("Vox", font_size=64, weight=BOLD, color=CORAL)
        vid = Text("ID", font_size=64, weight=BOLD, color=TEAL)
        vid.next_to(vox, RIGHT, buff=0.05)
        logo = VGroup(vox, vid).move_to(DOWN * 1)

        self.play(
            tagline.animate.shift(UP * 0.5).set_opacity(0.6),
            FadeIn(logo, scale=1.2),
            run_time=1,
        )
        self.wait(1.5)

    # ══════════════════════════════════════════
    # Scene 2: The Problem (8–25s)
    # ══════════════════════════════════════════
    def scene_problem(self):
        # Left: person with varied speech styles (colored bubbles)
        person = Circle(radius=0.4, color=TEAL, fill_opacity=0.3).shift(LEFT * 3.5 + UP * 0.5)
        person_label = Text("You", font_size=20, color=TEAL).next_to(person, DOWN, buff=0.2)

        bubbles = VGroup()
        bubble_data = [
            (LEFT * 1.8 + UP * 2.2, 0.7, TEAL),
            (LEFT * 1.2 + UP * 1.2, 0.5, AMBER),
            (LEFT * 2.5 + UP * 0.2, 0.6, EMERALD),
            (LEFT * 1.5 + DOWN * 0.5, 0.4, BLUE_ACC),
        ]
        for pos, size, color in bubble_data:
            b = RoundedRectangle(
                width=size * 2.5, height=size, corner_radius=0.15,
                color=color, fill_opacity=0.15,
            ).move_to(pos)
            bubbles.add(b)

        # Right: robot with identical bars (flat, single-style output)
        robot = Square(side_length=0.7, color=MUTED, fill_opacity=0.3).shift(RIGHT * 3.5 + UP * 0.5)
        robot_label = Text("Clone", font_size=20, color=MUTED).next_to(robot, DOWN, buff=0.2)

        bars = VGroup()
        for i in range(4):
            bar = Rectangle(
                width=1.2, height=0.25, color=MUTED, fill_opacity=0.2,
            ).shift(RIGHT * 1.8 + UP * (1.8 - i * 0.5))
            bars.add(bar)

        # Divider
        divider = DashedLine(UP * 3, DOWN * 2.5, color=MUTED, dash_length=0.15)

        self.play(
            FadeIn(person), FadeIn(person_label),
            FadeIn(robot), FadeIn(robot_label),
            Create(divider),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.8) for b in bubbles], lag_ratio=0.15),
            LaggedStart(*[FadeIn(b, shift=RIGHT * 0.3) for b in bars], lag_ratio=0.15),
            run_time=1.5,
        )
        self.wait(1)

        # Caption
        caption = Text(
            "One recording. One style. One output.",
            font_size=28, color=CORAL,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.8)
        self.wait(2)

        # Collapse into a dot
        dot = Dot(ORIGIN, radius=0.08, color=CORAL)
        all_elements = VGroup(person, person_label, bubbles, robot, robot_label, bars, divider, caption)
        self.play(
            *[m.animate.move_to(ORIGIN).scale(0.01) for m in all_elements],
            FadeIn(dot),
            run_time=1.2,
        )
        self.wait(0.5)

    # ══════════════════════════════════════════
    # Scene 3: Enrollment (25–45s)
    # ══════════════════════════════════════════
    def scene_enrollment(self):
        title = Text("Guided Enrollment", font_size=36, weight=BOLD, color=TXT)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.5)

        # Script panel with annotation markers
        script_bg = RoundedRectangle(
            width=10, height=1.6, corner_radius=0.15,
            color=SURFACE, fill_opacity=0.8,
        ).shift(UP * 1.2)

        words = [
            ("Hey, ", TXT), ("I was ", TXT), ("actually ", CORAL),
            ("thinking ", TXT), ("we could ", TXT), ("completely ", CORAL),
            ("rethink ", TXT), ("the approach.", TXT),
        ]
        word_group = VGroup()
        x_offset = -4.2
        for word_text, color in words:
            w = Text(word_text, font_size=22, color=color)
            w.move_to(UP * 1.2 + RIGHT * x_offset, aligned_edge=LEFT)
            x_offset += w.width + 0.05
            word_group.add(w)

        # Pause bars
        pause1 = Rectangle(width=0.06, height=0.4, color=BLUE_ACC, fill_opacity=0.7)
        pause1.move_to(word_group[1].get_right() + RIGHT * 0.1)
        pause2 = Rectangle(width=0.12, height=0.4, color=BLUE_ACC, fill_opacity=0.7)
        pause2.move_to(word_group[7].get_right() + RIGHT * 0.15)

        self.play(FadeIn(script_bg), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(w, shift=UP * 0.1) for w in word_group], lag_ratio=0.06),
            run_time=1.5,
        )
        self.play(FadeIn(pause1), FadeIn(pause2), run_time=0.3)

        # Emphasis underlines
        emphasis_boxes = VGroup()
        for idx in [2, 5]:
            underline = Line(
                word_group[idx].get_corner(DL),
                word_group[idx].get_corner(DR),
                color=CORAL, stroke_width=3,
            )
            emphasis_boxes.add(underline)
        self.play(*[Create(u) for u in emphasis_boxes], run_time=0.5)
        self.wait(0.5)

        # Waveform bars
        wave_bars = VGroup()
        for i in range(40):
            h = 0.1 + 0.5 * abs(np.sin(i * 0.3))
            bar = Rectangle(
                width=0.18, height=h, color=TEAL, fill_opacity=0.7,
            ).shift(LEFT * 4.5 + RIGHT * i * 0.23 + DOWN * 0.8)
            wave_bars.add(bar)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in wave_bars], lag_ratio=0.03),
            run_time=2,
        )

        # Quality meters
        meter_labels = ["SNR", "RMS", "Speech"]
        meter_colors = [EMERALD, AMBER, TEAL]
        meter_fills = [0.85, 0.65, 0.72]
        meters_group = VGroup()

        for i, (label, color, fill) in enumerate(zip(meter_labels, meter_colors, meter_fills)):
            lbl = Text(label, font_size=16, color=MUTED).shift(LEFT * 5.5 + DOWN * (2.0 + i * 0.45))
            bg_bar = Rectangle(width=3, height=0.2, color=SURFACE, fill_opacity=0.5)
            bg_bar.next_to(lbl, RIGHT, buff=0.3)
            fill_bar = Rectangle(width=0, height=0.2, color=color, fill_opacity=0.7)
            fill_bar.align_to(bg_bar, LEFT)
            meters_group.add(lbl, bg_bar, fill_bar)

        self.play(FadeIn(meters_group), run_time=0.3)

        fill_anims = []
        for i in range(3):
            fill_bar = meters_group[i * 3 + 2]
            target_w = 3 * meter_fills[i]
            fill_anims.append(fill_bar.animate.stretch_to_fit_width(target_w).align_to(meters_group[i * 3 + 1], LEFT))

        self.play(*fill_anims, run_time=1.5)

        # Phoneme coverage
        cov_label = Text("Phoneme Coverage", font_size=18, color=MUTED).shift(RIGHT * 2.5 + DOWN * 2.0)
        cov_bg = Rectangle(width=3.5, height=0.25, color=SURFACE, fill_opacity=0.5)
        cov_bg.next_to(cov_label, DOWN, buff=0.15)
        cov_fill = Rectangle(width=0, height=0.25, color=EMERALD, fill_opacity=0.7)
        cov_fill.align_to(cov_bg, LEFT)
        cov_pct = Text("0%", font_size=18, color=EMERALD)
        cov_pct.next_to(cov_bg, RIGHT, buff=0.2)

        self.play(FadeIn(cov_label), FadeIn(cov_bg), FadeIn(cov_fill), FadeIn(cov_pct), run_time=0.3)
        self.play(
            cov_fill.animate.stretch_to_fit_width(3.5 * 0.92).align_to(cov_bg, LEFT),
            ReplacementTransform(cov_pct, Text("92%", font_size=18, color=EMERALD).next_to(cov_bg, RIGHT, buff=0.2)),
            run_time=2,
        )

        enroll_caption = Text(
            "Guided enrollment captures every sound in your voice",
            font_size=24, color=TXT,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(enroll_caption, shift=UP * 0.2), run_time=0.5)
        self.wait(2)

    # ══════════════════════════════════════════
    # Scene 4: Persona Dashboard (45–60s)
    # ══════════════════════════════════════════
    def scene_persona(self):
        title = Text("Persona Dashboard", font_size=36, weight=BOLD, color=TXT)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.5)

        # 3 persona cards
        cards = VGroup()
        names = ["Alice", "Tom", "Sarah"]
        style_counts = [3, 2, 4]
        colors = [TEAL, CORAL, AMBER]

        for i, (name, count, color) in enumerate(zip(names, style_counts, colors)):
            card = RoundedRectangle(
                width=3.2, height=2, corner_radius=0.15,
                color=SURFACE, fill_opacity=0.8,
            )
            avatar = Circle(radius=0.3, color=color, fill_opacity=0.3)
            avatar.shift(UP * 0.3)
            name_txt = Text(name, font_size=22, weight=BOLD, color=TXT)
            name_txt.next_to(avatar, DOWN, buff=0.15)
            style_txt = Text(f"{count} styles", font_size=16, color=MUTED)
            style_txt.next_to(name_txt, DOWN, buff=0.1)
            initial = Text(name[0], font_size=24, weight=BOLD, color=color)
            initial.move_to(avatar)
            card_group = VGroup(card, avatar, initial, name_txt, style_txt)
            cards.add(card_group)

        cards.arrange(RIGHT, buff=0.5)
        cards.move_to(UP * 0.3)

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cards], lag_ratio=0.2),
            run_time=1.5,
        )
        self.wait(1)

        # Zoom into Tom
        self.play(
            cards[0].animate.set_opacity(0.2).scale(0.8),
            cards[2].animate.set_opacity(0.2).scale(0.8),
            cards[1].animate.scale(1.3).move_to(LEFT * 2),
            run_time=1,
        )

        # Style detail panel
        detail = VGroup()
        styles_data = [
            ("conversational", "qwen3-tts", TEAL),
            ("technical", "fish-speech", CORAL),
        ]
        for j, (sname, engine, color) in enumerate(styles_data):
            row = VGroup(
                Circle(radius=0.12, color=color, fill_opacity=0.5),
                Text(sname, font_size=18, color=TXT),
                Text(engine, font_size=14, color=MUTED),
            ).arrange(RIGHT, buff=0.3)
            detail.add(row)

        consent_badge = VGroup(
            RoundedRectangle(width=1.4, height=0.35, corner_radius=0.1, color=EMERALD, fill_opacity=0.2),
            Text("consent ✓", font_size=14, color=EMERALD),
        )
        consent_badge[1].move_to(consent_badge[0])
        detail.add(consent_badge)
        detail.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        detail.move_to(RIGHT * 2.5)

        self.play(
            LaggedStart(*[FadeIn(d, shift=RIGHT * 0.2) for d in detail], lag_ratio=0.15),
            run_time=1,
        )

        caption = Text(
            "Persistent personas — styles, engines, and consent in one place",
            font_size=24, color=TXT,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.5)
        self.wait(2.5)

    # ══════════════════════════════════════════
    # Scene 5: Generation (60–75s)
    # ══════════════════════════════════════════
    def scene_generation(self):
        title = Text("Smart Generation", font_size=36, weight=BOLD, color=TXT)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.5)

        # Text input
        input_bg = RoundedRectangle(
            width=10, height=1, corner_radius=0.1,
            color=SURFACE, fill_opacity=0.8,
        ).shift(UP * 1.5)
        input_text = Text(
            "The architecture handles load balancing across three regions.",
            font_size=18, color=TXT,
        ).move_to(input_bg)

        self.play(FadeIn(input_bg), run_time=0.3)
        self.play(Write(input_text), run_time=2)
        self.wait(0.5)

        # Style score chips
        scores = [
            ("technical", 0.89, True),
            ("conversational", 0.12, False),
            ("narration", 0.06, False),
            ("emphatic", 0.02, False),
        ]
        chips = VGroup()
        for style, score, selected in scores:
            color = CORAL if selected else MUTED
            fill_op = 0.3 if selected else 0.1
            chip = VGroup(
                RoundedRectangle(
                    width=2.2, height=0.45, corner_radius=0.15,
                    color=color, fill_opacity=fill_op,
                ),
                Text(f"{style}: {score:.2f}", font_size=16, color=color),
            )
            chip[1].move_to(chip[0])
            chips.add(chip)

        chips.arrange(RIGHT, buff=0.3)
        chips.move_to(UP * 0.3)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.8) for c in chips], lag_ratio=0.1),
            run_time=1,
        )

        self.play(Indicate(chips[0], color=CORAL, scale_factor=1.1), run_time=0.6)
        self.wait(0.5)

        # Generated waveform
        gen_wave = VGroup()
        for i in range(50):
            h = 0.05 + 0.4 * abs(np.sin(i * 0.25 + 0.5))
            bar = Rectangle(
                width=0.15, height=h, color=TEAL, fill_opacity=0.7,
            ).shift(LEFT * 4.5 + RIGHT * i * 0.18 + DOWN * 1.5)
            gen_wave.add(bar)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in gen_wave], lag_ratio=0.02),
            run_time=2,
        )

        caption = Text(
            "Auto-routes to the right speaking style",
            font_size=24, color=TXT,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.5)
        self.wait(2)

    # ══════════════════════════════════════════
    # Scene 6: Call to Action (75–90s)
    # ══════════════════════════════════════════
    def scene_cta(self):
        # Left: waveform bars (teal) representing voice output
        left_bars = VGroup()
        for i in range(12):
            h = 0.3 + 1.2 * abs(np.sin(i * 0.4))
            bar = Rectangle(
                width=0.25, height=h, color=TEAL, fill_opacity=0.7,
            ).shift(LEFT * 5 + RIGHT * i * 0.35)
            left_bars.add(bar)

        # Right: persona style arcs — layered voice registers (not fingerprint)
        style_arcs = VGroup()
        arc_colors = [TEAL, CORAL, AMBER, EMERALD]
        arc_labels = ["conv", "tech", "narr", "emph"]
        center = RIGHT * 3
        for i, (color, label) in enumerate(zip(arc_colors, arc_labels)):
            r = 0.6 + i * 0.45
            arc = Arc(
                radius=r,
                start_angle=-PI / 3,
                angle=2 * PI / 3,
                color=color,
                stroke_width=4 - i * 0.5,
            )
            arc.move_to(center)
            style_arcs.add(arc)

        # Persona dot at center of arcs
        persona_dot = Dot(center, radius=0.15, color=TXT)
        persona_label = Text("persona", font_size=14, color=MUTED)
        persona_label.next_to(persona_dot, DOWN, buff=0.3)

        # Transition elements in the middle — gradient bars
        mid_elements = VGroup()
        for i in range(5):
            h = 0.2 + 0.8 * abs(np.sin(i * 0.5))
            bar = Rectangle(
                width=0.15, height=h,
                color=_lerp_color(_hex_to_rgb(TEAL), _hex_to_rgb(CORAL), i / 4),
                fill_opacity=0.4,
            ).shift(LEFT * 1 + RIGHT * i * 0.4)
            mid_elements.add(bar)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in left_bars], lag_ratio=0.05),
            run_time=1.5,
        )
        self.play(
            LaggedStart(*[Create(a) for a in style_arcs], lag_ratio=0.1),
            FadeIn(persona_dot),
            FadeIn(persona_label),
            FadeIn(mid_elements),
            run_time=1.5,
        )
        self.wait(0.5)

        # Logo
        vox = Text("Vox", font_size=72, weight=BOLD, color=CORAL)
        vid = Text("ID", font_size=72, weight=BOLD, color=TEAL)
        vid.next_to(vox, RIGHT, buff=0.05)
        logo = VGroup(vox, vid).move_to(ORIGIN)

        self.play(
            left_bars.animate.set_opacity(0.2),
            style_arcs.animate.set_opacity(0.2),
            persona_dot.animate.set_opacity(0.15),
            persona_label.animate.set_opacity(0.15),
            mid_elements.animate.set_opacity(0.15),
            FadeIn(logo, scale=1.3),
            run_time=1.2,
        )

        # Subtitle and install command
        subtitle = Text(
            "Voice Persona Platform", font_size=28, color=MUTED,
        ).shift(DOWN * 1)
        install = Text(
            "uv add voxid", font_size=24, color=MUTED,
        ).shift(DOWN * 1.8)
        self.play(
            FadeIn(subtitle, shift=UP * 0.2),
            run_time=0.5,
        )
        self.play(FadeIn(install, shift=UP * 0.2), run_time=0.5)
        self.wait(2)

        # Fade to black
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.5,
        )
        self.wait(1)


def _hex_to_rgb(hex_color):
    """Convert hex string to RGB tuple for interpolation."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _lerp_color(c1, c2, t):
    """Interpolate between two RGB tuples."""
    r = c1[0] + (c2[0] - c1[0]) * t
    g = c1[1] + (c2[1] - c1[1]) * t
    b = c1[2] + (c2[2] - c1[2]) * t
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
