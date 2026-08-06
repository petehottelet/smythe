"""Deterministic cyber-glyph provider and digital-rain asset assembly.

The default 192 glyphs and extended 256-glyph benchmark catalog are
fictional procedural marks built from a calligraphic
stroke grammar: horizontal bars, vertical stems, hooks, enclosures, press
diagonals, bowls, tail sweeps, and diacritic dots, composed on an ideograph
grid with occasional serif nubs. The vocabulary is reminiscent of hand-drawn
ideographs and romanesque letterforms without reproducing any real character,
font, logo, or source image. The assembled visuals use the general vocabulary
of green digital rain -- black ground, glowing descending columns, bright
heads, and varied trails -- without copying exact reference pixels.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from smythe.provider import Artifact, CompletionResult, Provider


GLYPH_COUNT = 192
MAX_GLYPH_COUNT = 256
TILE_SIZE = 128
PREVIEW_SIZE = (1920, 1080)
GIF_SIZE = (640, 360)
GIF_FRAMES = 12
ATLAS_GRID = (16, 12)
ATLAS_SIZE = (ATLAS_GRID[0] * TILE_SIZE, ATLAS_GRID[1] * TILE_SIZE)
DEFAULT_SEED = 0x5A17_2026

# Stroke canvas: an upright ideograph cell, wider than a roman em box.
CANVAS_W = 100.0
CANVAS_H = 140.0
_X = (10.0, 28.0, 50.0, 72.0, 90.0)
_Y = (12.0, 40.0, 70.0, 100.0, 128.0)
_STROKE_KINDS = {"l": 5, "q": 7, "d": 3}


@dataclass(frozen=True, slots=True)
class GlyphSpec:
    """One fictional stroke-built glyph plus animation characteristics.

    ``strokes`` is a tuple of stroke tuples in glyph canvas coordinates:
    ``("l", x1, y1, x2, y2, width)`` straight stroke,
    ``("q", x1, y1, cx, cy, x2, y2, width)`` quadratic stroke, and
    ``("d", x, y, radius)`` dot.
    """

    id: str
    strokes: tuple[tuple, ...]
    speed: float
    trail_length: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"glyph-[0-9]{3}", self.id):
            raise ValueError(f"invalid glyph id: {self.id!r}")
        if not isinstance(self.strokes, tuple) or not 2 <= len(self.strokes) <= 9:
            raise ValueError("strokes must be a tuple of two to nine strokes")
        for stroke in self.strokes:
            if not isinstance(stroke, tuple) or not stroke:
                raise ValueError("each stroke must be a non-empty tuple")
            kind, *values = stroke
            if kind not in _STROKE_KINDS or len(values) != _STROKE_KINDS[kind]:
                raise ValueError(f"malformed stroke: {stroke!r}")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values
            ):
                raise ValueError(f"non-numeric stroke value: {stroke!r}")
            *coords, width = values
            if width <= 0:
                raise ValueError("stroke width must be positive")
            xs = coords[0::2]
            ys = coords[1::2]
            if any(not -10 <= x <= 110 for x in xs) or any(
                not -12 <= y <= 152 for y in ys
            ):
                raise ValueError(f"stroke leaves the glyph canvas: {stroke!r}")
        if isinstance(self.speed, bool) or not isinstance(self.speed, (int, float)):
            raise TypeError("speed must be numeric")
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        if isinstance(self.trail_length, bool) or not isinstance(self.trail_length, int):
            raise TypeError("trail_length must be an integer")
        if self.trail_length < 2:
            raise ValueError("trail_length must be at least two")


@dataclass(frozen=True, slots=True)
class OutputReceipt:
    """Objective, hash-bound facts about one generated output."""

    path: str
    sha256: str
    byte_size: int
    format: str
    width: int
    height: int
    frames: int = 1


@dataclass(frozen=True, slots=True)
class GlyphSuiteReceipt:
    """Receipts for a complete tile set and its four assembled views."""

    tiles: tuple[OutputReceipt, ...]
    preview: OutputReceipt
    animation: OutputReceipt
    atlas: OutputReceipt
    html: OutputReceipt
    unique_tile_hashes: int


def _r(value: float) -> float:
    return round(float(value), 1)


def _line(x1, y1, x2, y2, width) -> tuple:
    return ("l", _r(x1), _r(y1), _r(x2), _r(y2), _r(width))


def _curve(x1, y1, cx, cy, x2, y2, width) -> tuple:
    return ("q", _r(x1), _r(y1), _r(cx), _r(cy), _r(x2), _r(y2), _r(width))


def _dot(x, y, radius) -> tuple:
    return ("d", _r(x), _r(y), _r(radius))


def _quadratic(x1, y1, cx, cy, x2, y2, t):
    u = 1 - t
    return (
        u * u * x1 + 2 * u * t * cx + t * t * x2,
        u * u * y1 + 2 * u * t * cy + t * t * y2,
    )


# Both weights sit in a narrow band: every stroke reads as the same brush,
# with "light" only slightly finer than "heavy" (per Pete's direction, no
# thin or wispy strokes anywhere in the catalog).
def _heavy(rng: random.Random) -> float:
    return rng.uniform(10.5, 12.0)


def _light(rng: random.Random) -> float:
    return rng.uniform(9.0, 10.0)


def _serif_nubs(rng: random.Random, bar: tuple, width: float) -> list[tuple]:
    """Small perpendicular caps on a horizontal bar's ends (romanesque)."""
    _, x1, y1, x2, y2, _ = bar
    nub = rng.uniform(6.0, 9.0)
    return [
        _line(x1, y1 - nub / 2, x1, y1 + nub / 2, width),
        _line(x2, y2 - nub / 2, x2, y2 + nub / 2, width),
    ]


def _hooked_stem(rng: random.Random, x, y1, y2, width) -> list[tuple]:
    """A vertical stem whose foot sweeps into a short leftward hook."""
    hook_w = rng.uniform(12.0, 20.0)
    hook_h = rng.uniform(8.0, 13.0)
    return [
        _line(x, y1, x, y2, width),
        _curve(x, y2, x, y2 + hook_h, x - hook_w, y2 + hook_h - 2.0, width * 0.85),
    ]


def _archetype_stacked_bars(rng: random.Random) -> list[tuple]:
    heavy = _heavy(rng)
    light = _light(rng)
    strokes: list[tuple] = []
    bar_ys = sorted(rng.sample(_Y, rng.randint(2, 4)))
    spans = [(_X[0], _X[4]), (_X[0] + 6, _X[4] - 6), (_X[1], _X[3]), (_X[0], _X[3]),
             (_X[1], _X[4])]
    longest = None
    for y in bar_ys:
        x1, x2 = rng.choice(spans)
        bar = _line(x1 + rng.uniform(-3, 3), y + rng.uniform(-2, 2),
                    x2 + rng.uniform(-3, 3), y + rng.uniform(-2, 2),
                    heavy if rng.random() < 0.6 else light)
        strokes.append(bar)
        if longest is None or (bar[3] - bar[1]) > (longest[3] - longest[1]):
            longest = bar
    if rng.random() < 0.8:
        x = rng.choice((_X[2], _X[2], _X[1], _X[3])) + rng.uniform(-3, 3)
        y1 = min(bar_ys) - rng.uniform(6, 14)
        y2 = max(bar_ys) + rng.uniform(10, 24)
        if rng.random() < 0.45:
            strokes.extend(_hooked_stem(rng, x, y1, min(y2, _Y[4]), heavy))
        else:
            strokes.append(_line(x, y1, x, min(y2, _Y[4] + 6), heavy))
    if rng.random() < 0.35:
        # A short press accent crossing one bar corner.
        strokes.append(
            _curve(_X[3] + rng.uniform(0, 8), _Y[0] - 2,
                   _X[3] - 4, _Y[0] + 12,
                   _X[2] + rng.uniform(-6, 6), _Y[1] + rng.uniform(0, 8),
                   light)
        )
    if rng.random() < 0.3:
        strokes.append(_dot(rng.choice((_X[1], _X[3])) + rng.uniform(-4, 4),
                            min(bar_ys) - rng.uniform(8, 12),
                            rng.uniform(5.2, 6.6)))
    if longest is not None and rng.random() < 0.35:
        strokes.extend(_serif_nubs(rng, longest, light))
    return strokes


def _archetype_enclosure(rng: random.Random) -> list[tuple]:
    heavy = _heavy(rng)
    light = _light(rng)
    left = _X[0] + rng.uniform(0, 6)
    right = _X[4] - rng.uniform(0, 6)
    top = _Y[0] + rng.uniform(0, 10)
    bottom = _Y[2] + rng.uniform(10, _Y[4] - _Y[2])
    sides = {
        "top": _line(left, top, right, top, heavy),
        "left": _line(left, top, left, bottom, heavy),
        "right": _line(right, top, right, bottom, heavy),
        "bottom": _line(left, bottom, right, bottom, heavy),
    }
    # Leave zero or one side open, so boxes vary between full frames and
    # bracket forms opening in any direction.
    open_side = rng.choice((None, None, "bottom", "left", "right", "top"))
    strokes = [stroke for side, stroke in sides.items() if side != open_side]
    inner = rng.randint(1, 3)
    mid_y = (top + bottom) / 2
    for _ in range(inner):
        pick = rng.random()
        if pick < 0.35:
            strokes.append(
                _line(left + 12, mid_y + rng.uniform(-14, 14),
                      right - 12, mid_y + rng.uniform(-14, 14), light)
            )
        elif pick < 0.6:
            strokes.append(
                _line(_X[2] + rng.uniform(-6, 6), top + 8,
                      _X[2] + rng.uniform(-6, 6), bottom - 8, light)
            )
        elif pick < 0.82:
            strokes.append(_dot(_X[2] + rng.uniform(-16, 16),
                                mid_y + rng.uniform(-12, 12),
                                rng.uniform(5.2, 6.8)))
        else:
            strokes.append(
                _curve(left + 12, mid_y - 8, _X[2], mid_y + 14,
                       right - 12, mid_y - 8, light)
            )
    if bottom < _Y[3] and rng.random() < 0.6:
        # A leg or under-mark grounds a high box (two-component feel).
        if rng.random() < 0.5:
            strokes.append(_line(_X[2] + rng.uniform(-4, 4), bottom,
                                 _X[2] + rng.uniform(-4, 4), _Y[4] + 4, heavy))
        else:
            strokes.append(_line(_X[1], _Y[4] + rng.uniform(-4, 4),
                                 _X[3], _Y[4] + rng.uniform(-4, 4), heavy))
    return strokes


def _archetype_radical_split(rng: random.Random) -> list[tuple]:
    heavy = _heavy(rng)
    light = _light(rng)
    stem_x = _X[0] + rng.uniform(2, 7)
    strokes = [_line(stem_x, _Y[0], stem_x, _Y[4] + rng.uniform(0, 6), heavy)]
    for _ in range(rng.randint(1, 2)):
        y = rng.uniform(_Y[1], _Y[3])
        strokes.append(_line(stem_x - 6, y, stem_x + rng.uniform(10, 16), y, light))
    right_x1 = _X[2] + rng.uniform(-4, 2)
    right_x2 = _X[4] + rng.uniform(-2, 2)
    pick = rng.random()
    if pick < 0.4:
        for y in sorted(rng.sample(_Y[1:4], 2)):
            strokes.append(_line(right_x1, y, right_x2, y, heavy))
        strokes.append(
            _line((right_x1 + right_x2) / 2, _Y[1] - 8,
                  (right_x1 + right_x2) / 2, _Y[3] + rng.uniform(8, 22), light)
        )
    elif pick < 0.7:
        strokes.append(
            _curve(right_x1, _Y[1], right_x2 + 6, (_Y[1] + _Y[3]) / 2,
                   right_x1 + 4, _Y[3] + 12, heavy)
        )
        if rng.random() < 0.6:
            strokes.append(_dot(right_x2 - 4, _Y[0] + 4, rng.uniform(5.2, 6.6)))
    else:
        mid = (right_x1 + right_x2) / 2
        strokes.append(_curve(mid, _Y[1] - 6, right_x2 + 8, _Y[2], mid,
                              _Y[3] + 14, heavy))
        strokes.append(_line(right_x1, _Y[2], right_x2, _Y[2], light))
    return strokes


def _archetype_cross_diagonals(rng: random.Random) -> list[tuple]:
    heavy = _heavy(rng)
    light = _light(rng)
    apex_x = _X[2] + rng.uniform(-10, 10)
    apex_y = _Y[0] + rng.uniform(0, 12)
    left_foot = _Y[3] + rng.uniform(0, _Y[4] - _Y[3] + 8)
    right_foot = _Y[3] + rng.uniform(0, _Y[4] - _Y[3] + 8)
    left_bow = rng.uniform(6, 20)
    right_bow = rng.uniform(6, 20)
    strokes = [
        _curve(apex_x, apex_y, apex_x - left_bow, (apex_y + left_foot) / 2,
               _X[0] + rng.uniform(0, 8), left_foot, heavy),
    ]
    if rng.random() < 0.3:
        # Replace the right leg with a vertical drop for asymmetric forms.
        strokes.append(_line(apex_x + rng.uniform(6, 14), apex_y,
                             apex_x + rng.uniform(6, 14),
                             right_foot, heavy))
    else:
        strokes.append(
            _curve(apex_x, apex_y, apex_x + right_bow, (apex_y + right_foot) / 2,
                   _X[4] - rng.uniform(0, 8), right_foot, heavy)
        )
    if rng.random() < 0.7:
        y = rng.uniform(_Y[1], _Y[2] + 10)
        bar = _line(_X[1] + rng.uniform(-6, 6), y, _X[3] + rng.uniform(-6, 6), y,
                    light if rng.random() < 0.5 else heavy)
        strokes.append(bar)
    if rng.random() < 0.45:
        strokes.append(_dot(apex_x, apex_y - 8, rng.uniform(5.0, 6.6)))
    if rng.random() < 0.3:
        strokes.append(_line(_X[1], _Y[4] + rng.uniform(0, 6),
                             _X[3], _Y[4] + rng.uniform(0, 6), light))
    return strokes


def _archetype_arc_form(rng: random.Random) -> list[tuple]:
    heavy = _heavy(rng)
    light = _light(rng)
    strokes: list[tuple] = []
    orientation = rng.random()
    if orientation < 0.22:
        # Left- or right-opening C-curve with a crossing bar.
        side = rng.choice((-1, 1))
        x_open = _X[0] + 6 if side > 0 else _X[4] - 6
        x_far = _X[4] - 2 if side > 0 else _X[0] + 2
        strokes.append(
            _curve(x_open, _Y[0] + 6, x_far + side * 10, _Y[2],
                   x_open, _Y[4] - 4, heavy)
        )
        strokes.append(_line(_X[1], _Y[2] + rng.uniform(-8, 8),
                             _X[3], _Y[2] + rng.uniform(-8, 8), light))
        if rng.random() < 0.5:
            strokes.append(_dot(x_far - side * 6, _Y[1], rng.uniform(5.2, 6.4)))
    elif orientation < 0.6:
        # Dome with a hanging stem.
        strokes.append(
            _curve(_X[0] + 4, _Y[2], _X[2], _Y[0] - 6, _X[4] - 4, _Y[2], heavy)
        )
        strokes.extend(
            _hooked_stem(rng, _X[2] + rng.uniform(-3, 3), _Y[1],
                         _Y[3] + rng.uniform(4, 16), heavy)
            if rng.random() < 0.5
            else [_line(_X[2] + rng.uniform(-3, 3), _Y[1], _X[2],
                        _Y[4] + rng.uniform(0, 6), heavy)]
        )
    else:
        # Open bowl with a baseline tail sweep (arabesque flavor).
        strokes.append(
            _curve(_X[0] + 6, _Y[1], _X[2], _Y[3] + 16, _X[4] - 4, _Y[1], heavy)
        )
        strokes.append(
            _curve(_X[4] - 4, _Y[3], _X[2], _Y[4] + rng.uniform(10, 18),
                   _X[0] + rng.uniform(0, 8), _Y[4] - 2, heavy * 0.9)
        )
    for _ in range(rng.randint(0, 2)):
        strokes.append(
            _dot(_X[2] + rng.uniform(-24, 24),
                 rng.choice((_Y[0] - 4, _Y[4] + 10)) + rng.uniform(-3, 3),
                 rng.uniform(5.0, 6.6))
        )
    if rng.random() < 0.4:
        y = _Y[3] + rng.uniform(-6, 6)
        strokes.append(_line(_X[1], y, _X[3], y, light))
    return strokes


def _archetype_hooked_stem(rng: random.Random) -> list[tuple]:
    heavy = _heavy(rng)
    light = _light(rng)
    x = _X[2] + rng.uniform(-6, 6)
    strokes = _hooked_stem(rng, x, _Y[0] + rng.uniform(0, 6),
                           _Y[3] + rng.uniform(8, 20), heavy)
    for _ in range(rng.randint(1, 3)):
        side = rng.choice((-1, 1))
        y = rng.uniform(_Y[0] + 6, _Y[3])
        if rng.random() < 0.5:
            strokes.append(_line(x + side * 6, y, x + side * rng.uniform(16, 26),
                                 y + rng.uniform(-4, 4), light))
        else:
            strokes.append(_dot(x + side * rng.uniform(14, 24), y,
                                rng.uniform(5.2, 6.8)))
    if rng.random() < 0.45:
        y = _Y[0] + rng.uniform(0, 6)
        bar = _line(_X[1], y, _X[3], y, heavy)
        strokes.append(bar)
        if rng.random() < 0.4:
            strokes.extend(_serif_nubs(rng, bar, light))
    return strokes


def _archetype_stacked_composite(rng: random.Random) -> list[tuple]:
    """Two stacked sub-components, echoing multi-radical ideographs."""
    heavy = _heavy(rng)
    light = _light(rng)
    strokes: list[tuple] = []
    split = _Y[1] + rng.uniform(4, 22)
    # Top component: crown bar with dot, small box, or twin ticks.
    pick = rng.random()
    if pick < 0.38:
        y = _Y[0] + rng.uniform(0, 6)
        strokes.append(_line(_X[1] - 4, y, _X[3] + 4, y, heavy))
        strokes.append(_dot(_X[2] + rng.uniform(-6, 6), y - rng.uniform(8, 11),
                            rng.uniform(5.2, 6.6)))
    elif pick < 0.7:
        left, right = _X[1] + rng.uniform(-6, 0), _X[3] + rng.uniform(0, 6)
        top = _Y[0] + rng.uniform(0, 4)
        strokes.extend((
            _line(left, top, right, top, heavy),
            _line(left, top, left, split - 6, heavy),
            _line(right, top, right, split - 6, heavy),
        ))
    else:
        for side in (-1, 1):
            x = _X[2] + side * rng.uniform(10, 18)
            strokes.append(_line(x, _Y[0] + 2, x + side * 4, split - 10, light))
    # Bottom component: wide bar + legs, bowl, or barred stem.
    pick = rng.random()
    if pick < 0.4:
        strokes.append(_line(_X[0] + 2, split, _X[4] - 2, split, heavy))
        for side in (-1, 1):
            x = _X[2] + side * rng.uniform(14, 24)
            strokes.append(
                _curve(_X[2] + side * 4, split + 4,
                       x, (split + _Y[4]) / 2,
                       x + side * rng.uniform(0, 6), _Y[4] + rng.uniform(0, 6),
                       heavy)
            )
    elif pick < 0.72:
        strokes.append(
            _curve(_X[0] + 8, split + 8, _X[2], _Y[4] + rng.uniform(8, 16),
                   _X[4] - 8, split + 8, heavy)
        )
        if rng.random() < 0.5:
            strokes.append(_line(_X[2] + rng.uniform(-3, 3), split + 2,
                                 _X[2] + rng.uniform(-3, 3), _Y[4], light))
    else:
        x = _X[2] + rng.uniform(-4, 4)
        strokes.extend(_hooked_stem(rng, x, split, _Y[4] - 4, heavy))
        y = (split + _Y[4]) / 2
        strokes.append(_line(x - rng.uniform(16, 24), y,
                             x + rng.uniform(16, 24), y, light))
    return strokes


_ARCHETYPES = (
    (_archetype_stacked_bars, 18),
    (_archetype_enclosure, 18),
    (_archetype_radical_split, 16),
    (_archetype_cross_diagonals, 12),
    (_archetype_arc_form, 13),
    (_archetype_hooked_stem, 11),
    (_archetype_stacked_composite, 12),
)


def _stroke_points(stroke: tuple) -> list[tuple[float, float]]:
    """Sample a stroke's actual extent, not just its control cage."""
    if stroke[0] == "l":
        _, x1, y1, x2, y2, _ = stroke
        return [(x1, y1), ((x1 + x2) / 2, (y1 + y2) / 2), (x2, y2)]
    if stroke[0] == "q":
        _, x1, y1, cx, cy, x2, y2, _ = stroke
        return [
            _quadratic(x1, y1, cx, cy, x2, y2, step / 8) for step in range(9)
        ]
    _, x, y, radius = stroke
    return [(x - radius, y), (x + radius, y), (x, y - radius), (x, y + radius)]


def _ink_length(strokes: Sequence[tuple]) -> float:
    """Total drawn path length, a proxy for the glyph's visual mass."""
    total = 0.0
    for stroke in strokes:
        points = _stroke_points(stroke)
        if stroke[0] == "d":
            total += 2 * math.pi * stroke[3]
            continue
        total += sum(
            math.hypot(bx - ax, by - ay)
            for (ax, ay), (bx, by) in zip(points, points[1:])
        )
    return total


def _coverage_ok(strokes: Sequence[tuple]) -> bool:
    """Reject skinny or underfilled marks before they enter the catalog.

    The rendered ink must span most of the glyph box in both axes and carry
    enough total path length that no tile reads as a sliver or a floating
    fragment at rain size.
    """
    xs = [x for stroke in strokes for x, _ in _stroke_points(stroke)]
    ys = [y for stroke in strokes for _, y in _stroke_points(stroke)]
    return (
        (max(xs) - min(xs)) >= 58
        and (max(ys) - min(ys)) >= 88
        and _ink_length(strokes) >= 240
    )


def _build_glyph_specs(
    count: int = GLYPH_COUNT,
    seed: int = DEFAULT_SEED,
) -> tuple[GlyphSpec, ...]:
    functions = [fn for fn, weight in _ARCHETYPES for _ in range(weight)]
    specs: list[GlyphSpec] = []
    seen: set[tuple[tuple, ...]] = set()
    for index in range(count):
        nonce = 0
        while True:
            rng = random.Random(seed ^ (index * 0x9E3779B1) ^ (nonce * 0x85EBCA6B))
            strokes = tuple(rng.choice(functions)(rng))
            if (
                2 <= len(strokes) <= 9
                and _coverage_ok(strokes)
                and strokes not in seen
            ):
                seen.add(strokes)
                break
            nonce += 1
        motion_rng = random.Random(seed + index * 104729)
        specs.append(
            GlyphSpec(
                id=f"glyph-{index:03d}",
                strokes=strokes,
                speed=round(motion_rng.uniform(0.65, 2.35), 3),
                trail_length=motion_rng.randint(8, 26),
            )
        )
    return tuple(specs)


GLYPH_CATALOG_SPECS = _build_glyph_specs(MAX_GLYPH_COUNT)
GLYPH_SPECS = GLYPH_CATALOG_SPECS[:GLYPH_COUNT]
_SPEC_BY_ID = {spec.id: spec for spec in GLYPH_CATALOG_SPECS}
_PROMPT_ID_RE = re.compile(r"CYBER_GLYPH_ID=(glyph-[0-9]{3})(?:\b|$)")


def get_glyph_specs(count: int = GLYPH_COUNT) -> tuple[GlyphSpec, ...]:
    """Return the stable catalog prefix for a supported benchmark width."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if not 1 <= count <= MAX_GLYPH_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_GLYPH_COUNT}")
    return GLYPH_CATALOG_SPECS[:count]


def glyph_prompt(spec: GlyphSpec) -> str:
    """Return a stable prompt that lets a concurrent provider select a glyph."""

    return (
        f"CYBER_GLYPH_ID={spec.id}\n"
        "Draw one fictional calligraphic cyber glyph: bold luminous green "
        "brush strokes (bars, stems, hooks, curves) centered on a plain "
        "solid black background. Flat 2D mark only - no tile, no frame, no "
        "border, no glass, no 3D object, no scene, no background art, and "
        "no real letters, kanji, kana, arabic script, existing symbols, "
        "logos, or branded marks. Invent the mark."
    )


def _select_spec(prompt: str) -> GlyphSpec:
    match = _PROMPT_ID_RE.search(prompt)
    if match:
        return _SPEC_BY_ID[match.group(1)]
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return GLYPH_SPECS[int.from_bytes(digest[:4], "big") % GLYPH_COUNT]


def _pillow():
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageOps
    except ImportError as exc:  # pragma: no cover - optional benchmark dependency
        raise ImportError(
            "Glyph asset rendering requires Pillow; install smythe[benchmarks]"
        ) from exc
    return Image, ImageDraw, ImageFilter, ImageOps


def _draw_capped_segment(draw, a, b, width) -> None:
    draw.line((a, b), fill=255, width=max(1, round(width)))
    radius = width / 2
    for x, y in (a, b):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)


def _glyph_core_mask(spec: GlyphSpec, size: int):
    """Rasterize the stroke program as an anti-aliasable coverage mask."""
    Image, ImageDraw, _, _ = _pillow()
    # Render at 2x and downscale for smooth calligraphic edges.
    scale = 2
    big = size * scale
    mask = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(mask)
    glyph_h = big * 0.86
    glyph_w = glyph_h * (CANVAS_W / CANVAS_H)
    x0 = (big - glyph_w) / 2
    y0 = (big - glyph_h) / 2

    def point(x: float, y: float) -> tuple[float, float]:
        return (x0 + x / CANVAS_W * glyph_w, y0 + y / CANVAS_H * glyph_h)

    stroke_scale = glyph_w / CANVAS_W

    def tapered(width: float, t: float) -> float:
        # A gentle brush taper. The band is deliberately narrow so stroke
        # weight stays consistent across the catalog.
        return width * stroke_scale * (1.03 - 0.20 * t)

    for stroke in spec.strokes:
        if stroke[0] == "l":
            _, x1, y1, x2, y2, width = stroke
            steps = 6
            previous = point(x1, y1)
            for step in range(1, steps + 1):
                t = step / steps
                current = point(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
                _draw_capped_segment(draw, previous, current, tapered(width, t))
                previous = current
        elif stroke[0] == "q":
            _, x1, y1, cx, cy, x2, y2, width = stroke
            steps = 22
            previous = point(x1, y1)
            for step in range(1, steps + 1):
                t = step / steps
                current = point(*_quadratic(x1, y1, cx, cy, x2, y2, t))
                _draw_capped_segment(draw, previous, current, tapered(width, t))
                previous = current
        else:
            _, x, y, radius = stroke
            cx_p, cy_p = point(x, y)
            r = max(1.0, radius * stroke_scale)
            draw.ellipse((cx_p - r, cy_p - r, cx_p + r, cy_p + r), fill=255)
    return mask.resize((size, size), Image.Resampling.LANCZOS)


def render_glyph_tile(spec: GlyphSpec, *, size: int = TILE_SIZE) -> bytes:
    """Render one deterministic transparent PNG tile."""

    if not isinstance(spec, GlyphSpec):
        raise TypeError("spec must be a GlyphSpec")
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 16:
        raise ValueError("size must be at least 16 pixels")
    Image, _, ImageFilter, _ = _pillow()
    core = _glyph_core_mask(spec, size)
    glow = core.filter(ImageFilter.GaussianBlur(max(1.0, size / 22)))
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    aura = Image.new("RGBA", (size, size), (32, 255, 92, 0))
    aura.putalpha(glow.point(lambda value: round(value * 0.42)))
    image.alpha_composite(aura)
    body = Image.new("RGBA", (size, size), (116, 255, 142, 0))
    body.putalpha(core)
    image.alpha_composite(body)
    # A small deterministic highlight gives each tile a luminous leading edge.
    highlight = core.crop((0, 0, size, max(1, size // 2)))
    white = Image.new("RGBA", (size, max(1, size // 2)), (224, 255, 232, 0))
    white.putalpha(highlight.point(lambda value: round(value * 0.34)))
    image.alpha_composite(white, (0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


class ProceduralGlyphProvider(Provider):
    """Offline provider returning one deterministic glyph PNG per prompt."""

    def __init__(self, *, latency_s: float = 0.0, tile_size: int = TILE_SIZE) -> None:
        if isinstance(latency_s, bool) or not isinstance(latency_s, (int, float)):
            raise TypeError("latency_s must be numeric")
        if latency_s < 0:
            raise ValueError("latency_s must be non-negative")
        if isinstance(tile_size, bool) or not isinstance(tile_size, int):
            raise TypeError("tile_size must be an integer")
        if tile_size < 16:
            raise ValueError("tile_size must be at least 16")
        self.latency_s = float(latency_s)
        self.tile_size = tile_size
        self.calls: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        spec = _select_spec(prompt)
        self.calls.append(spec.id)
        return CompletionResult(
            text=json.dumps(
                {
                    "glyph_id": spec.id,
                    "procedural": True,
                    "tile_size": self.tile_size,
                },
                sort_keys=True,
            ),
            artifacts=[
                Artifact(
                    data=render_glyph_tile(spec, size=self.tile_size),
                    mime_type="image/png",
                )
            ],
            cost_usd=0.0,
        )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            temp.write(data)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _encode_image(image, image_format: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **kwargs)
    return buffer.getvalue()


def _image_receipt(path: Path) -> OutputReceipt:
    Image, _, _, _ = _pillow()
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        frames = int(getattr(image, "n_frames", 1))
        image_format = image.format or "UNKNOWN"
        width, height = image.size
    return OutputReceipt(
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        format=image_format,
        width=width,
        height=height,
        frames=frames,
    )


def _text_receipt(path: Path, *, width: int, height: int) -> OutputReceipt:
    data = path.read_bytes()
    return OutputReceipt(
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        format="HTML",
        width=width,
        height=height,
        frames=0,
    )


def normalize_tile(
    source: str | os.PathLike[str] | bytes,
    destination: str | os.PathLike[str],
    *,
    size: int = TILE_SIZE,
) -> OutputReceipt:
    """Normalize arbitrary raster input to a centered transparent square PNG."""

    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 16:
        raise ValueError("size must be at least 16")
    Image, _, _, ImageOps = _pillow()
    stream = io.BytesIO(source) if isinstance(source, bytes) else Path(source)
    with Image.open(stream) as loaded:
        loaded.load()
        tile = loaded.convert("RGBA")
    tile.putalpha(_extract_glyph_mask(tile))
    contained = ImageOps.contain(
        tile,
        (size, size),
        method=Image.Resampling.LANCZOS,
    )
    normalized = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    normalized.alpha_composite(
        contained,
        ((size - contained.width) // 2, (size - contained.height) // 2),
    )
    path = Path(destination)
    _atomic_write_bytes(path, _encode_image(normalized, "PNG", compress_level=6))
    return _image_receipt(path)


def _extract_glyph_mask(rgba):
    """Return a bounded foreground mask, including for opaque model output.

    GPT Image can return an opaque canvas even when transparency is requested.
    In that case, using alpha directly turns the whole tile into a rectangle.
    We instead measure distance from the corner-estimated background and reject
    masks that are empty or cover most of the tile.
    """

    Image, _, ImageFilter, _ = _pillow()
    alpha = rgba.getchannel("A")
    if alpha.getextrema() == (255, 255):
        from PIL import ImageChops

        rgb = rgba.convert("RGB")
        corners = (
            rgb.getpixel((0, 0)),
            rgb.getpixel((rgb.width - 1, 0)),
            rgb.getpixel((0, rgb.height - 1)),
            rgb.getpixel((rgb.width - 1, rgb.height - 1)),
        )
        background_color = tuple(
            round(sum(pixel[channel] for pixel in corners) / len(corners))
            for channel in range(3)
        )
        background = Image.new("RGB", rgb.size, background_color)
        difference = ImageChops.difference(rgb, background)
        red, green, blue = difference.split()
        alpha = ImageChops.lighter(red, ImageChops.lighter(green, blue)).point(
            lambda value: 0 if value < 20 else min(255, (value - 20) * 3)
        )
        alpha = alpha.filter(ImageFilter.GaussianBlur(0.5))

    histogram = alpha.histogram()
    visible_fraction = sum(histogram[17:]) / (alpha.width * alpha.height)
    if visible_fraction < 0.002:
        raise ValueError("tile has no separable glyph foreground")
    if visible_fraction > 0.60:
        raise ValueError(
            "tile foreground covers most of the canvas; refusing an opaque "
            "background or rectangular pseudo-glyph"
        )
    return alpha


def _load_tile_masks(tile_paths: Sequence[str | os.PathLike[str]], size: int):
    get_glyph_specs(len(tile_paths))
    Image, _, _, _ = _pillow()
    masks = []
    for path in tile_paths:
        with Image.open(path) as tile:
            tile.load()
            rgba = tile.convert("RGBA")
        mask = _extract_glyph_mask(rgba).resize(
            (size, size), Image.Resampling.LANCZOS
        )
        if mask.getbbox() is None:
            raise ValueError(f"tile has no visible pixels: {path}")
        masks.append(mask)
    return masks


def _render_rain_layer(
    *,
    overlay,
    heads,
    masks,
    specs: Sequence[GlyphSpec],
    width: int,
    height: int,
    cell: int,
    glyph_size: int,
    seed: int,
    frame_index: int,
    level: float,
    spacing: float,
):
    """Draw one depth layer of columns onto the shared overlays."""
    Image, _, ImageFilter, _ = _pillow()
    rng = random.Random(seed)
    step = max(6, round(cell * spacing))
    column_count = math.ceil(width / step) + 1
    glyph_count = len(masks)
    for column in range(column_count):
        spec = specs[(column * 31) % glyph_count]
        trail = max(
            7,
            round(spec.trail_length * 1.6 * height / PREVIEW_SIZE[1]),
        )
        speed = max(1, round(spec.speed * cell * 0.68))
        cycle = height + (trail + 2) * cell
        start = rng.randrange(cycle)
        # Keep some part of every column visible. As a head moves below the
        # viewport, its trailing glyphs continue to drain before it wraps.
        head_y = (start + frame_index * speed) % cycle
        x = column * step + rng.randint(-2, 2)
        base_glyph = rng.randrange(glyph_count)
        for tail_index in range(trail, -1, -1):
            y = head_y - tail_index * cell
            if y < -glyph_size or y >= height:
                continue
            mask = masks[
                (base_glyph + tail_index * 7 + frame_index // 2) % glyph_count
            ]
            if tail_index == 0:
                glow = mask.filter(ImageFilter.GaussianBlur(max(1.0, glyph_size / 5)))
                halo = Image.new("RGBA", (glyph_size, glyph_size), (64, 255, 118, 0))
                halo.putalpha(glow.point(lambda value: round(value * 0.72 * level)))
                heads.alpha_composite(halo, (x, y))
                head = Image.new("RGBA", (glyph_size, glyph_size), (225, 255, 232, 0))
                head.putalpha(mask.point(lambda value: round(value * level)))
                heads.alpha_composite(head, (x, y))
            else:
                proximity = 1 - tail_index / max(1, trail)
                # High floors keep every visible trail glyph bright; nothing
                # in the field reads as a dark glyph.
                green = round((116 + 134 * proximity) * level)
                alpha = round((96 + 159 * proximity * proximity) * level)
                body = Image.new("RGBA", (glyph_size, glyph_size), (38, green, 84, 0))
                body.putalpha(mask.point(lambda value, a=alpha: value * a // 255))
                overlay.alpha_composite(body, (x, y))


def _render_rain_frame(
    tile_paths: Sequence[str | os.PathLike[str]],
    *,
    width: int,
    height: int,
    seed: int,
    frame_index: int,
    masks_by_layer=None,
):
    Image, ImageDraw, ImageFilter, _ = _pillow()
    image = Image.new("RGB", (width, height), (0, 2, 1))
    # Two depth layers: a dimmer, smaller, tighter far field behind a bright
    # near field. Overlapping spacing (< 1.0) packs columns like heavy rain.
    specs = get_glyph_specs(len(tile_paths))
    layer_params = _rain_layer_params(width)
    if masks_by_layer is None:
        masks_by_layer = [
            _load_tile_masks(tile_paths, params["glyph_size"])
            for params in layer_params
        ]
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    heads = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for layer_index, params in enumerate(layer_params):
        _render_rain_layer(
            overlay=overlay,
            heads=heads,
            masks=masks_by_layer[layer_index],
            specs=specs,
            width=width,
            height=height,
            cell=params["cell"],
            glyph_size=params["glyph_size"],
            seed=seed + layer_index * 7919,
            frame_index=frame_index,
            level=params["level"],
            spacing=params["spacing"],
        )
    reference_glyph = layer_params[-1]["glyph_size"]
    glow = overlay.filter(ImageFilter.GaussianBlur(max(1.0, reference_glyph / 8)))
    glow.putalpha(glow.getchannel("A").point(lambda value: round(value * 0.62)))
    image = Image.alpha_composite(image.convert("RGBA"), glow)
    image = Image.alpha_composite(image, overlay)
    image = Image.alpha_composite(image, heads)
    # Subtle dark scan lines add display texture without importing pixels.
    draw = ImageDraw.Draw(image)
    for y in range(2, height, 4):
        draw.line((0, y, width, y), fill=(0, 7, 3, 42), width=1)
    return image.convert("RGB")


def _rain_layer_params(width: int) -> list[dict]:
    # Near-uniform glyph sizes: depth reads through brightness, not scale.
    # Cell sizes track the approved web screensaver look (~24-28 px at 1080p).
    near_cell = max(12, round(width / 72))
    far_cell = max(10, round(width / 80))
    return [
        {
            "cell": far_cell,
            "glyph_size": max(9, far_cell - 1),
            "level": 0.78,
            "spacing": 0.60,
        },
        {
            "cell": near_cell,
            "glyph_size": max(10, near_cell - 2),
            "level": 1.0,
            "spacing": 0.66,
        },
    ]


def assemble_preview(
    tile_paths: Sequence[str | os.PathLike[str]],
    destination: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
) -> OutputReceipt:
    """Assemble the full-resolution 1920x1080 still preview."""

    frame = _render_rain_frame(
        tile_paths,
        width=PREVIEW_SIZE[0],
        height=PREVIEW_SIZE[1],
        seed=seed,
        frame_index=29,
    )
    path = Path(destination)
    _atomic_write_bytes(path, _encode_image(frame, "PNG", compress_level=6))
    return _image_receipt(path)


def assemble_animation(
    tile_paths: Sequence[str | os.PathLike[str]],
    destination: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
    frames: int = GIF_FRAMES,
) -> OutputReceipt:
    """Assemble a compact looping GIF proof of varied column motion."""

    if isinstance(frames, bool) or not isinstance(frames, int):
        raise TypeError("frames must be an integer")
    if frames < 2:
        raise ValueError("frames must be at least two")
    masks_by_layer = [
        _load_tile_masks(tile_paths, params["glyph_size"])
        for params in _rain_layer_params(GIF_SIZE[0])
    ]
    rendered = [
        _render_rain_frame(
            tile_paths,
            width=GIF_SIZE[0],
            height=GIF_SIZE[1],
            seed=seed,
            frame_index=index * 3,
            masks_by_layer=masks_by_layer,
        )
        for index in range(frames)
    ]
    data = _encode_image(
        rendered[0],
        "GIF",
        save_all=True,
        append_images=rendered[1:],
        duration=85,
        loop=0,
        optimize=True,
        disposal=2,
    )
    path = Path(destination)
    _atomic_write_bytes(path, data)
    return _image_receipt(path)


def assemble_atlas(
    tile_paths: Sequence[str | os.PathLike[str]],
    destination: str | os.PathLike[str],
) -> OutputReceipt:
    """Assemble a 16-column contact-sheet atlas of normalized tiles."""

    get_glyph_specs(len(tile_paths))
    Image, ImageDraw, _, _ = _pillow()
    columns = ATLAS_GRID[0]
    rows = math.ceil(len(tile_paths) / columns)
    atlas = Image.new(
        "RGB",
        (columns * TILE_SIZE, rows * TILE_SIZE),
        (0, 3, 1),
    )
    draw = ImageDraw.Draw(atlas)
    for index, path_value in enumerate(tile_paths):
        with Image.open(path_value) as source:
            tile = source.convert("RGBA")
        x = (index % columns) * TILE_SIZE
        y = (index // columns) * TILE_SIZE
        atlas.paste(tile, (x, y), tile)
        draw.rectangle((x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1), outline=(7, 42, 18))
    path = Path(destination)
    _atomic_write_bytes(path, _encode_image(atlas, "PNG", compress_level=6))
    return _image_receipt(path)


def _html_document(seed: int, specs: Sequence[GlyphSpec]) -> str:
    strokes = [[list(stroke) for stroke in spec.strokes] for spec in specs]
    speeds = [spec.speed for spec in specs]
    trails = [spec.trail_length for spec in specs]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fictional Cyber Glyph Rain</title>
<style>
html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#000}}
canvas{{display:block;width:100vw;height:100vh;background:#000}}
</style>
</head>
<body>
<canvas id="rain" width="1920" height="1080" aria-label="Animated fictional green cyber glyph rain"></canvas>
<script>
const strokes={json.dumps(strokes, separators=(",", ":"))};
const speeds={json.dumps(speeds, separators=(",", ":"))};
const trails={json.dumps(trails, separators=(",", ":"))};
const COUNT=strokes.length;
let state={seed & 0xFFFFFFFF};
function rand(){{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296}}
const canvas=document.getElementById('rain'),ctx=canvas.getContext('2d');
function glyphPath(g,x,y,size){{
  const gh=size,gw=size*100/140,sx=gw/100,sy=gh/140;
  ctx.lineCap='round';ctx.lineJoin='round';
  for(const s of strokes[g]){{
    if(s[0]==='d'){{ctx.beginPath();ctx.arc(x+s[1]*sx,y+s[2]*sy,Math.max(.9,s[3]*sx),0,7);ctx.fill();continue}}
    ctx.beginPath();ctx.lineWidth=Math.max(1.3,s[s.length-1]*sx);
    ctx.moveTo(x+s[1]*sx,y+s[2]*sy);
    if(s[0]==='l')ctx.lineTo(x+s[3]*sx,y+s[4]*sy);
    else ctx.quadraticCurveTo(x+s[3]*sx,y+s[4]*sy,x+s[5]*sx,y+s[6]*sy);
    ctx.stroke();
  }}
}}
const layers=[
  {{cell:20,spacing:.6,speed:.62,level:.78}},
  {{cell:22,spacing:.66,speed:1,level:1}},
];
for(const L of layers){{
  L.columns=[];const step=Math.max(6,L.cell*L.spacing);
  for(let x=0;x<canvas.width+step;x+=step){{
    const g=Math.floor(rand()*COUNT);
    L.columns.push({{x:x+(rand()*4-2),y:rand()*canvas.height*2-canvas.height,
      glyph:g,speed:speeds[g]*66*L.speed,length:trails[g],phase:Math.floor(rand()*COUNT)}});
  }}
}}
let previous=performance.now();
function draw(now){{
  const dt=Math.min(.05,(now-previous)/1000);previous=now;
  ctx.globalAlpha=1;ctx.shadowBlur=0;ctx.fillStyle='rgba(0,2,1,.28)';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  for(const L of layers)for(const column of L.columns){{
    column.y+=column.speed*dt;
    if(column.y-column.length*L.cell>canvas.height){{column.y=-L.cell;column.glyph=(column.glyph+17)%COUNT}}
    for(let tail=column.length;tail>=0;tail--){{
      const y=column.y-tail*L.cell;if(y<-L.cell||y>canvas.height)continue;
      const index=(column.glyph+column.phase+tail*7)%COUNT,near=1-tail/column.length;
      if(tail===0){{ctx.shadowColor='#63ff8d';ctx.shadowBlur=12*L.level;
        ctx.strokeStyle=ctx.fillStyle='#e3ffe9';ctx.globalAlpha=L.level;
        glyphPath(index,column.x,y,L.cell-2)}}
      else{{ctx.shadowBlur=0;
        const c=`rgb(34,${{Math.round((116+120*near)*L.level)}},80)`;
        ctx.strokeStyle=ctx.fillStyle=c;ctx.globalAlpha=(.34+.58*near*near)*L.level;
        glyphPath(index,column.x,y,L.cell-2)}}
    }}
  }}
  ctx.shadowBlur=0;ctx.globalAlpha=1;requestAnimationFrame(draw);
}}
requestAnimationFrame(draw);
</script>
</body>
</html>
"""


def assemble_html(
    destination: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
    glyph_count: int = GLYPH_COUNT,
) -> OutputReceipt:
    """Write a self-contained animated 1920x1080 HTML canvas screensaver."""

    path = Path(destination)
    specs = get_glyph_specs(glyph_count)
    _atomic_write_bytes(path, _html_document(seed, specs).encode("utf-8"))
    return _text_receipt(path, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])


def build_glyph_screensaver_assets(
    output_dir: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
) -> GlyphSuiteReceipt:
    """Build all 192 tiles and the four deterministic default outputs."""

    root = Path(output_dir)
    tile_dir = root / "tiles"
    tile_receipts = tuple(
        normalize_tile(
            render_glyph_tile(spec),
            tile_dir / f"{spec.id}.png",
        )
        for spec in GLYPH_SPECS
    )
    unique = len({receipt.sha256 for receipt in tile_receipts})
    if unique != GLYPH_COUNT:
        raise RuntimeError(f"expected {GLYPH_COUNT} unique tiles, got {unique}")
    tile_paths = [receipt.path for receipt in tile_receipts]
    preview = assemble_preview(tile_paths, root / "glyph-rain-preview.png", seed=seed)
    animation = assemble_animation(tile_paths, root / "glyph-rain-loop.gif", seed=seed)
    atlas = assemble_atlas(tile_paths, root / "glyph-atlas.png")
    html = assemble_html(root / "glyph-rain.html", seed=seed)
    return GlyphSuiteReceipt(
        tiles=tile_receipts,
        preview=preview,
        animation=animation,
        atlas=atlas,
        html=html,
        unique_tile_hashes=unique,
    )


__all__ = [
    "ATLAS_GRID",
    "ATLAS_SIZE",
    "CANVAS_H",
    "CANVAS_W",
    "DEFAULT_SEED",
    "GIF_FRAMES",
    "GIF_SIZE",
    "GLYPH_CATALOG_SPECS",
    "GLYPH_COUNT",
    "GLYPH_SPECS",
    "MAX_GLYPH_COUNT",
    "PREVIEW_SIZE",
    "TILE_SIZE",
    "GlyphSpec",
    "GlyphSuiteReceipt",
    "OutputReceipt",
    "ProceduralGlyphProvider",
    "assemble_animation",
    "assemble_atlas",
    "assemble_html",
    "assemble_preview",
    "build_glyph_screensaver_assets",
    "get_glyph_specs",
    "glyph_prompt",
    "normalize_tile",
    "render_glyph_tile",
]
