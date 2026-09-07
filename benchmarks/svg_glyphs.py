"""Original filled SVG glyphs governed by the measured Glyph Rain style brief.

Geometry is composed from independently authored writing gestures. No reference
image, font, traced outline, or upstream implementation is read by this module.
Each index has a stable family, profile, seed, and structural recipe. Curves are
flattened to fine polygonal contours before union so every consumer uses the
same explicit nonzero fill geometry, including counters and detached marks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import LineString, Point, Polygon, box, orient_polygons, set_precision, union_all
from shapely.affinity import scale, translate
from shapely.ops import nearest_points

VERSION = "glyph-svg-v1"
GLYPH_COUNT = 192
FAMILY_QUOTAS = {
    "numeral_operator": 45,
    "bar_hook": 50,
    "stacked_marks": 3,
    "diagonal_lozenge": 17,
    "roofed_curves": 14,
    "rounded_loop_interlock": 39,
    "mixed": 24,
}
BASE_SEED = 0x534D59544845
# Fixed authoring parameters chosen during the unlit optical calibration. These
# are contour-design choices, never cached SVGs or per-run measurement feedback.
AUTHORED_PARAMETERS = {1: {'apertures': [(70.703, 30.859, 6.0)], 'weight': 1.16},
 2: {'gap': 6.5},
 3: {'gap': 11, 'weight': 1.08},
 4: {'weight': 0.94},
 6: {'counter_scale': 0.65},
 7: {'weight': 1.08},
 11: {'counter_scale': 0.65},
 13: {'weight': 0.94},
 14: {'weight': 1.16},
 18: {'counter_scale': 0.65, 'structure': 'offset-double-counter'},
 19: {'weight': 1.16},
 24: {'weight': 1.08},
 25: {'weight': 1.24},
 26: {'seed_offset': 2},
 27: {'counter_scale': 1.5},
 28: {'seed_offset': 1},
 33: {'seed_offset': 8},
 35: {'counter_scale': 1.8},
 41: {'seed_offset': 3},
 43: {'counter_scale': 1.25},
 44: {'structure': 'split-roof-sweep'},
 45: {'gap': 10, 'weight': 1.16},
 48: {'structure': 'diagonal-bracket-fork', 'weight': 1.08},
 51: {'seed_offset': 8},
 52: {'weight': 0.94},
 55: {'structure': 'crossing-fork'},
 56: {'seed_offset': 3},
 57: {'weight': 0.94},
 58: {'seed_offset': 3},
 59: {'counter_scale': 0.65, 'structure': 'counter-with-left-prong'},
 64: {'seed_offset': 8},
 65: {'structure': 'three-offset-bars'},
 66: {'gap': 6.5},
 68: {'structure': 'double-stem-sweep', 'weight': 0.94},
 70: {'seed_offset': 2},
 75: {'structure': 'upper-arc-lower-fork'},
 76: {'weight': 1.24},
 79: {'seed_offset': 8, 'weight': 1.24},
 83: {'counter_scale': 0.65, 'structure': 'diagonal-ascender-loop'},
 85: {'weight': 1.08},
 87: {'apertures': [(47, 58, 7)]},
 88: {'weight': 1.24},
 89: {'seed_offset': 1, 'weight': 1.16},
 90: {'counter_scale': 0.65, 'structure': 'lower-counter-ascender'},
 91: {'structure': 'crook-with-floating-square'},
 94: {'gap': 6.5},
 95: {'structure': 'open-triple-prong'},
 96: {'apertures': [(54.297, 56.641, 6.0)]},
 97: {'weight': 1.16},
 99: {'apertures': [(33.984, 65.234, 6.0)], 'structure': 'roof-with-left-descender'},
 100: {'weight': 1.08},
 103: {'gap': 10},
 108: {'counter_scale': 0.65, 'structure': 'upper-counter-descender', 'weight': 0.87},
 112: {'structure': 'roof-with-two-shoulders'},
 115: {'opening': 1.8, 'structure': 'counter-and-lower-crook'},
 116: {'seed_offset': 3, 'weight': 1.08},
 117: {'structure': 'side-open-loop-with-diagonal'},
 119: {'seed_offset': 8},
 121: {'apertures': [(69.141, 69.141, 6.0)]},
 123: {'structure': 'open-crossing-crook'},
 124: {'weight': 1.24},
 126: {'structure': 'roof-with-descending-loop'},
 127: {'weight': 1.16},
 128: {'gap': 8, 'seed_offset': 1},
 129: {'apertures': [(66.016, 63.672, 8.5)]},
 130: {'weight': 1.16},
 131: {'gap': 9.5},
 133: {'counter_scale': 0.65, 'structure': 'interlocked-counters'},
 134: {'structure': 'sloped-ladder-hook'},
 135: {'gap': 6.5, 'seed_offset': 8},
 138: {'weight': 1.24},
 139: {'structure': 'open-bowl-descender'},
 140: {'weight': 1.16},
 144: {'weight': 0.87},
 146: {'structure': 'upright-loop-side-crook'},
 147: {'structure': 'sweep-with-lozenge'},
 148: {'gap': 11, 'weight': 1.24},
 150: {'gap': 11, 'seed_offset': 1, 'weight': 1.16},
 151: {'apertures': [(67.578, 66.016, 6.0)]},
 152: {'gap': 10, 'mark_size': 10.5, 'seed_offset': 5},
 155: {'structure': 'crooked-upper-counter'},
 156: {'apertures': [(67.578, 66.797, 6.0)], 'seed_offset': 3},
 157: {'weight': 1.08},
 159: {'weight': 1.16},
 160: {'counter_scale': 0.65, 'structure': 'rounded-counter-with-fork'},
 161: {'seed_offset': 1},
 162: {'apertures': [(47.266, 41.797, 6.0), (48.047, 55.078, 6.0)],
       'structure': 'diagonal-switchback'},
 164: {'weight': 1.16},
 166: {'apertures': [(67, 66, 8.5)]},
 172: {'seed_offset': 8},
 173: {'apertures': [(51.953, 27.734, 8.5)], 'structure': 'open-diamond-hook'},
 174: {'structure': 'split-numeral-hook'},
 176: {'structure': 'double-elbow-operator'},
 178: {'counter_scale': 1.25, 'gap': 6.5},
 179: {'structure': 'split-crossbar-crook'},
 180: {'apertures': [(51.172, 71.484, 6.0)], 'structure': 'crossed-open-bowls'},
 181: {'apertures': [(41.797, 62.891, 6.0)],
       'seed_offset': 3,
       'structure': 'roof-and-reversed-crook'},
 182: {'counter_scale': 0.65, 'gap': 10, 'structure': 'tilted-counter-descender', 'weight': 0.87},
 183: {'seed_offset': 3},
 184: {'structure': 'pierced-chevron', 'weight': 1.08},
 186: {'structure': 'open-rounded-ladder'},
 188: {'structure': 'stepped-stem-with-hook'},
 189: {'gap': 11, 'structure': 'lower-loop-overhanging-arc'},
 190: {'weight': 1.08}}
SVG_NS = "http://www.w3.org/2000/svg"
_TOKEN = re.compile(r"[MLZ]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _schedule() -> tuple[tuple[str, int], ...]:
    """Weighted distribution makes the first 24 a cross-family calibration set."""
    used = dict.fromkeys(FAMILY_QUOTAS, 0)
    result = []
    for _ in range(GLYPH_COUNT):
        family = min(
            (name for name, quota in FAMILY_QUOTAS.items() if used[name] < quota),
            key=lambda name: (used[name] / FAMILY_QUOTAS[name], list(used).index(name)),
        )
        result.append((family, used[family]))
        used[family] += 1
    return tuple(result)


SCHEDULE = _schedule()


def _stroke(points, width, *, rounded=False, angular=False):
    return LineString(points).buffer(
        width / 2, quad_segs=16,
        cap_style="round" if rounded else "flat",
        join_style="mitre" if angular else "round",
    )


def _curve(points, width, *, rounded=False):
    a, b, c, d = points
    samples = []
    for i in range(49):
        t = i / 48
        u = 1 - t
        samples.append(tuple(u**3*a[k] + 3*u*u*t*b[k] + 3*u*t*t*c[k] + t**3*d[k]
                             for k in (0, 1)))
    # A writing gesture changes pressure through the bend: a broad shoulder
    # narrows toward its terminals. This is authored contour mass, not glow.
    left, right = [], []
    for i, point in enumerate(samples):
        before, after = samples[max(0, i-1)], samples[min(48, i+1)]
        dx, dy = after[0]-before[0], after[1]-before[1]
        length = math.hypot(dx, dy)
        radius = width*(.62+.68*math.sin(math.pi*i/48)**.8)/2
        nx, ny = -dy/length*radius, dx/length*radius
        left.append((point[0]+nx, point[1]+ny))
        right.append((point[0]-nx, point[1]-ny))
    shape = Polygon(left+right[::-1]).buffer(0)
    if rounded:
        shape = union_all([shape, Point(samples[0]).buffer(width*.31, quad_segs=16),
                           Point(samples[-1]).buffer(width*.31, quad_segs=16)])
    return shape


def _round_box(left, top, right, bottom, radius):
    radius = min(radius, (right-left)/2, (bottom-top)/2)
    return box(left+radius, top+radius, right-radius, bottom-radius).buffer(
        radius, quad_segs=16, join_style="round"
    )


def _ring(left, top, right, bottom, weight, radius):
    return _round_box(left, top, right, bottom, radius).difference(
        _round_box(left+weight, top+weight, right-weight, bottom-weight,
                   max(1, radius-weight*.65))
    )


def _individual_gesture(recipe, w):
    """Additional authored structures used where the base grammar repeated itself."""
    if recipe == "lower-counter-ascender":
        return union_all([_ring(35, 43, 78, 86, w*.8, 17),
                          _stroke([(30, 17), (30, 76)], w),
                          _stroke([(18, 24), (65, 24)], w*.85)])
    if recipe == "upper-counter-descender":
        return union_all([_ring(20, 14, 77, 56, w*.8, 16),
                          _stroke([(70, 45), (70, 80), (31, 80)], w, angular=True)])
    if recipe == "diagonal-ascender-loop":
        return union_all([_ring(17, 32, 69, 83, w*.8, 18),
                          _stroke([(53, 47), (77, 19), (50, 19)], w*.9)])
    if recipe == "open-crossing-crook":
        return union_all([_stroke([(23, 17), (72, 17), (57, 48), (73, 84)], w),
                          _curve([(76, 40), (20, 24), (13, 75), (48, 81)], w),
                          _stroke([(25, 62), (60, 62)], w*.85)])
    if recipe == "double-stem-sweep":
        return union_all([_stroke([(20, 18), (20, 80), (50, 80)], w),
                          _curve([(20, 27), (90, 1), (85, 61), (49, 48)], w),
                          _stroke([(65, 46), (65, 81), (83, 81)], w*.88)])
    if recipe == "sweep-with-lozenge":
        return union_all([_curve([(24, 20), (80, 0), (83, 53), (42, 53)], w),
                          _stroke([(42, 53), (74, 83)], w),
                          Polygon([(18, 75), (28, 66), (37, 75), (28, 87)])])
    if recipe == "counter-and-lower-crook":
        return union_all([_ring(25, 14, 72, 53, w*.8, 14),
                          _stroke([(31, 43), (21, 79)], w),
                          _curve([(21, 79), (84, 100), (88, 45), (64, 58)], w)])
    if recipe == "roof-with-descending-loop":
        return union_all([_stroke([(17, 19), (83, 19)], w),
                          _stroke([(37, 19), (37, 48)], w*.85),
                          _curve([(73, 19), (90, 88), (18, 91), (29, 51)], w),
                          _stroke([(29, 51), (59, 51)], w*.85)])
    if recipe == "roof-with-two-shoulders":
        return union_all([_stroke([(17, 20), (82, 20)], w),
                          _curve([(34, 20), (16, 62), (46, 43), (47, 82)], w),
                          _curve([(66, 20), (89, 56), (53, 57), (78, 82)], w*.9)])
    if recipe == "roof-and-reversed-crook":
        return union_all([_stroke([(18, 19), (83, 19), (83, 47)], w),
                          _curve([(35, 20), (7, 70), (73, 96), (70, 57)], w),
                          _stroke([(70, 57), (44, 57)], w*.8)])
    if recipe == "pierced-chevron":
        return union_all([_stroke([(25, 18), (73, 51), (25, 83)], w, angular=True),
                          _stroke([(28, 44), (57, 44)], w*.8),
                          _stroke([(55, 51), (55, 84)], w*.85)])
    if recipe == "open-diamond-hook":
        return union_all([_stroke([(51, 16), (20, 49), (50, 82), (80, 49)], w,
                                 angular=True),
                          _stroke([(80, 49), (56, 49), (56, 28)], w*.85)])
    if recipe == "split-crossbar-crook":
        return union_all([_stroke([(19, 20), (19, 76), (50, 76)], w),
                          _stroke([(18, 43), (79, 43), (79, 21)], w),
                          _curve([(77, 44), (91, 84), (56, 89), (52, 75)], w)])
    if recipe == "stepped-stem-with-hook":
        return union_all([_stroke([(22, 18), (22, 49), (51, 49), (51, 81)], w,
                                 angular=True),
                          _curve([(22, 18), (83, 5), (92, 66), (68, 62)], w),
                          _stroke([(26, 80), (79, 80)], w*.85)])
    if recipe == "open-triple-prong":
        return union_all([_stroke([(20, 79), (34, 18)], w),
                          _stroke([(48, 77), (48, 39)], w*.85),
                          _stroke([(76, 79), (65, 18)], w),
                          _stroke([(24, 63), (74, 63)], w*.85)])
    if recipe == "sloped-ladder-hook":
        return union_all([_stroke([(20, 24), (72, 24), (78, 81)], w),
                          _stroke([(23, 52), (76, 45)], w*.85),
                          _curve([(23, 21), (38, 36), (12, 67), (37, 83)], w)])
    if recipe == "crossing-fork":
        return union_all([_stroke([(20, 24), (73, 78)], w),
                          _stroke([(76, 20), (24, 79)], w),
                          _stroke([(49, 22), (49, 52)], w*.85),
                          _stroke([(22, 78), (75, 78)], w*.8)])
    if recipe == "counter-with-left-prong":
        return union_all([_ring(39, 25, 80, 81, w*.75, 18),
                          _stroke([(22, 16), (22, 59), (48, 59)], w),
                          _stroke([(22, 32), (53, 32)], w*.8)])
    if recipe == "upper-arc-lower-fork":
        return union_all([_curve([(21, 45), (11, 2), (84, 0), (79, 42)], w),
                          _stroke([(47, 38), (47, 70), (23, 83)], w),
                          _stroke([(47, 70), (79, 81)], w*.9)])
    if recipe == "interlocked-counters":
        return union_all([_ring(19, 15, 60, 58, w*.75, 16),
                          _ring(40, 44, 81, 87, w*.75, 16),
                          _stroke([(30, 76), (70, 24)], w*.55)])
    if recipe == "open-bowl-descender":
        return union_all([_curve([(23, 18), (5, 75), (91, 89), (78, 23)], w),
                          _stroke([(51, 46), (51, 85)], w*.85),
                          _stroke([(38, 20), (75, 20)], w*.8)])
    if recipe == "crooked-upper-counter":
        return union_all([_ring(24, 15, 74, 55, w*.8, 17),
                          _stroke([(30, 47), (22, 81), (78, 72)], w),
                          _stroke([(70, 50), (80, 83)], w*.75)])
    if recipe == "diagonal-switchback":
        return union_all([_stroke([(22, 19), (73, 46), (27, 81)], w, angular=True),
                          _stroke([(22, 19), (22, 48), (48, 48)], w*.85),
                          _stroke([(52, 66), (78, 80)], w*.8)])
    if recipe == "crossed-open-bowls":
        return union_all([_curve([(69, 19), (10, -2), (2, 62), (64, 52)], w),
                          _curve([(35, 48), (94, 38), (92, 100), (31, 80)], w),
                          _stroke([(51, 25), (51, 75)], w*.8)])
    if recipe == "roof-with-left-descender":
        return union_all([_stroke([(18, 20), (81, 20)], w),
                          _stroke([(26, 20), (26, 83)], w*.9),
                          _curve([(67, 20), (89, 83), (24, 81), (48, 48)], w)])
    if recipe == "split-roof-sweep":
        return union_all([_stroke([(18, 19), (46, 19)], w),
                          _stroke([(63, 19), (82, 19), (82, 53)], w),
                          _curve([(45, 38), (9, 42), (21, 89), (72, 79)], w),
                          _stroke([(46, 38), (65, 61)], w*.8)])
    if recipe == "upright-loop-side-crook":
        return union_all([_ring(18, 18, 59, 82, w*.8, 19),
                          _curve([(57, 40), (91, 32), (87, 81), (68, 83)], w),
                          _stroke([(23, 51), (53, 51)], w*.7)])
    if recipe == "lower-loop-overhanging-arc":
        return union_all([_ring(33, 44, 80, 83, w*.75, 16),
                          _curve([(19, 43), (6, 5), (75, 0), (73, 40)], w),
                          _stroke([(20, 44), (20, 81)], w*.85)])
    if recipe == "open-rounded-ladder":
        return union_all([_curve([(23, 18), (11, 75), (56, 100), (72, 69)], w),
                          _stroke([(72, 18), (72, 70)], w),
                          _stroke([(25, 43), (70, 43)], w*.8),
                          _stroke([(36, 64), (70, 64)], w*.8)])
    if recipe == "split-numeral-hook":
        return union_all([_curve([(25, 20), (79, 1), (91, 51), (54, 51)], w),
                          _stroke([(53, 52), (23, 78), (76, 78)], w),
                          _stroke([(19, 45), (34, 45)], w*.85)])
    if recipe == "tilted-counter-descender":
        return union_all([_ring(24, 17, 73, 59, w*.8, 18),
                          _stroke([(33, 55), (54, 83), (81, 65)], w),
                          _stroke([(32, 81), (16, 66)], w*.85)])
    if recipe == "double-elbow-operator":
        return union_all([_stroke([(20, 20), (77, 20), (77, 48), (47, 48)], w),
                          _stroke([(22, 42), (22, 81), (74, 81)], w),
                          _stroke([(50, 49), (50, 67)], w*.8)])
    if recipe == "offset-double-counter":
        return union_all([_ring(19, 14, 72, 53, w*.8, 17),
                          _ring(29, 47, 83, 86, w*.8, 17),
                          _stroke([(18, 65), (39, 65)], w*.8)])
    if recipe == "side-open-loop-with-diagonal":
        bowl = _ring(18, 16, 81, 85, w, 24).difference(box(15, 36, 43, 57))
        return union_all([bowl, _stroke([(30, 68), (73, 34)], w*.85),
                          _stroke([(20, 21), (40, 21)], w*.75)])
    if recipe == "diagonal-bracket-fork":
        return union_all([_stroke([(23, 19), (70, 45), (26, 83)], w, angular=True),
                          _stroke([(70, 45), (78, 79)], w*.8),
                          _stroke([(24, 19), (24, 48)], w*.8)])
    if recipe == "rounded-counter-with-fork":
        return union_all([_ring(27, 15, 74, 69, w*.8, 21),
                          _stroke([(50, 61), (31, 86)], w*.85),
                          _stroke([(50, 61), (77, 84)], w*.85),
                          _stroke([(18, 40), (35, 40)], w*.75)])
    if recipe == "three-offset-bars":
        return union_all([box(18, 17, 55, 17+w), box(31, 45, 81, 45+w),
                          box(20, 73, 67, 73+w)])
    if recipe == "crook-with-floating-square":
        return union_all([_stroke([(20, 17), (20, 80), (70, 80)], w),
                          _stroke([(20, 45), (55, 45)], w*.85),
                          _stroke([(43, 21), (84, 21)], w), box(73, 45, 87, 59)])
    raise ValueError(f"unknown authored structure {recipe}")


def _gesture(family: str, ordinal: int, rng: random.Random, profile: str, parameters=None):
    """Construct a writing gesture, not a transformed copy of another glyph."""
    parameters = parameters or {}
    w = rng.uniform(13.4, 15.7) if profile == "classic" else rng.uniform(11.8, 14.0)
    w *= parameters.get("weight", 1)
    left, top, right, bottom = 18.0, 14.0, 82.0, 86.0
    mid = rng.uniform(42, 56)
    arm = rng.uniform(36, 65)
    variant = ordinal % 10
    pieces = []
    if parameters.get("structure"):
        pieces = [_individual_gesture(parameters["structure"], w)]
    elif family == "bar_hook":
        if variant == 0:
            pieces = [_stroke([(left, top+w/2), (right, top+w/2)], w),
                      _curve([(arm, top), (arm+7, bottom-9), (left+5, bottom-5),
                              (left+3, bottom)], w)]
            pieces.append(_stroke([(arm-5, mid), (right-9, mid-5)], w*.82))
        elif variant == 1:
            pieces = [_stroke([(left+w/2, top), (left+w/2, bottom-w/2),
                               (right, bottom-w/2)], w, angular=True),
                      _stroke([(left, mid), (right-9, mid)], w*.91)]
        elif variant == 2:
            pieces = [_stroke([(left, top+w/2), (arm, top+w/2),
                               (arm, bottom)], w, angular=True),
                      _curve([(arm, mid), (right+2, mid-3), (right, bottom-5),
                              (right-10, bottom-5)], w*.87)]
        elif variant == 3:
            pieces = [_curve([(left, top+5), (left+3, bottom+2), (right-8, bottom),
                              (right-5, mid)], w),
                      _stroke([(left+2, top+7), (right, top+7)], w),
                      _stroke([(arm, top+7), (arm, mid-9)], w*.9)]
        elif variant == 4:
            pieces = [_stroke([(left, bottom-2), (mid, top), (right, bottom-2)], w,
                               angular=True),
                      _stroke([(mid-9, mid+7), (right-1, mid+7)], w*.85),
                      _stroke([(right-3, bottom-6), (right-20, bottom-6)], w*.85)]
        elif variant == 5:
            pieces = [_stroke([(right-w/2, top), (right-w/2, bottom-w/2),
                               (left, bottom-w/2)], w, angular=True),
                      _stroke([(left, top+8), (right-1, top+8)], w),
                      _stroke([(left+6, mid), (arm, mid)], w*.85)]
        elif variant == 6:
            pieces = [_curve([(right, top+6), (left-3, top-9), (left-3, bottom+5),
                              (right-6, bottom-5)], w),
                      _stroke([(mid-4, top+3), (mid-4, mid+5)], w*.95),
                      _stroke([(mid-4, mid+5), (right-7, mid+5)], w*.85)]
        elif variant == 7:
            pieces = [_stroke([(left, top+7), (right, top+7)], w),
                      _stroke([(arm, top), (arm-12, bottom)], w),
                      _curve([(left+8, mid), (mid, mid-6), (right-6, bottom-4),
                              (right, bottom-5)], w*.9)]
        elif variant == 8:
            pieces = [_stroke([(left+7, top), (left+7, bottom)], w),
                      _curve([(left, mid), (right+5, mid-7), (right, top+8),
                              (right-15, top+7)], w),
                      _stroke([(left+4, bottom-7), (arm+8, bottom-7)], w*.9)]
        else:
            pieces = [_stroke([(left, top+6), (right, top+6)], w),
                      _stroke([(left+12, top), (mid+4, bottom-4)], w),
                      _stroke([(left+3, mid+3), (right-6, mid-4),
                               (right-6, bottom)], w*.9, angular=True)]
    elif family == "numeral_operator":
        kind = ordinal % 15
        if kind < 3:
            # Sparse punctuation occupies its cell optically, without full-height fitting.
            y = 34 + kind*7
            pieces = [box(22, y, 52+kind*8, y+w)]
            if kind != 1:
                pieces.append(box(31+kind*4, y+w+9, 64, y+2*w+9))
        elif kind < 6:
            pieces = [_stroke([(41, top), (41, bottom)], w),
                      _stroke([(41, 27+kind*3), (62, 22+kind*3)], w*.9)]
            if kind == 4:
                pieces.append(_stroke([(41, 65), (66, 65)], w*.85))
            if kind == 5:
                pieces.append(box(61, 68, 75, 83))
        elif kind == 6:
            pieces = [_stroke([(left, mid), (right, mid)], w),
                      _stroke([(arm, top+3), (arm-9, bottom-3)], w)]
        elif kind == 7:
            pieces = [_stroke([(right-2, top+4), (left+4, mid),
                               (right-6, bottom-4)], w, angular=True),
                      _stroke([(left+9, mid), (arm+8, mid)], w*.8)]
        elif kind == 8:
            pieces = [_ring(25, top, 75, bottom, w, 18),
                      _stroke([(left, mid+9), (38, mid+9)], w*.85)]
        elif kind == 9:
            pieces = [_curve([(left+5, top+9), (right+7, top-12), (right+5, mid+9),
                              (left+5, mid)], w),
                      _stroke([(left+5, mid), (arm, bottom-7), (right, bottom-7)], w,
                               angular=True)]
        elif kind == 10:
            pieces = [_curve([(right-6, top+6), (left-3, top-4), (left, bottom+6),
                              (right-7, bottom-5)], w),
                      _stroke([(mid, mid+3), (right-7, mid+3), (right-7, bottom-5)], w)]
        elif kind == 11:
            pieces = [_stroke([(left+6, top), (left+6, bottom-w/2),
                               (right, bottom-w/2)], w, angular=True),
                      _stroke([(left+6, mid), (right-6, mid-7), (right-6, top+4)], w)]
        elif kind == 12:
            pieces = [_ring(left, top, right-6, bottom-19, w, 17),
                      _stroke([(mid, bottom-24), (mid+9, bottom)], w)]
        elif kind == 13:
            pieces = [_curve([(left+6, bottom-5), (right+7, bottom+3),
                              (right-5, top-5), (left+6, top+7)], w),
                      _stroke([(left+6, top+7), (left+6, mid), (arm, mid)], w)]
        else:
            pieces = [_stroke([(left, top+w/2), (right, top+w/2),
                               (mid, mid), (mid, bottom)], w, angular=True),
                      _stroke([(mid, bottom-7), (right-5, bottom-7)], w*.85)]
    elif family == "stacked_marks":
        pieces = [box(30+ordinal*3, 20, 46+ordinal*2, 35),
                  box(46-ordinal*4, 47, 68-ordinal*4, 61),
                  box(33+ordinal*5, 73, 48+ordinal*5, 85)]
        if ordinal == 1:
            pieces[1] = box(35, 47, 70, 60)
        elif ordinal == 2:
            pieces[0] = Polygon([(31, 21), (52, 17), (47, 35), (33, 37)])
            pieces[2] = box(35, 73, 65, 85)
    elif family == "diagonal_lozenge":
        if variant < 5:
            pieces = [_stroke([(left+2, mid+7), (mid-4, top+7),
                               (right-2, mid-1)], w, rounded=True)]
            if variant % 2:
                pieces.append(_stroke([(mid+6, top+17), (right-5, bottom-5)], w*.9,
                                      rounded=True))
            else:
                pieces.append(_stroke([(left+10, bottom-5), (mid-5, mid+17),
                                       (right-12, bottom-4)], w*.95, rounded=True))
        else:
            outer = Polygon([(mid-4, top), (right, mid-3), (mid+5, bottom),
                             (left, mid+5)])
            inner = scale(outer, xfact=.43, yfact=.43, origin=(50, 50))
            shape = outer.difference(inner)
            if variant == 6:
                shape = shape.difference(box(left-2, mid-8, mid+3, mid+10))
            elif variant == 7:
                shape = shape.difference(box(mid-6, top-2, mid+7, mid))
                pieces.append(_stroke([(left+1, mid+4), (right-3, mid+4)], w*.65))
            elif variant == 8:
                shape = shape.difference(box(mid-3, mid+3, right+2, bottom+2))
                pieces.append(box(68, 72, 84, 87))
            elif variant == 9:
                shape = shape.difference(box(left-2, mid-7, mid+1, mid+9))
                pieces.append(_stroke([(mid+2, mid), (right+1, mid+7)], w*.8))
            pieces.append(shape)
        # This detached comma belongs to the composition, with a stable clear gap.
        if variant < 5:
            pieces.append(Polygon([(77, 67), (88, 76), (80, 89), (70, 79)]))
            if ordinal >= 10:
                pieces[0] = pieces[0].difference(box(mid-9, top, mid+5, mid+1))
                pieces.append(_stroke([(left+4, mid+11), (mid+7, mid+11)], w*.8))
    elif family == "roofed_curves":
        pieces = [_stroke([(left, top+7), (right, top+7)], w)]
        if variant % 3 == 0:
            pieces += [_curve([(left+11, top+5), (left-5, bottom), (right, bottom),
                               (right-6, mid)], w),
                       _stroke([(right-6, mid), (arm-4, mid)], w*.9)]
        elif variant % 3 == 1:
            pieces += [_curve([(right-9, top+5), (right+4, bottom+7),
                               (left+8, bottom+2), (left+7, mid)], w),
                       _stroke([(left+7, mid), (arm+9, mid+5)], w*.95)]
        else:
            pieces += [_curve([(mid, top+6), (left-6, mid), (right+7, mid),
                               (right-7, bottom)], w),
                       _curve([(left+4, bottom), (left, mid+5), (mid-2, mid-7),
                               (right-7, bottom)], w*.95)]
        if ordinal % 4 == 0:
            pieces.append(_stroke([(mid-8, top+4), (mid-8, mid-5)], w*.85))
    elif family == "rounded_loop_interlock":
        if variant < 4:
            pieces = [_ring(left, top, right, bottom-4, w, 21)]
            # Opening and bridge position are independent structural choices.
            if variant == 0:
                pieces[0] = pieces[0].difference(box(right-w-2, mid-7, right+2, mid+11))
                pieces.append(_stroke([(left+w/2, bottom-12), (mid+6, bottom-12),
                                       (mid+6, mid+4)], w*.9, rounded=True))
            elif variant == 1:
                pieces.append(_stroke([(left+w/2, mid-3), (right-w/2, mid+4)], w*.84,
                                      rounded=True))
            elif variant == 2:
                pieces[0] = pieces[0].difference(box(mid-8, top-2, mid+8, top+w+2))
                pieces.append(_curve([(right-w/2, mid-2), (mid, mid-2), (mid, bottom+3),
                                      (left-1, bottom-2)], w*.9, rounded=True))
            else:
                pieces.append(_stroke([(right-w/2, mid+4), (mid, mid+4),
                                       (mid, top+9)], w*.85, rounded=True))
        elif variant < 7:
            pieces = [_curve([(left+8, top+9), (right+4, top-10), (right+1, mid+9),
                              (mid-2, mid)], w, rounded=True),
                      _curve([(mid-2, mid), (left-3, mid-4), (left-5, bottom+8),
                              (right-6, bottom-8)], w, rounded=True)]
            if variant == 4:
                pieces.append(_stroke([(right-7, bottom-8), (right-7, mid+8)], w,
                                      rounded=True))
            elif variant == 5:
                pieces.append(_stroke([(left+8, top+9), (left+8, mid-8)], w,
                                      rounded=True))
            else:
                pieces.append(_stroke([(mid+9, mid), (mid+9, bottom-8)], w*.85,
                                      rounded=True))
        else:
            pieces = [_curve([(left+7, top), (left-5, bottom+7), (right+2, bottom+1),
                              (right-7, top+7)], w, rounded=True),
                      _curve([(right-7, top+7), (mid+1, top-4), (mid-10, mid-4),
                              (mid+9, mid+7)], w*.95, rounded=True)]
            if variant == 8:
                pieces.append(_stroke([(left+6, mid+3), (mid+9, mid+7)], w*.85,
                                      rounded=True))
            elif variant == 9:
                pieces.append(_stroke([(right-6, bottom-10), (right+3, bottom+1)], w*.8))
    else:  # mixed asymmetric structures: loop, shaft, and a deliberate open terminal.
        if variant < 5:
            pieces = [_ring(left+10, top, right-9, mid+12, w, 15),
                      _stroke([(mid-6, mid+7), (mid-6, bottom-6)], w),
                      _curve([(mid-6, bottom-12), (right+5, bottom+4),
                              (right-3, mid+15), (right, mid+13)], w*.9)]
            if variant % 2:
                pieces.append(_stroke([(left+5, bottom-6), (right-4, bottom-6)], w*.83))
            if variant == 2:
                pieces[0] = pieces[0].difference(box(left+8, mid-6, mid-1, mid+14))
                pieces.append(_stroke([(left+5, top+8), (mid+2, top+8)], w*.85))
            elif variant == 3:
                pieces.append(_stroke([(mid-6, mid+12), (left+5, mid+12),
                                       (left+5, bottom-6)], w*.8))
            elif variant == 4:
                pieces.append(_stroke([(mid-4, top+19), (right+1, top+19)], w*.8))
        else:
            pieces = [_curve([(left+8, top+5), (left-8, mid+12),
                              (right+5, mid-9), (right-6, bottom-5)], w),
                      _stroke([(right-6, bottom-5), (mid-7, bottom-5),
                               (mid-7, mid)], w, rounded=True)]
            pieces.append(_stroke([(left+6, top+6), (arm+10, top+6)], w))
            if variant % 2 == 0:
                pieces.append(_stroke([(mid-7, mid), (left+3, mid+5)], w*.85))

    geometry = union_all(pieces)
    # Recipes gain a distinct cut or terminal choice, never only an affine variation.
    if (parameters.get("structure") and ordinal >= 10
            and family not in {"stacked_marks", "diagonal_lozenge"} and ordinal // 10 % 3 == 1):
        # The fit-jitter stream is part of the fixed authoring specification.
        # Consume its reserved cut-position draw without constructing an unused base shape.
        rng.uniform(.38, .62)
    if (ordinal >= 10 and family not in {"stacked_marks", "diagonal_lozenge"}
            and not parameters.get("structure")):
        bounds = geometry.bounds
        side = ordinal // 10 % 3
        if side == 1:
            cut_y = top + (bottom-top)*rng.uniform(.38, .62)
            geometry = geometry.difference(box(bounds[0]-1, cut_y, mid-1, cut_y+8))
            geometry = union_all([geometry, _stroke([(mid-5, bottom-6),
                                                     (right+3, bottom-6)], w*.78)])
        elif side == 2:
            geometry = geometry.difference(box(mid-7, top-1, mid+5, top+18))
            geometry = union_all([geometry, _stroke([(left+4, mid+9),
                                                     (right-3, mid+9)], w*.72)])
        else:
            geometry = geometry.difference(box(right-17, bottom-22, right+3, bottom+3))
            geometry = union_all([geometry, _stroke([(right-7, top+5),
                                                     (right-7, mid+13)], w*.76)])
    # Close sub-pixel nicks and remove tiny slivers left by contour intersections.
    # Deliberate counters remain; incidental acute voids are filled before export.
    geometry = geometry.buffer(.85, quad_segs=12).buffer(-.85, quad_segs=12)
    # Contour unions must not leave hairline protrusions at opened terminals.
    # A small geometric opening removes those artifacts before optical fitting.
    opening = parameters.get("opening", 1.35)
    geometry = geometry.buffer(-opening, quad_segs=12).buffer(opening, quad_segs=12)
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    rebuilt = []
    for polygon in polygons:
        if polygon.area < 12:
            continue
        holes = []
        for ring in polygon.interiors:
            hole = Polygon(ring)
            if hole.area < (100 if profile == "classic" else 40):
                continue
            if profile == "expanded":
                hole = hole.buffer(-.7, quad_segs=12)
            if parameters.get("counter_scale"):
                hole = scale(hole, xfact=parameters["counter_scale"],
                             yfact=parameters["counter_scale"], origin="centroid")
            if not hole.is_empty and hole.geom_type == "Polygon":
                holes.append(list(hole.exterior.coords))
        rebuilt.append(Polygon(polygon.exterior.coords, holes))
    geometry = union_all(rebuilt)
    if parameters.get("mark_size") and geometry.geom_type == "MultiPolygon":
        parts = []
        for part in geometry.geoms:
            if part.area < 110:
                x, y = part.centroid.coords[0]
                radius = parameters["mark_size"]/2
                part = box(x-radius, y-radius, x+radius, y+radius)
            parts.append(part)
        geometry = union_all(parts)
    if parameters.get("gap") and geometry.geom_type == "MultiPolygon":
        parts = sorted(geometry.geoms, key=lambda polygon: -polygon.area)
        adjusted = [parts[0]]
        for part in parts[1:]:
            fixed = union_all(adjusted)
            a, b = nearest_points(fixed, part)
            distance = a.distance(b)
            if distance > 0:
                offset = parameters["gap"]-distance
                part = translate(part, xoff=(b.x-a.x)*offset/distance,
                                 yoff=(b.y-a.y)*offset/distance)
            # Recompute the nearest boundary after each move: a concave arm
            # can become the closest feature as a detached mark moves outward.
            for _ in range(16):
                a, b = nearest_points(fixed, part)
                distance = a.distance(b)
                if distance >= parameters["gap"]-.01 or distance == 0:
                    break
                offset = parameters["gap"] - distance
                part = translate(part, xoff=(b.x-a.x)*offset/distance,
                                 yoff=(b.y-a.y)*offset/distance)
            adjusted.append(part)
        geometry = union_all(adjusted)
    # Fit the authored envelope uniformly. Narrow marks and punctuation keep their scale.
    min_x, min_y, max_x, max_y = geometry.bounds
    ratio = min(rng.uniform(65, 69)/(max_x-min_x), rng.uniform(70, 74)/(max_y-min_y), 1.06)
    geometry = scale(geometry, xfact=ratio, yfact=ratio, origin=(50, 50))
    min_x, min_y, max_x, max_y = geometry.bounds
    geometry = translate(geometry, xoff=50-(min_x+max_x)/2+rng.uniform(-1.3, 1.3),
                         yoff=50-(min_y+max_y)/2+rng.uniform(-.8, .8))
    # Fixed optical corrections widen specific open entrances and counter
    # throats in the authored 100-unit canvas. They are independent design data.
    if parameters.get("apertures"):
        geometry = geometry.difference(union_all([
            Point(x, y).buffer(radius, quad_segs=16)
            for x, y, radius in parameters["apertures"]
        ]))
    return orient_polygons(set_precision(geometry, .001)), w


def _ring_path(coordinates) -> str:
    points = [(round(float(x), 3), round(float(y), 3)) for x, y in coordinates[:-1]]
    start = min(range(len(points)), key=lambda i: points[i])
    points = points[start:] + points[:start]
    return "M" + " L".join(f"{x:g} {y:g}" for x, y in points) + " Z"


def generate_glyph(index: int, attempt: int = 0) -> dict:
    """Generate one deterministic original SVG, with no artifact cache or I/O."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < GLYPH_COUNT:
        raise ValueError(f"index must be an integer from 0 to {GLYPH_COUNT-1}")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or not 0 <= attempt <= 1000:
        raise ValueError("attempt must be an integer from 0 to 1000")
    family, ordinal = SCHEDULE[index]
    profile = "classic" if family == "bar_hook" or (family == "numeral_operator" and ordinal < 28) else "expanded"
    parameters = AUTHORED_PARAMETERS.get(index, {})
    seed = BASE_SEED + index*104729 + (attempt+parameters.get("seed_offset", 0))*1000003
    geometry, weight = _gesture(family, ordinal, random.Random(seed), profile, parameters)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"invalid generated geometry for glyph {index}")
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    polygons.sort(key=lambda p: (-p.area, p.bounds))
    paths = []
    for polygon in polygons:
        rings = [_ring_path(list(polygon.exterior.coords))]
        rings.extend(_ring_path(list(hole.coords)) for hole in polygon.interiors)
        paths.append(" ".join(rings))
    glyph_id = f"GLYPH-{index:03d}"
    svg = (f'<svg xmlns="{SVG_NS}" viewBox="0 0 100 100" width="128" height="128">'
           + ''.join(f'<path fill="#000000" fill-rule="nonzero" d="{path}"/>' for path in paths)
           + '</svg>\n')
    return {"glyph_id": glyph_id, "index": index, "family": family, "profile": profile,
            "seed": seed, "attempt": attempt, "version": VERSION,
            "authoring_parameters": dict(parameters),
            "recipe": f"{family}/{parameters.get('structure', ordinal % (15 if family == 'numeral_operator' else 10))}",
            "ordinal": ordinal, "authoring_weight": round(weight, 4),
            "svg": svg, "paths": paths, "svg_sha256": hashlib.sha256(svg.encode()).hexdigest()}


def _parse_paths(svg: str) -> list[list[list[tuple[float, float]]]]:
    if not isinstance(svg, str) or len(svg.encode("utf-8")) > 262144:
        raise ValueError("SVG must be text of at most 256 KiB")
    if "<!" in svg or "<?" in svg:
        raise ValueError("DTD, entities, comments, and processing instructions are not allowed")
    root = ET.fromstring(svg)
    if root.tag != f"{{{SVG_NS}}}svg" or root.attrib.get("viewBox") != "0 0 100 100":
        raise ValueError("expected an SVG with viewBox 0 0 100 100")
    if set(root.attrib) - {"viewBox", "width", "height"}:
        raise ValueError("unsupported root attributes")
    if any(root.attrib.get(key) != "128" for key in ("width", "height")):
        raise ValueError("source SVG dimensions must be 128 by 128")
    paths = []
    for child in root:
        if child.tag != f"{{{SVG_NS}}}path" or len(child):
            raise ValueError("only leaf path elements are accepted")
        if set(child.attrib) != {"d", "fill", "fill-rule"}:
            raise ValueError("paths require only d, fill, and fill-rule")
        if child.attrib["fill"] != "#000000" or child.attrib["fill-rule"] != "nonzero":
            raise ValueError("glyphs require black nonzero fills")
        data = child.attrib["d"]
        tokens = _TOKEN.findall(data)
        if re.sub(r"[\s,]", "", _TOKEN.sub("", data)):
            raise ValueError("only explicit absolute M/L/Z path commands are supported")
        rings, ring = [], None
        position = 0
        while position < len(tokens):
            token = tokens[position]
            if token == "Z":
                if ring is None or len(ring) < 3:
                    raise ValueError("closed contours need at least three vertices")
                rings.append(ring)
                ring = None
                position += 1
                continue
            if token not in {"M", "L"} or position+2 >= len(tokens):
                raise ValueError("malformed path command")
            if (token == "M" and ring is not None) or (token == "L" and ring is None):
                raise ValueError("each contour must start with M and close with Z")
            point = (float(tokens[position+1]), float(tokens[position+2]))
            if any(not math.isfinite(v) or not 0 <= v <= 100 for v in point):
                raise ValueError("coordinates must be finite and inside the viewBox")
            if token == "M":
                ring = []
            ring.append(point)
            position += 3
        if ring is not None or not rings:
            raise ValueError("unclosed or empty path")
        paths.append(rings)
    if not paths or sum(len(r) for p in paths for r in p) > 16000:
        raise ValueError("empty or excessively complex SVG")
    return paths


def _signed_area(points):
    return sum(x*y2-x2*y for (x, y), (x2, y2) in zip(points, points[1:]+points[:1]))/2


def validate_svg(svg: str) -> dict:
    """Fail closed on unsupported content, invalid bounds, or degenerate contours."""
    try:
        paths = _parse_paths(svg)
        if any(abs(_signed_area(ring)) < .01 for path in paths for ring in path):
            raise ValueError("degenerate contour")
        for path in paths:
            if _signed_area(path[0]) <= 0 or any(_signed_area(r) >= 0 for r in path[1:]):
                raise ValueError("exterior and counter contours need opposite winding")
            if not Polygon(path[0], path[1:]).is_valid:
                raise ValueError("intersecting or invalid filled contour")
        return {"passed": True, "errors": [], "paths": len(paths),
                "contours": sum(len(p) for p in paths),
                "vertices": sum(len(r) for p in paths for r in p),
                "sha256": hashlib.sha256(svg.encode()).hexdigest()}
    except (ValueError, TypeError, ET.ParseError, OverflowError) as error:
        return {"passed": False, "errors": [str(error)]}


def render_svg(svg: str, size: int = 128) -> Image.Image:
    """Rasterize the restricted filled-path format with nonzero winding and 4x AA."""
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 2048:
        raise ValueError("size must be an integer from 1 to 2048")
    paths = _parse_paths(svg)
    side = size*4
    result = np.zeros((side, side), dtype=bool)
    factor = side/100
    for path in paths:
        winding = np.zeros((side, side), dtype=np.int16)
        for ring in path:
            mask = Image.new("L", (side, side), 0)
            ImageDraw.Draw(mask).polygon([(x*factor, y*factor) for x, y in ring], fill=1)
            winding += np.asarray(mask, dtype=np.int16) * (1 if _signed_area(ring) > 0 else -1)
        result |= winding != 0
    pixels = np.where(result, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels).resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")


def write_catalog(destination: Path, *, count: int = GLYPH_COUNT, overwrite=False) -> dict:
    """Export original sources, deterministic contact sheets, and a manifest."""
    if not 1 <= count <= GLYPH_COUNT:
        raise ValueError("count must be from 1 to 192")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}; choose a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    glyphs = [generate_glyph(index) for index in range(count)]
    cells = 144
    columns = 8 if count <= 24 else 16
    rows = math.ceil(count/columns)
    sheet = Image.new("RGB", (columns*cells, rows*164), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=13)
    records = []
    for index, glyph in enumerate(glyphs):
        report = validate_svg(glyph["svg"])
        if not report["passed"]:
            raise ValueError(report["errors"])
        (destination / f'{glyph["glyph_id"]}.svg').write_text(glyph["svg"], encoding="utf-8", newline="\n")
        tile = render_svg(glyph["svg"])
        tile.save(destination / f'{glyph["glyph_id"]}.png')
        x, y = (index % columns)*cells+8, (index//columns)*164+4
        sheet.paste(tile.convert("RGB"), (x, y))
        draw.text((x+18, y+133), glyph["glyph_id"], fill="black", font=font)
        records.append({k: v for k, v in glyph.items() if k not in {"svg", "paths"}})
    sheet.save(destination / "contact-sheet.png")
    # Small-render review uses nearest-neighbor enlargement only for inspection:
    # the actual raster is generated at the labeled native resolution first.
    for size, magnification in ((16, 3), (32, 2), (64, 1)):
        side = size*magnification
        cell = max(80, side+16)
        review = Image.new("RGB", (columns*cell, rows*(cell+20)), "white")
        labels = ImageDraw.Draw(review)
        for index, glyph in enumerate(glyphs):
            tile = render_svg(glyph["svg"], size).convert("RGB")
            if magnification > 1:
                tile = tile.resize((side, side), Image.Resampling.NEAREST)
            x, y = (index % columns)*cell, (index//columns)*(cell+20)
            review.paste(tile, (x+(cell-side)//2, y+8))
            labels.text((x+12, y+cell), f"{index:03d} / {size}px", fill="black", font=font)
        review.save(destination / f"contact-sheet-{size}.png")
    manifest = {"version": VERSION, "count": count, "canvas": [100, 100],
                "method": "independent deterministic filled-contour grammar",
                "reference_artwork_included": False, "api_calls": 0,
                "glyphs": records}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+'\n', encoding="utf-8", newline="\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write_catalog(args.out, count=args.count, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    sys.exit(main())
