"""Exercise an actual Linux ELF renderer against an X11 display (use Xvfb)."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops


REPO_ROOT = Path(__file__).resolve().parents[2]
GLYPH_COUNT, REFERENCE_COUNT, ORIGINAL_COUNT, BLANK_INDEX = 249, 57, 192, 4
SHEET_TILE, SHEET_COLUMNS = 128, 16


def inspect_catalog(catalog: dict, root: Path = REPO_ROOT) -> dict:
    """Bind compiled identities to both published source catalogs."""
    receipt_path = root / "screensaver/native-catalog.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    reference_path = root / "screensaver/svg-preview/reference/catalog.json"
    original_path = root / "benchmarks/partitions/glyph_svg_v1/catalog/manifest.json"
    expected = {
        "version": "native-mixed-svg-v1", "glyph_count": GLYPH_COUNT,
        "reference_count": REFERENCE_COUNT, "reference_visible_count": 56,
        "original_count": ORIGINAL_COUNT, "original_offset": REFERENCE_COUNT,
        "blank_index": BLANK_INDEX, "canvas": [100, 100], "fill_rule": "nonzero",
        "default_original_mix": 0.1, "catalog_sha256": receipt["catalog_sha256"],
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
    }
    for key, value in expected.items():
        if catalog.get(key) != value:
            raise AssertionError(f"Compiled catalog mismatch: {key}")
    return {**expected, "native_catalog_receipt_sha256": hashlib.sha256(
        receipt_path.read_bytes()).hexdigest()}


def source_sheet_svg(root: Path = REPO_ROOT, *, tile_size: int = SHEET_TILE,
                     margin: float = 4, white_ink: bool = False) -> bytes:
    """Build a verification sheet from source SVGs, never the C header."""
    directory = root / "benchmarks/partitions/glyph_svg_v1/catalog"
    originals = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    reference = json.loads((root / "screensaver/svg-preview/reference/catalog.json").read_text(
        encoding="utf-8"))
    if len(originals["glyphs"]) != ORIGINAL_COUNT or len(reference["glyphs"]) != 56:
        raise AssertionError("Source SVG catalog count changed")
    namespaces = "http://www.w3.org/2000/svg"
    if tile_size not in (64, 128) or not 0 <= margin < tile_size / 2:
        raise ValueError("Expected 64px or 128px cells with a bounded inner margin")
    size = SHEET_COLUMNS * tile_size
    foreground, background = ("white", "black") if white_ink else ("black", "white")
    sheet = ET.Element("svg", xmlns=namespaces, width=str(size), height=str(size),
                       viewBox=f"0 0 {size} {size}")
    ET.SubElement(sheet, "rect", width="100%", height="100%", fill=background)

    def group(index: int) -> ET.Element:
        x, y = (index % SHEET_COLUMNS) * tile_size + margin, (index // SHEET_COLUMNS) * tile_size + margin
        scale = (tile_size - 2 * margin) / 100
        return ET.SubElement(sheet, "g", transform=f"translate({x:g} {y:g}) scale({scale:g})")

    slots = {glyph["source_sequence_index"] for glyph in reference["glyphs"]}
    if slots != set(range(REFERENCE_COUNT)) - {BLANK_INDEX}:
        raise AssertionError("Reference blank or sequence positions changed")
    for glyph in reference["glyphs"]:
        element = group(glyph["source_sequence_index"])
        ET.SubElement(element, "path", d=" ".join(glyph["paths"]), fill=foreground,
                      **{"fill-rule": "nonzero"})
    for index, glyph in enumerate(originals["glyphs"]):
        if glyph["glyph_id"] != f"GLYPH-{index:03d}":
            raise AssertionError("Original source sequence changed")
        svg = (directory / glyph["file"]).read_bytes()
        if hashlib.sha256(svg).hexdigest() != glyph["svg_sha256"]:
            raise AssertionError(f"Original source SVG hash mismatch: {glyph['glyph_id']}")
        element = group(REFERENCE_COUNT + index)
        for path in ET.fromstring(svg):
            if path.tag != f"{{{namespaces}}}path":
                raise AssertionError("Expected source SVG path geometry")
            path.set("fill", foreground)
            element.append(path)
    return ET.tostring(sheet, encoding="utf-8")


def render_source_sheet(svg: bytes, path: Path) -> None:
    """Use librsvg's SVG parser as an independent compiled-contour oracle."""
    pointer, integer, number = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    cairo = ctypes.CDLL(ctypes.util.find_library("cairo") or "libcairo.so.2")
    rsvg = ctypes.CDLL(ctypes.util.find_library("rsvg-2") or "librsvg-2.so.2")
    gobject = ctypes.CDLL(ctypes.util.find_library("gobject-2.0") or "libgobject-2.0.so.0")
    cairo.cairo_image_surface_create.argtypes = [integer, integer, integer]
    cairo.cairo_image_surface_create.restype = pointer
    cairo.cairo_create.argtypes, cairo.cairo_create.restype = [pointer], pointer
    cairo.cairo_surface_write_to_png.argtypes = [pointer, ctypes.c_char_p]
    cairo.cairo_surface_write_to_png.restype = integer
    cairo.cairo_destroy.argtypes = cairo.cairo_surface_destroy.argtypes = [pointer]
    rsvg.rsvg_handle_new_from_data.argtypes = [ctypes.c_char_p, ctypes.c_size_t, pointer]
    rsvg.rsvg_handle_new_from_data.restype = pointer
    rsvg.rsvg_handle_render_document.argtypes = [pointer, pointer, pointer, pointer]
    rsvg.rsvg_handle_render_document.restype = integer
    gobject.g_object_unref.argtypes = [pointer]

    class Rectangle(ctypes.Structure):
        _fields_ = [(name, number) for name in ("x", "y", "width", "height")]

    size = int(ET.fromstring(svg).get("width", "0"))
    if size not in (64 * SHEET_COLUMNS, 128 * SHEET_COLUMNS):
        raise ValueError("Expected a bounded 64px or 128px source sheet")
    surface = cairo.cairo_image_surface_create(1, size, size)
    context = cairo.cairo_create(surface)
    handle = rsvg.rsvg_handle_new_from_data(svg, len(svg), None)
    try:
        viewport = Rectangle(0, 0, size, size)
        if not handle or not rsvg.rsvg_handle_render_document(handle, context,
                                                             ctypes.byref(viewport), None):
            raise AssertionError("librsvg could not render the source SVG sheet")
        if cairo.cairo_surface_write_to_png(surface, os.fsencode(path)):
            raise AssertionError("Could not save the independent source sheet")
    finally:
        if handle:
            gobject.g_object_unref(handle)
        cairo.cairo_destroy(context)
        cairo.cairo_surface_destroy(surface)


def compare_glyph_sheet(actual_path: Path, expected_path: Path, *, tile_size: int = SHEET_TILE,
                        margin: float = 4, white_ink: bool = False, minimum_iou: float = .99) -> dict:
    """Check every shape, including counters and the intentionally blank slot."""
    with Image.open(actual_path) as actual_image, Image.open(expected_path) as expected_image:
        if any(image.convert("RGBA").getchannel("A").getextrema() != (255, 255)
               for image in (actual_image, expected_image)):
            raise AssertionError("Glyph sheet must be opaque")
        actual, expected = actual_image.convert("L"), expected_image.convert("L")
    if tile_size not in (64, 128) or not .9 <= minimum_iou <= 1:
        raise ValueError("Unsupported glyph-sheet comparison parameters")
    size = (tile_size * SHEET_COLUMNS,) * 2
    if actual.size != size or expected.size != size:
        raise AssertionError("Glyph sheet dimensions changed")
    threshold = [255 if (value >= 128 if white_ink else value < 128) else 0 for value in range(256)]
    checks = []
    for index in range(GLYPH_COUNT):
        x, y = index % SHEET_COLUMNS * tile_size, index // SHEET_COLUMNS * tile_size
        box = (x, y, x + tile_size, y + tile_size)
        actual_tile, expected_tile = actual.crop(box), expected.crop(box)
        ink = actual_tile.point(threshold, mode="1")
        source_ink = expected_tile.point(threshold, mode="1")
        union = ImageChops.logical_or(ink, source_ink).histogram()[255]
        intersection = ImageChops.logical_and(ink, source_ink).histogram()[255]
        if index == BLANK_INDEX:
            background = 0 if white_ink else 255
            if actual_tile.getextrema() != (background, background) or union:
                raise AssertionError("The reference blank slot contains geometry")
            iou = 1.0
        else:
            if union < 20 or not intersection:
                raise AssertionError(f"Glyph slot {index} is missing its source shape")
            iou = intersection / union
            if iou < minimum_iou:
                raise AssertionError(f"Glyph slot {index} differs from source SVG: IoU={iou:.6f}")
        checks.append({"slot": index, "family": "reference" if index < REFERENCE_COUNT else "original",
                       "ink_pixels": ink.histogram()[255], "bounds": ink.getbbox(),
                       "source_silhouette_iou": iou,
                       "pixel_sha256": hashlib.sha256(actual_tile.tobytes()).hexdigest()})
    return {"status": "passed", "glyph_count": GLYPH_COUNT, "reference_visible_count": 56,
            "original_count": ORIGINAL_COUNT, "blank_index": BLANK_INDEX,
            "method": "Compiled native contours versus librsvg source SVGs; no alignment, scaling, or flip correction",
            "tile_size": tile_size, "inner_margin": margin, "white_ink": white_ink,
            "ink_threshold": 128, "minimum_required_iou": minimum_iou,
            "minimum_measured_iou": min(check["source_silhouette_iou"] for check in checks),
            "actual_sha256": hashlib.sha256(actual_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(), "glyphs": checks}


def selection_counts(stdout: str, expected_mix: float | None = None) -> dict:
    match = re.search(r"selected_reference=(\d+) selected_original=(\d+) selected_blank=(\d+) mix=([\d.]+)", stdout)
    if match is None:
        raise AssertionError("Missing native family-selection accounting")
    reference, original, blank = map(int, match.groups()[:3])
    mix = float(match[4])
    if not reference + original or blank > reference:
        raise AssertionError("Invalid native family-selection accounting")
    if expected_mix is not None and mix != expected_mix:
        raise AssertionError("Native mix option was not applied")
    if mix == 0 and (original or not reference):
        raise AssertionError("Reference-only mode selected an original glyph")
    if mix == 1 and (reference or not original):
        raise AssertionError("Original-only mode selected a reference glyph")
    share = original / (reference + original)
    if mix == 0.1 and not 0.06 < share < 0.14:
        raise AssertionError(f"Default per-cell mix is not approximately 90/10: {share}")
    return {"reference": reference, "original": original, "blank": blank,
            "configured_original_mix": mix, "observed_original_share": share}


def inspect_frame(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        pixels = list(rgb.getdata())
        black = sum(max(pixel) < 12 for pixel in pixels)
        green = sum(g > 20 and g > r * 1.15 and g > b * 1.15 for r, g, b in pixels)
        body_hues = [120 + 60 * (b - r) / (g - r) for r, g, b in pixels
                     if g > 35 and r < g * .35 and r < b < g * .55]
        mint = sum(g > 40 and .55 * g < r < .72 * g and .75 * g < b < .95 * g
                   for r, g, b in pixels)
        total = len(pixels)
        if black / total < 0.35 or green / total < 0.015:
            raise AssertionError(f"Frame lacks black field or green glyphs: {path}")
        if not body_hues or not 136 <= statistics.median(body_hues) <= 139:
            raise AssertionError(f"Frame body hue does not match the 137-degree grade: {path}")
        return {
            "width": image.width,
            "height": image.height,
            "black_fraction": round(black / total, 5),
            "green_fraction": round(green / total, 5),
            "body_hue_median": statistics.median(body_hues),
            "mint_pixels": mint,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


class XHost:
    """Own a real preview window and inspect pixels sent to the X server."""

    def __init__(self) -> None:
        self.x = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        pointer, ulong, uint, integer = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_uint, ctypes.c_int
        signatures = {
            "XOpenDisplay": ([ctypes.c_char_p], pointer),
            "XDefaultRootWindow": ([pointer], ulong),
            "XCreateSimpleWindow": ([pointer, ulong, integer, integer, uint, uint, uint, ulong, ulong], ulong),
            "XMapWindow": ([pointer, ulong], integer),
            "XResizeWindow": ([pointer, ulong, uint, uint], integer),
            "XDestroyWindow": ([pointer, ulong], integer),
            "XCloseDisplay": ([pointer], integer),
            "XSync": ([pointer, integer], integer),
            "XGetImage": ([pointer, ulong, integer, integer, uint, uint, ulong, integer], pointer),
            "XGetPixel": ([pointer, integer, integer], ulong),
            "XDestroyImage": ([pointer], integer),
            "XGetInputFocus": ([pointer, ctypes.POINTER(ulong), ctypes.POINTER(integer)], integer),
            "XSetInputFocus": ([pointer, ulong, integer, ulong], integer),
            "XKeysymToKeycode": ([pointer, ulong], ctypes.c_ubyte),
            "XSendEvent": ([pointer, ulong, integer, ctypes.c_long, pointer], integer),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.x, name)
            function.argtypes = arguments
            function.restype = result
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("No X display: run this harness under xvfb-run")
        self.root = self.x.XDefaultRootWindow(self.display)
        self.window = self.x.XCreateSimpleWindow(self.display, self.root, 0, 0, 320, 240, 0, 0, 0)
        self.x.XMapWindow(self.display, self.window)
        self.x.XSync(self.display, False)

    def escape(self, window: int) -> None:
        focus, revert = ctypes.c_ulong(), ctypes.c_int()
        self.x.XGetInputFocus(self.display, ctypes.byref(focus), ctypes.byref(revert))
        if focus.value != window:
            raise AssertionError("Fullscreen window did not acquire keyboard focus")

        class KeyEvent(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int), ("serial", ctypes.c_ulong),
                ("send_event", ctypes.c_int), ("display", ctypes.c_void_p),
                ("window", ctypes.c_ulong), ("root", ctypes.c_ulong),
                ("subwindow", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("x", ctypes.c_int), ("y", ctypes.c_int),
                ("x_root", ctypes.c_int), ("y_root", ctypes.c_int),
                ("state", ctypes.c_uint), ("keycode", ctypes.c_uint),
                ("same_screen", ctypes.c_int),
            ]

        storage = ctypes.create_string_buffer(24 * ctypes.sizeof(ctypes.c_long))
        event = KeyEvent.from_buffer(storage)
        event.type, event.display, event.window, event.root = 2, self.display, window, self.root
        event.keycode = self.x.XKeysymToKeycode(self.display, 0xFF1B)
        event.same_screen = 1
        self.x.XSendEvent(self.display, window, False, 1, ctypes.byref(storage))
        self.x.XSync(self.display, False)

    def pixels(self, width: int = 320, height: int = 240) -> dict:
        self.x.XSync(self.display, False)
        image = self.x.XGetImage(self.display, self.window, 0, 0, width, height,
                                 ctypes.c_ulong(-1).value, 2)
        if not image:
            raise AssertionError("Unable to read embedded X11 framebuffer")
        try:
            samples = [self.x.XGetPixel(image, x, y)
                       for y in range(0, height, 4) for x in range(0, width, 4)]
        finally:
            self.x.XDestroyImage(image)
        green = sum(((p >> 8) & 255) > max((p >> 16) & 255, p & 255) + 10 for p in samples)
        black = sum((p & 0xFFFFFF) == 0 for p in samples)
        if green / len(samples) < 0.01 or black / len(samples) < 0.3:
            raise AssertionError("The target X11 window did not receive black/green glyph pixels")
        return {"sampled_pixels": len(samples), "green_pixels": green, "black_pixels": black}

    def close(self) -> None:
        self.x.XDestroyWindow(self.display, self.window)
        self.x.XCloseDisplay(self.display)


def run_smoke(binary: Path, out: Path) -> dict:
    binary = binary.resolve()
    if binary.read_bytes()[:4] != b"\x7fELF":
        raise AssertionError("Expected a compiled Linux ELF executable")
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("XSCREENSAVER_WINDOW", None)

    def invoke(*args: str, env: dict | None = None, expected: int = 0) -> subprocess.CompletedProcess:
        result = subprocess.run([str(binary), *args], env=env or environment,
                                text=True, capture_output=True, timeout=30)
        if result.returncode != expected:
            raise AssertionError(f"{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
        return result

    version = invoke("--version").stdout.strip()
    if "glyphs=249 reference=57 original=192" not in version:
        raise AssertionError("Compiled catalog is not the mixed SVG catalog")
    no_display = dict(environment)
    no_display.pop("DISPLAY", None)
    catalog = inspect_catalog(json.loads(invoke("--catalog", env=no_display).stdout))
    license_text = invoke("--license", env=no_display).stdout
    if "Copyright (c) 2018 Rezmason" not in license_text or "Permission is hereby granted" not in license_text:
        raise AssertionError("Compiled reference MIT notice is missing")
    sheet, source_sheet = out / "glyph-sheet.png", out / "source-glyph-sheet.png"
    invoke("--glyph-sheet", str(sheet), env=no_display)
    render_source_sheet(source_sheet_svg(), source_sheet)
    contours = compare_glyph_sheet(sheet, source_sheet)
    frames = {}
    selections = {}
    for name, count in (("preview-start", 1), ("preview-moving", 40), ("preview-repeat", 1)):
        path = out / f"{name}.png"
        rendered = invoke("--window", "--width", "640", "--height", "480", "--seed", "42",
                          "--frames", str(count), "--snapshot", str(path))
        selections[name] = selection_counts(rendered.stdout, expected_mix=0.1)
        frames[name] = inspect_frame(path)
        if (frames[name]["width"], frames[name]["height"]) != (640, 480):
            raise AssertionError("Standalone dimensions were not honored")
    if frames["preview-start"]["sha256"] == frames["preview-moving"]["sha256"]:
        raise AssertionError("Animated frames did not change")
    if frames["preview-start"]["sha256"] != frames["preview-repeat"]["sha256"]:
        raise AssertionError("Fixed seed is not deterministic")
    if not frames["preview-moving"]["mint_pixels"]:
        raise AssertionError("Animated frame has no mint leading glyphs")
    for name, mix in (("reference-only", "0"), ("original-only", "1")):
        path = out / f"{name}.png"
        rendered = invoke("--window", "--width", "640", "--height", "480", "--seed", "42",
                          "--mix", mix, "--frames", "1", "--snapshot", str(path))
        frames[name] = inspect_frame(path)
        selections[name] = selection_counts(rendered.stdout, expected_mix=float(mix))
    if frames["reference-only"]["sha256"] == frames["original-only"]["sha256"]:
        raise AssertionError("Reference and original family renderings are identical")

    invoke("--width", "0", expected=2)
    for value in ("-0.1", "1.1", "nan", "inf", "invalid", ""):
        invoke("--mix", value, expected=2)
    invoke("-display", ":987", "--frames", "1", expected=1)
    invoke("-window-id", "0x7ffffffe", "--frames", "1", expected=1)
    host = XHost()
    try:
        embedded = out / "embedded.png"
        invoke("-window-id", hex(host.window), "--seed", "42", "--frames", "3",
               "--snapshot", str(embedded))
        frames["embedded"] = inspect_frame(embedded)
        framebuffer = host.pixels()
        embedded_env = dict(environment, XSCREENSAVER_WINDOW=hex(host.window))
        invoke("-root", "--frames", "3", env=embedded_env)
        host.pixels()
        # A long-running embedded instance must follow host resizes and stop
        # without destroying the window owned by the screensaver manager.
        process = subprocess.Popen([str(binary), "-window-id", hex(host.window)],
                                   env=environment, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        try:
            ready, _, _ = select.select([process.stdout], [], [], 20)
            if not ready or "window=" not in process.stdout.readline():
                raise AssertionError("Renderer did not initialize before signal test")
            host.x.XResizeWindow(host.display, host.window, 480, 320)
            host.x.XSync(host.display, False)
            time.sleep(0.4)
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            if process.returncode != 0 or "width=480 height=320 stopped=1" not in stdout:
                raise AssertionError(f"Resize/signal failure: {process.returncode} {stdout} {stderr}")
            resized_pixels = host.pixels(480, 320)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
        for mode in ("fullscreen_escape", "preview_destroy"):
            option = "--fullscreen" if mode == "fullscreen_escape" else "--window"
            host.x.XSetInputFocus(host.display, host.window, 2, 0)
            host.x.XSync(host.display, False)
            process = subprocess.Popen([str(binary), option, "--width", "320", "--height", "240"],
                                       env=environment, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            try:
                ready, _, _ = select.select([process.stdout], [], [], 20)
                if not ready:
                    raise AssertionError(f"{mode} did not start")
                line = process.stdout.readline()
                if not line.startswith("window="):
                    raise AssertionError(f"{mode} has no target window: {line}")
                target = int(line.split()[0].split("=")[1], 16)
                if mode == "fullscreen_escape":
                    host.escape(target)
                else:
                    host.x.XDestroyWindow(host.display, target)
                    host.x.XSync(host.display, False)
                stdout, stderr = process.communicate(timeout=10)
                if process.returncode != 0:
                    raise AssertionError(f"{mode} failed: {stdout} {stderr}")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)
    finally:
        host.close()
    root_path = out / "root.png"
    invoke("-root", "--seed", "42", "--frames", "3", "--snapshot", str(root_path))
    frames["root"] = inspect_frame(root_path)
    receipt = {
        "status": "passed",
        "binary": binary.name,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "version": version,
        "checks": ["ELF", "mixed_249_catalog_hashes", "black_green_render", "animation",
                   "deterministic_seed", "window_id_embedding", "XSCREENSAVER_WINDOW",
                   "actual_X11_pixels", "resize", "SIGTERM_clean_exit", "root_render",
                   "fullscreen_focus_escape", "destroyed_preview_clean_exit",
                   "invalid_display", "invalid_window", "invalid_arguments",
                   "all_249_source_svg_silhouettes", "nonzero_fill_and_blank_slot",
                   "both_catalog_families_render", "weighted_90_10_selection",
                   "emerald_body_mint_heads", "embedded_MIT_notice"],
        "catalog": catalog,
        "contours": contours,
        "selections": selections,
        "frames": frames,
        "embedded_framebuffer": framebuffer,
        "resized_framebuffer": resized_pixels,
    }
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                           encoding="utf-8", newline="\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, nargs="?")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-sheet", type=Path,
                        help="Compare a compiled native atlas against source SVGs, without running a binary")
    parser.add_argument("--tile-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--margin", type=float, default=4)
    parser.add_argument("--white-ink", action="store_true")
    parser.add_argument("--minimum-iou", type=float, default=.99)
    arguments = parser.parse_args()
    if arguments.compare_sheet:
        arguments.out.mkdir(parents=True, exist_ok=True)
        source = arguments.out / "source-glyph-sheet.png"
        options = {"tile_size": arguments.tile_size, "margin": arguments.margin,
                   "white_ink": arguments.white_ink}
        svg = source_sheet_svg(**options)
        render_source_sheet(svg, source)
        receipt = compare_glyph_sheet(arguments.compare_sheet, source, **options,
                                      minimum_iou=arguments.minimum_iou)
        receipt["source_svg_sheet_sha256"] = hashlib.sha256(svg).hexdigest()
        (arguments.out / "contour-comparison.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({key: value for key, value in receipt.items() if key != "glyphs"}, indent=2))
    elif arguments.binary:
        receipt = run_smoke(arguments.binary, arguments.out)
        print(json.dumps({"status": receipt["status"], "checks": receipt["checks"],
                          "catalog": receipt["catalog"],
                          "minimum_contour_iou": receipt["contours"]["minimum_measured_iou"],
                          "frames": receipt["frames"], "selections": receipt["selections"]}, indent=2))
    else:
        parser.error("a compiled binary or --compare-sheet is required")


if __name__ == "__main__":
    main()
