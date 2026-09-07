"""Exercise an actual Linux ELF renderer against an X11 display (use Xvfb)."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

from PIL import Image


def inspect_frame(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        pixels = list(rgb.getdata())
        black = sum(max(pixel) < 12 for pixel in pixels)
        green = sum(g > 20 and g > r * 1.15 and g > b * 1.15 for r, g, b in pixels)
        total = len(pixels)
        if black / total < 0.35 or green / total < 0.015:
            raise AssertionError(f"Frame lacks black field or green glyphs: {path}")
        return {
            "width": image.width,
            "height": image.height,
            "black_fraction": round(black / total, 5),
            "green_fraction": round(green / total, 5),
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
    if "glyphs=192" not in version:
        raise AssertionError("Compiled catalog is not the 192-glyph shared catalog")
    frames = {}
    for name, count in (("preview-start", 1), ("preview-moving", 40), ("preview-repeat", 1)):
        path = out / f"{name}.png"
        invoke("--window", "--width", "640", "--height", "480", "--seed", "42",
               "--frames", str(count), "--snapshot", str(path))
        frames[name] = inspect_frame(path)
        if (frames[name]["width"], frames[name]["height"]) != (640, 480):
            raise AssertionError("Standalone dimensions were not honored")
    if frames["preview-start"]["sha256"] == frames["preview-moving"]["sha256"]:
        raise AssertionError("Animated frames did not change")
    if frames["preview-start"]["sha256"] != frames["preview-repeat"]["sha256"]:
        raise AssertionError("Fixed seed is not deterministic")

    invoke("--width", "0", expected=2)
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
        "checks": ["ELF", "shared_192_catalog", "black_green_render", "animation",
                   "deterministic_seed", "window_id_embedding", "XSCREENSAVER_WINDOW",
                   "actual_X11_pixels", "resize", "SIGTERM_clean_exit", "root_render",
                   "fullscreen_focus_escape", "destroyed_preview_clean_exit",
                   "invalid_display", "invalid_window", "invalid_arguments"],
        "frames": frames,
        "embedded_framebuffer": framebuffer,
        "resized_framebuffer": resized_pixels,
    }
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_smoke(arguments.binary, arguments.out), indent=2))


if __name__ == "__main__":
    main()
