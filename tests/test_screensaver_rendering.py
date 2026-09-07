"""Regression checks for the web and Windows screensaver renderers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_web_refresh_rate_does_not_change_rendered_trails(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to exercise the canvas simulation")
    script = tmp_path / "check.cjs"
    script.write_text(
        r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = process.argv[2];
const html = fs.readFileSync(root + '/screensaver/index.html', 'utf8');
const source = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1];
const catalog = fs.readFileSync(root + '/screensaver/glyphs.js', 'utf8');

function scene(reduced = false) {
  const events = {};
  let rafCalls = 0, cancelled = [];
  function canvas() {
    const context = {
      stamps: 0, rendered: [],
      fillRect() { this.rendered = []; },
      drawImage(sprite, x, y) {
        this.stamps++; this.rendered.push([sprite.id, x, y, this.globalAlpha]);
      },
      clearRect() { this.rendered = []; }, beginPath() {}, arc() {}, fill() {}, moveTo() {},
      lineTo() {}, quadraticCurveTo() {}, stroke() {}
    };
    return { getContext: () => context };
  }
  const surfaces = { far: canvas(), mid: canvas(), near: canvas() };
  const sandbox = {
    Math, innerWidth: 800, innerHeight: 600,
    window: { devicePixelRatio: 2 },
    matchMedia: () => ({ matches: reduced }),
    performance: { now: () => 0 },
    OffscreenCanvas: function() { return canvas(); },
    requestAnimationFrame: () => ++rafCalls,
    cancelAnimationFrame: handle => { cancelled.push(handle); },
    setTimeout: callback => { events.timeout = callback; }, clearTimeout() {},
    addEventListener: (name, callback) => { events[name] = callback; },
    document: {
      hidden: false,
      getElementById: id => surfaces[id],
      addEventListener: (name, callback) => { events[name] = callback; },
      body: { classList: { remove() {}, add() {} } }
    }
  };
  vm.createContext(sandbox);
  vm.runInContext(catalog + '\n' + source, sandbox);
  return { run: code => vm.runInContext(code, sandbox), surfaces, events,
           rafCalls: () => rafCalls, cancelled: () => cancelled };
}

function sample(fps) {
  const s = scene();
  return s.run(`(() => {
    const layer = LAYERS[0];
    layer.sprites.head.forEach((sprite, i) => sprite.id = 'head' + i);
    layer.sprites.trail.forEach((sprite, i) => sprite.id = 'trail' + i);
    layer.columns = [{ x: 30, y: 0, glyph: 0, phase: 0, rate: 4.25, acc: .37, burst: 0 }];
    for (let i = 0; i < ${fps}; i++) stepLayer(layer, 1 / ${fps});
    return { rendered: layer.ctx.rendered,
             y: layer.columns[0].y, phase: layer.columns[0].phase };
  })()`);
}
const slow = sample(30), fast = sample(144);
assert.equal(JSON.stringify(slow.rendered), JSON.stringify(fast.rendered));
assert.equal(slow.rendered.filter(glyph => glyph[0].startsWith('head')).length, 1);
assert.ok(slow.rendered.length > 1);
assert.equal(slow.y, fast.y);
assert.equal(slow.phase, fast.phase);

// Resizing a reduced-motion scene used to leave three blank canvases.
const still = scene(true);
assert.equal(still.rafCalls(), 0);
const before = still.surfaces.near.getContext().stamps;
still.events.resize();
still.events.timeout();
assert.ok(still.surfaces.near.getContext().stamps > before);
assert.equal(still.rafCalls(), 0);

// Pause preserves the scene; resume permits the next animation frame.
const active = scene();
active.events.keydown({ code: 'Space', preventDefault() {} });
assert.equal(active.run('paused'), true);
active.events.keydown({ code: 'Space', preventDefault() {} });
assert.equal(active.run('paused'), false);

// Hidden tabs stop requesting frames, then restart with a fresh time origin.
active.run('document.hidden = true');
active.events.visibilitychange();
assert.deepEqual(active.cancelled(), [1]);
assert.equal(active.run('frameHandle'), null);
active.run('document.hidden = false');
active.events.visibilitychange();
assert.equal(active.rafCalls(), 2);
console.log(JSON.stringify({ slow, fast, reducedMotionResize: true, pause: true,
                             visibility: true }));
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [node, str(script), str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert json.loads(result.stdout)["reducedMotionResize"] is True


@pytest.mark.skipif(os.name != "nt", reason="Windows GDI+ renderer")
def test_windows_preserves_stroke_widths_and_disposes_resized_scene(tmp_path):
    compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    if not compiler.is_file():
        pytest.skip("Windows .NET Framework compiler is unavailable")
    harness = tmp_path / "RenderCheck.cs"
    harness.write_text(
        r"""
using System;
using System.Collections;
using System.Drawing;
using System.Reflection;
using SmytheGlyphRain;

internal class RenderCheck
{
    private static int Ink(Bitmap image)
    {
        int count = 0;
        for (int y = 0; y < image.Height; y++)
            for (int x = 0; x < image.Width; x++)
                if (image.GetPixel(x, y).A > 128) count++;
        return count;
    }

    [STAThread]
    private static void Main()
    {
        float[][] original = GlyphData.Strokes[0];
        GlyphData.Strokes[0] = new float[][] { new float[] { 0, 20, 60, 80, 60, 4 } };
        int thin;
        using (Bitmap image = Sprites.Render(0, 140, 70, 1f, false)) { thin = Ink(image); }
        GlyphData.Strokes[0][0][5] = 16;
        using (Bitmap image = Sprites.Render(0, 140, 70, 1f, false))
            if (Ink(image) < thin * 2) throw new Exception("Authored widths were flattened");
        GlyphData.Strokes[0] = original;

        using (var form = new SaverForm(new Rectangle(0, 0, 200, 150), true))
        {
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
            var build = typeof(SaverForm).GetMethod("BuildScene", flags);
            build.Invoke(form, null);
            var layers = (IList)typeof(SaverForm).GetField("_layers", flags).GetValue(form);
            var sprites = (Bitmap[])typeof(Layer).GetField("_trail", flags).GetValue(layers[0]);
            Bitmap old = sprites[0];
            build.Invoke(form, null);
            bool disposed = false;
            try { old.GetPixel(0, 0); } catch (ArgumentException) { disposed = true; }
            if (!disposed) throw new Exception("Resize leaked the old sprite atlas");
        }
    }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "RenderCheck.exe"
    subprocess.run(
        [
            str(compiler), "/nologo", "/target:exe", "/main:RenderCheck",
            "/reference:System.dll", "/reference:System.Drawing.dll",
            "/reference:System.Windows.Forms.dll", f"/out:{executable}",
            str(ROOT / "screensaver/windows/GlyphRainSaver.cs"),
            str(ROOT / "screensaver/windows/GlyphData.cs"), str(harness),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=120)
