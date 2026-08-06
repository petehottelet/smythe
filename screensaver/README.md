# Glyph Rain screensaver

A high-fan-out example of Smythe's general-purpose execution model, built from
the 192 original procedural cyber glyphs that the
[glyph screensaver benchmark](../benchmarks/glyph_screensaver_benchmark.md)
generates as one 192-node parallel Smythe run. The stroke programs, fall
speeds, and trail lengths in every port are the benchmark's generated
`GLYPH_SPECS` values, exported verbatim by [export_glyphs.py](export_glyphs.py).

The aesthetic is reference-inspired, not copied: original stroke-grammar
marks, a black field, luminous descending columns, and bright leading glyphs.
No font, logo, screenshot, or film frame is reproduced.

## Ports

| Port | Where | Run it |
|---|---|---|
| Web (this directory) | [index.html](index.html) + [glyphs.js](glyphs.js) | open `index.html` directly or deploy this static directory |
| Windows 11 (`.scr`) | source [windows/](windows/), binary [dist/SmytheGlyphRain.scr](dist/SmytheGlyphRain.scr) | download the `.scr`, right-click → **Install** |
| macOS 12+ (`.saver`) | source [macos/](macos/) | download the `GlyphRain-macos-saver` artifact from the [screensavers build](https://github.com/petehottelet/smythe/actions/workflows/screensavers.yml), or build on a Mac with `macos/build_macos.sh` |

All three implement the same simulation: three depth layers of overlapping
columns, persistence-fade trails, glowing white-green heads, and per-glyph
speeds and trail lengths from the generated catalog.

### Windows notes

`windows/build_windows.cmd` compiles `windows/GlyphRainSaver.cs` +
`windows/GlyphData.cs` with the C# compiler that ships inside Windows — no
SDK, no NuGet, no network. The committed binary in `dist/` is exactly that
build; rebuild it yourself to verify. Screensaver arguments `/s` (run),
`/p <hwnd>` (settings preview), and `/c` (about) are implemented; `/w` runs
in a window for debugging.

### macOS notes

`macos/build_macos.sh` builds a universal (arm64 + x86_64) `GlyphRain.saver`
with only the Xcode command-line tools and applies an ad-hoc signature.
Double-click to install, or copy to `~/Library/Screen Savers/`. The
`screensavers` workflow builds the same bundle on GitHub's macOS runners and
publishes it as a downloadable artifact.

## Web controls

- **Click / F** — toggle fullscreen
- **Space** — pause
- Cursor and badge auto-hide when idle; `prefers-reduced-motion` renders a
  static frame instead of animating.

## Regenerating the data

After any change to the glyph grammar in
`benchmarks/glyph_screensaver_assets.py`:

```bash
python screensaver/export_glyphs.py
```

writes `glyphs.js` (web), `windows/GlyphData.cs`, and `macos/glyphs.json`
from the same generated catalog.

## Deploy (web)

The directory is a self-contained static site:

```bash
cd screensaver
vercel deploy --prod
```

`glyph-rain-preview.png` (the Open Graph card) is the benchmark's assembled
1920×1080 preview, copied from the committed benchmark artifacts.
