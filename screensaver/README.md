# Glyph Rain screensaver

192 procedural glyphs, generated in a parallel Smythe run, descend through
three depth planes. Bold angular strokes, luminous green bodies, and varied
bloom give each stream a distinct weight against black space.

The native ports and original web view use the [glyph benchmark's](../benchmarks/glyph_screensaver_benchmark.md)
original stroke programs, speeds, and trail lengths. The renderer draws each
bounded trail from cached sprites, keeping glyphs sharp as the code falls.

The separate [web explorer](svg-preview/README.md) adapts the MIT-licensed
rain, bloom, and palette renderer from m8e/Rezmason. It combines the reference's
56 visible base glyphs and blank slot with Smythe's 192 original SVGs, using
the originals for 10% of selections by default. Its
[renderer interaction checkpoint](../benchmarks/partitions/glyph_rain_reference_v1/pixel-preview-review.json)
passed before the final borderless styling; rendering performance remains unmeasured.
The native downloads below retain the earlier stroke
catalog and renderer.

## Reference-based web explorer

Classic uses the reference's fixed 2D grid. The 3D preset adds arrow-key travel;
Operator is a separate visual preset. S opens settings, Space pauses playback,
R resets the viewpoint, and F toggles fullscreen. Settings edit a draft and
apply together; the URL retains the selected configuration.
Simple VT323 pixel controls and a Trajan Bold outline logo sit within generous black
padding. The interface uses Hottelet green `#37FF6E` and bright `#9CFFBC`, with
borderless text controls and no glow or corner ornaments. The rain retains its 137°
Matrix green grade and mint `#A2FFD8` highlights; Reference body colors remain
selectable.

The renderer and reference artwork are licensed imports. Smythe's original
192-glyph catalog remains an independently generated artifact, with separate
hashes and acceptance records.

[Complete 192-glyph contact sheet](../benchmarks/partitions/glyph_svg_v1/catalog/contact-sheet.png) ·
[24-glyph calibration sheet](../benchmarks/partitions/glyph_svg_v1/catalog/calibration-sheet.png) ·
[Individual SVGs and manifest](../benchmarks/partitions/glyph_svg_v1/catalog/) ·
[Run the explorer](svg-preview/README.md).

The [new benchmark](../benchmarks/svg_glyph_benchmark.md) measures fresh original
SVG geometry, style validation, duplicate detection, and catalog assembly
through Smythe. Its 4.03-second median workflow result excludes imported base
artwork and browser rendering. The adapted renderer has not been timed; the
[previous renderer's records](../benchmarks/partitions/glyph_svg_v1/renderer-v1/)
remain superseded diagnostics.

## From goal to screensaver

The workflow generates each tile independently, validates dimensions and
uniqueness, and assembles the accepted artifacts:

![Glyph generation workflow](../assets/glyph_rain/glyph_pipeline.svg)

Twelve specimens from the native ports' 192-glyph stroke catalog:

![Original glyph specimens](../assets/glyph_rain/glyph_specimens.svg)

## Ports

| Port | Where | Run it |
|---|---|---|
| Legacy web (this directory) | [index.html](index.html) + [glyphs.js](glyphs.js) | open `index.html` directly or deploy this static directory |
| Reference-based web explorer | [svg-preview/](svg-preview/README.md) | serve the repository locally; Classic, 3D, and Operator presets; browser interaction checks pass |
| Windows 11 (`.scr`) | source [windows/](windows/), binary [dist/SmytheGlyphRain.scr](dist/SmytheGlyphRain.scr) | download the `.scr`, right-click → **Install** |
| macOS 12+ (`.saver`) | source [macos/](macos/), [universal ZIP](dist/GlyphRain-macos-universal.zip) | unzip, then double-click `GlyphRain.saver`; build locally with `macos/build_macos.sh` |
| Linux x86-64 / X11 | source and setup [linux/](linux/README.md), [compiled archive](dist/SmytheGlyphRain-linux-x86_64.tar.gz) | extract and run `./smythe-glyph-rain-linux-x86_64 --window`; build with `sh screensaver/linux/build_linux.sh` |

The native ports and legacy web view share layer sizes, column spacing, stroke weights, colors, and
motion rules. Foreground glyphs are larger and brighter; distant streams are
finer and slower. Trail brightness depends on position within the stream,
so display refresh rate does not accumulate glow or leave faded ghost columns.

### Windows notes

`windows/build_windows.cmd` compiles `windows/GlyphRainSaver.cs` +
`windows/GlyphData.cs` with the C# compiler that ships inside Windows — no
SDK, no NuGet, no network. The committed binary in `dist/` comes from that
build in CI and passed native checks there and on Windows 11. Screensaver arguments `/s` (run),
`/p <hwnd>` (settings preview), and `/c` (about) are implemented; `/w` runs
in a window for debugging.

### macOS notes

`macos/build_macos.sh` builds a universal (arm64 + x86_64) `GlyphRain.saver`
with only the Xcode command-line tools and applies an ad-hoc signature.
Double-click to install, or copy to `~/Library/Screen Savers/`. The
`screensavers` workflow builds the universal bundle and tests it on Apple
Silicon and Intel. The committed ZIP preserves the bundle and executable permissions.

The download is an archive containing the `.saver` bundle. It is ad-hoc signed;
Developer ID signing and notarization are planned. Build locally if macOS
blocks installation of the downloaded bundle.

### Linux notes

The native ELF executable uses X11 and Cairo. Extract the archive, then run
`./smythe-glyph-rain-linux-x86_64 --window` or configure it in XScreenSaver.
The download targets x86-64 and Ubuntu 22.04-compatible system libraries.
An XWayland preview is supported through X11; native Wayland screensaver and
lock-screen integration is planned. [Dependencies and integration](linux/README.md).

## Native verification

The [published build](https://github.com/petehottelet/smythe/actions/runs/34093505361)
passed all five native jobs. Downloads in `dist/` are the artifacts from that run.
[SHA-256 checksums](dist/SHA256SUMS) and [build provenance](dist/BUILD_INFO.json)
identify the source commit, packages, and individual [verification receipts](dist/verification/).

The [build workflow](../.github/workflows/screensavers.yml) validates compiled
artifacts before uploading them:

- Windows loads the compiled `.scr`, checks rendering, motion and resize, and
  launches its `/p` preview process inside a hidden host window through clean exit.
- macOS loads the same universal bundle on Apple Silicon and Intel, then checks
  preview/fullscreen rendering, motion, resize, and stop behavior.
- Linux checks the same ELF on Ubuntu 22.04 and 24.04 under Xvfb, including
  visible frames, animation, embedding, resizing, invalid input, and shutdown.

Run the local checks with `windows/smoke_windows.ps1`,
`bash macos/smoke_macos.sh`, or the [Linux smoke command](linux/README.md).
Each check produces a rendering receipt. These checks validate native execution;
OS installation policy and session locking remain separate concerns.
Windows `/s` multi-monitor dispatch is not covered by these checks.

## Legacy web controls

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

writes `glyphs.js` (web), `windows/GlyphData.cs`, `macos/glyphs.json`, and
`linux/glyph_data.h` from the same generated catalog.

## Deploy (web)

The directory is a self-contained static site:

```bash
cd screensaver
vercel deploy --prod
```

`glyph-rain-preview.png` is a 1920×1080 capture of the web renderer. Benchmark
artifacts retain their original renderings and hash-bound receipts.

## Credits and references

Glyph Rain's current web explorer adapts the renderer and classic artwork from
[m8e/matrix-rain](https://github.com/m8e/matrix-rain), a fork of
[Rezmason/matrix](https://github.com/Rezmason/matrix). Credit to Rezmason and
the project's contributors for the reference implementation of glyph
presentation, traveling illumination, green palettes, bloom, and depth.

The imported source is pinned to [revision 5ba9049](https://github.com/m8e/matrix-rain/tree/5ba90490453ceceb6812d6b1bc658a99a92411d0).
Its [MIT license](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/LICENSE)
credits **Copyright (c) 2018 Rezmason**. Copies of the notice accompany the
[engine](svg-preview/engine/LICENSE) and [base artwork](svg-preview/reference/LICENSE).
The [artwork provenance](svg-preview/reference/provenance.json) identifies the
source atlas and extracted outlines. The REGL rain, bloom, and palette passes
are adapted licensed code; the additional 192 Smythe glyphs are independently
authored and retain their own generation receipts.

The [style and implementation plan](../docs/glyph-rain-plan.md) records the
reference measurements and the specification for new original SVG glyphs.
