# Glyph Rain screensaver

192 procedural glyphs, generated in a parallel Smythe run, descend through
three depth planes. Bold angular strokes, luminous green bodies, and varied
bloom give each stream a distinct weight against black space.

Every port uses the [glyph benchmark's](../benchmarks/glyph_screensaver_benchmark.md)
original stroke programs, speeds, and trail lengths. The renderer draws each
bounded trail from cached sprites, keeping glyphs sharp as the code falls.
The display changes preserve the benchmark catalog and its historical results.

## Ports

| Port | Where | Run it |
|---|---|---|
| Web (this directory) | [index.html](index.html) + [glyphs.js](glyphs.js) | open `index.html` directly or deploy this static directory |
| Windows 11 (`.scr`) | source [windows/](windows/), binary [dist/SmytheGlyphRain.scr](dist/SmytheGlyphRain.scr) | download the `.scr`, right-click → **Install** |
| macOS 12+ (`.saver`) | source [macos/](macos/), [universal ZIP](dist/GlyphRain-macos-universal.zip) | unzip, then double-click `GlyphRain.saver`; build locally with `macos/build_macos.sh` |
| Linux x86-64 / X11 | source and setup [linux/](linux/README.md), [compiled archive](dist/SmytheGlyphRain-linux-x86_64.tar.gz) | extract and run `./smythe-glyph-rain-linux-x86_64 --window`; build with `sh screensaver/linux/build_linux.sh` |

The ports share layer sizes, column spacing, stroke weights, colors, and
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
