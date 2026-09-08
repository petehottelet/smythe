# Linux Glyph Rain

Native C/X11 screensaver with Cairo rendering. The executable embeds the
browser explorer's exact filled SVG contours: 56 reference shapes, their blank
slot, and 192 original Smythe glyphs. Each cell selects the reference catalog
90% of the time and the original catalog 10% of the time.

Compound paths retain their cubic curves and nonzero fill, including counters.
Bodies use HSL(137°, 80%, 50%) with layer attenuation; heads use mint `#A2FFD8`.
The native renderer retains its three depth layers and cached Cairo glow. It
does not implement the browser's REGL rain simulation, bloom, settings, or 3D
navigation. No Python or network connection is needed to run the executable.

Build on Debian/Ubuntu:

```sh
sudo apt-get install build-essential pkg-config libx11-dev libcairo2-dev
sh screensaver/linux/build_linux.sh
screensaver/dist/smythe-glyph-rain-linux-x86_64 --window
```

Runtime packages are `libx11-6` and `libcairo2`. The binary is architecture
specific; rebuild on another architecture. Precompiled binaries are not
distributed. The build script sets up the executable in `screensaver/dist/`.

Use `--fullscreen` for a standalone fullscreen preview. Escape closes the
preview; SIGTERM stops it cleanly. Window resizing rebuilds the three depth
layers. This renderer does not lock the session.

For XScreenSaver, add a program entry pointing to the installed executable:

```text
"Smythe Glyph Rain"  /absolute/path/smythe-glyph-rain-linux-x86_64 -root
```

XScreenSaver supplies `XSCREENSAVER_WINDOW`; the renderer paints that window
without owning or destroying it. `-window-id 0x12345` embeds into an explicit
preview window. `-root` without that environment variable paints the X11 root
window. Run only one renderer per target window.

This port requires an X11 display. A preview can run through XWayland on a
Wayland desktop; native Wayland lock-screen integration is not implemented.

Validate the compiled binary in a virtual X server:

```sh
sudo apt-get install xvfb xauth python3-pil librsvg2-2
xvfb-run -a python3 screensaver/linux/smoke_linux.py \
  screensaver/dist/smythe-glyph-rain-linux-x86_64 --out /tmp/smythe-linux-smoke
```

The smoke harness checks the compiled catalog hashes, all 249 contour slots,
the blank slot, both catalog families, the measured selection mix, body/head
colors, animation, embedding, resizing, and clean shutdown. Its independent
librsvg render of the published SVG sources must match every compiled
silhouette at an intersection-over-union of at least 0.99. It writes PNG frames
and a JSON receipt. Bounded rendering is also available directly:

```sh
./smythe-glyph-rain-linux-x86_64 --seed 42 --frames 40 --snapshot frame.png
```

Inspect the embedded catalog without an X server:

```sh
./smythe-glyph-rain-linux-x86_64 --catalog
./smythe-glyph-rain-linux-x86_64 --glyph-sheet glyph-sheet.png
./smythe-glyph-rain-linux-x86_64 --license
```

The diagnostic sheet uses 16 columns of 128px cells, a 4px inner margin, and
black silhouettes on white. `--mix 0` renders only reference glyphs;
`--mix 1` renders only originals. The normal default is `--mix 0.1`.

[Catalog provenance](https://github.com/petehottelet/smythe/blob/main/screensaver/native-catalog.json) binds the source SVGs and generated
native data. [Third-party notices](https://github.com/petehottelet/smythe/blob/main/screensaver/svg-preview/THIRD_PARTY_NOTICES.md) identify
the imported artwork and its license. Both MIT notices are embedded in the
executable. Retain their license notices when redistributing a local build.
