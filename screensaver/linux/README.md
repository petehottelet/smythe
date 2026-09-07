# Linux Glyph Rain

Native C/X11 screensaver with Cairo rendering. The executable contains the
same 192 vector glyphs as the web, Windows, and macOS ports; no Python or
network connection is needed to run it.

Build on Debian/Ubuntu:

```sh
sudo apt-get install build-essential pkg-config libx11-dev libcairo2-dev
sh screensaver/linux/build_linux.sh
screensaver/dist/smythe-glyph-rain-linux-x86_64 --window
```

Runtime packages are `libx11-6` and `libcairo2`. The binary is architecture
specific; rebuild on another architecture. The GitHub workflow builds the
x86-64 Linux download on Ubuntu 22.04. Make a downloaded binary executable
with `chmod +x smythe-glyph-rain-linux-x86_64`.

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
sudo apt-get install xvfb python3-pil
xvfb-run -a python3 screensaver/linux/smoke_linux.py \
  screensaver/dist/smythe-glyph-rain-linux-x86_64 --out /tmp/smythe-linux-smoke
```

The smoke harness checks the ELF signature, visible black/green frames,
animation, explicit and environment-based embedding, resize handling, invalid
display/window errors, and clean signal shutdown. It writes PNG frames and a
JSON receipt. Bounded rendering is also available directly:

```sh
./smythe-glyph-rain-linux-x86_64 --seed 42 --frames 40 --snapshot frame.png
```
