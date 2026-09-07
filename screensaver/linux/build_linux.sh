#!/bin/sh
# Native X11 ELF executable; Debian/Ubuntu: build-essential pkg-config libx11-dev libcairo2-dev.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT=${1:-"$HERE/../dist/smythe-glyph-rain-linux-x86_64"}
if [ "$(uname -m)" != x86_64 ] && [ "$#" -eq 0 ]; then
    OUT="$HERE/../dist/smythe-glyph-rain-linux-$(uname -m)"
fi
if ! pkg-config --exists x11 cairo; then
    echo 'Install X11 and Cairo development packages (libx11-dev libcairo2-dev).' >&2
    exit 1
fi
mkdir -p "$(dirname -- "$OUT")"
${CC:-cc} -std=c11 -O2 -Wall -Wextra -Werror -Wpedantic \
    "$HERE/glyph_rain.c" -o "$OUT" $(pkg-config --cflags --libs x11 cairo) -lm
"$OUT" --version
printf 'Built %s\n' "$OUT"
