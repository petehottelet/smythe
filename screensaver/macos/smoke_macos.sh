#!/bin/bash
# Compile a native smoke host, load the universal saver, and save receipts.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIST="$HERE/../dist"
BUNDLE="${1:-$DIST/GlyphRain.saver}"
RECEIPTS="${2:-$DIST/macos-smoke}"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "The native ScreenSaverView smoke test requires macOS." >&2
    exit 1
fi
mkdir -p "$DIST" "$RECEIPTS"
xcrun clang -fobjc-arc -Wall -Wextra \
    -framework AppKit -framework ScreenSaver \
    "$HERE/smoke_macos.m" -o "$DIST/GlyphRain-macos-smoke"
"$DIST/GlyphRain-macos-smoke" "$BUNDLE" "$RECEIPTS"
