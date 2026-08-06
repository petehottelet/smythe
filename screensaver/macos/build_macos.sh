#!/bin/bash
# Build GlyphRain.saver on macOS. Requires only the Xcode command-line tools.
#     screensaver/macos/build_macos.sh
# Output: screensaver/dist/GlyphRain.saver
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIST="$HERE/../dist"
BUNDLE="$DIST/GlyphRain.saver"
MIN="12.0"

rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"

build_slice() {
    local arch="$1"
    xcrun swiftc -O -parse-as-library \
        -target "$arch-apple-macos$MIN" \
        -emit-library -module-name GlyphRain \
        -framework AppKit -framework ScreenSaver \
        -o "$DIST/GlyphRain-$arch" \
        "$HERE/GlyphRainView.swift"
}

build_slice arm64
build_slice x86_64
lipo -create -output "$BUNDLE/Contents/MacOS/GlyphRain" \
    "$DIST/GlyphRain-arm64" "$DIST/GlyphRain-x86_64"
rm -f "$DIST/GlyphRain-arm64" "$DIST/GlyphRain-x86_64"

cp "$HERE/Info.plist" "$BUNDLE/Contents/Info.plist"
cp "$HERE/glyphs.json" "$BUNDLE/Contents/Resources/glyphs.json"

# Ad-hoc signature so Gatekeeper loads the bundle locally. Distributing
# outside the repo checkout still benefits from a real Developer ID.
codesign --force --deep --sign - "$BUNDLE"

echo "Built $BUNDLE"
echo "Install: double-click GlyphRain.saver, or copy it to ~/Library/Screen Savers/"
