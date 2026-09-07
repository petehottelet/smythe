#!/bin/bash
# Build GlyphRain.saver on macOS. Requires only the Xcode command-line tools.
#     screensaver/macos/build_macos.sh
# Outputs: screensaver/dist/GlyphRain.saver and GlyphRain-macos-universal.zip
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIST="$HERE/../dist"
BUNDLE="$DIST/GlyphRain.saver"
MIN="12.0"
ARCHIVE="$DIST/GlyphRain-macos-universal.zip"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Build this native bundle on macOS with the Xcode command-line tools." >&2
    exit 1
fi

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
lipo -verify_arch arm64 x86_64 "$BUNDLE/Contents/MacOS/GlyphRain"
rm -f "$DIST/GlyphRain-arm64" "$DIST/GlyphRain-x86_64"

cp "$HERE/Info.plist" "$BUNDLE/Contents/Info.plist"
cp "$HERE/glyphs.json" "$BUNDLE/Contents/Resources/glyphs.json"
chmod 755 "$BUNDLE/Contents/MacOS/GlyphRain"
plutil -lint "$BUNDLE/Contents/Info.plist"

# Bind bundle resources with an ad-hoc signature. Trusted downloaded
# distribution requires a Developer ID signature and notarization.
codesign --force --deep --sign - "$BUNDLE"
codesign --verify --strict --verbose=2 "$BUNDLE"

# Preserve the .saver hierarchy and executable modes across artifact downloads.
ditto -c -k --sequesterRsrc --keepParent "$BUNDLE" "$ARCHIVE"

echo "Built $BUNDLE"
echo "Packaged $ARCHIVE"
echo "Verify: bash $HERE/smoke_macos.sh"
echo "Install: double-click GlyphRain.saver, or copy it to ~/Library/Screen Savers/"
