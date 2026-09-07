# Native contour verification

The Windows, Apple Silicon, and Intel Mac atlases from
[CI run 34123023804](https://github.com/petehottelet/smythe/actions/runs/34123023804)
are compared with an independent librsvg render of the published SVGs. The
compiled source is `a0327aa58f80e6c7206f5c3eeeae1b48985953e6`.
All 248 visible shapes and the intentional blank slot are present.

| Measurement | [Windows](windows-contour-diagnostic.json) | [Mac arm64](macos-arm64-contour-diagnostic.json) | [Mac x86-64](macos-x86_64-contour-diagnostic.json) |
|---|---:|---:|---:|
| Minimum raw silhouette IoU | 0.949748744 | 0.891666667 | 0.891666667 |
| Mean raw IoU, all 249 slots | 0.993977965 | 0.982755825 | 0.982755825 |
| Visible glyphs below 0.99 IoU | 31 | 163 | 163 |
| Maximum symmetric foreground / boundary distance | 1px / 1px | 1px / 1px | 1px / 1px |
| Pixels farther than 1px, either direction | 0 | 0 | 0 |
| Maximum bounding-box edge difference | 1px | 1px | 1px |
| Components, actual / source | 336 / 336 | 336 / 336 | 336 / 336 |
| Counters ≥4px, actual / source | 84 / 84 | 84 / 84 | 84 / 84 |
| Counters of any size, actual / source | 86 / 86 | 84 / 86 | 84 / 86 |

macOS loses two one-pixel counters at this resolution: slots 185 (`GLYPH-128`)
and 207 (`GLYPH-150`). The lowest Mac IoU, slot 25 (`BASE-024`), has identical
bounds in both images. Its 52 differing pixels have intensity 128 in the source
and 109 on Mac. Thresholding at 128 amplifies this fractional edge difference.
The measurements are consistent with antialiasing coverage differences; they
do not establish pixel-identical rendering or identical subpixel placement.

## Method and scope

Inputs are unscaled 1024×1024 images: 16 columns of 64px cells, zero inner
margin, white ink on opaque black. No shift, scale, rotation, flip, or alignment
correction is applied. Foreground is grayscale ≥128. For each foreground pixel
center, measure its Euclidean distance to foreground in the other image, then
repeat in reverse. Report the larger maximum. Boundary distances use the same
procedure on eight-neighbor foreground boundaries. Horizontal or vertical
adjacent pixels are within 1px; diagonal separation of √2px is outside it.

Components use eight-connected foreground. Counters are four-connected
background regions enclosed within the cell. Raw counts and counts after an
explicit four-pixel area cutoff are both retained. Masks are not altered to
improve agreement. **These are spatial diagnostics.** The raw 0.99 IoU failures
remain failures; Linux's existing 128px source-silhouette gate stays at 0.99.

## Artifacts and reproduction

[Source oracle](catalog-source-64.png) · [Windows atlas](windows-atlas.png) ·
[Mac arm64 atlas](macos-arm64-atlas.png) · [Mac x86-64 atlas](macos-x86_64-atlas.png).
The records bind these bytes, the [native catalog](../../native-catalog.json),
the [analysis script](../../verification/analyze_native_masks.py), published
execution receipts, and original downloaded CI receipt hashes. Mac atlas bytes
are identical across architectures. The oracle was produced by the
[librsvg source-sheet helper](../../linux/smoke_linux.py).

From the repository root, install `pip install -e ".[glyphs]"`, then run:

```sh
python screensaver/verification/analyze_native_masks.py \
  --actual screensaver/dist/verification/windows-atlas.png \
  --source screensaver/dist/verification/catalog-source-64.png \
  --artifact-receipt screensaver/dist/verification/windows.json \
  --out smythe_artifacts/windows-contour-recheck.json \
  --label windows
```

For either Mac architecture, substitute `macos-arm64` or `macos-x86_64` for
`windows`. Existing receipts require explicit `--overwrite`. Reproduction
preserves the per-glyph measurements; environment and script metadata identify
the new analysis run.
