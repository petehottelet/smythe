# Classic reference artwork

These 56 visible glyphs are imported from
[m8e/matrix-rain](https://github.com/m8e/matrix-rain), a fork of
[Rezmason/matrix](https://github.com/Rezmason/matrix), pinned to
`5ba90490453ceceb6812d6b1bc658a99a92411d0`. This directory contains artwork, not upstream rendering code.

The original [classic atlas](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/svg%20sources/texture_simplified.svg)
is included as `texture_simplified.svg`. It has a 512×512 viewBox and an 8×8 grid
of 64×64 cells. Its 57-slot active sequence includes blank slot 4 and 56 visible
glyphs; slots 57–63 are unused. `BASE-000.svg` through `BASE-055.svg` preserve
the visible cells in row-major source order. Each cell is translated to the
origin and uniformly scaled by 100/64. Cubic curves, contour order, winding,
spacing, and handedness are retained. No glyph has been redrawn or fitted to
its ink bounding box. `contact-sheet.png` shows the full imported catalog.

## Credit and stated origins

The upstream repository carries the MIT License with
**Copyright (c) 2018 Rezmason**. Its exact pinned notice is included in
[LICENSE](LICENSE). The source artwork and normalized derivatives retain this
attribution. Smythe's importer is independently implemented. The adapted
renderer is documented separately in [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The upstream [Goals section](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/README.md#goals)
states that the classic vectors were cleaned from an archived SWF on the
official *The Matrix: Path of Neo* promotional website. It identifies
katakana-derived forms and characters from Susan Kare's Chicago typeface.
That stated history is recorded here rather than assigning Smythe authorship
to these shapes. The repository MIT notice does not separately establish
clearance of every underlying film or typeface element.

Only the classic atlas is imported. The Coptic, Gothic, Huberfish, and
*Resurrections* assets and their separate stated origins are outside this
import. Smythe's 192 original SVGs remain a separate, unchanged catalog.

## Reproduction

Run `python screensaver/import_reference_glyphs.py --check` from the repository
root. The check verifies pinned source and license hashes, all 84 contours,
all 56 visible cells, the blank slot, normalized SVGs, metadata, and browser data.
It needs the repository's `glyphs` or `dev` extra for the inspection PNG.
`provenance.json` records source hashes and exact transformation details.
