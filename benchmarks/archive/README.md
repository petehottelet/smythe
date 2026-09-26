# Retired benchmark evidence

Superseded, diagnostic and historical records leave the tree once a newer
campaign replaces them. They stay available, byte for byte, in one archive
pinned by SHA-256:

| Archive | Size | SHA-256 |
|---|---|---|
| [smythe-evidence-2026-09-25.zip](https://github.com/petehottelet/smythe/releases/download/v0.9.0/smythe-evidence-2026-09-25.zip) | 28.3 MB | `28f9029d80939e4c85eefccf77822e367c25ba7987901ac203cfb6a83708ea73` |

The archive holds 115 files read from tag `v0.8.2` (commit
`c11bbe0f751b0c3b0514052fc207e1131da5850e`), under `files/` at their original
repository paths. Its `MANIFEST.json` lists each file's size and SHA-256.
[The pointer](evidence-2026-09-25.json) records the archive checksum, the
manifest checksum and every group below.

| Group | Status | Contents |
|---|---|---|
| First glyph fan-out campaign | Superseded | The August 2026 sweep at 64, 128, 192 and 256 nodes, its zero-latency run, one partial and three diagnostic live-provider runs, the 256-node partition's assembled assets, and their report. The [Noumenon sweep](../noumenon_benchmark.md) re-measured every width under the same protocol. |
| Browser renderer timing study | Diagnostic | The 7 September 2026 protocol, results, cadence control, verifier, test, timing records and frozen renderer source. All six primary sessions missed the pacing target. The renderer now lives in the [Noumenon repository](https://github.com/petehottelet/noumenon). |
| Superseded glyph partitions | Superseded | The stroke-based v1 partition and the renderer copy kept beside the v1 SVG partition. The v1 SVG catalog and its [workflow records](../svg_glyph_benchmark.md) remain in the tree. |
| Materials and catalog review records | Historical | The 13 September 2026 materials check, the catalog review record and the web explorer's browser checks. The [catalog](../noumenon/catalog/README.md), its measurements and contact sheets are unchanged. |
| Withdrawn native package receipts | Historical | Build information, checksums and rendering receipts for the precompiled screensaver packages withdrawn on 7 September 2026, and the 0.7.0 verification guide that listed them. |

## Verify

Download the archive, then run from a Smythe checkout with tags fetched:

```bash
python tools/evidence_archive.py verify smythe-evidence-2026-09-25.zip --against-git
```

The command checks the archive checksum against the pointer, every member
against the manifest and, with `--against-git`, every file against the
recorded commit. The same files remain in the repository history at that tag.
