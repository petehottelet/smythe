# Windows Glyph Rain Defender notice — 7 September 2026

**Status: unresolved; Windows installation and execution are paused pending
review.** Microsoft Defender detected the native Windows `SmytheGlyphRain.scr`
distributed with Smythe v0.7.0 as `Trojan:Win32/Wacatac.H!ml` on 7 September
2026. The downloaded copy and the repository distribution copy were
quarantined. Keep those files quarantined; do not restore them, add Defender
exclusions, or disable protection to run this release.

## Affected artifact and evidence

- Release: Smythe v0.7.0; native Windows version `1.1.0.0`.
- Artifact: `SmytheGlyphRain.scr`.
- Recorded SHA-256:
  `c7660ee87d6d0e7774b1ee0c044d66dabf8cd92ebd8b8d193a55e6621ac177ea`.
- Public release metadata reports the same SHA-256 as the committed
  [build record](../screensaver/dist/BUILD_INFO.json). This confirms metadata
  consistency; it does not independently verify the quarantined copies or
  resolve the detection.
- Defender records a real-time file detection (`FastPath`) and currently
  reports `IsActive=false` and `DidThreatExecute=false`. These fields do not
  prove that no earlier execution occurred.
- Source inspection found no process launcher, network downloader, startup
  persistence, or dynamic code loader. Byte-level inspection matched all 249
  glyph-loading methods and their geometry strings to the source. Managed
  references and the five native window-control imports match the expected
  drawing and preview functions. Full renderer IL equivalence was not verified.
  These findings support investigation; they do not settle the classification.
- Existing native rendering and host checks establish their recorded
  functional results. The Windows CI checks included neither signing nor an
  explicit antivirus gate; they do not constitute malware clearance.
- No Microsoft review determination has been obtained for this investigation.
  The cause remains unresolved; it has not been established as a false positive.

This observation concerns the native Windows screensaver. It does not
establish a detection in the Python package, web explorer, macOS bundle, or
Linux archive, and it does not provide a security assessment of those
artifacts. Historical build and rendering receipts remain available with
their original scope.

## Review and next steps

1. Preserve the Defender detection record and artifact identity while leaving
   affected files quarantined.
2. Submit the affected artifact and source/build provenance to Microsoft for
   analysis through an appropriate isolated review process. Record the
   submission identity, outcome, and any changed detection status here.
3. Investigate the compiled artifact and build provenance. If changes are
   required, publish a separately identified replacement with its own hashes
   and verification records; do not relabel the existing artifact as cleared.
4. Add an explicit antivirus gate for exact Windows release candidates before
   native execution and publication, recording the engine and intelligence
   versions, file hash, scan outcome, and any analyst determination. A scan
   failure or unavailable scanner must prevent publication.
5. Update the Windows download and installation guidance only after the
   review establishes an evidence-backed resolution.

The [screensaver guide](../screensaver/README.md) and
[roadmap](../ROADMAP.md#coming-soon) track the distribution status. This notice
does not change the glyph-generation or renderer benchmark records.
