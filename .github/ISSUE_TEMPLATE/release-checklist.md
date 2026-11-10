---
name: Release checklist
about: Verify a Rust source or binary release
title: "Release vX.Y.Z"
labels: roadmap
---

# Release checklist

Follow [RELEASING.md](https://github.com/petehottelet/smythe/blob/main/RELEASING.md).

- [ ] Workspace versions and tested commit identified.
- [ ] Rust format, Clippy, full offline suite, release build and both performance gates pass.
- [ ] Linux and Windows CI pass for the exact commit.
- [ ] All 256 Noumenon glyphs validate and completed recovery reuses the results.
- [ ] Source and binary provenance are retained; benchmark claims match eligible records.
- [ ] README, guides, roadmap, examples and changelog agree.
- [ ] Released changelog sections and existing tags remain unchanged.
- [ ] No credentials, private notes or local outputs enter artifacts.
- [ ] Publication is explicitly authorized, with the supplied release title.
- [ ] Exact source/binary artifacts have verified hashes and usable download locations.

The active Rust preview has no PyPI publication workflow. The historical
Python `v0.2.0` tag and Rust `rust-v0.2.0` tag identify different immutable
revisions. Stable Rust 1.0.0 remains ahead.
