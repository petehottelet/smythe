---
name: Bug report
about: Report incorrect or unexpected behavior
title: "[bug] "
labels: bug
assignees: ''
---

## Summary

A clear, one-paragraph description of the bug.

## Reproduction

Minimal code that reproduces the issue. A failing test against the smythe
test suite is ideal.

```rust
# Smallest possible repro
```

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Include the full traceback if there is one.

```
# Paste traceback here
```

## Environment

- smythe version (or commit SHA if installed from source):
- Rust version (`rustc --version`):
- OS:
- Provider implementation (OfflineProvider, Noumenon or custom pure provider):
- Journal schema and fresh/resumed execution:

## Additional context

Anything else that might help — relevant logs, configuration snippets,
related issues, hypotheses.
