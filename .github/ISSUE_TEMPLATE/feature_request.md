---
name: Feature request
about: Propose a new capability or enhancement
title: "[feature] "
labels: enhancement
assignees: ''
---

## Problem

What problem are you trying to solve? What use case is currently awkward
or impossible? Include the intended operator (library user, autonomous agent,
benchmark author, or framework integrator) and the scale of the workload.

## Proposed solution

What you'd like to see. Sketch the API or behavior if you have one in mind.

```python
# Optional: sketch of the proposed API
```

## Alternatives considered

Other approaches you thought about, and why this one is preferred.

## Success criteria

How would we know the change works? Prefer observable acceptance criteria such
as throughput, cost completeness, recovery behavior, artifact validity, or a
small reproducible benchmark. Separate measured results from estimates.

## Safety and operational impact

- Could this execute tools, persist artifacts/checkpoints, or expose secrets?
- Could it increase provider spend, fan-out, retries, or concurrency?
- What should happen after cancellation, a partial failure, or resume?
- Can the behavior be exercised offline in tests or a benchmark smoke run?

## Scope and breaking changes

- Does this change the public API? If so, how?
- Are there backward-compatible incremental steps?
- Roughly how big is this — a single function, a new module, or a multi-sprint
  initiative?
- Does it overlap an item in [ROADMAP.md](../../ROADMAP.md)?

## Additional context

Links to relevant issues, prior art in other frameworks, benchmark records, or
specifications. Remove credentials, private prompts, and sensitive generated
artifacts before attaching logs.
