<!-- Thanks for contributing to smythe! Please fill out the sections below. -->

## Summary

What does this PR change, and why? One or two sentences focused on the "why"
rather than the "what".

## Linked issue

Closes #<issue-number> <!-- or "Refs #..." if not closing -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (API or behavior change — call out the migration path)
- [ ] Performance / benchmark change
- [ ] Security / reliability hardening
- [ ] Documentation only
- [ ] Test / CI / tooling only

## Testing

How did you verify this change?

- [ ] Added new tests covering this change
- [ ] Existing tests cover this change
- [ ] Manually exercised — describe how:
- [ ] Benchmark evidence added or updated — describe protocol and repetitions:

```
# Paste relevant test output, or describe the manual run
```

## Checklist

- [ ] `ruff check smythe/ tests/ benchmarks/ examples/` passes locally
- [ ] `pytest tests/` passes locally on at least one supported Python version
- [ ] [CHANGELOG.md](../CHANGELOG.md) updated under `[Unreleased]` for any
      user-facing change
- [ ] Public API additions/changes are documented in docstrings and
      [README.md](../README.md) where appropriate
- [ ] New provider calls have conservative budget admission and preserve
      incurred cost on post-billing failures
- [ ] Parallel or resumable behavior has cancellation, partial-failure, and
      duplicate-side-effect semantics documented and tested
- [ ] Benchmark claims identify the protocol, environment, raw records, and
      important confounds; estimates are not presented as measurements
- [ ] Logs, fixtures, traces, checkpoints, and artifacts contain no credentials
      or private customer data
- [ ] No "created with X" attribution in commit messages
- [ ] No emojis in code or commit messages (unless explicitly requested)

## Breaking changes

If this is a breaking change, describe the migration path users should follow.
