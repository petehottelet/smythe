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
- [ ] Documentation only
- [ ] Test / CI / tooling only

## Testing

How did you verify this change?

- [ ] Added new tests covering this change
- [ ] Existing tests cover this change
- [ ] Manually exercised — describe how:

```
# Paste relevant test output, or describe the manual run
```

## Checklist

- [ ] `ruff check smythe/ tests/` passes locally
- [ ] `pytest tests/` passes locally on at least one supported Python version
- [ ] [CHANGELOG.md](../CHANGELOG.md) updated under `[Unreleased]` for any
      user-facing change
- [ ] Public API additions/changes are documented in docstrings and
      [Readme.md](../Readme.md) where appropriate
- [ ] No "created with X" attribution in commit messages
- [ ] No emojis in code or commit messages (unless explicitly requested)

## Breaking changes

If this is a breaking change, describe the migration path users should follow.
