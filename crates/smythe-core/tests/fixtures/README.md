# Schema-3 compatibility fixtures

The four `schema3-*.json` files contain journals and a saved projection for
every prefix, produced with the runtime and transitions at
`a6f4ef31307425410cf5aa382163e1ab6ce60363`. Inputs and clock values are synthetic.

- `retry-revision`: failed attempt, explicit retry, successful revision.
- `raised-budget`: blocked admission, raised ceiling and completed resume.
- `failed-quote`: the original terminal quote failure with no attempts.
- `revision-crash-limit`: a crash replay at the original lifetime attempt
  limit after earlier completed work and two recorded revisions.

Compatibility tests compare all original fields at every prefix. The derived
`quote_blocked` and attempt `revision_base` fields are additive. Original
`local_replay` events lack a revision-base marker and retain their original
limit semantics; new events carry the checked marker. Do not regenerate these
expectations with the amended implementation.

`revision-attempt-limit.json` records the ChaCha8 seed that caught an actual
temporary mutation restoring the lifetime attempt guard in the coordinator.
The simulation replays it with per-revision limits and checks that revised
work dispatches beyond the old lifetime limit.
