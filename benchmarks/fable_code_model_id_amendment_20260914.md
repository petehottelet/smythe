# Code Workflow model-identifier amendment

The Code pilot and initial main tasks repeatedly generated child-agent model
names such as `fable-5.1` and `Fable 5.1`. Claude Code rejected those names and
relaunched the workflows. The gateway admitted paid requests only for the exact
`claude-fable-5-1` model, so no other provider model was billed.

This exposed an avoidable ambiguity in the Code prompt: it said to use
“Fable 5.1,” while the native adapter always supplied the exact API identifier.
The initial Code main batch is diagnostic. Its attempts, outputs, recovery
events, and charges remain recorded and are excluded from the primary Code
comparison. The native 12-run pilot and 100-position schedule are unchanged.

Before the revised Code schedule begins, replace that instruction with:

> Use the exact model identifier `claude-fable-5-1` for every agent. Do not use a display name or shorthand model alias.

Keep the same ten tasks, task order, one repetition, medium effort, Claude Code
and SDK versions, tool restrictions, output limit, accounting gateway, and
per-workflow allowance. Run the entire ten-task Code schedule under this
declared instruction. Do not select favorable rows from the earlier batch.

All earlier Code main charges count against the existing $15 Code main
allocation. Before each new workflow, reserve its full $5 ceiling. Stop when
that reservation no longer fits or new billing becomes unresolved. No extra
budget or transfer between stages is introduced.

Retain the exact original and amended prompts and runtime sources with their
respective evidence. Report observed Workflow launches, automatic recovery,
output checks, quality scores and all paid calls for each cohort. The results
remain a separate comparison with Claude Code's Workflow runtime.

See the [Fable protocol](fable_51_benchmark_plan.md) and
[completed pilot evidence](results/fable_20260914_pilot/README.md).
