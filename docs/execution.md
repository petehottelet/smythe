# Execution and failure policies

Serial and parallel execution apply the same node failure policies. A terminal
failure stops new provider dispatch, preserves the original exception, and
retains completed results and recorded charges.

| Policy | After a provider error |
|---|---|
| `HALT` | Fail the node and stop the run immediately |
| `RETRY` | Make up to `max_retries` additional attempts; stop if all fail |
| `SKIP` | Mark the node skipped and continue eligible work |

A node timeout follows its failure policy. Invalid accounting, budget
admission or reconciliation failures, and failures persisting a billed result
are terminal regardless of that policy. See [Cost guardrails](budgets.md).

In serial execution, every later node remains pending after a terminal
failure, including independent siblings and descendants with `SKIP` policies.
A queued node's policy does not override an earlier node's halt.

In parallel execution, calls already in flight can finish before a failure is
observed. The executor cancels and awaits the remaining active tasks, settles
completed work, and starts no queued sibling after observing the failure.
Cancellation does not establish that a remote provider issued no charge.

An explicit [resume](checkpoint-format.md#resume-semantics) preserves completed
nodes and their costs, resets failed or interrupted nodes, and continues the
pending graph. Invalid-accounting markers require reconciliation first.
Unfinished [verification transitions](verifier.md#recovery-and-concurrent-work)
are recovered before new work or cached-output return.

## Deep graphs

Validation, cycle detection, dependency ordering, and depth calculation use
iterative traversal. A chain does not depend on Python's recursion limit or
the order in which its nodes appear. Serial execution preserves depth-first
dependency order, including the declared order of sibling dependencies.

Regression tests validate 5,000-node chains in forward, reverse, and shuffled
order, check a 5,000-node fork/join graph, and execute all 5,000 steps of a
reverse-ordered chain with an offline provider. Revision tests cover deep
rewiring and reject cycles or missing dependencies before changing the graph.
Reading `depth` on a cyclic graph raises `ValueError`.

These are correctness checks, not throughput measurements. Serial execution
still rechecks pending work after each step so supervision and verification
can change the remaining graph. Flat artifact fan-out has a separate
[Jobs scale and recovery protocol](../benchmarks/jobs_scale_benchmark.md).

[YAML example](../examples/01_pipeline.yaml) ·
[Crash and resume example](../examples/04_resume_after_crash.py) ·
[Architecture](architecture.md).
