# Execution and failure policies

Serial and parallel execution apply the same node failure policies. A terminal
failure stops new provider dispatch, preserves the original exception, and
retains completed results and recorded charges.

| Policy | After a provider error |
|---|---|
| `HALT` | Fail the node and stop the run immediately |
| `RETRY` | Make up to `max_retries` additional attempts; stop if all fail |
| `SKIP` | Mark the node skipped and continue eligible work |

A skipped node's dependents receive `[skipped: this step did not complete]` in
place of its result. The error text stays in the node's `result`, the trace,
and checkpoints for diagnosis, but it never enters a dependent's prompt or the
synthesized output. Resumed runs apply the same rule to saved skipped nodes.

Truncated output is a provider error. When a response stops at its output
limit (Anthropic `stop_reason` `max_tokens` or `model_context_window_exceeded`,
OpenAI `finish_reason` `length`, Gemini `finish_reason` `MAX_TOKENS`), the
executor records the call's charge and then raises `OutputTruncatedError`. The
node's failure policy applies, and no tool call from the truncated turn runs.
An `LLM_MERGE` synthesis raises the same error after recording its charge.
Raising the provider's `max_tokens` is the usual fix.

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

## Provider connections

Serial execution runs each node under its own `asyncio.run()`, as do
`Swarm.plan()`, `execute()`, and `resume()`. `AnthropicProvider`,
`OpenAIProvider`, `OpenAIImageProvider`, and `GeminiProvider` therefore keep
one SDK client per event loop. Calls on the same loop share its connection
pool, so a parallel run reuses connections across its fan-out. A client is
never used on another loop, and it closes on its own loop when that loop shuts
down (`asyncio.run()` does this before closing it).

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
