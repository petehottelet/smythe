# Astra study amendment — 13 September 2026

Human calibration passed: five real pilot answers received 4/4, and the
planted-error control received 1/4. The submitted receipt is bound to the six
anonymous samples. The original quality and success gates remain unchanged.

The first main schedule stopped after 31 complete workflows. Its native
charges total **$1.7646655**; all 31 outcomes, including 12 strict-check
failures, remain diagnostic evidence. No result from that run qualifies as a
performance claim. It exposed two input-contract defects:

1. Graph instructions in `Task.constraints` reached fixed executors. Some
   answers included `max_retries`, `max_regenerations`, and node metadata as
   extra JSON fields.
2. The arithmetic checker required exact decision strings while the prompt
   also allowed ordinary prose such as “Use overtime.”

A separate 12-workflow planner-only pilot completed with one numeric-type
mismatch: the correct percentage appeared as a JSON string. Its charge was
**$0.6192880**, retained within the pilot allocation. The full task pack was
then audited for field types before the final pilot freeze.

## Amended inputs

- Put graph policy in `LLMArchitect.planning_instructions`, with durable
  binding and the existing enforced eight-node, zero-retry graph policy.
- Keep task sources and answer requirements identical across all four arms.
- For object tasks, require exactly the requested top-level fields. Array
  tasks retain their original array-only instructions.
- State every already-checked numeric and nonempty-string field type.
  Numeric values must be JSON numbers. Paths identify fields, not answers.
- State both allowed decision labels for each arithmetic task. The model
  must derive the applicable label from the supplied sources.

Factual answers, rubrics, expected values, task sources, arm order, models,
medium reasoning effort, output caps and retry policy remain unchanged.
The revised pilot must satisfy all 12 deterministic output contracts before
the complete main schedule starts. Paid work is never repeated silently.

**Final pilot result:** all 12 contracts passed, with zero execution failures
and $0.5807450 in workflow charges. The four pilot stages total $2.6596199.
The amended 200-workflow schedule then started under a new source/input freeze.
A later [connection failure and approved continuation](astra_connection_continuation_20260913.md)
preserved the failed row and completed the exact schedule. [Complete results](results/astra_20260913_main/README.md).

## Accounting and interpretation

Every earlier pilot charge is deducted from the same **$60 pilot allocation**.
The stopped main run leaves **$198.2353345** of the **$200 main allocation**.
Judging shares its original **$40 allocation** across pilot and main outputs.
Each workflow remains capped at $5; the total campaign ceiling is $300.

The amended main schedule retains all **200 workflows**: two models × two
strategies × ten tasks × five repetitions. Earlier outcomes do not fill its
cells. All failures in the amended schedule remain in its denominators and
cost totals. Source and input hashes are frozen again before execution.

Because some main tasks were examined during diagnosis, this is an
**amended comparison on reused project-authored tasks**, not an untouched
holdout or an external benchmark. Report all repetitions, task-clustered
intervals, model contrasts and the model-by-strategy interaction. Human
review of disputed main outputs and an evidence audit still precede claims.

[Original protocol and fixed gates](astra_benchmark_plan.md) ·
[Runner and recovery](astra_runtime.md) ·
[Original pilot evidence](results/astra_20260913/README.md).
