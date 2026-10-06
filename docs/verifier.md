# Verification that gates

A verifier can reject an output and trigger bounded regeneration. Verification
is an ordinary node: planned, budgeted, traced, and checkpointed. It declares
which node it judges:

> **Lead with objective checks.** Image dimensions, schema conformance,
> required sections, hashes, and domain rules have definitive verdicts that a
> local function can compute. Use `CallableVerifier` for those contracts. Only
> the verdict is model-free: the gate node still runs like any other node and
> makes its own provider call ([cost](#cost)). Reserve `TokenVerifier` for a
> criterion that genuinely requires model judgment; the
> [control ablation](../benchmarks/control_ablation.md) shows why generic prose
> review is not the default gate.

```python
from smythe.graph import Node

draft = Node(id="draft", label="Write the launch brief")
check = Node(
    id="check",
    label="Verify every claim in the brief is supported. Reply PASS or FAIL.",
    depends_on=["draft"],
    verifies="draft",          # which node this judges
    max_regenerations=2,       # how many times it may send that node back
)
```

When `check` fails `draft`, the executor records the rejection, cancels and
awaits affected active work, then resets `draft` **and everything downstream
of it** to pending. Regeneration starts after that reset is saved. Completed
charges remain recorded; outputs and artifact references from the rejected
generation are cleared. Under a [strict acceptance policy](#rejected-generations)
a bounded record of each cleared generation is kept instead.

A [supervisor](supervisor.md#what-a-revision-may-change) cannot revise a gate
away: a revision that drops `check` or `draft`, rewires `check` without
keeping `draft` among its dependencies, or rewires another step so that
`check` no longer depends on `draft` at all, is rejected. A revision can still
add a step after `draft`. The gate does not judge that step, and under
`DELIVERABLE` synthesis a step that depends on `draft` becomes the returned
output instead of `draft`
([steps added after a gate](supervisor.md#steps-added-after-a-gate)). A
[strict acceptance policy](#strict-delivery-acceptance) accepts such output
only through a final validator, and without one refuses the revision.

## Recovery and concurrent work

A completed judge records the generation it inspected. A rejection produces
an idempotent regeneration intent before cancellation begins. The intent binds
the affected nodes and the next regeneration count, so resuming it cannot
renew the allowance or increment it twice.

Control transitions force checkpoints even when ordinary node snapshots are
batched. Resume processes unfinished verification decisions and regeneration
intents before dispatching work or returning a stored final output. A second
judge whose input was invalidated cannot apply its stale verdict.

The executor waits for billed artifact finalization before resetting an
affected node. Invalid accounting and persistence failures stop the current
run; regeneration cannot hide them. After repairing a local write failure,
an explicit resume may regenerate the output while retaining its earlier
charge. Unresolved accounting blocks resume. Cancellation alone does not prove that a remote
provider issued no charge. See [Cost guardrails](budgets.md).

A judge skipped after a provider failure has no completed verdict and does
not trigger regeneration from its error text. Advisory and exhausted gates
retain their existing policy. Checkpoint-version compatibility is documented
in the [recovery guide](checkpoint-format.md#version-compatibility).

## Reading a verdict

`TokenVerifier` (the default) accepts JSON or prose:

```text
{"passed": false, "reason": "two claims lack sources"}
FAIL - two claims lack sources
```

**JSON.** Every JSON object in the reply is examined, whether it is bare,
surrounded by text, or in a code fence with any tag (`json`, `JSON`, or
none). A top-level object with a `passed` key is a verdict:

- `passed` must be a JSON boolean; `"false"`, `1`, or `null` is not a
  verdict, and an object that repeats `passed` with different values is
  contradictory.
- Every verdict object in the reply must agree, so an echoed format
  example such as `{"passed": true, "reason": "..."}` next to the real
  `{"passed": false}` fails.
- An object nested inside another JSON value, including an array, never
  passes a reply, but a nested `"passed": false` fails it.
- A `"passed":` key outside valid JSON, as in truncated or malformed
  output, fails the reply.
- Text inside JSON is never read as prose, so a reason that mentions PASS
  cannot override `"passed": false`. A label whose value is JSON, such as
  `Verdict: {"passed": false}`, is read from the JSON.

**Prose.** A prose verdict is the uppercase word `PASS` or `FAIL`
(`PASSED`, `PASSES`, `FAILED`, and `FAILS` also count) standing on its
own. It counts when it:

- is alone on its line, or leads its line and is followed by a separator
  and a reason: `**PASS**`, `PASS.`, `FAIL - two claims lack sources`,
  `PASS (3/3 criteria met)`;
- starts a new sentence after `.`, `!`, `?`, or `…`, as in
  `No criterion is unmet. PASS`; or
- follows a label that names the final verdict: `Verdict: PASS`,
  `**Final verdict** - PASS`, `Overall: PASS`, `PASS/FAIL: PASS`,
  `Verdict (PASS or FAIL): PASS`, `My final verdict is PASS`,
  `- Verdict: PASS`. The label has at most six words and must contain
  `verdict`, `final`, `overall`, or the choice `PASS/FAIL`; its other
  words may only be my, our, the, is, gate, judge, result, answer,
  decision, conclusion, outcome, assessment, evaluation, grade, status,
  summary, judgment, or judgement.

Markdown emphasis, headings, table pipes, and check marks such as ✅
around the word or label are ignored. A reply that is only the verdict
word may use any case and may be wrapped in quotes, backticks, `*`, `_`,
`#`, or `>`: `Pass.`, `"PASS"`. Otherwise lowercase words are prose, so
"does not fail any criterion. PASS" passes.

A PASS anywhere else is not a verdict: inside a sentence
(`the draft does not PASS`), in quotation marks, joined to another word
(`PASS-THROUGH`, `PASS.md`), in a list item without a verdict label
(`- PASS`, `1. PASS`), after any other label (`Criterion 2: PASS`,
`Result: PASS`), or in quoted material, meaning a fenced code block or a
`>` blockquote. A criterion-by-criterion reply therefore needs a final
verdict line.

Verdicts **fail closed**. The reply is read as FAIL, and the regeneration
span records why, when:

- it is empty or has no verdict;
- a JSON verdict is invalid, as described above;
- its verdicts disagree: JSON with JSON, JSON with prose, or prose lines
  with each other;
- the prose contains an uppercase `FAIL`, `FAILED`, or `FAILS` anywhere,
  including quoted material, except where the two words are offered as a
  choice: `PASS or FAIL`, `PASS/FAIL`, `PASS|FAIL`, or `PASS-FAIL`, in
  either order;
- an uppercase PASS is negated or conditional, even when another line
  says PASS. It is negated when one of the three words before it in the
  same clause is not, no, never, cannot, nor, neither, hardly, a form of
  fail, or a contraction ending in n't (`NOT PASS`, `cannot PASS`). It is
  conditional when the next word is if, unless, provided, providing,
  assuming, pending, once, conditional, conditionally, or contingent
  (`PASS if …`, `PASS - provided …`);
- a line outside quoted material that is labelled with `verdict` or
  `PASS/FAIL`, such as `Verdict:` or `Final verdict -`, states something
  other than a verdict, as in `Verdict: not met`; or
- it is longer than 200,000 characters.

Reading takes time linear in the reply's length and never raises. At most
64 JSON candidates are decoded; a `"passed"` key beyond them fails the
reply. A gate must not approve work because its judge was unreadable.

The cost of failing closed is bounded by the gate itself. A failed verdict
triggers at most `max_regenerations` send-backs; after that, the run
finishes with the last output rather than stopping. A gate with
`max_regenerations=0` is advisory and its verdict is never read. Ask the
judge for JSON, or for a single PASS or FAIL on its own line, so that a
readable verdict is the norm. Under a [strict acceptance
policy](#strict-delivery-acceptance) the final verdict is read and recorded
in both cases.

A verdict that fails because no verdict could be read, rather than because
the reply states FAIL, is marked `invalid` (`Verdict.invalid`): an empty,
overlong or unreadable reply, an invalid or contradictory JSON verdict,
disagreeing verdicts, or a reply with no standalone verdict. Only a JSON
`"passed": false` or a stated FAIL with no PASS beside it is a valid
rejection. The mark changes nothing for an ordinary gate and does not take
part in comparing verdicts.

## Gating without a model

Most useful gates are objective, and their verdict needs no LLM — image
dimensions, JSON schema conformance, a required section:

```python
from smythe import CallableVerifier, Swarm

long_enough = CallableVerifier(
    lambda verifier_node, target: len(str(target.result)) > 500,
)
swarm = Swarm(verifier=long_enough, ...)
```

The function decides the gate only through an explicit bool. It may return:

- `True` or `False`;
- a `Verdict(passed=..., reason=...)` whose `passed` is a bool, when you want
  the reason recorded in the trace; or
- an object with a bool `passed` attribute, such as a
  [`ValidationReport`](#validation-reports). Its `reason` or `summary` string
  becomes the verdict's reason; a failing report without either lists up to
  four of its hard findings as `code: message`. A report whose `outcome` is
  `"error"` could not decide, so its reason is `validator error`, followed by
  the report's validator name in parentheses and up to four of its findings
  when it has them: `validator error (required-section): unreadable: output
  is not UTF-8`.

Anything else **fails** the gate, with a reason that names the returned type
(`verifier returned str`). The string `"FAIL"`, a non-empty list of
findings, `None`, `1`, a NumPy boolean, and a `passed` that is not a bool are
not verdicts, so return `bool(...)` when a check produces another type. The
function is called synchronously: an `async def` function's coroutine is
closed without running, and the gate fails. These results, and a report
whose `outcome` is `error`, give a verdict marked `invalid`.

`smythe.design.design_verifier(system, width=..., height=...)` is a ready-made
objective gate for images. It reads each file the judged node wrote once,
fails a file that is missing, unreadable or no longer holds the bytes whose
`sha256` its record carries, and fails blank frames, wrong dimensions and
near-duplicates; palette and flat-region findings block only with
`include_advisory=True`. A node without files passes unless
`require_artifacts=True`, which a gate on an image deliverable should set.
The palette is checked when the verifier is built, and the verifier's
`identity` is a `ValidatorIdentity` named `smythe.design` whose version and
configuration digest cover its settings and detector thresholds.

The executor holds every `Verifier` to the same rule: a verdict passes only
when its `passed` is exactly `True`. A custom verifier's
`Verdict(passed="false")` or `Verdict(passed=1)` is a failed verdict, and the
regeneration span records `verdict 'passed' must be a bool`.

### Validation reports

`smythe.validation` defines `ValidationReport`, `ValidationFinding` and
`FindingSeverity` without importing an image library; `smythe.assets`
re-exports the same classes for image validation. A report's `outcome` is
`passed`, `rejected` or `error`:

- Without a stated outcome, a report with no hard findings is `passed` and
  one with any hard finding is `rejected`. Advisory findings never decide it.
- A validator that could not decide states `outcome="error"`. An `error`
  report never passes, so as a gate result it fails.
- A stated outcome must agree with the findings: a `passed` report has no
  hard findings and a `rejected` report has at least one.
- `dataclasses.replace(report, findings=...)` derives an outcome that was not
  stated again from the new findings, and checks a stated outcome against
  them. A report read with `from_dict()` states the outcome its data records.

`smythe.assets.validate_image(path, spec, ...)` reads the file once and checks
that snapshot, so the bytes hashed for a finishing receipt are the bytes
decoded. `validate_image_bytes(data, spec, ...)` does the same for bytes
already in memory; it resolves a relative logo or font path only against an
explicit `base_dir` and otherwise reports the master unavailable, where
`validate_image` falls back to the working directory.

`report.validator` optionally holds a `ValidatorIdentity`: the validator's
name, version and `config_sha256`, the SHA-256 of its configuration as
canonical JSON (`ValidatorIdentity.for_config(name, version, config)` computes
it). `to_dict()` and `from_dict()` convert reports, findings and identities to
and from JSON data; a tuple in a finding's `expected` or `observed` value
becomes a JSON array, and `from_dict()` rejects unknown keys and outcomes.

### Gate validators in durable workflows

A [durable text workflow](workflow-accounting.md) accepts no `CallableVerifier`,
because its function has no version a saved run could be checked against.
Declare a validator instead, and pass the declaration as `verifier`:

```python
from smythe import LocalOnly, Swarm

sources = LocalOnly(lambda: RequiredSection("## Sources"), "required-section", "1",
                    role="validator")
swarm = Swarm(run_store=store, verifier=sources, ...)
```

`RequiredSection` is the validator shown under
[delivery acceptance records](#delivery-acceptance-records). The factory
builds a fresh validator for each run; its `ValidatorIdentity` must have the
declared name and version. Each gate then passes when the validator's report
on the exact text of the node it judges is `passed`, and fails with the
report's hard findings when it is `rejected`. A validator that raises, returns
anything but its own `ValidationReport` or reports `error` could not decide:
the gate ends at once without a verdict, its receipt consumed with `outcome`
`error`, so it spends no regeneration and buys no further judge call, with or
without a strict policy. The run continues with the judged output, which a
strict policy decides `on_invalid`. The gate node still runs and makes its
provider call; only the verdict comes from the validator.

The declaration refuses a `Provider`, a coroutine function or an awaitable as
the factory, and refuses what the factory builds when it is a provider, an
awaitable, a class, an object without a synchronous `validate()` or one whose
identity has another name or version. The run's recipe records
`"verifier": "validator-v1"` and the validator's name, version and
`config_sha256` under `validators`, so resuming the run with another
validator, version or configuration raises `WorkflowConflictError` before any
call. Recipes without a validator keep their bytes. An ordinary `Swarm`
refuses a `LocalOnly` verifier with `TypeError`; there, a `CallableVerifier`
whose function returns
`validator.validate(ValidationSubject.from_text(target.result))` runs the
same check.

## Delivery acceptance records

A gate decides whether one node's output may move on. Delivery acceptance
decides whether the run's delivered output is accepted, and is reported apart
from execution and cost. `smythe.acceptance` defines its types:

- `AcceptancePolicy()` is advisory: gates advise, as described above, and no
  record is made. `AcceptancePolicy(mode="strict", ...)` takes
  `final_validators`, which check the exact delivered bytes;
  `allow_gate_acceptance` (default `True`), which lets a passed gate you
  supplied accept output that is byte for byte its target's result;
  `allow_generated_gates` (default `False`), which lets a gate that a
  generated plan or a supervisor revision added do the same
  ([which gates count](#which-gates-count)); `on_exhausted`
  (`"rejected"`, or `"needs_review"`) for an exhausted gate whose final
  verdict fails; and `on_invalid` (`"indeterminate"`, or `"needs_review"`)
  for a missing, malformed or erroring verdict. An advisory policy refuses
  these settings, and so does a strict policy that could never accept.
  `policy.to_dict()` lists validator identities sorted by name, never code,
  and includes `allow_generated_gates` only when it is set;
  `policy.sha256` hashes it.
- `DeliveryAcceptance` is the versioned record of one decision:
  `not_evaluated`, `accepted`, `rejected`, `needs_review` or `indeterminate`.
  It binds the SHA-256 of the delivered bytes, the synthesis strategy, the
  source nodes and their generations, the task and policy hashes, and each
  validator's outcome; version 2 records an
  [operator's decision](#operator-decisions). Checkpoints and JSON keep it under
  `delivery_acceptance`; the [checkpoint format](checkpoint-format.md#delivery-acceptance-version-5)
  lists its fields.
- A validator is trusted local code: an object with an `identity`
  (`ValidatorIdentity`) and a synchronous `validate(subject)` that returns a
  `ValidationReport`, with `outcome="error"` when it cannot decide. `subject`
  is a `ValidationSubject`: the exact bytes, their `sha256` and `text()`.
  Registration refuses a `Provider`, a class, a coroutine function, an
  awaitable and an `async def validate`, including a `validate` object whose
  `__call__` is async. That structural check enforces a caller contract; it
  is not a sandbox. It cannot see a synchronous `validate()` that makes
  network requests or paid calls, which a validator must not do.

```python
from smythe import AcceptancePolicy, ValidationReport, ValidatorIdentity
from smythe.validation import hard_finding

class RequiredSection:
    def __init__(self, heading):
        self.heading = heading
        self.identity = ValidatorIdentity.for_config(
            "required-section", "1", {"heading": heading},
        )

    def validate(self, subject):
        if self.heading in subject.text():
            return ValidationReport("output", validator=self.identity)
        missing = hard_finding("section_missing", f"no {self.heading!r} section")
        return ValidationReport("output", (missing,), validator=self.identity)

policy = AcceptancePolicy(mode="strict", final_validators=[RequiredSection("## Sources")])
```

The [built-in final validators](#built-in-final-validators) check required
sections, length bounds and JSON paths without a class of your own.

`SwarmResult.acceptance` holds the record, or `None` when there is none.
`SwarmResult.accepted` is true only for an `accepted` record. When a record
exists and does not accept, `output` is `""` and the text is in
`rejected_output`. `SwarmResult.to_dict()` returns the result as JSON data
with the record under `delivery_acceptance`, and the `execution`, `delivery`,
`unknown_exposure` and `next_action` blocks that
[`smythe runs inspect`](runs.md) reports for a saved run. Advisory and legacy
runs have
`acceptance=None`; a run under a [strict policy](#strict-delivery-acceptance)
always has a record, and a strict `Swarm` withholds the output of a completed
run resumed without one. `PlannerMemory` stores the record's decision as
`acceptance`, or `not_evaluated` without one. Jobs run the same validator
objects on artifact bytes when a manifest names them; see
[registered validators](jobs.md#registered-validators).

## Strict delivery acceptance

```python
from smythe import AcceptancePolicy, Swarm

policy = AcceptancePolicy(mode="strict", final_validators=[RequiredSection("## Sources")])
swarm = Swarm(acceptance=policy, ...)
result = swarm.execute(task)
if result.accepted:
    publish(result.output)
else:
    print(result.acceptance.decision, result.acceptance.reasons)
```

[`examples/15_strict_acceptance.py`](../examples/15_strict_acceptance.py) runs
this offline: a draft passes its gate, a later step drops the required
section, and the strict run keeps the defective text as `rejected_output`.

`Swarm(acceptance=...)` and `Swarm.from_yaml(..., acceptance=...)` take the
policy. It comes only from that argument: a YAML file, a graph, a `Task`, a
generated plan or a supervisor revision cannot set or weaken it. `None`, the
default, and `AcceptancePolicy()` keep the behavior described above. Strict
acceptance runs on ordinary runs and on
[durable text workflows](#strict-acceptance-in-durable-workflows).

**The delivered output decides.** After synthesis, the run decides on exactly
the text it delivers, as UTF-8 bytes. Every final validator runs once on those
bytes as local code, with no provider call and no graph node, and every gate
in the graph is read from its recorded final verdict. The decision is:

1. `rejected` when a final validator rejects the output. A final rejection is
   terminal: nothing is regenerated.
2. Otherwise `on_exhausted` when a gate's final verdict fails.
3. Otherwise `on_invalid` when a final validator or a gate could not decide,
   or a delivered node wrote files that no check covers (see below).
4. Otherwise `accepted` when every final validator passed, or, without final
   validators and with `allow_gate_acceptance`, when the output is byte for
   byte the result of the one node `DELIVERABLE` synthesis returned and a
   gate that [counts](#which-gates-count) passed that node at its current
   generation.
5. Otherwise `on_invalid`: no check covers the delivered output.

A gate therefore accepts only the node it judged. A later step, including one
a supervisor adds after the gated node, two deliverable nodes, the
`DELIVERABLE` fallback to every completed node, and `CONCATENATE`,
`STRUCTURED`, `LLM_MERGE` or a custom synthesizer all need a final validator.
A `Swarm` whose strict policy has none raises `ValueError` when it is built
with a synthesis other than `DELIVERABLE`, and refuses the other cases
[before any call](#acceptance-obligations). `DELIVERABLE` output to which
`retain_contradictions` appends a
[contradictions section](source-evidence.md#keep-contradictions-through-synthesis)
is not covered by the gate either. Whether the section is appended depends on
results, so such a run is not refused up front: without a final validator it
makes its calls and its output is decided by `on_invalid`.

<a id="which-gates-count"></a>
**Which gates count.** Acceptance criteria are inputs you control, so a gate
accepts output on its own only when you supplied it: in a graph you pass to
`execute()` or load from YAML, in a plan that a `DeterministicArchitect`
builds or a `ConstrainedArchitect` composes from your templates, or, in a
durable run, in a plan that a `LocalOnly` architect builds in local code. A
strict `Swarm` marks each gate of a plan that any other architect, such as
`LLMArchitect`, generates with `generated_gate: true` in its metadata, and
`apply_revision` marks every node a supervisor revision adds with
`added_by_revision: true`. A passed verdict from such a gate does not accept
the output: without a final validator the decision is `on_invalid`. Set
`allow_generated_gates=True` to count them as well. Their failing verdicts
always count. A plan made by a `Swarm` without a strict policy carries no
mark, so a graph you take from it and pass to a strict `Swarm` counts as
yours.

<a id="acceptance-obligations"></a>
**Obligations checked before any call.** A strict run that could never
decide its output is refused with `ValueError` before it spends anything.
`smythe.acceptance.validate_acceptance_obligations(graph, policy,
strategy="deliverable")` reads the graph's structure, not its results, and
lists every unmet obligation:

- a gate that verifies itself or a node that is not in the graph, or that
  does not depend on the node it verifies, directly or through other nodes,
  and so could judge it before it runs;
- a gate with `FailurePolicy.SKIP`, which would leave it without a verdict
  when its judge fails;
- without final validators, a delivered output that no counted gate covers:
  `DELIVERABLE` synthesis must deliver exactly one node, and a gate that
  counts must judge it.

A strict `Swarm` checks a graph you pass to `execute()` or load from YAML
before any provider call, and a generated plan after planning, before any
node runs. Without final validators or `allow_generated_gates`, it refuses
before planning a run whose architects all generate their gates, such as the
default `LLMArchitect`, because no gate of that plan could accept; `plan()`
still returns the graph for inspection, and executing it is refused. A
[supervisor revision](supervisor.md#steps-added-after-a-gate) that would
break an obligation the graph met is refused like any invalid revision. A
resumed run is not checked again: its graph met the obligations when it
started. A [durable run](#strict-acceptance-in-durable-workflows) applies
the same checks at the same points.

A final validator that raises, returns an awaitable or anything other than a
`ValidationReport`, reports `outcome="error"`, or returns a report naming
another validator has outcome `error`.

**Files a delivered node wrote.** Final validators check only the output text.
A delivered node's files are covered when a passed gate that the policy
counts judged that node at its current generation, such as
[`design_verifier(require_artifacts=True)`](#gating-without-a-model). Each file's
record must carry the `sha256` and `size_bytes` written with it, and the file
must still hold exactly those bytes when the run decides. A file that is not
hashed, is missing or has changed, or files that no passed gate on their node
covers, give `on_invalid` unless an earlier rule applies. The record binds
every verified file under `artifacts` by name, SHA-256, size and MIME type.

**Gates under a strict policy.** A valid rejection before exhaustion
regenerates as usual. The judge's last verdict is then read, without another
call, and recorded in its consumed receipt with `outcome` (`passed`,
`rejected` or `error`), `passed`, `reason` and, for an exhausted failing
verdict, `exhausted: true`; that verdict decides `on_exhausted` (`rejected`
by default, or `needs_review`). A verdict marked `invalid`, a `passed` that is
not a bool, a result that is not a `Verdict`, and a verifier that raises have
outcome `error`: the receipt is consumed at once, no regeneration is spent,
the run continues, and the decision is `on_invalid` (`indeterminate` by
default, or `needs_review`). A gate with `max_regenerations=0` is read once
when its judge completes and its verdict recorded under `advisory_verdict`
in the judge's metadata; a failing verdict counts as exhausted. A judge that
was skipped or failed, or whose target was unavailable, leaves the gate
undecided. Each recorded verdict names the judge's and its target's
generations, and regeneration clears it.

Only a verdict the run's own judges produce decides it. Each attempt of a
judge replaces the verdict record it carries, so the record holds that
attempt's verdict even when a graph is run again or a YAML file sets the
metadata. A judge that completed before a fresh run does not run again, so a
strict `Swarm` refuses with `ValueError`, before any provider call, a graph in
which any judge has already completed: whether it holds a consumed or pending
receipt, an `advisory_verdict`, a bound `verification_target_generation` or
only an earlier reply, the run would decide on a verdict it did not produce.
Resume the run's checkpoint, or set the judge back to pending so it judges
its target in the run. Without a strict policy none of this changes:
an exhausted or advisory gate's verdict is never read and a verifier that
raises stops the run.

**Results and recovery.** A strict run's result always carries its record: an
accepted result has the exact text in `output`; any other decision returns
normally with `output=""` and the text in `rejected_output`. Every checkpoint
of a strict run is [version 5](checkpoint-format.md#delivery-acceptance-version-5)
and carries the record, `not_evaluated` until the run finishes. An unfinished
strict run resumes only under a policy with the same `sha256`, and is refused
before any provider call otherwise; a strict `Swarm` refuses an unfinished
checkpoint written without one. Resuming a completed strict run checks the
saved output against the record's digest, re-hashes every file the record
binds, and checks the task, source generations and, when the resuming `Swarm`
has a strict policy, its `sha256` against the record; any mismatch returns an
`indeterminate` record for the saved output, never acceptance.

A completed checkpoint without a record (versions 1 to 4, or version 5
without `delivery_acceptance` in `requires`) was never decided. A `Swarm`
without a strict policy returns its text in `output`, as before. A strict
`Swarm` withholds it: the result has `output=""`, the saved text in
`rejected_output` and `acceptance=None`, and a warning on the `smythe.swarm`
logger says how to obtain a decision. Re-run the task under the strict
policy, or decide the saved output yourself with
`evaluate_delivery(policy, result.graph, result.rejected_output, ...)`. The
checkpoint is not changed, and [re-evaluation](#re-evaluation) refuses it:
nothing bound its output.

The trace gains one span with `node_id` `acceptance:delivery`, status
`acceptance_<decision>` and the first reason as its label; `Tracer.summary()`
keys are unchanged, and the validators' time counts as verification in
[`SwarmResult.timing`](execution.md#timing).
`smythe.acceptance.evaluate_delivery(policy, graph, output, strategy=...,
sources=...)` makes the same decision outside a run, and
`smythe.synthesizer.deliverable_sources(graph)` returns the nodes
`DELIVERABLE` synthesis delivers and whether it fell back.

### Re-evaluation

Resuming a completed run never decides it again. To decide its saved output
under a fixed or changed validator, call
`swarm.reevaluate(execution_id, acceptance=policy)`, or
`await swarm.areevaluate(...)`. Without `acceptance` it uses the `Swarm`'s
own policy, which must then be strict.

```python
fixed = AcceptancePolicy(mode="strict", final_validators=[RequiredSection("## Sources")])
result = swarm.reevaluate(execution_id, acceptance=fixed)
```

- The policy's final validators run on the saved output, with no provider
  call, and every gate's recorded final verdict is read, as when the run
  completed; no judge runs again. No node runs: costs, the budget, retry and
  regeneration counters and node generations stay as saved, and so do a
  durable run's calls and charges.
- Only a completed run whose record holds a decision can be re-evaluated,
  and only while everything that record binds still holds: the saved output
  must match its digest, and its task, source generations and delivered files
  must be unchanged. Otherwise `reevaluate` raises `ValueError`
  (`WorkflowStateError` on a durable run) and writes nothing, so changed
  output is never accepted. A completed run without a record, such as an
  advisory run or a version 4 checkpoint, has nothing that binds its output,
  and its gates were not marked by origin
  ([which gates count](#which-gates-count)): run the task again under a
  strict policy. An unfinished run is resumed, not re-evaluated.
- The new decision becomes the run's current decision, and the decisions
  before it are kept. Sequence 0 is the decision the run made when it
  completed, and each re-evaluation adds one, so the same bytes re-evaluated
  under an unchanged policy still get a new decision. An ordinary checkpoint
  keeps the last 16 superseded records under `delivery_acceptance_history`
  ([checkpoint format](checkpoint-format.md#decision-history)); a durable run
  keeps every decision in its journal
  ([durable re-evaluation](#strict-acceptance-in-durable-workflows)).
- Resuming the run returns its current decision, checked as any completed
  resume checks its record. An ordinary checkpoint does not record its
  policy, so, as before, a strict `Swarm` whose policy did not make the
  current decision gets `indeterminate`. A durable run still resumes only
  under the policy its recipe records, which re-evaluation does not change,
  and returns the current decision, whichever policy made it.
  [`smythe runs inspect`](runs.md#decision-history) lists the decisions and
  exits by the current one.

### Re-evaluation and republication

A durable run with a [renderer](workflow-accounting.md#published-documents)
delivers a published document, and two different repairs apply to it:

| What changed | Repair | What is decided |
|---|---|---|
| A validator or the acceptance policy | `swarm.reevaluate(run_id, acceptance=policy)` | The same saved bytes, again, under the given policy. The document must still be intact. |
| The published file: missing or changed | `swarm.republish(run_id)` | Without [lineage](lineage.md), nothing: the saved bytes go to a new file and the decision, which binds them, stands. With lineage, the bytes [rendered again](lineage.md#repair-a-run-in-place) from the saved synthesis, under the `Swarm`'s policy. |
| A source, the renderer or the synthesizer | Neither: a new run | |

A lineage run's republication never copies a decision: the new decision
binds the new publication, so even identical bytes get a decision of their
own, and the earlier ones stay in the decision history. On a lineage run,
every new decision also supersedes the run's lineage record. A re-evaluation
refuses a damaged document, so republish first and then re-evaluate when
both changed; on a lineage run, republishing under the new policy does both
in one decision.

### Operator decisions

A policy that selects `needs_review`, or a decision left `indeterminate`,
withholds output until someone decides it. An authorized operator can
resolve it, as accepted or rejected, through the same entry point:
`swarm.reevaluate(execution_id, operator=decision)`, or
`await swarm.areevaluate(...)`. `decision` is an
`smythe.acceptance.OperatorDecision`; it takes no `acceptance` policy, and the
`Swarm` needs none.

Smythe does not authenticate operators. Your host's approval verifier does,
through the [`ApprovalVerifier`](actions.md#connectors-and-approval-verifiers)
contract that external actions use: it issues an assertion for one digest and
returns the `Principal` behind it from `verify()`.
`operator_decision_sha256(run_id, sequence, record, decision,
evidence_sha256)` is the digest to approve. It binds the run, the current
decision's sequence and record, the operator's `"accepted"` or `"rejected"`,
and the SHA-256 of the evidence the operator decided on, which your host
keeps; Smythe stores only the digest.

```python
from smythe.acceptance import OperatorDecision, operator_decision_sha256
from smythe.checkpoint import delivery_acceptance_sequence

record = result.acceptance                       # needs_review
sequence = delivery_acceptance_sequence(checkpoint_store.load(execution_id))
digest = operator_decision_sha256(execution_id, sequence, record, "accepted", evidence_sha256)
assertion = identity_host.issue(digest, ...)     # your host's approval flow
decided = swarm.reevaluate(execution_id, operator=OperatorDecision(
    "accepted", evidence_sha256, assertion=assertion, verifier=identity_host))
assert decided.accepted
```

On a durable run, `smythe.workflow.current_decision(store, run_id)` gives the
`sequence` and `record`.

- Only a current decision of `needs_review` or `indeterminate` can be
  resolved. An `accepted` or `rejected` decision, including an operator's,
  stands: `reevaluate` raises `ValueError` (`WorkflowStateError` on a durable
  run) before the assertion is checked. As with re-evaluation, the run must
  be completed and everything its record binds must still hold.
- The verifier must accept the assertion for exactly that digest. A forged,
  expired or revoked assertion, or one issued for another run, decision,
  sequence, record or evidence, makes it raise `ApprovalRejected`, which
  propagates; nothing is written.
- No node runs, no validator or provider is called, and costs, the budget,
  retry and regeneration counters and node generations stay as saved.
- The new decision is a version 2 record with the bindings of the one it
  resolves, no validator outcomes, and an `operator` block: the principal's
  `issuer`, `subject` and `tenant`, the verifier's `identity`, and
  `evidence_sha256`. A reader that knows only version 1 records refuses it.
  It becomes the current decision and the earlier one is kept, as for a
  re-evaluation: an ordinary checkpoint also lists `operator_decision` in
  [`requires`](checkpoint-format.md#required-state-features-version-5) while
  it holds such a record, and a durable run journals it as the next
  `acceptance` operation, whose inputs also bind the decision and the
  operator. Delivery, resume and `smythe runs inspect` follow it.
- A later re-evaluation under a policy still records a new decision, and
  keeps the operator's among the earlier ones.

## Built-in final validators

`TextValidator` and `JSONValidator` (in `smythe.validators`, also exported
from `smythe`) are final validators configured by a specification: JSON data
that versioned library code interprets. A validator's `identity` is its name,
the version of that code (`"1"`), and the SHA-256 of its specification with
every default filled in, so a policy's `sha256` and each acceptance record pin
the exact checks; `validator.spec` returns that specification. Both run
locally with no provider call and report `rejected`, never `error`, for output
that fails a check, including bytes that are not UTF-8. An invalid
specification raises `ValueError` when the validator is built: an unknown key
or kind, a missing key, a character count outside 0 to 1,000,000,000 or a
check's count outside 0 to 10,000, a non-finite number, or a repeated section
or check id. `name=` renames a validator, so that one policy
can hold two of the same kind; a policy refuses two validators with one name.

```python
from smythe import AcceptancePolicy, JSONValidator, Swarm, TextValidator

complete = TextValidator({"sections": ["Argument", "Critique"], "max_chars": 12_000})
swarm = Swarm(acceptance=AcceptancePolicy(mode="strict", final_validators=[complete]), ...)
```

**Text.** `TextValidator(spec)`, named `smythe.text-spec`, takes these keys,
each optional:

| Key | Default | Requires |
|---|---|---|
| `min_chars` | `1` | at least this many characters after surrounding whitespace is removed, so empty output fails |
| `max_chars` | `None` | at most this many characters, when set |
| `sections` | `[]` | each listed heading, as a Markdown ATX heading (`#` to `######`) outside fenced code, at any level |
| `ordered` | `true` | the sections in the listed order; other headings may come between them |
| `min_section_chars` | `1` | at least this many characters in each required section, up to the next heading at its level or above |

Headings compare ignoring case, spacing, `*` and `_` emphasis and a final
colon, so `## **Critique:**` is the section `Critique`; a bold line or a
heading inside a code block is not a heading. A section may be written as a
heading line, such as `## Sources`: its `#` marker and the spaces around it
are removed when the validator is built, so `spec` lists `Sources` and a
heading at any level satisfies it. Findings are `text_too_short`,
`text_too_long`, `section_missing`, `section_out_of_order`, `section_empty`
and `not_utf8`; a `section_out_of_order` finding observes each required
heading once, in the order it first appears.

A sections check is how strict acceptance holds a run to its whole
deliverable. With the policy above, a final step that returns only its own
increment, such as `## Critique` without the argument it critiques, fails
with `section_missing: required section 'Argument' is missing`: the decision
is `rejected`, `output` is `""` and the increment is in `rejected_output`.

**JSON.** `JSONValidator(spec)`, named `smythe.json-spec`, requires the whole
output to be strict JSON: a code fence around it, a repeated object key,
`NaN`, `Infinity`, a number too large for a float, or nesting too deep for
the interpreter to parse or compare fails with `json_invalid`, and output
larger than `max_bytes` (default 1,048,576) with `json_too_large`. The
nesting depth that counts as too deep depends on the Python version and the
caller's stack; such output is rejected either way. Its `checks` (default `[]`) each have an `id` (2 to 64
lowercase letters, digits or hyphens, starting with a letter), a `kind` and a
`path`: a list of object keys and array indexes from the document root, `[]`
for the root itself. A path that does not resolve fails with
`json_path_missing`; any other failed check gives `json_check_failed`.

| Kind | Other keys | Passes when the value at `path` |
|---|---|---|
| `present` | | exists |
| `equals` | `expected` | equals `expected` as canonical JSON, so `1` does not equal `1.0` or `true` |
| `number` | `expected`, `tolerance` (default 0) | is a finite number, not a bool, within `tolerance` of `expected` |
| `nonempty` | `min_chars` (default 1) | is a string with at least `min_chars` characters after surrounding whitespace is removed |
| `object` | `minimum` (default 1) | is an object with at least `minimum` keys |
| `length` | `expected` | is an array or object with exactly `expected` items |
| `keys` | `expected` | is an object whose keys are exactly `expected` |
| `required_keys` | `expected` | is an object holding every key in `expected` |
| `citations` | `allowed`, `minimum` (default 1) | is an array of distinct strings, each in `allowed`, with at least `minimum` of them |

```python
summary = JSONValidator({"checks": [
    {"id": "fields", "kind": "required_keys", "path": [], "expected": ["total", "sources"]},
    {"id": "total", "kind": "number", "path": ["total"], "expected": 41},
    {"id": "sources", "kind": "citations", "path": ["sources"], "allowed": ["doc-1", "doc-2"]},
]})
```

**Source evidence.** `SourceEvidenceValidator(sources)`, named
`smythe.source-evidence`, checks the [source evidence](source-evidence.md)
that an artifact carries in a `source-evidence` block against the sources the
run was given: each cited source exists, its stated version and digest match,
each quoted passage appears verbatim in its source's retained text, no passage
cites a source the evidence does not list, and each contradiction the
evidence lists appears in the artifact. A passage from a tombstoned source
cannot be checked and gives an `error` outcome, never acceptance. The checks
are mechanical: whether a passage supports its claim stays a judgement for a
gate or a reviewer. `Synthesizer(retain_contradictions=True)`
[keeps listed contradictions through synthesis](source-evidence.md#keep-contradictions-through-synthesis),
and these checks then run on the output it delivers.

### Rejected generations

Under a strict policy a regeneration keeps a record of what it discards. Each
reset node that had produced a result or files gains an entry in the
`rejected_generations` list in its metadata: the generation, its status, the
judge and its reason, the SHA-256 and UTF-8 size of the result, an `excerpt`
of its first 1,024 bytes, and the generation's artifact records with their
digests. A longer result is written in full to a `sidecar` file beside the
run's artifacts, such as `draft_g0_rejected.txt`, when the run has an
`artifact_dir`. A regenerated node's files are named for its generation and
never replace a file ([artifacts](architecture.md#artifacts-and-jobs)), so a
record's paths still hold the bytes that were judged.

Ordinary checkpoints rewrite the whole graph on every save, so the records of
all nodes together add at most 64 KiB to a checkpoint
(`smythe.executor_base.REJECTED_GENERATIONS_MAX_BYTES`). Past that a record
keeps no excerpt, and one that still does not fit is not kept; the
regeneration span in the trace still names its node. Without a strict policy
a regeneration clears rejected results and artifact references as before. The
[checkpoint format](checkpoint-format.md#verification-state) lists the record's
fields.

### Strict acceptance in durable workflows

A `Swarm` with a durable `run_store` decides the same way, on the same
policy, and records the decision in its journal:

- **The same gates count and the same obligations hold.** The run marks each
  gate of a generated plan `generated_gate`, in the planning result its
  journal saves, and each node a supervisor revision adds `added_by_revision`,
  as an ordinary run does ([which gates count](#which-gates-count)). It
  checks the [obligations](#acceptance-obligations) before any node call: a
  graph you pass to `execute()` before its run is created, and a generated
  plan, or a planned graph you edit, before its first node runs. Without
  final validators or `allow_generated_gates`, `execute()` with a task is
  refused before the planning call when every architect that could plan
  generates its gates. A refusal raises `ValueError`. A refused plan keeps
  its planning charge and stays `planned`, and resuming it is refused the
  same way. A supervisor revision that would break an obligation is refused
  and traced, and the run continues on its plan.
- **The policy is part of the run.** The recipe records `policy.to_dict()`
  under `acceptance`: validator names, versions and configuration digests,
  never code. Final validators are the policy's validator objects, registered
  in-process as on ordinary runs. Resuming under another policy, or without
  one, raises `WorkflowConflictError` before any call, and so does a strict
  `Swarm` resuming a run that began without one. Gates use `TokenVerifier`
  or a [gate validator](#gate-validators-in-durable-workflows);
  `CallableVerifier` is refused with `WorkflowBindingError`.
- **The decision is a journal operation.** After synthesis, an operation of
  kind `acceptance`, keyed by the output digest and everything else the
  record binds, runs the final validators and reads the gates, and the
  completed checkpoint consumes it in the same transaction as the output.
  A run stopped after synthesis and before the decision is committed decides
  again when it resumes: the validators run on the same saved output, which
  is never accepted for having been synthesized. A committed decision is read
  back without running a validator or calling a provider.
- **Checkpoints.** A run whose recipe has a strict policy or a gate validator
  writes checkpoint version 2, which adds `delivery_acceptance`: the
  `not_evaluated` record until the run completes, then the decision, and
  `null` without a strict policy. Other runs still write version 1, and a
  checkpoint whose version does not match its recipe raises
  `WorkflowStateError`. Earlier releases cannot resume version 2 runs; they
  already refuse their recipes.
- **Completed runs.** Resuming a completed run checks the saved output
  against the record's digest, the task and source generations against the
  record, and the record against the journal's current decision. Any
  mismatch returns an `indeterminate` record with the saved text in
  `rejected_output`, never acceptance. A run without a strict policy returns
  `acceptance=None`. [`smythe runs inspect`](runs.md) makes the same checks
  without resuming the run.
- **Re-evaluation.** `Swarm(run_store=...).reevaluate(run_id,
  acceptance=policy)`, or `WorkflowRuntime.reevaluate(run_id)`,
  [decides a completed run again](#re-evaluation). The saved recipe may
  differ from the `Swarm`'s only in its acceptance policy and gate verifier,
  since no gate judges again; anything else raises `WorkflowConflictError`.
  The decision is a new `acceptance` operation keyed
  `acceptance/<sequence>/<digest>`, the digest covering the output, the
  policy and everything else the decision binds, and a new completed
  checkpoint applies it in one transaction. The run's calls, charges,
  budget, counters and recipe are unchanged. A re-evaluation stopped before
  that checkpoint is finished by the next one under the same policy, without
  validating again, unless a source was [removed](tasks.md#remove-a-source)
  from the run after it decided; the next one then decides again.
  `smythe.workflow.current_decision(store, run_id)` returns the run's
  current decision, and `decision_history(store, run_id)` every
  decision a checkpoint applied, by sequence; resume, re-evaluation and
  `smythe runs inspect` read the decision there. Resuming under a changed
  policy is still refused with `WorkflowConflictError`. A run with a renderer
  is re-evaluated only while its published document is intact; when the
  file is missing or changed, `republish()` publishes it again first
  ([re-evaluation and republication](#re-evaluation-and-republication)). An
  [operator's decision](#operator-decisions) is journaled and applied the
  same way, by a `Swarm` with or without a policy. On a run with a
  [lineage policy](lineage.md#repair-a-run-in-place), each new decision also
  journals a lineage record that supersedes the run's earlier one, in the
  same checkpoint.
- **Synthesis keys.** Such a run journals its synthesis under `synthesis`
  while every node is at generation 0, and otherwise under
  `synthesis/<digest>`, the first 16 hex digits of the SHA-256 of every node
  id and its generation, so synthesis for one graph state never replays the
  output of another, even when their generations add up to the same number.
  Other runs keep the key `synthesis`.

**Verdicts left pending.** A judge's receipt stays `pending` until its verdict
is read, and resume then reads the judge's saved reply with the rules of the
release that resumes. A pending receipt never passes unread. The rules
[above](#reading-a-verdict) date from 0.8.1: earlier releases passed replies
such as `the draft does not PASS` or an empty reply, while durable recipes
kept the verifier identity `token-v1`. A receipt that such a release left
pending is read with the current rules and fails closed, so the gate
regenerates within its allowance. Under a strict policy an unreadable verdict
is an `error` outcome that makes the decision `indeterminate`. Strict recipes
name the current rules `token-v2` (`smythe.verifier.TOKEN_VERIFIER_IDENTITY`).
A release that changes the rules changes that identity, and with it the
recipe, so a strict run is read only with the rules it started with.

## Gating from YAML and from a generated plan

`verifies` and `max_regenerations` are ordinary node fields, so a YAML
DAG declares a gate the same way Python does:

```yaml
topology: serial
nodes:
  - id: draft
    label: "Write the launch brief"
  - id: check
    label: "Verify every claim is supported. Reply PASS or FAIL."
    depends_on: [draft]
    verifies: draft
    max_regenerations: 2
```

A `verifies` that names a node the file does not define is a load
error, not a silently inert gate — the failure mode worth protecting
against is believing a run is checked when nothing is checking it.

For generated plans, state the criteria on the task and the Architect
may add the gate itself:

```python
task = Task(
    goal="Write the launch brief",
    done_when=["every claim cites a source", "under 800 words"],
)
```

`done_when` reaches the planner (which shapes the plan around it and
may add a verifier node) and every executing node (so the work knows
the bar it is held to). The Architect is instructed to add at most one
gate, on the node that produces the deliverable. A generated plan may
contain at most one gate, the gate must list the node it verifies in
`depends_on`, and it may set `max_regenerations` to at most 2. A plan
that breaks these rules is rejected and the planner asks the model for a
corrected plan. Graphs you write in Python or YAML may declare several
gates. Under a strict acceptance policy a generated plan's gate still
regenerates and can reject, but does not accept output on its own
([which gates count](#which-gates-count)).

The verifier's verdict is never the deliverable: `DELIVERABLE`
synthesis excludes verifier nodes, so a gated run returns the artefact
rather than the "PASS" that approved it.

### The final step

The executor tells each node whose output `DELIVERABLE` synthesis returns, when
it has dependencies, that it is the final step: its output is the deliverable
and must carry forward everything the task asked for, not only its own
contribution. Prompts follow the same rule as that synthesis. A judge is never
told it is the final step, and its edge to the node it verifies does not count,
so in a draft-to-judge graph the draft receives the instruction. A step that
depends on the draft is the final step instead. A step that depends only on
the judge leaves the draft deliverable, so both are final steps and
`DELIVERABLE` synthesis returns both. Executors take a keyword-only
`prompt_assembly`; `1` restores the earlier placement, which counted the
judge's edge and so told a judge that nothing depends on, rather than the
draft, that it was the final step.
[Durable text workflows](workflow-accounting.md#recovery-boundaries) keep the
earlier placement.

## Cost

A gate node is executed like any other node: it makes a provider call. When
the gate has `max_regenerations` above 0, the verifier then computes the
verdict from that call's result. A durable workflow, which accepts
`TokenVerifier` or a [gate validator](#gate-validators-in-durable-workflows),
journals the call in the `verification` phase. A `CallableVerifier` or a
validator replaces only the verdict computation, so an objective gate still
pays for its node's call, and again after each regeneration.

Regeneration buys another provider call, so it is bounded per verifier
by `max_regenerations` (default `0`, which makes the verdict advisory
only) and every attempt goes through the normal Sentinel reservation. A
verifier cannot spend past `max_budget_usd`; it can only make the run
stop sooner.

## Observability

Every send-back emits a trace span:

```python
for span in result.trace:
    if "regeneration" in span:
        print(span["regeneration"])
# {'verifier': 'check', 'target': 'draft', 'attempt': 1,
#  'limit': 2, 'reset': ['draft', 'summary']}
```

`reset` lists everything that was re-run, which is the quickest way to
see whether a gate is doing more work than you intended.

## Relationship to select-from-N

Generating several candidates and keeping the best is the image-pipeline
version of this idea; `verifies=` is the general form. The same
machinery regenerates a defective image, an unsupported claim, or a
draft that missed a constraint — the only difference is what the
verifier node looks at.
