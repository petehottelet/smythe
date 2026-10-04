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
generation are cleared.

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

`report.validator` optionally holds a `ValidatorIdentity`: the validator's
name, version and `config_sha256`, the SHA-256 of its configuration as
canonical JSON (`ValidatorIdentity.for_config(name, version, config)` computes
it). `to_dict()` and `from_dict()` convert reports, findings and identities to
and from JSON data; a tuple in a finding's `expected` or `observed` value
becomes a JSON array, and `from_dict()` rejects unknown keys and outcomes.

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
  validator's outcome. Checkpoints and JSON keep it under
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

`SwarmResult.acceptance` holds the record, or `None` when there is none.
`SwarmResult.accepted` is true only for an `accepted` record. When a record
exists and does not accept, `output` is `""` and the text is in
`rejected_output`. `SwarmResult.to_dict()` returns the result as JSON data
with the record under `delivery_acceptance`. Advisory and legacy runs have
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

`Swarm(acceptance=...)` and `Swarm.from_yaml(..., acceptance=...)` take the
policy. It comes only from that argument: a YAML file, a graph, a `Task`, a
generated plan or a supervisor revision cannot set or weaken it. `None`, the
default, and `AcceptancePolicy()` keep the behavior described above. Strict
acceptance runs on ordinary runs; a `Swarm` with a durable `run_store` refuses
a strict policy with `WorkflowBindingError`.

**The delivered output decides.** After synthesis, the run decides on exactly
the text it delivers, as UTF-8 bytes. Every final validator runs once on those
bytes as local code, with no provider call and no graph node, and every gate
in the graph is read from its recorded final verdict. The decision is:

1. `rejected` when a final validator rejects the output. A final rejection is
   terminal: nothing is regenerated.
2. Otherwise `on_exhausted` when a gate's final verdict fails.
3. Otherwise `on_invalid` when a final validator or a gate could not decide,
   or a delivered node wrote files.
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
[before any call](#acceptance-obligations).

<a id="which-gates-count"></a>
**Which gates count.** Acceptance criteria are inputs you control, so a gate
accepts output on its own only when you supplied it: in a graph you pass to
`execute()` or load from YAML, or in a plan that a `DeterministicArchitect`
builds or a `ConstrainedArchitect` composes from your templates. A strict
`Swarm` marks each gate of a plan that any other architect, such as
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
started.

A final validator that raises, returns an awaitable or anything other than a
`ValidationReport`, reports `outcome="error"`, or returns a report naming
another validator has outcome `error`. Artifact deliverables are not supported
yet: artifact records hold no hash of their files, so when a delivered node
wrote files the decision is `on_invalid` whatever the validators report.

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
saved output against the record's digest, and the task, source generations
and, when the resuming `Swarm` has a strict policy, its `sha256` against the
record; any mismatch returns an `indeterminate` record for the saved output,
never acceptance.

A completed checkpoint without a record (versions 1 to 4, or version 5
without `delivery_acceptance` in `requires`) was never decided. A `Swarm`
without a strict policy returns its text in `output`, as before. A strict
`Swarm` withholds it: the result has `output=""`, the saved text in
`rejected_output` and `acceptance=None`, and a warning on the `smythe.swarm`
logger says how to obtain a decision. Re-run the task under the strict
policy, or decide the saved output yourself with
`evaluate_delivery(policy, result.graph, result.rejected_output, ...)`. The
checkpoint is not changed.

The trace gains one span with `node_id` `acceptance:delivery`, status
`acceptance_<decision>` and the first reason as its label; `Tracer.summary()`
keys are unchanged, and the validators' time counts as verification in
[`SwarmResult.timing`](execution.md#timing).
`smythe.acceptance.evaluate_delivery(policy, graph, output, strategy=...,
sources=...)` makes the same decision outside a run, and
`smythe.synthesizer.deliverable_sources(graph)` returns the nodes
`DELIVERABLE` synthesis delivers and whether it fell back.

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

## Cost

A gate node is executed like any other node: it makes a provider call. When
the gate has `max_regenerations` above 0, the verifier then computes the
verdict from that call's result. A durable workflow, which accepts only
`TokenVerifier`, journals the call in the `verification` phase. A
`CallableVerifier` replaces only the verdict computation, so an objective gate
still pays for its node's call, and again after each regeneration.

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
