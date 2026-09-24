# Verification that gates

A verifier can reject an output and trigger bounded regeneration. Verification
is an ordinary node: planned, budgeted, traced, and checkpointed. It declares
which node it judges:

> **Lead with objective checks.** Image dimensions, schema conformance,
> required sections, hashes, and domain rules have definitive verdicts and no
> judging cost. Use `CallableVerifier` for those contracts. Reserve
> `TokenVerifier` for a criterion that genuinely requires model judgment; the
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
away: a revision that drops `check` or `draft`, or rewires `check` so that it
no longer depends on `draft`, is rejected.

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
readable verdict is the norm.

## Gating without a model

Most useful gates are objective and need no LLM at all — image
dimensions, JSON schema conformance, a required section:

```python
from smythe import CallableVerifier, Swarm

long_enough = CallableVerifier(
    lambda verifier_node, target: len(str(target.result)) > 500,
)
swarm = Swarm(verifier=long_enough, ...)
```

A `CallableVerifier` can return a bool or a `Verdict(passed=..., reason=...)`
when you want the reason recorded in the trace.

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
gates.

The verifier's verdict is never the deliverable: `DELIVERABLE`
synthesis excludes verifier nodes, so a gated run returns the artefact
rather than the "PASS" that approved it.

## Cost

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
