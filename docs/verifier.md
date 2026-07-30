# Verification that gates

Smythe could always *score* work — a red-team node, a vision judge. What
it could not do was act on the score. A judge that found a misspelling
baked into a generated ad recorded its verdict, and the pipeline carried
on regardless.

A verifier closes that loop. Verification is an ordinary node, so it is
planned, budgeted, traced, and checkpointed like any other work. It just
declares which node it judges:

> **Use this for objective checks.** Measured on judged prose, gating
> fired in 4 of 30 runs, cost 34% more per run, and did not reduce the
> number of bad runs at all
> ([benchmarks/control_ablation.md](../benchmarks/control_ablation.md)).
> An LLM verifier asked "does this meet the criteria" reads a fluent
> partial deliverable and passes it — it catches *malformed* output, not
> *absent* output. The value is in the deterministic gates below: image
> dimensions, schema conformance, a required section. Reach for
> `CallableVerifier` before `TokenVerifier`.

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

When `check` fails `draft`, the executor resets `draft` **and everything
downstream of it** to pending and runs them again. Resetting only the
draft would leave a summary that was written from a rejected draft — the
run would be internally inconsistent.

## Reading a verdict

`TokenVerifier` (the default) accepts either strict JSON or prose:

```text
{"passed": false, "reason": "two claims lack sources"}
FAIL - two claims lack sources
```

An output that says neither is treated as a **pass**. This matters: a
confused judge must not be able to burn a run's regeneration budget in a
loop, and silence is not evidence of failure.

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
gate, on the node that produces the deliverable.

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
