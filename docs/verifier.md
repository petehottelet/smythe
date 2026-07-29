# Verification that gates

Smythe could always *score* work — a red-team node, a vision judge. What
it could not do was act on the score. A judge that found a misspelling
baked into a generated ad recorded its verdict, and the pipeline carried
on regardless.

A verifier closes that loop. Verification is an ordinary node, so it is
planned, budgeted, traced, and checkpointed like any other work. It just
declares which node it judges:

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
