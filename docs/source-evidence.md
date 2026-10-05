# Source evidence

Source evidence records what a deliverable's claims rest on: the sources a
run drew on, each claim, the passages quoted for and against it, and the
questions it leaves open. An artifact carries its records, and a final
validator checks them mechanically against the sources the run was given and
against the artifact itself. These records are separate from the durable
journal's response evidence, which holds raw provider and read responses.

## Records

`smythe.evidence` defines three versioned JSON records, each also exported
from `smythe`:

- **`SourceRecord`** identifies one source by `source_id` and its
  `source_version` or `sha256` (the SHA-256 of its text as UTF-8). It holds
  the retained `text`, or, without text, a `reference` to where the text is
  kept, or a `tombstone`: why the text is gone and, optionally, when
  (`removed_at`). A tombstone keeps the id, version and digest and nothing of
  the content. `SourceRecord.from_context_source(source)` retains a run's
  [`ContextSource`](tasks.md#context-snapshots), `without_text()` keeps its
  id, version and digest, and `tombstoned(reason)` replaces it with a
  tombstone.
- **`ClaimRecord`** states one claim: a `claim_id`, its `text`, the
  `supporting` and `contradicting` passages, each a `ClaimPassage` of a
  `source_id` and a `quote` copied exactly from that source, an optional
  `scope`, the `unresolved` questions and an optional proposed `conclusion`.
- **`SourceEvidence`** holds an artifact's `sources` and `claims`, each id
  once. Its `sha256` is the digest of its canonical JSON.

`to_dict()` writes each record with `"version": 1`, and `from_dict()` rejects
another version, a missing or unknown key, or a retained text that does not
match its digest, rather than guessing. `summary()` replaces each source's
text with its UTF-8 size.

```python
from smythe import ClaimPassage, ClaimRecord, ContextSource, SourceEvidence, SourceRecord

poll = ContextSource("survey-2019", "An informal 2019 poll: most customers prefer annual billing.",
                     version="2019")
audit = ContextSource("renewals-2026", "Audited 2026 renewals: 71% chose monthly billing.",
                      version="2026-q2")
claim = ClaimRecord(
    "billing-preference", "Most customers prefer annual billing.",
    supporting=[ClaimPassage("survey-2019", "most customers prefer annual billing")],
    contradicting=[ClaimPassage("renewals-2026", "71% chose monthly billing")],
    unresolved=["Whom did the 2019 poll ask?"], conclusion="Treat the preference as contested.",
)
evidence = SourceEvidence([SourceRecord.from_context_source(item).without_text()
                           for item in (poll, audit)], [claim])
```

## Carry evidence in an artifact

A text artifact carries its evidence in one fenced block whose info string is
`source-evidence`, holding a `SourceEvidence` record as strict JSON.
`smythe.evidence.render_source_evidence(evidence)` writes that block, as
indented JSON that no line of can close early; records meant for an artifact
usually leave out source text. `read_source_evidence(text)` returns the
evidence and the artifact's *body*: the text without the block. Text without
a block gives `(None, text)`. Two blocks, a block that is not closed, or a
block that is not one strict JSON record (a repeated key or a non-finite
number is not strict) raises `ValueError`. Fences follow Markdown, so a fence
line inside another fenced block is content.

````text
# Billing

A 2019 poll suggested most customers prefer annual billing, but audited
renewals show that 71% chose monthly billing.

```source-evidence
{ ... the SourceEvidence record ... }
```
````

## Check evidence mechanically

`SourceEvidenceValidator(sources)` (in `smythe.validators`, also exported
from `smythe`) is a [final validator](verifier.md#built-in-final-validators).
*sources* are the sources the run was given: a `ContextSnapshot`, or a
sequence of `ContextSource` or `SourceRecord` values, each with a digest.
Build it from the same sources the run records, so that its checks compare
the evidence with what the run's nodes were shown.

```python
from smythe import AcceptancePolicy, SourceEvidenceValidator, Swarm

checks = SourceEvidenceValidator([poll, audit])
swarm = Swarm(..., run_store=store, context_sources=[poll, audit],
              acceptance=AcceptancePolicy(mode="strict", final_validators=[checks]))
```

The output must be UTF-8 and carry one readable evidence block; otherwise it
is rejected with `not_utf8`, `evidence_missing` or `evidence_invalid`. The
specification is a JSON object with two optional keys: `min_claims` (default
1) rejects evidence with fewer claims with `claims_too_few`, so an empty block
does not pass, and `checks` selects the checks to run, all by default:

| Check | Requires | Finding |
|---|---|---|
| `source_ids` | every source the evidence lists or a passage cites is one of the validator's sources | `source_unknown` |
| `source_versions` | each listed source's stated version and SHA-256 equal those of the source with its id | `source_version_mismatch`, `source_digest_mismatch` |
| `passages` | each supporting and contradicting passage is quoted verbatim from the retained text of its source | `passage_not_found` |
| `references` | each passage cites a source the evidence lists | `reference_dangling` |
| `contradictions` | each contradicting passage appears verbatim in the artifact's body | `contradiction_missing` |

Matching is exact: a paraphrase, a changed space or a quote from another
source is not found, and a contradiction quoted only inside the evidence block
has not reached the artifact. A claim that cites no passage passes these
checks. A failed check rejects the output. A report
lists at most 32 findings, rejections first, and a last `findings_truncated`
finding counts the rest.

The validator's identity is `smythe.source-evidence`, version `1`, with a
configuration digest of its specification and each source's id, version,
digest and state, never its text. A policy or record that names the validator therefore
pins the sources it checks against, and a source whose text changes, or that
becomes a tombstone, gives a different identity. `name=` renames it.

### Unavailable sources

A tombstoned source, or one whose text the validator does not retain, is
unavailable: passages quoted from it cannot be checked, and the report names
it with a `source_unavailable` finding. Without a rejection found elsewhere,
the report's outcome is `error`, which strict acceptance decides as
`indeterminate` (or `needs_review` under `on_invalid`), never as accepted;
nothing is read from a cache in its place. The tombstone keeps the source's
version and digest, so the other checks still run.

### What the checks do not prove

These checks are mechanical. A passage quoted exactly can still fail to
support its claim, and a poll of 40 visitors passes as easily as an audit of
18,400 customers. Whether evidence supports a claim, or a source deserves its
weight, is a judgement for a [gate](verifier.md) whose judge reads the
artifact, such as one read with `TokenVerifier`, or for a reviewer. Under a
strict policy a failing gate still decides `rejected` when every mechanical
check passed.

## Durable runs

A durable run whose strict policy has a `SourceEvidenceValidator` journals
the evidence its delivered output carries. After the `synthesis` operation
and before the `acceptance` decision, an operation of kind `source_evidence`,
keyed `source-evidence/<sha256>`, records:

- **inputs**: the output's SHA-256 and the run's source set, as the
  checkpoint's `{"revision", "sha256"}` record, or `null` for a run without
  [context assembly](tasks.md#context-snapshots). The key's digest is of these
  inputs as canonical JSON (`smythe.evidence.source_evidence_key`);
- **result**: the `SourceEvidence` record read from the output, or `null`
  with a `problem` that says why the output carries none.

The completed checkpoint consumes it together with the synthesis and the
decision. A resumed run reads the recorded result without reading the output
again; the validators, not this record, decide acceptance. Other runs record
no such operation. The journal format is unchanged: operation kinds are free
text.

In a run with context assembly, each `SourceEvidenceValidator` must check
exactly the run's current source set: the same source ids, each with the
same version and digest. A validator built from other sources, such as the
sources a run had before [`Swarm.refresh_context`](tasks.md#context-snapshots)
gave it new ones, could accept evidence for sources the run's nodes were
never shown, so it does not run: its outcome is `error`, the decision follows
`on_invalid` (`indeterminate` by default), and the reason names the sources
that differ. The recipe keeps the policy the run started with, so re-evaluate
such a run with [`Swarm.reevaluate`](verifier.md#re-evaluation) and a
validator built from its current sources. A run without context assembly
records no source set, and its validators check the sources they were given.

### Inspect evidence

`smythe.evidence.load_source_evidence(store, run_id)` returns the evidence a
completed run delivered, or `None` when it recorded none: the operation's key
and state, the output digest, the run's source set, any `problem`, the claim
records, and each source record's `summary()` with `available`: whether its
text is retained, in the record or in the run's recorded sources with the
same id, version and digest, and `removed`: the reason and time of the
tombstone of a source [removed from the run](tasks.md#remove-a-source), or
`None`. A tombstoned or removed source is never available, even when the
journal or the evidence record still holds its text. Source text is left out
unless the caller asks with `include_text=True`, which adds each available
source's `text` and `None` for the others. `smythe runs inspect` shows the
same evidence, summarized, with each claim's unresolved questions as
[open issues](runs.md#research-runs).

```python
from smythe.evidence import load_source_evidence

evidence = load_source_evidence(store, result.execution_id)
for claim in evidence["claims"]:
    print(claim["claim_id"], claim["unresolved"])
```

`ValueError` means the journal holds another record under the evidence key.
Explicit `include_text=True` calls expose source text; protect the journal as
the [inspection notes](workflow-accounting.md#inspect-a-run) describe.
