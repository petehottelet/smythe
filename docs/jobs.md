# Durable artifact jobs

Smythe Jobs v1 runs large sets of independent artifact operations through a
strict, inspectable contract:

```text
manifest -> preflight -> approval -> dispatch journal -> artifacts -> selective reroll
```

It is designed for work where a duplicate provider call can cost money and a
partially completed run is still valuable. The ordinary `Swarm` checkpoint
system remains the right surface for resuming a dependency graph; Jobs v1 adds
a durable dispatch boundary, per-attempt lineage, and operator commands for
wide artifact production.

## Install

```bash
pip install "smythe[jobs]"
```

Add the provider extra needed by a live manifest, for example:

```bash
pip install "smythe[jobs,openai]"
pip install "smythe[jobs,gemini]"
```

The installed command is `smythe`. Job state defaults to
`~/.smythe/jobs.sqlite3`; use `--store PATH` to choose another SQLite file.

## Manifest v1

Jobs accept strict JSON or YAML. Unknown fields are errors, USD values support
at most six decimal places, output and attachment paths must remain beneath the
manifest directory, and secret-like provider option keys are rejected.
Credentials belong in provider environment variables, never in a manifest or
export.

This zero-cost manifest exercises the complete workflow offline:

```yaml
version: 1
name: offline-artifact-smoke

profiles:
  - name: local
    provider: offline
    model: offline-image
    max_cost_per_call_usd: "0"
    options:
      artifacts_per_call: 1

operations:
  - key: tile
    count: 16
    prompt: Render one deterministic fixture tile.
    profile: local
    artifact:
      mime_type: image/png
      width: 1
      height: 1
      filename: tile.png

execution:
  max_concurrency: 8
  max_attempts: 2
  max_budget_usd: "0"
  call_timeout_s: 300
  max_wall_seconds: 21600
  output_directory: outputs
```

The contract has four sections:

- `profiles` select `offline`, `openai_image`, or `gemini_image`, a model,
  provider options, and an inclusive maximum cost for one call. Billable
  profiles require a positive ceiling; offline profiles require exactly zero.
- `operations` define a stable key, prompt, provider profile, optional count,
  relative attachments, and the expected artifact MIME type, dimensions, and
  filename. A count of 16 expands to `tile[0]` through `tile[15]`.
- `execution.max_concurrency` bounds active calls. `max_attempts` bounds the
  complete attempt lineage, including later rerolls.
- `execution.call_timeout_s` bounds each provider attempt (default 300 seconds,
  maximum 3,600). `max_wall_seconds` bounds the whole run (default 21,600
  seconds/six hours, maximum 604,800 seconds/seven days). Both values are part
  of the approved plan identity.
- `execution.max_budget_usd` is the outer policy ceiling. The expanded
  worst-case cost is every operation multiplied by its per-call ceiling and
  maximum attempts; preflight fails if that total does not fit.

Version 1 admits at most 64 profiles, 512 operation templates, and 5,000
expanded operations. Per-template count is capped at 5,000, concurrency at 256,
attempts at 10, and prompts and profile options at 64 KiB each. Manifests are
capped at 2 MiB; duplicate JSON/YAML keys and YAML aliases are rejected at the
parser boundary. Raster requests are capped at 16,384 pixels per edge and
67,108,864 total pixels. Monetary values must fit a signed 63-bit micro-USD
integer.

Requested artifact names must be portable across Windows, macOS, and Linux.
Reserved DOS device names, control/forbidden characters, trailing dots or
spaces, non-NFC names, and names beyond 255 UTF-8 bytes are rejected. Raster
filename extensions must match the declared PNG, JPEG, GIF, or WebP MIME type.

Print the maintained JSON Schema at any time:

```bash
smythe jobs schema --json
```

### Attachments

Attachment paths are relative to the manifest. Preflight rejects traversal and
symlink escapes, records SHA-256, size, and MIME fingerprints, and includes
those fingerprints in the plan hash. A changed input therefore invalidates the
old approval. Each operation can reference at most 16 attachments; a job can
reference at most 128 unique attachments. Preflight caps each attachment at 8
MiB and their aggregate size at 64 MiB before bounded streaming hashes begin.

Gemini image profiles can receive reference images. The OpenAI image generation
adapter currently rejects operations with attachments rather than silently
discarding them; reference-image support there requires the Image Edits path.

## Validate, plan, and approve

Validation performs strict parsing, deterministic expansion, attachment
fingerprinting, provider-option checks, and whole-job cost preflight without
dispatching an artifact call:

```bash
smythe jobs validate job.yaml
```

Planning prints the expanded plan and an approval token:

```bash
smythe jobs plan job.yaml --max-spend-usd 0.64
```

The approved ceiling must be at least the complete worst-case cost and no more
than the manifest budget. If `--max-spend-usd` is omitted, it defaults to the
manifest budget.

The token binds all of the following:

- the canonical manifest hash;
- the expanded plan hash, including provider configuration and attachment
  fingerprints; and
- the exact approved spend ceiling.

Changing any of them produces a different token. `jobs run` recomputes the
plan and refuses a mismatch before creating a run:

```bash
smythe jobs run job.yaml \
  --approve approve_v1_... \
  --max-spend-usd 0.64
```

Use the same `--max-spend-usd` value for `plan` and `run`. The token is an
explicit plan-and-spend acknowledgment, not an authentication credential or a
signature from a secret key.

## The dispatch journal

Each run is stored incrementally in SQLite using WAL mode. Before a provider
call, Smythe creates an attempt, reserves the call's inclusive ceiling, records
a prepared call and event, and only then marks it dispatched. The journal
tracks integer micro-USD values to avoid floating-point accounting drift.

The distinction between `prepared` and `dispatched` controls recovery:

- A process that stops while a call is only `prepared` is safe to resume; the
  provider was not dispatched.
- A process or connection that stops after `dispatched` becomes
  `unknown_outcome`. The provider may have completed and billed the request, so
  Smythe does not issue an automatic duplicate.
- A completed response is inspected, written atomically beneath
  `artifacts/<operation-key>/attempt-NNNN/`, and committed with its SHA-256,
  MIME type, byte size, and observed dimensions.

The local journal derives an idempotency key for every attempt, but Jobs v1
does not yet transmit that key to every upstream provider. It is durable local
identity, not a claim that an external API will deduplicate a repeated call.

Inspect current state and, optionally, the append-only event sequence:

```bash
smythe jobs status RUN_ID
smythe jobs status RUN_ID --events --json
```

Snapshots distinguish confirmed cost, reserved cost, and unresolved exposure.
`cost_is_complete` is false while an unknown or incomplete provider charge
remains exposed; `cost_contains_estimates` reports whether any recorded call
used a conservative estimate.

## Resume and unknown outcomes

Resume revalidates the stored manifest, rebuilds the same plan, verifies the
stored approval, rechecks provider inputs, and runs only pending work:

```bash
smythe jobs resume RUN_ID
```

Start, resume, and reroll hold a renewable SQLite run lease for the entire
state transition and execution window. A second process cannot recover or
reroll the same live run; it receives a job-state error until the owner
releases the lease or its heartbeat expires after a crash.

A completed run starts zero new operations. Safe pre-dispatch interruptions
return to pending. Dispatched-but-uncommitted calls remain `unknown_outcome`,
and the run becomes `needs_attention`.

Do not resolve an unknown outcome by ordinary `resume`; that is intentionally a
no-op for the ambiguous operation. First investigate the provider account and
artifact destination. If a duplicate call is acceptable, acknowledge that
risk explicitly during a selective reroll.

## Selective rerolls

Only `failed` and `rejected` operations are normally rerollable. Specify an
operation key such as `tile[3]` or its stable operation ID, plus a reason:

```bash
smythe jobs reroll RUN_ID "tile[3]" --reason "artifact failed review"
```

An unknown outcome additionally requires explicit duplicate-spend
acknowledgment:

```bash
smythe jobs reroll RUN_ID "tile[3]" \
  --reason "provider shows no completed request" \
  --acknowledge-unknown
```

Rerolls consume the same original approval and attempt allowance. The selected
operation receives a new immutable attempt linked to its predecessor; other
operations are not reset, and prior artifact records remain in the journal.
Successful operations cannot be casually rerolled, which preserves accepted
outputs byte-for-byte.

## Export

Export includes the manifest and plan hashes, operation and attempt lineage,
cost state, artifact receipts, and the event journal:

```bash
smythe jobs export RUN_ID --out run-export.json
```

The file is written atomically. Exported artifact paths use forward-slash
relative paths beneath an explicit `artifact_root` such as
`outputs/RUN_ID`. The export declares `manifest_root: "."` as its relocatable
base and `paths_relative_to: "artifact_root"`; join the chosen relocated
manifest root, `artifact_root`, and each artifact `relative_path` to locate its
bytes.

## Provider profiles

| Provider | Environment | Supported options |
|---|---|---|
| `offline` | none | `artifacts_per_call`, `echo_prefix` |
| `openai_image` | `OPENAI_API_KEY` | `size`, `quality`, `output_format`, `output_compression`, `moderation`, `n` |
| `gemini_image` | `GOOGLE_API_KEY` | `response_modalities`, `image_config` |

Provider endpoints are a trusted operator setting, not manifest data.
`openai_image` rejects a manifest-level `base_url`; an operator who
intentionally uses an OpenAI-compatible endpoint may set `OPENAI_BASE_URL` in
the trusted process environment.

Live pricing is deliberately not hard-coded. Set
`max_cost_per_call_usd` from current provider pricing and include every input
and output charge that one request can incur.

## CLI exit codes

| Code | Meaning |
|---:|---|
| 0 | command succeeded and the resulting job is not in a failed terminal state |
| 1 | job finished `failed`, `partial`, `needs_attention`, or `budget_overrun` |
| 2 | invalid manifest, preflight input, or command value |
| 3 | provider configuration or optional dependency failure |
| 4 | approval mismatch |
| 5 | budget admission failure |
| 6 | unknown run or invalid state transition |
| 7 | local SQLite or filesystem failure |

Use `--json` for one compact document on stdout, including structured errors.

## Python API

The same workflow is available without the CLI:

```python
import asyncio

from smythe.jobs.loading import load_manifest
from smythe.jobs.preflight import make_approval, preflight_job
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import SQLiteRunStore

manifest, root = load_manifest("job.yaml")
plan = preflight_job(manifest, manifest_root=root)
approval = make_approval(plan, approved_max_cost_usd="0")

with SQLiteRunStore("jobs.sqlite3") as store:
    result = asyncio.run(
        JobRunner(store).start(plan, approval, manifest_root=root)
    )
```

## Current boundaries

- Jobs v1 is a local SQLite worker, not a distributed queue or hosted service.
- Built-in artifact acceptance checks MIME type and optional pixel dimensions.
  The public `smythe.assets` package adds typed brand modes, deterministic
  finishing, hash-bound receipts, and richer image validation for custom
  pipelines; those policies are not automatically inferred from a generic job
  manifest.
- Rerolls are operator-selected; there is no automatic vision-judge promotion
  policy in the runtime.
- Unknown provider outcomes require human investigation or explicit duplicate
  acknowledgment.

## Flagship fan-out workload

The related [glyph screensaver benchmark](../benchmarks/glyph_screensaver_benchmark.md)
uses a 64-node `BROADCAST_REDUCE` graph to generate 64 original cyber-glyph
tiles, validate dimensions and uniqueness, and assemble a 1920x1080 still,
animated GIF, contact-sheet atlas, and standalone animated HTML canvas. Its
offline lane is deterministic and free; its optional GPT Image lane requires
explicit whole-call and whole-run ceilings. The benchmark currently exercises
the `Swarm` scheduler directly to isolate concurrency, while Jobs v1 supplies
the durable operator semantics intended for production artifact runs.
