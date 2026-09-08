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

Install Smythe 0.7.0 with the Jobs dependencies:

```bash
pip install "smythe[jobs]==0.7.0"
```

Add the provider extra needed by a live manifest, for example:

```bash
pip install "smythe[jobs,openai]==0.7.0"
pip install "smythe[jobs,gemini]==0.7.0"
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
  maximum 3,600). `max_wall_seconds` bounds the worker execution window for
  each start, resume, or reroll (default 21,600 seconds/six hours, maximum
  604,800 seconds/seven days). It includes artifact-directory preparation;
  it is not cumulative elapsed time across separate invocations. Both values
  are part of the approved plan identity.
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
- A completed response is inspected, published without replacing an existing
  file beneath `<artifact_root>/artifacts/<operation-directory>/attempt-NNNN/`, and committed with its SHA-256,
  MIME type, byte size, and observed dimensions.

Each new run receives a persistent artifact namespace. Its files live beneath
`<output_directory>/run-<namespace>`; snapshots expose that final component as
`artifact_directory`. Run IDs remain the public identifiers, so `Run`, `run`,
and `run.` can stay distinct on Windows. Separate databases also receive
separate namespaces when they share an output directory.

Before provider dispatch, the worker verifies the directory's persistent owner
marker and probes exclusive file publication. Jobs requires a local filesystem
with file hard-link support. An unsupported filesystem, conflicting owner, or
existing destination fails without replacing earlier bytes. A collision found
after a provider response remains an unknown outcome for explicit investigation.

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

## List and inspect runs

Find a run, inspect its attempts, or write a local report:

```bash
smythe jobs list --limit 50
smythe jobs list --status needs_attention --json
smythe jobs inspect RUN_ID
smythe jobs inspect RUN_ID --operation "tile[3]" --json
smythe jobs inspect RUN_ID --out job-report.html
```

The HTML report is a self-contained black-and-white document. It shows
whole-run state and exact USD balances, then operation prompts, responses,
attempts, call records, validation errors, artifact receipts, and recent
events. Expand records to read their text and attempt history. Rejected and unknown
attempts remain visible alongside accepted outputs. The report includes
stored prompts, responses, and local paths; it is intended for local review.
It runs no scripts and loads no remote resources.

Lists are ordered newest first. Both commands accept `--limit` (1–500,
default 50) and `--offset` (default 0). Inspection includes complete attempt
lineage for the selected operation page; its cost and status totals still
cover the entire run. `--operation` accepts an exact key or stable ID.
`--events-limit` controls the most recent events (1–1,000, default 100), with
explicit counts when earlier events are omitted. Select another operation
page or use `export` for the complete journal.
Each command reads a fresh snapshot; new runs or status changes can shift
list offsets between requests.

Artifact integrity is a timestamped observation of the files on disk after
the ledger snapshot. Inspection compares size and SHA-256, reading at most
64 MiB per invocation and 32 MiB per file. Missing, changed, unreadable, and
unchecked files are labeled. Unsafe paths, symbolic links, and junctions are
not followed. These checks preserve the recorded acceptance decision and do
not repeat image decoding or quality evaluation.

`list`, `inspect`, `status`, `attach`, and `export` open the existing Jobs database in
read-only mode, without provider initialization or schema migration. Missing,
foreign, and unsupported databases fail closed. A live WAL reader sees one
consistent committed ledger snapshot while the worker continues. `--out`
writes the requested report atomically and rejects the database and its WAL
or SHM files as output destinations.

Successful `list` and `inspect` commands return exit code 0 even when a run
needs attention. Their JSON output retains the run's recorded status. The
existing `status` and `export` commands retain their job-state exit codes.

## Detached execution and attachment

Add `--detach` to start an approved job in a separate worker:

```bash
smythe jobs run job.yaml --approve approve_v1_... --detach
smythe jobs attach RUN_ID
smythe jobs resume RUN_ID --detach
```

The launcher returns after the worker proves current lease ownership and the
launcher records dispatch authorization. It reports the run ID, actual worker PID, and
private log and receipt paths. The worker uses the launching Python environment.
On supported hosts, the worker survives its launcher exiting and detaches from
the console. Windows requires permission to break away from the launcher's
process container. A bounded, isolated no-op probe checks that permission
before the actual worker is launched once. A denied probe refuses detached
startup before worker creation or dispatch authorization and retains the run
for recovery. Run `jobs resume RUN_ID` in the foreground, or retry detached
resume from a host that permits breakaway. Smythe does not weaken host policy
or fall back to a console-only launch that could die with its launcher.
[Windows process-container rules](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
also apply to containers created by Python launchers.
[Windows qualification and controlled refusal evidence](../benchmarks/results/windows_operator_20260907/README.md).

Failure before authorization prevents a later dispatch. An interruption or
I/O error after authorization may leave
the worker active; error metadata distinguishes that boundary and retains
recovery information for the saved run. `--startup-timeout-s` accepts 0.1–300
seconds and defaults to 30.

`attach` observes saved state without initializing providers, recovering work,
or controlling the worker. It follows for up to 30 seconds by default; set
`--timeout-s` from 0 to 3,600 and `--poll-interval-s` from 0.05 to 60 seconds
(default 1). A timeout ends the attachment. Ctrl+C disconnects with exit code
130. Both leave the job running. `--json` returns one document, including the
final attachment state. Worker failure and expired ownership return code 7.

Launch receipts live beside the database in `.DATABASE.workers/RUN_ID_SHA256/LAUNCH_ID`.
Hashing the exact run ID keeps distinct IDs separate on case-insensitive filesystems.
Directories restrict access to the current user (and SYSTEM on Windows).
Logs can contain provider errors and local paths. The latest launch receipt
and current lease are separate observations: a competing failed launch does
not identify the active worker. Saved PIDs are never used to signal a process.

## Stop and resume safely

Stop requests a durable pause, then observes the drain:

```bash
smythe jobs stop RUN_ID --reason "Review the accepted outputs"
smythe jobs status RUN_ID --json
smythe jobs resume RUN_ID --detach
```

The request blocks new attempt, reservation, and dispatch admission. Calls
already marked dispatched finish through the normal accounting and artifact
path. A pause does not cancel a provider request. Safe prepared calls return
to pending; accepted outputs and attempt history remain intact. The run becomes
`paused` after admitted work drains and pending work remains. Completed work,
unknown outcomes, and budget overruns retain their corresponding statuses.

`stop` uses the same timeout and polling ranges as `attach`. Set `--timeout-s 0`
to return immediately after saving the request. A timeout or Ctrl+C ends only
the observation; the pause request persists. An idle approved run can retain
`pause_requested: true` without starting a worker. JSON includes the captured
`stop_request` and current `control`, so a later operator action remains visible.

Each stop advances a durable pause generation. Explicit resume captures that
generation before preflight and clears only the matching request when it
acquires ownership. A stop issued after that resume intent survives. Initial
detached launch never clears a pause. Reroll queues selected work under the
same pause control; explicit resume is needed to reopen admission.

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

Each new ownership period receives a higher lease epoch. Renewals retain it. Every worker
write checks the live owner and epoch inside the same SQLite transaction as the
mutation. Attempts retain their originating owner and epoch. An expired worker
cannot dispatch, settle a call, replace an accepted result, or finalize the run
after another worker takes over. Lease time is sampled after the transaction
acquires its write lock, so lock contention cannot revive an expired lease.

This fences durable dispatch admission and journal writes. A request already
marked dispatched can still reach or finish at the provider after ownership
expires. Its uncommitted outcome remains unknown until investigated; the lease
does not cancel an external request or make it safe to repeat automatically.

A completed run starts zero new operations. Safe pre-dispatch interruptions
return to pending. Dispatched-but-uncommitted calls remain `unknown_outcome`,
and the run becomes `needs_attention`.

Invalid provider cost or token usage also becomes `unknown_outcome` and stops
queued dispatches. Already-dispatched calls finish and retain their charges.
This stop survives process restart: ordinary resume dispatches no further work
while an invalid-accounting outcome remains unresolved. Repair the provider
adapter and investigate the charge before explicitly acknowledging a reroll.
The earlier unknown exposure remains in the ledger after that reroll.

Do not resolve an unknown outcome by ordinary `resume`; that is intentionally a
no-op for the ambiguous operation. First investigate the provider account and
artifact destination. If a duplicate call is acceptable, acknowledge that
risk explicitly during a selective reroll.

## Jobs database upgrades

**Unreleased after 0.7.0:** concurrent openers use bounded retries for WAL
setup, then revalidate and create or migrate the schema in one transaction.
Failed initialization rolls back. This repair preserves schema version 4 and
existing records; read-only inspection performs no migration or WAL setup.

The writable store upgrades earlier Jobs databases to schema version 4. Stop
older workers using their existing controls before upgrading; an unexpired
earlier-version lease blocks migration. The new `jobs stop` command requires
the upgraded store and cannot pause a live older worker. Version 3 workers
enforce ownership but cannot honor durable pause requests.
Existing attempt provenance is preserved, including unbound pre-v3 attempts. Runs that held a
legacy lease remain fenced after upgrade, including after lease release.
Read-only inspection supports versions 2 and 3 without migration and reports
whether lease fencing and durable pause control are supported in that snapshot.

Migrated runs retain their original `<output_directory>/<RUN_ID>` directories
and artifact receipts. Before the first upgraded execution, Smythe verifies
their accepted bytes and claims the directory with a persistent owner marker.
Ambiguous legacy directories shared by multiple runs fail closed. Keep the
original database and outputs together; inspection and export report the saved
artifact location without moving files.

Code using `JobRunner` receives these checks automatically. Direct store users
must retain the `RunLease` returned by `acquire_run_lease` and pass it as `lease=`
to worker mutations, heartbeat, and release. Omitting the token is supported
only for a run that has never acquired a lease. A new owner's token cannot
settle an earlier owner's attempt; use recovery to classify the old work.

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
`outputs/run-<namespace>` for a new run or `outputs/RUN_ID` for a migrated run.
The export declares `manifest_root: "."` as its relocatable
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
| 0 | command succeeded; `list` and `inspect` succeed regardless of recorded job status |
| 1 | job finished `failed`, `partial`, `needs_attention`, or `budget_overrun` |
| 2 | invalid manifest, preflight input, or command value |
| 3 | provider configuration or optional dependency failure |
| 4 | approval mismatch |
| 5 | budget admission failure |
| 6 | unknown run or invalid state transition |
| 7 | local SQLite/filesystem failure, detached startup failure, or attachment detects a failed worker or expired lease |
| 130 | detached startup interrupted, or attachment disconnected with Ctrl+C |

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

## High-fan-out example workload

The [Jobs scale and recovery protocol](../benchmarks/jobs_scale_benchmark.md)
executes actual offline artifact operations, hard-kills an owned worker,
preserves completed outputs during resume, and explicitly rerolls unknown
calls. It verifies durable call identity and artifact receipts using tiny PNG
fixtures. Its evidence is separate from glyph generation and provider latency.
The [completed 5,000-operation campaign](../benchmarks/jobs_scale_5000_20260907_results.md)
preserved accepted work and recovered 2,500 pending operations. Eight interrupted
operations required explicit rerolls; their original unknown call records remain
in the ledger. Its frozen schema-v3 runtime predates
the schema-v4 namespaces and operator commands documented above.

The related [glyph screensaver benchmark](../benchmarks/glyph_screensaver_benchmark.md)
is one concrete use of Smythe's general-purpose artifact execution model. It
uses a 192-node `BROADCAST_REDUCE` graph to generate 192 original cyber-glyph
tiles, validate dimensions and uniqueness, and assemble a 1920x1080 still,
animated GIF, contact-sheet atlas, and standalone animated HTML canvas. Its
offline lane is deterministic and free; its live Gemini and GPT Image lanes require
explicit whole-call and whole-run ceilings. The benchmark currently exercises
the `Swarm` scheduler directly to isolate concurrency, while Jobs v1 supplies
the durable operator semantics intended for production artifact runs.
