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

Install Smythe 0.9.1 with the Jobs dependencies:

```bash
pip install "smythe[jobs]==0.9.1"
```

Add the provider extra needed by a live manifest, for example:

```bash
pip install "smythe[jobs,openai]==0.9.1"
pip install "smythe[jobs,gemini]==0.9.1"
```

The installed command is `smythe`. Job state defaults to
`~/.smythe/jobs.sqlite3`; use `--store PATH` to choose another SQLite file.
The journal stores prompts and responses, so a new database file is created
readable only by its owner (mode 0600, which SQLite also applies to its `-wal`
and `-shm` files), and a missing parent directory is created as 0700. Existing
files and directories keep their permissions.

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
  relative attachments, and the expected artifact MIME type, dimensions,
  filename, and optional [registered validators](#registered-validators). A
  count of 16 expands to `tile[0]` through `tile[15]`.
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

Validation also rejects an artifact the profile cannot produce. An `offline`
profile emits 1x1 PNGs, so it accepts only `image/png` with no dimensions or
1x1. A `gemini_image` profile accepts only PNG, JPEG, GIF or WebP. The
[support matrix](support-matrix.md) lists every profile's checks.

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

An approval token creates one run. The token is derived only from the manifest
hash, plan hash and ceiling, so approving the same plan at the same ceiling
again yields the same token. A Jobs database refuses a token that already created a run, before
creating anything, with exit code 4 and an error that names the earlier run;
continue that run with `smythe jobs resume RUN_ID` against the same `--store`. To start another run with
its own full ceiling on purpose, add `--new-run`. The new run's `run_created`
event records how many earlier runs used the token as
`approval_reuse.prior_run_count` and the five most recent run IDs as
`approval_reuse.prior_run_ids`. A
different ceiling produces a different token. Resume and reroll continue an
existing run and need no new approval. The check covers one database, and the
token remains an acknowledgment rather than a credential.

### Operation identity and input digest

Each planned operation has two identities. Its `operation_id` (`op_...`) names
one expanded operation: it covers the template, including its key, count,
prompt and artifact spec, the ordinal, the profile, including its name and
per-call ceiling, and the attachment paths. `jobs inspect --operation` and
`jobs reroll` accept it.

Its input digest describes what the operation asks its provider for.
`operation_input_digest(operation, plan.attachments)` returns `sha256:`
followed by the hash of version 1 of a canonical JSON object holding:

- the provider kind and its adapter version from
  `smythe.jobs.providers.ADAPTER_VERSIONS`, which increases whenever Smythe
  changes how that provider's requests are built;
- the model, provider options and prompt;
- the artifact spec; and
- each attachment's SHA-256 and MIME type, in the operation's order. The
  MIME type comes from a fixed table of extensions (`.gif`, `.jpg` and
  `.jpeg`, `.png` and `.webp`, in any case); any other extension counts as
  `application/octet-stream`. The plan's fingerprints record the type the
  platform's MIME database guesses, which differs between platforms and
  Python versions, so the digest does not use it.

The template key, count, ordinal, profile name, `max_attempts`, cost ceilings
and attachment paths are left out. Every sample of a template therefore shares
one digest, and renaming a profile or moving an attachment to another path
with the same bytes and extension keeps it. A different prompt, model, option,
artifact spec, attachment content or attachment order changes it.
`operation_input_digest` raises `ValueError` when the operation's attachments
are not an array or an attachment has no fingerprint in the list it is given.

The digest is computed on demand and never stored. It is not part of the plan
hash or approval token, so plans, exports, approvals and the Jobs schema are
unchanged. `jobs inspect` adds `input_digest` to each operation, computed from
the run's stored plan, prints it on an `Input digest:` line under each
operation, and the HTML report shows it in an Input digest column. It is
`null` (`unavailable` in the printed output) for an operation whose stored
specification cannot produce one, such as one whose stored attachments are
not an array of planned paths, or that sends attachments when the plan's
stored attachment fingerprints cannot be read; inspection itself still
succeeds.
Jobs does not reuse artifacts across runs by digest.

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
  MIME type, byte size, and observed dimensions. Image bytes are decoded only
  as PNG, JPEG, GIF, or WebP, whatever MIME type the provider declares; other
  formats are refused before any other Pillow decoder runs. Like undecodable
  data, they are never written: the attempt is `rejected` at the cost the
  provider reported, with the reason in the attempt error, and a reroll needs
  no duplicate-spend acknowledgment. Invalid provider accounting still makes
  the call an unknown outcome. See [delivery acceptance](#delivery-acceptance).

Inspection applies its own limits, whatever the manifest requests: 32 MiB per
artifact, 67,108,864 pixels per image or animation frame, 256 frames, and
268,435,456 decoded pixels across all frames. Declared sizes, including a WebP
canvas and the canvas every GIF frame requires, are checked from the bytes
before Pillow parses them, and each frame is checked again before its pixels
are decoded. An artifact over a limit is refused like undecodable data. The
limits hold whatever `PIL.Image.MAX_IMAGE_PIXELS` allows, including `None`,
and inspection leaves the process's warning filters unchanged. An animated GIF
whose frames extend past its logical screen reports the enlarged canvas as its
dimensions.

To find every GIF frame, inspection reads the blocks the way Pillow does.
Pillow reads on past a data sub-block terminator in two places where the GIF
format ends the block: after an extension other than a comment whose first
sub-block is empty, and after a NETSCAPE2.0 extension before the first frame
whose looping sub-block is empty. A GIF with more bytes after such a
terminator is refused like undecodable data before Pillow reads it.

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

The local journal derives an idempotency key for every attempt from its run,
operation, and attempt number. It is a local correlation key: Jobs transmits it
to no provider, so no external API can use it to deduplicate a repeated call.

Inspect current state and, optionally, the append-only event sequence:

```bash
smythe jobs status RUN_ID
smythe jobs status RUN_ID --events --json
```

Snapshots distinguish confirmed cost, reserved cost, and unresolved exposure.
`cost_is_complete` is false while an unknown or incomplete provider charge
remains exposed; `cost_contains_estimates` reports whether any recorded call
used a conservative estimate. The `*_microusd` amounts are whole micro-USD,
which the cost block's `unit` field names: `"microUSD"`. See
[money units](budgets.md#money-units).

## Delivery acceptance

Every completed call is decided by checks on the exact response bytes. The
built-in artifact check, `smythe.jobs.artifact-spec` version 1, decodes each
artifact within the limits above and compares it with the operation's
`artifact` spec. Each finding has a stable code:

| Code | Finding |
|---|---|
| `no_artifacts` | The response has no artifact |
| `artifact_count_limit` | The response has more than 10 artifacts |
| `artifact_bytes_limit` | The response's artifacts exceed 128 MiB together |
| `artifact_too_large` | One artifact exceeds 32 MiB |
| `artifact_undecodable` | The bytes cannot be decoded, use a refused format, or exceed a pixel or frame limit |
| `mime_type_mismatch` | The decoded MIME type differs from `mime_type` |
| `dimensions_mismatch` | The decoded size differs from `width` and `height` |

Any finding rejects the attempt; the operation becomes `rejected` and can be
rerolled. Bytes that decode are published either way, with `accepted=0` when
rejected, so they can be reviewed. The call's `call_completed` or
`artifact_rejected` event records the decision under `delivery_acceptance`,
in the transaction that settles the call's cost:

```json
{
  "version": 1,
  "scope": "artifacts",
  "attempt_id": "...",
  "decision": "rejected",
  "reasons": ["tile.png: expected 1x1, observed 2x3"],
  "subjects": [{"path": "artifacts/tile_0/attempt-0001/tile.png",
                "sha256": "...", "size_bytes": 78, "published": true}],
  "validators": [{
    "validator": {"name": "smythe.jobs.artifact-spec", "version": "1",
                  "config_sha256": "..."},
    "outcome": "rejected",
    "reports": [{"path": "artifacts/tile_0/attempt-0001/tile.png",
                 "outcome": "rejected", "validator": {"...": "..."},
                 "findings": [{"code": "dimensions_mismatch", "severity": "hard",
                               "message": "tile.png: expected 1x1, observed 2x3",
                               "expected": [1, 1], "observed": [2, 3]}]}]
  }]
}
```

`decision` is `accepted`, `rejected`, or `indeterminate`. `subjects` lists
each artifact's recorded path, SHA-256 and size; `published` is false for
bytes that were not written. Findings about the whole response are reported
for the path `response`. The check's `config_sha256` hashes the MIME type and
dimensions it compared. The attempt error keeps the finding messages, and the
operation's status and recorded `accepted` flags mean what they did before.
Calls recorded by earlier releases have no `delivery_acceptance` block. The
Jobs database schema is unchanged.

### Registered validators

An operation's `artifact.validators` names further checks by name and
version:

```yaml
    artifact:
      mime_type: image/png
      validators:
        - name: brand-palette
          version: "1"
```

A manifest never contains or imports validator code. A validator is trusted
local code registered in-process, the same object the
[delivery acceptance types](verifier.md#delivery-acceptance-records) describe:
an `identity` (`ValidatorIdentity`) and a synchronous `validate(subject)` that
returns a `ValidationReport`. Pass validators to both `preflight_job(...,
validators=[...])` and `JobRunner(..., validators=[...])`; the CLI registers
none, so it refuses a manifest that names one. Registration refuses a
`Provider`, a class, a coroutine function, an awaitable, an `async def
validate`, duplicate names, and names starting with `smythe.`. That check is a
caller contract, not a sandbox: a validator must make no network requests or
provider calls.

Preflight refuses a name that is not registered, or registered at another
version, before any run exists. The plan records each resolved identity,
including its `config_sha256`, so a changed validator changes the plan hash:
its approval no longer matches, and `resume` and `reroll` of a run approved
under the old validator are refused before any provider call;
[recheck](#recheck-accepted-outputs) decides that run's accepted outputs again
under the changed validator without a call. A runner that
lacks a validator the plan names, or has it with another identity, refuses
to start the run. Manifests without validators keep their manifest hash, plan
hash, operation IDs and approval tokens.

Each registered validator runs after the built-in check on the exact
in-memory bytes of every decodable artifact, as a `ValidationSubject`, and
its reports are recorded beside the built-in check's, at most 32 findings
each with hard findings first. A hard finding rejects the attempt. A
validator that raises, returns anything but its own JSON-serializable
`ValidationReport`, or reports `error` could not decide: its outcome is
`error` and the decision is `indeterminate`, unless another check rejected
the bytes. The call still settles at its reported cost, the operation becomes
`rejected` and can be rerolled without a duplicate-spend acknowledgment; it
never becomes an unknown outcome. The attempt error names the validator and
the exception type, never the exception's message.

### Delivery eligibility

`jobs inspect` adds each attempt's recorded `delivery_acceptance`, and to each
operation a projected `delivery_acceptance` and a derived
`delivery_eligibility`. The projection describes the accepted attempt, or the
latest completed one: its `source`, `attempt_id`, `decision`, the recorded
`accepted` flag, each validator's identity and outcome, and the reasons. An
attempt completed by an earlier release has no record and is projected with
`source: "legacy_built_in"`, `decision: null` and `validators: null`: a
legacy built-in decision whose validator is unrecorded. It is never reported
as `accepted` or as `not_evaluated`, and it keeps the recorded flag as
provenance.

An accepted attempt's decisions are its call's recorded one and any later
[rechecks](#recheck-accepted-outputs) of the same bytes, which inspection
lists, oldest first, as the attempt's `delivery_reevaluations`. Eligibility
uses the latest of them whose validators are exactly the ones the operation
requires now: this release's built-in check and the registered validators
its plan names. `jobs inspect` takes those from the plan;
`inspect_job(store, run_id, validators=[...])` and a `JobRunner` take, for
each name, the validator registered now. The projection describes that
decision, with `source: "reevaluated"` for a recheck; without one, it
describes the latest decision.

`delivery_eligibility` is `{"eligible": true, "reasons": []}` only when that
decision is `accepted` and every accepted artifact is `verified` on disk with
the SHA-256 the decision names. Otherwise `reasons` lists codes:
`no_decision`, `not_accepted`, `validator_unrecorded`, `validator_changed`
(no decision used the current validators: a recheck is required),
`record_invalid`, `record_mismatch`, or `artifact_` followed by the integrity
status (`artifact_changed`, `artifact_missing`, `artifact_unsafe`,
`artifact_unreadable`, `artifact_not_checked`). Eligibility is derived on
every read and never written back; recorded statuses and `accepted` flags are
unchanged. The HTML report shows it in a Delivery column, with the decision,
validator results and rechecks beside each attempt.

`resume` and `reroll` recheck the SHA-256 of every file of each operation's
accepted attempt after execution, without the 64 MiB inspection limit, and
return the same fields with an `artifact_integrity` summary. A changed or
missing accepted file is reported there and makes its operation ineligible;
the run's status and the operation's recorded acceptance stay as they were.
[Repair](#repair-corrupted-accepted-outputs) regenerates such an operation.

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
not repeat image decoding or quality evaluation. They feed each operation's
[delivery eligibility](#delivery-eligibility).

`list`, `inspect`, `review`, `status`, `attach`, and `export` open the existing Jobs database in
read-only mode, without provider initialization or schema migration. Missing,
foreign, and unsupported databases fail closed. A live WAL reader sees one
consistent committed ledger snapshot while the worker continues. `--out`
writes the requested report atomically and rejects the database and its WAL
or SHM files as output destinations.

Successful `list` and `inspect` commands return exit code 0 even when a run
needs attention. Their JSON output retains the run's recorded status. The
existing `status` and `export` commands retain their job-state exit codes.

## Review what needs attention

`jobs review` lists only what in one run needs an operator's decision, and the
command that carries each decision out:

```bash
smythe jobs review RUN_ID
smythe jobs review RUN_ID --limit 100 --offset 100 --json
```

Each item comes from a state the run already records:

| Kind | Holds back | State | Resolve with |
|---|---|---|---|
| `budget_overrun` | admission | The run spent past its approved ceiling | Nothing in this run: a new plan and approval |
| `invalid_accounting` | admission | An [unknown outcome](#resume-and-unknown-outcomes) whose provider reported cost or usage that cannot be accounted for; the run dispatches nothing more | `reroll --acknowledge-unknown`, after the adapter is repaired and the charge investigated |
| `pause_requested` | admission | A [stop request](#stop-and-resume-safely), while admitted calls drain or once the run is `paused` | `resume` |
| `worker_failed` | completion | The latest detached worker failed, was aborted or never started, and no live lease holds the run | `resume` |
| `lease_expired` | completion | The run is recorded as `running`, but no live lease holds it | `resume` |
| `unknown_outcome` | completion | A dispatched call whose outcome is unknown, with the exposure it keeps | [`adopt`](#adopt-published-bytes), or `reroll --acknowledge-unknown` |
| `rejected`, `failed` | completion | An operation with an attempt left | [`reroll`](#selective-rerolls) |
| `attempts_exhausted` | completion | A rejected or failed operation with no attempt left | Nothing in this run: a new plan and approval |
| `artifact_integrity` | delivery | An accepted file of a succeeded operation is missing, changed, unreadable or unsafe | [`reroll --repair-integrity`](#repair-corrupted-accepted-outputs) for a missing or changed file; a person restores the others |
| `delivery_ineligible` | delivery | A succeeded operation is not [eligible for delivery](#delivery-eligibility) for another reason | [`recheck`](#recheck-accepted-outputs) for a changed check; a person otherwise |

`worker_failed` and `lease_expired` are the states for which `jobs attach`
exits with code 7. A reroll or repair is offered only while an attempt is left
and the run's budget has not overrun. `recheck` is offered when the plan names
no registered validators, since the CLI registers none; otherwise the item
names `JobRunner(store, validators=[...]).recheck()`.

With `--json` the command prints one compact document:

```text
{
  "review_version": 1,
  "run_id": "...", "name": "...", "status": "needs_attention", "schema_version": 4,
  "reviewed_at_ns": ...,
  "operation_counts": {"succeeded": 3, "unknown_outcome": 1},
  "cost": {"approved_microusd", "confirmed_microusd", "exposure_microusd",
           "reserved_microusd", "cost_is_complete", "cost_contains_estimates", "unit"},
  "page": {"limit": 50, "offset": 0, "total": 4, "returned": 4, "has_more": false},
  "totals": {"budget_overrun": 0, "invalid_accounting": 0, "...": 0, "unknown_outcome": 1},
  "items": [{
    "item_id": "<64 hex characters>",
    "kind": "unknown_outcome",
    "blocks": "completion",
    "subject": {"operation_key": "tile[3]", "operation_id": "op_...",
                "attempt_id": "...", "attempt_number": 1, "call_id": "..."},
    "observed": {"status": "unknown_outcome", "error_class": null,
                 "excerpt": "process stopped after dispatch",
                 "exposure_microusd": 100000, "attempts_remaining": 1},
    "reasons": [],
    "decision_needed": "The call may have completed and been charged. ...",
    "resolve_with": [["jobs", "adopt", "RUN_ID", "tile[3]", "--reason", "REASON"],
                     ["jobs", "reroll", "RUN_ID", "tile[3]", "--acknowledge-unknown",
                      "--reason", "REASON"]]
  }],
  "artifact_integrity": {"counts": {"verified": 3}, "bytes_hashed": 312,
                         "hash_byte_limit": 67108864}
}
```

- **Items.** `subject` names an operation's key, ID, the attempt and call
  the item is about, the run for items about the whole run, or a failed
  worker's `launch_id`. `observed` holds the recorded `status`, an
  `error_class` (a failed worker's error type, or
  `invalid_provider_accounting`) and an `excerpt` of the stored error or stop
  reason of at most 300 characters, with account identifiers redacted and
  the manifest root, the store's directory and the home directory written as
  `<manifest root>`, `<store directory>` and `~`; some kinds add the kept
  `exposure_microusd`, `attempts_remaining`, the damaged `artifacts`, the
  `eligibility_reasons` or the `pause_generation`. `artifact_integrity` and
  `delivery_ineligible` items give `reasons` from the
  [reuse decisions' vocabulary](lineage.md#reason-codes), as the
  [repair preview](#preview-repairs) does.
- **Commands.** Each `resolve_with` entry is the argument list of one
  `smythe jobs` command, in the order to try; it parses as given, without
  `--store`, and `REASON` stands for your own reason. An empty list means
  this run has no command for it. The printed output shows each item with
  its decision and commands, and reminds you to add `--store` when you used
  one.
- **Stable identity.** `item_id` is the SHA-256 of the item's kind, run,
  subject and the recorded state it depends on. It stays the same across
  reviews until that state changes, as a reroll or a new attempt does, so a
  review can be repeated at any time, for example after reconnecting to a
  detached run, to see which decisions are still open.
- **Order and pages.** Items that hold back admission come first, then
  completion, then delivery; within each, items about the whole run come
  before operations, by key. `--limit` (1–500, default 50) and `--offset`
  page the operations; items about the whole run appear on every page,
  `totals` counts each kind on this page, and `operation_counts` covers the
  whole run.
- **Bounded reads.** The accepted files of the page's succeeded operations
  are hashed, at most 64 MiB per review. A file past that limit is counted
  as `not_checked` in `artifact_integrity` and is not an item;
  `jobs reroll RUN_ID --dry-run` hashes every file.

A review opens the database read-only, writes nothing, never migrates a
schema-2 or schema-3 database or enables WAL, and constructs no provider. It
also reads the latest detached launch's receipts beside the database. The
packet holds no local paths, so it has no `manifest_root`; recorded
acceptance flags and statuses are never changed. `review_version` versions
the JSON, and later versions only add fields. In Python,
`smythe.jobs.inspection.review_job(store, run_id, limit=50, offset=0,
validators=None)` returns the same document; `validators` judge delivery
eligibility as for `inspect_job`.

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
path. Without a drain deadline, a pause does not cancel a provider request.
Safe prepared calls return to pending; accepted outputs and attempt history
remain intact. The run becomes
`paused` after admitted work drains and pending work remains. Completed work,
unknown outcomes, and budget overruns retain their corresponding statuses.

`--drain-timeout-s` (0 to 3,600 seconds) bounds the drain:

```bash
smythe jobs stop RUN_ID --drain-timeout-s 60
```

That long after the request, the worker cancels each call still awaiting its
provider and records it as `unknown_outcome`, keeping its full per-call
ceiling as exposure; the run then becomes `needs_attention`. A call whose
response has arrived settles normally. The deadline counts from the request,
and the worker checks for it at least once a second; a worker from an earlier
release ignores it and waits for every call. A check that cannot read the
database, for example while it is locked, changes nothing: the run continues,
any deadline the worker has already read still applies, and the next check
reads again. The request's `pause_requested`
event and the JSON `stop_request` record it as `drain_deadline_ns`; the Jobs
schema is unchanged. A later stop can bring the deadline forward but not
extend it, and resume clears it with the pause. Investigate abandoned calls as
described under [unknown outcomes](#resume-and-unknown-outcomes). In Python,
pass `drain_timeout_s` to `SQLiteRunStore.request_pause` or
`smythe.jobs.operator.stop_job`.

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

A completed run starts zero new operations, and resume rechecks the bytes of
its accepted artifacts before reporting their
[delivery eligibility](#delivery-eligibility). Safe pre-dispatch interruptions
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
artifact destination. If the attempt already published the whole response,
[adopt those bytes](#adopt-published-bytes) without another call. If a
duplicate call is acceptable, acknowledge that risk explicitly during a
selective reroll.

## Jobs database upgrades

**Added in 0.8.0:** concurrent openers use bounded retries for WAL
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

Operation keys may come before, between or after the options; every plain
argument after the run ID is an operation key. `--dry-run`
[previews](#preview-repairs) what each operation calls for without changing
anything.

Rerolls consume the same original approval and attempt allowance. The selected
operation receives a new immutable attempt linked to its predecessor that
records the reroll reason, which `smythe jobs inspect` shows with that attempt;
other operations are not reset, and prior artifact records remain in the
journal. A safe retry after recovery starts another attempt without a reason,
and attempts recorded by earlier versions keep an empty reason.
Successful operations cannot be casually rerolled, which preserves accepted
outputs byte-for-byte.

### Repair corrupted accepted outputs

A succeeded operation whose accepted file was later deleted, truncated or
overwritten is not delivered: its [delivery eligibility](#delivery-eligibility)
reports `artifact_missing` or `artifact_changed`. Regenerate it with an
integrity repair:

```bash
smythe jobs reroll RUN_ID "tile[3]" --repair-integrity --reason "tile.png was overwritten"
```

Under the run's lease, immediately before queueing, the runner hashes every
file of the operation's accepted attempt again, without the 64 MiB inspection
limit. The repair is queued only when at least one of those files is `missing`
or `changed` and the others are `verified`. Bytes that all verify are refused,
so a verified success still cannot be rerolled. A file that is unreadable or
unsafe, such as a symbolic link or a directory, an exhausted attempt allowance
and another worker's lease also refuse it with exit code 6, and a budget
overrun with exit code 5. Before hashing, the runner claims the run's artifact
directory as execution does, so a directory that holds files without its owner
marker, or whose marker names another run, refuses the repair. A
[migrated run](#jobs-database-upgrades) whose directory has not been claimed
yet verifies its accepted bytes in that first claim: like a resume, its repair
is refused until the original bytes are restored. A refused repair queues
nothing and changes no operation, attempt or artifact record.
`--repair-integrity` cannot be combined with `--acknowledge-unknown`.

A repair uses the run's original approval and attempt allowance; it needs no
new approval. The `integrity_repair_queued` event records the reason and each
accepted file's finding: its `artifact_id`, `relative_path` and recorded
`sha256`, and the observed `status` and `detail`. The new attempt is linked to
the accepted attempt it replaces and records the reason. The operation keeps
naming that accepted attempt until a new attempt is accepted: a rejected or
failed repair attempt leaves it ineligible, and a reroll tries again. Earlier
artifact rows and their `accepted` flags never change. `resume` and `reroll`
recheck only each operation's current accepted attempt, so the files a repair
replaced are not reported again; `jobs inspect` still shows their integrity.
In Python, pass `repair_integrity=True` to `JobRunner.reroll`; it records the
findings with `SQLiteRunStore.queue_integrity_repair`. The Jobs schema stays at
version 4.

## Adopt published bytes

A response's bytes are published before its call settles in the journal. A
process that stops between the two leaves an unknown outcome whose attempt
may already hold the whole response on disk. When investigation shows the
provider completed that request, adopt those bytes instead of paying for a
duplicate call:

```bash
smythe jobs adopt RUN_ID "tile[3]" --reason "provider console shows one completed image"
```

Adoption applies only to an `unknown_outcome` operation. Its latest attempt
directory, `artifacts/<operation-directory>/attempt-NNNN/` beneath the run's
artifact root, must hold exactly the files of a response with the artifact
count the profile requests (`artifacts_per_call` for `offline`, `n` for
`openai_image`, one for `gemini_image`) and nothing else. The bytes then pass
the [built-in artifact check](#delivery-acceptance) and the plan's
[registered validators](#registered-validators) again; pass the runner the
same validators as for the run. Missing, extra, undecodable or wrong-MIME
bytes, a validator rejection, and any other operation state refuse adoption
with exit code 6, and symbolic links or junctions with exit code 2. A refusal
leaves the operation, its attempts, calls and artifact records unchanged.

The adopted operation becomes `succeeded` with that attempt accepted and its
artifact receipts recorded. No provider call is made and no attempt is used.
The attempt and its call keep their unknown outcome, and the call keeps its
full per-call ceiling as exposure: no settlement is invented. A run whose
operations have all succeeded is therefore `completed` with
`cost_is_complete: false` while that exposure remains. The
`unknown_outcome_adopted` event records the reason, call, attempt, kept
exposure, each artifact's path and SHA-256, and the delivery decision under
`delivery_acceptance`, from which [delivery eligibility](#delivery-eligibility)
is derived as for a completed call. `jobs inspect` adds that record, without
the decision, to the operation as `adoption`, prints an
`Adopted published bytes:` line, and the HTML report shows it with the
operation.

Adoption holds the run's lease, like a reroll, and dispatches no pending
work; resume the run for that. It prints the run and returns its exit code
like `reroll`. In Python, call `JobRunner.adopt(run_id, operation_key,
reason=...)`; it records the verified bytes with
`SQLiteRunStore.adopt_published_attempt`. The Jobs schema stays at version 4.

## Recheck accepted outputs

A changed check never inherits a decision made under the old one. After a
registered validator gets a new version or configuration, or a release ships
a new version of the built-in artifact check, each operation decided under
the old one reports `validator_changed` in its
[delivery eligibility](#delivery-eligibility): a recheck is required. A
recheck decides the accepted bytes again, without a provider call:

```bash
smythe jobs recheck RUN_ID "tile[3]" "tile[4]" --reason "artifact check version 2"
```

In Python, register the validators as they are now on the runner, such as
the [`MaxBytes`](#python-api) validator with a lower limit, which is a new
configuration:

```python
runner = JobRunner(store, validators=[MaxBytes(1_500_000)])
result = asyncio.run(runner.recheck(run_id, ["tile[3]"], reason="limit lowered"))
```

- **What decides.** This release's built-in check and, for each validator
  the plan names, the one registered under that name now, at any version or
  configuration. Nothing is dispatched, so the plan's approval is not used,
  and a recheck works where `resume` and `reroll` are refused under the
  changed validator. A validator the plan names that the runner has not
  registered refuses it with exit code 2. The CLI registers none, so
  `jobs recheck` covers runs whose plans name no registered validators.
- **What is read.** Only the accepted files of `succeeded` operations. Under
  the run's lease each is hashed again and must be `verified`; then each
  operation's files are read once into memory, at most 10 files of 32 MiB,
  and must still hold the recorded bytes. Every check decides that copy, as
  it decided the response.
- **What is recorded.** One `delivery_acceptance_reevaluated` event per
  operation, all in one transaction under the lease:

  ```json
  {
    "version": 1,
    "attempt_id": "...",
    "reason": "artifact check version 2",
    "outcome": "accepted",
    "validators": [{"validator": {"name": "smythe.jobs.artifact-spec", "version": "2",
                                  "config_sha256": "..."},
                    "outcome": "passed", "reports": [{"...": "..."}]}],
    "findings": [],
    "artifacts": [{"relative_path": "artifacts/tile_3/attempt-0001/tile.png", "sha256": "..."}]
  }
  ```

  `outcome` is decided as for a call: `accepted`, `rejected` or
  `indeterminate`. `validators` holds each check's identity, outcome and
  reports, `findings` the finding messages, and `artifacts` each accepted
  file's path and SHA-256. Earlier decisions are kept. Statuses, the accepted
  attempt, artifact rows and their `accepted` flags never change, and no
  budget, attempt or approval is used.
- **What follows.** Eligibility uses the latest decision made under the
  current validators. An accepting recheck makes the operation eligible
  again; a rejecting one leaves it `succeeded` but not eligible
  (`not_accepted`). Its bytes still verify, so an integrity repair refuses
  them, and regenerating them takes a new plan and approval.
- **Refusals.** Another operation state, an unknown operation, an accepted
  file that is missing, changed, unreadable or unsafe (an
  [integrity repair](#repair-corrupted-accepted-outputs) regenerates a
  missing or changed one) and another worker's lease refuse the whole
  recheck with exit code 6; nothing is recorded.

A recheck holds the run's lease, like a reroll, prints the run and returns
its exit code like `reroll`. `jobs inspect` lists each attempt's rechecks as
its `delivery_reevaluations`, with `reevaluated_at_ns`, and the HTML report
shows them beside the attempt. `JobRunner.recheck` records the decisions
with `SQLiteRunStore.record_reevaluation`. The Jobs schema stays at
version 4.

## Preview repairs

Before rerolling, repairing or rechecking anything, see what each operation
calls for:

```bash
smythe jobs reroll RUN_ID --dry-run
smythe jobs reroll RUN_ID "tile[3]" "tile[4]" --dry-run --json
```

`--dry-run` previews the named operations, or every operation of the run
when none is named; `--reason` and the other reroll options are ignored. It
opens the Jobs database read-only, takes no lease, writes nothing and calls
no provider, so it works while a worker holds the run, and it exits 0. It
hashes every accepted file of the previewed succeeded operations again,
without the 64 MiB inspection limit, and fingerprints every planned
attachment again. In Python, `JobRunner(store, validators=[...]).repair_preview(run_id,
operation_keys=None)` returns the same document, judged under that runner's
validators; the CLI registers none, so it uses those the plan names.

```text
{
  "version": 1,
  "run_id": "...",
  "status": "completed",
  "budget": {"approved_microusd", "confirmed_microusd", "exposure_microusd",
             "reserved_microusd", "remaining_microusd", "overrun", "unit"},
  "attachments": [{"relative_path", "sha256", "status"}],
  "counts": {"reuse": 3, "recheck": 0, "regenerate": 1, "review": 0},
  "operations": [{"operation_key", "operation_id", "status", "action", "reasons",
                  "attempts_remaining", "new_plan_required"}]
}
```

Each operation's `action` is the most conservative its reasons call for, and
its `reasons` are sorted codes from the
[reuse decisions' vocabulary](lineage.md#reason-codes):

| Operation | Action | Reasons | What carries it out |
|---|---|---|---|
| Succeeded, its latest decision under the current validators accepted, every accepted file verified | `reuse` | none | Nothing |
| Succeeded, no decision under the current validators | `recheck` | `validator_set_changed` | [Recheck](#recheck-accepted-outputs) |
| Succeeded, an accepted file missing or changed and the rest verified | `regenerate` | `artifact_missing`, `artifact_changed` | [Integrity repair](#repair-corrupted-accepted-outputs) |
| Succeeded, an accepted file unsafe, unreadable or not checked | `review` | `artifact_unsafe`, `artifact_unreadable`, `artifact_unchecked` | A person |
| Succeeded, its decision under the current validators not `accepted`, malformed, or not about its files | `review` | `decision_not_accepted` | A person; the bytes verify, so no repair in the run replaces them |
| Pending, failed or rejected | `regenerate` | `node_incomplete` | `resume` for pending work, a [reroll](#selective-rerolls) otherwise |
| Running | `review` | `run_in_progress` | The worker that holds it, or `resume` |
| Unknown outcome | `review` | `outcome_unknown` | Investigation, then [adoption](#adopt-published-bytes) or an acknowledged reroll |
| Sends an attachment whose bytes changed since the plan | `regenerate` | `source_changed` | A new plan and approval |
| Sends an attachment that is missing, unsafe or unreadable | `review` | `source_unavailable` | Restore it |

Jobs keep no rendering input, so a missing or changed accepted file calls for
`regenerate`, a new attempt, where a durable run would rerender.
`attempts_remaining` is the operation's attempt allowance left under the plan.
`new_plan_required` is true for a `regenerate` that the run's approval cannot
admit: the operation sends an attachment whose bytes changed, has no attempt
left, or its run's budget overran. The run's `attachments` give each planned
attachment's recorded SHA-256 and `status` (`verified`, `changed`, `missing`,
`unsafe` or `unreadable`); while any is not `verified`, `resume` and
`reroll` refuse the whole run until it is restored. `budget` is the run's
cost state with `remaining_microusd` of the approved ceiling and whether it
`overrun`. While every attachment verifies and the runner registers the
validators the plan names, a `regenerate` with `new_plan_required` false is
what an integrity repair, reroll or resume of that operation admits.

The preview repairs nothing and never offers another run's artifacts, even
when their input digest matches: Jobs reuse nothing across runs.

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
| 0 | command succeeded; `list`, `inspect` and `review` succeed regardless of recorded job status |
| 1 | job finished `failed`, `partial`, `needs_attention`, or `budget_overrun` |
| 2 | invalid manifest, preflight input, or command value |
| 3 | provider configuration or optional dependency failure |
| 4 | approval mismatch, or an already used approval without `--new-run` |
| 5 | budget admission failure |
| 6 | unknown run, invalid state transition, or refused adoption, repair or recheck |
| 7 | local SQLite/filesystem failure, detached startup failure, or attachment detects a failed worker or expired lease |
| 130 | detached startup interrupted, attachment disconnected with Ctrl+C, or review interrupted |

Use `--json` for one compact document on stdout, including structured errors.

`jobs review` returns 0 whether or not anything needs a decision, 2 for an
invalid `--limit` or `--offset`, 6 for an unknown run, 7 for a local SQLite or
filesystem failure, and 130 when interrupted. An interrupted review prints
no part of its packet: with `--json` it prints only the structured error. It
has no exit code of its own for items that need a decision; read `totals` or
`items` instead.

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

`JobRunner.start` and `SQLiteRunStore.create_run` raise `ApprovalError` for an
approval that already created a run in the store. Pass
`allow_approval_reuse=True` to start another run with it, as `--new-run` does.

A manifest that names [registered validators](#registered-validators) runs
with the same validator objects passed to preflight and to every runner that
starts, resumes or rerolls it:

```python
from smythe.validation import ValidationReport, ValidatorIdentity, hard_finding

class MaxBytes:
    def __init__(self, limit):
        self.limit = limit
        self.identity = ValidatorIdentity.for_config("max-bytes", "1", {"limit": limit})

    def validate(self, subject):
        if len(subject.data) <= self.limit:
            return ValidationReport("artifact", validator=self.identity)
        finding = hard_finding("too_large", f"{len(subject.data)} bytes",
                               expected=self.limit, observed=len(subject.data))
        return ValidationReport("artifact", (finding,), validator=self.identity)

validators = [MaxBytes(2_000_000)]
plan = preflight_job(manifest, manifest_root=root, validators=validators)
runner = JobRunner(store, validators=validators)
```

## Current boundaries

- Jobs v1 is a local SQLite worker, not a distributed queue or hosted service.
- Built-in artifact acceptance checks MIME type and optional pixel dimensions.
  Registered validators add developer-written checks by name and version. The
  public `smythe.assets` package adds typed brand modes, deterministic
  finishing, hash-bound receipts, and richer image validation for custom
  pipelines; those policies are not automatically inferred from a generic job
  manifest.
- Rerolls are operator-selected; there is no automatic vision-judge promotion
  policy in the runtime. Repairs are too: the
  [repair preview](#preview-repairs) shows what each operation calls for,
  and nothing repairs an operation automatically.
- Jobs write no [artifact lineage](lineage.md) record. Integrity repairs,
  rechecks and the repair preview work within one run, and no run reuses
  another run's accepted output, even for the same
  [input digest](#operation-identity-and-input-digest).
- Unknown provider outcomes require human investigation, then explicit
  duplicate acknowledgment or adoption of the bytes the attempt published.
- Jobs do not read a response's `stop_reason`. A response that the provider
  truncated, refused or filtered is judged by its artifacts and accounting
  alone: without an artifact it is `rejected` (`no_artifacts`) at its
  reported cost, and artifacts that pass every check are accepted.

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

The related [Noumenon fan-out benchmark](../benchmarks/noumenon_benchmark.md)
is one concrete use of Smythe's general-purpose artifact execution model. It
uses a 192-node `BROADCAST_REDUCE` graph to generate 192 original cyber-glyph
tiles, validate dimensions and uniqueness, and assemble a 1920x1080 still,
animated GIF, contact-sheet atlas, and standalone animated HTML canvas. Its
offline lane is deterministic and free; its live Gemini and GPT Image lanes require
explicit whole-call and whole-run ceilings. The benchmark currently exercises
the `Swarm` scheduler directly to isolate concurrency, while Jobs v1 supplies
the durable operator semantics intended for production artifact runs.
