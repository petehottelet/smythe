# Smythe 0.10.0

[GitHub release](https://github.com/petehottelet/smythe/releases/tag/v0.10.0) ·
[PyPI package](https://pypi.org/project/smythe/0.10.0/)

Smythe 0.10.0 carries the execution envelope from a finished run to an
accepted deliverable. A run can now be held to the exact output it returns,
a durable research workflow keeps what it read and claimed across an
interruption, and accepted artifacts record what they depend on so a changed
source or validator can be repaired selectively. New capabilities are opt-in
and every new scheduling and learning policy is off by default. Some earlier
behavior now fails closed; read [Upgrade from 0.9.1](#upgrade-from-091)
before upgrading.

```bash
pip install "smythe==0.10.0"
# With a provider, for example:
pip install "smythe[openai]==0.10.0"
```

## What changes

- **Strict delivery acceptance.** An opt-in `AcceptancePolicy` decides on the
  exact output a run returns, after synthesis and after any step a supervisor
  added. Exhausted regeneration, an unreadable verdict or a failed validator
  never produces an accepted result, and the rejected output stays
  inspectable. Built-in `TextValidator` and `JSONValidator` check required
  sections, paths and JSON shape. Ordinary runs, durable text workflows and
  Jobs record each decision apart from execution and cost.
  [Verification](verifier.md#strict-delivery-acceptance) ·
  [Jobs validators](jobs.md#registered-validators) ·
  [example 15](../examples/15_strict_acceptance.py).
- **Supported configurations.** The [support matrix](support-matrix.md) states
  which providers, verifiers, tools, stores and Jobs profiles each execution
  path accepts and who owns each execution responsibility, with the test
  behind each cell. Unsupported tools, image inputs, models and price ceilings
  are refused before any paid call, including planning.
- **Elapsed time.** `SwarmResult.timing` reports elapsed time apart from
  summed node time, with time per phase and waits by reason, and
  `PlannerMemory` records both. [Timing](execution.md#timing).
- **Durable research workflows.** Versioned HTTP read steps are journaled and
  reused on resume, workers see immutable context snapshots, and versioned
  source and claim records travel with an artifact so a final validator can
  reject quotes that are not in their sources. A removed source leaves a
  tombstone that withholds decisions resting on it, and a completed run can
  be re-evaluated or decided by an authorized operator. The offline
  [source-to-artifact example](../examples/16_source_to_artifact.py) plans,
  reads, is interrupted, resumes and delivers an accepted document.
  [Recorded reads](workflow-accounting.md#recorded-read-steps) ·
  [source evidence](source-evidence.md).
- **Inspect and review saved runs.** `smythe runs list`, `inspect` and
  `review` read ordinary checkpoints and durable journals without changing
  them; `smythe jobs review` lists what a Jobs run needs from an operator.
  `runs inspect` exits 10 when the output is not accepted, 11 when the
  decision is indeterminate and 12 when an accepted run has an unresolved
  external action. [Runs](runs.md).
- **External actions, in preview.** A strict durable run's accepted output
  can be delivered through a host connector under an approval bound to one
  action, with a stable key on every attempt, readback, reconciliation of a
  lost answer and revocation that takes precedence over dispatch. The API may
  change before it leaves preview, and `import smythe` does not load it.
  [External actions](actions.md) ·
  [example 19](../examples/19_governed_publication.py).
- **Artifact lineage and selective repair.** Strict durable runs can record
  the sources, validators and tools their accepted artifacts depend on. A
  read-only repair plan shows what will be reused, rechecked, rendered again
  or regenerated, and `Swarm.repair` applies a reviewed plan under an explicit
  allowance. Jobs can repair a damaged accepted file, recheck accepted files
  under changed validators and preview rerolls.
  [Lineage](lineage.md) · [example 18](../examples/18_archive_maintenance.py).
- **Failure-aware scheduling, experimental and off by default.** A classified
  [retry policy](execution.md#retry-policy) with one retry owner,
  per-model [capacity limits](execution.md#provider-capacity),
  [draining](execution.md#draining-in-flight-work) of in-flight work,
  [held allowance](budgets.md#held-allowance-for-checks) for checks,
  [security stops](execution.md#security-stops),
  [model profiles](execution.md#model-profiles) with fallback,
  [evidence-triggered supervision](supervisor.md#evidence-triggered-review),
  [routing rules](architecture.md#routing-rules) with a single-worker tier
  and [claim exchange](architecture.md#claim-exchange). None changes a
  default until a held-out comparison supports it.
- **Reviewable learning.** Distilled graphs become
  [versioned template records](learning.md#template-records), validated as
  untrusted candidates that cannot weaken their parent's gates. Autotune
  campaigns compare a version with its predecessor and a simpler reference
  arm, and `smythe optimize review`, `activate`, `rollback`, `retire` and
  `history` manage what is in effect through a separate policy registry.
  [Agent Skills directories](skills.md#agent-skills-directories) lend
  capability tags without granting tools.
  [Example 21](../examples/21_template_lifecycle.py).
- **Correctness.** `CallableVerifier` and gate verdicts fail closed,
  `distill_template` keeps gates and execution policies, durable journals stay
  readable after a price or model-list change, checkpoint saves no longer
  re-audit the whole journal, orphaned durable reservations are released, and
  stored error text no longer keeps provider account identifiers.

Full behavior and fixes are in the [changelog](../CHANGELOG.md#0100---2026-10-24).

## Upgrade from 0.9.1

Back up durable workflow journals before upgrading. Check these:

- **Durable workflow journals.** Opening a journal for writing upgrades it to
  format version 2 in one transaction, keeping every run, call, balance and
  evidence row. Smythe 0.9.x cannot open a version 2 journal, even read-only.
  Stop workers from earlier releases first: the upgrade raises
  `WorkflowLeaseError` while any run holds an unexpired lease.
  `SQLiteWorkflowStore(path, read_only=True)` reads a version 1 journal
  without upgrading it. [Journal upgrades](workflow-accounting.md#journal-upgrades).
- **Checkpoints.** Ordinary runs still write version 4 unless a run uses a
  feature that needs version 5, such as strict acceptance, persisted retry
  attempts, model profiles or a security stop. Earlier releases cannot read
  version 5. [Checkpoint format](checkpoint-format.md).
- **Jobs approval tokens create one run.** `jobs run` with a token that
  already created a run in the same database exits with code 4 and names that
  run. Pass `--new-run`, or `allow_approval_reuse=True` in Python, to start
  another run on purpose.
- **Jobs artifacts.** Preflight refuses an artifact its profile cannot
  produce, so an `offline` profile needs a 1x1 `image/png` and a
  `gemini_image` profile a PNG, JPEG, GIF or WebP artifact. A response whose
  bytes cannot be decoded or exceed a limit is a rejected attempt settled at
  the reported cost instead of an unknown outcome.
- **Autotune runner version 4.** Plan hashes and automatic campaign IDs
  change, so the same command without `--campaign-id` starts a new campaign,
  and an explicit campaign ID from an earlier release exits with code 8. Used
  holdouts stay used; contracts and the ledger schema are unchanged.
- **Checks move before paid calls.** An ordinary `Swarm` refuses a tool
  runtime, image attachments or a node model its provider does not support
  before any provider call, and raises `BudgetEstimateRequired` before routing
  or planning when a per-call ceiling is missing. Runs that used to succeed
  because no node used the unsupported feature are now refused.
  `max_concurrency`, `max_revisions` and `retry_backoff_s` are validated when
  the `Swarm` is constructed.
- **Verifiers fail closed.** A `CallableVerifier` function must return a bool,
  a `Verdict` or an object with a bool `passed`; return `bool(...)` where a
  function returned `1` or a NumPy boolean. A gate passes only when
  `Verdict.passed` is exactly `True`.
- **Gated graph prompts.** On ordinary runs, the node a judge verifies is told
  it is the final step instead of the judge. Prompts of gated graphs change,
  so benchmark results from before and after are not directly comparable.
  `prompt_assembly=1` on the executor restores the earlier placement; durable
  text workflows keep it. [The final step](verifier.md#the-final-step).
- **Smaller refusals.** `OpenAIImageProvider` refuses image attachments
  instead of dropping them, `ConstrainedArchitect` refuses duplicate template
  names, `distill_template` raises `DistillationError` for graphs it cannot
  record, `ValidationReport` accepts only `ValidationFinding` values, and an
  attached dependency image whose bytes no longer match its recorded digest
  raises `ArtifactIntegrityError`.
- **Imports.** `smythe.provider_messages` no longer imports `MODEL`; import it
  from `smythe.pricing_anthropic`.

## Verify a checkout or package

Install `.[dev,openai,anthropic]` from the tagged checkout and run:

```bash
python -m ruff check .
python -m mypy
python -m pytest tests/ -q
python examples/15_strict_acceptance.py
python examples/16_source_to_artifact.py
```

The examples use fixture responses and make no provider calls. The
[distribution guide](distribution.md) covers archive inventories, source-to-wheel
reproduction, clean installed-package checks and the source test profile.
Package README links resolve against `v0.10.0`. The release workflow retains
the actual published distributions and SHA-256 hashes; verify downloaded PyPI
bytes against that receipt as described in [Releasing](../RELEASING.md).
