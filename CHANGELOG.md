# Changelog

All notable changes to **smythe** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with the pre-1.0 stability note below.

## Versioning policy (pre-1.0)

While the project is on a `0.x` line, the public API is **not yet stable**:

- `0.x` minor bumps (e.g. `0.1.0` -> `0.2.0`) MAY include backward-incompatible changes.
  Each minor release will document its breaking changes in this file.
- `0.x.y` patch bumps (e.g. `0.1.0` -> `0.1.1`) are **non-breaking** and contain only
  bug fixes, documentation, or internal improvements.
- The first stable release will be `1.0.0`. Until then, pin to a specific minor
  version (`smythe~=0.1.0`) if API stability matters to you.

Releases 0.1.0 through 0.8.1 are recorded, unchanged, in the
[changelog archive](CHANGELOG-ARCHIVE.md).

---

## [Unreleased]

### Added

- Shared offline test fakes. `tests/fakes.py` has Responses and Messages SDK
  wire fakes with request logs and failure schedules, a manual clock, crash
  images and helpers that hard-kill a child process.
  `tests/fixture_service.py` is a fake external service that can commit a
  request and lose its acknowledgement, receive duplicate or reordered
  deliveries, and report its own state for comparison with what the caller
  saw. Its requests can name a tenant's collection, whose keys and objects
  other tenants cannot see or update, and it can send a notice for each
  acknowledged commit or record a forged one. `tests/fixture_connector.py`
  adapts it to the external-action connector contract in process and over
  loopback HTTP, and provides an HMAC approval verifier for tests.
- A [support matrix](docs/support-matrix.md) lists the providers, verifiers,
  tools, stores and Jobs profiles that ordinary runs, durable text workflows
  and Jobs accept, how each path records cost and who calls again after a
  failure, and names the test behind each cell.
  `tests/test_support_matrix.py` checks that every unsupported provider or
  component on a durable run, and every artifact a Jobs profile cannot
  produce, is refused before any provider call, and runs an ordinary `Swarm`
  with `AnthropicMessagesProvider`. A repository test checks that every test
  the page cites exists.
- `Provider.feature_support(model)` returns a `ProviderFeatures` declaring
  whether the provider accepts a model, writes text, calls tools, takes image
  attachments, returns images and needs a per-call cost ceiling under
  `max_budget_usd`. Each built-in provider declares its features; the native
  providers take their models from `workflow_descriptor()`. A custom
  provider, or a subclass that replaces a built-in provider's `chat()`,
  declares nothing unless it overrides the method, and runs as before. An
  ordinary `Swarm` checks the declaration before any provider call: a
  `tool_runtime` its provider cannot serve and an `LLM_MERGE` synthesis
  before routing or planning; the run's model too, before planning, when
  `LLMArchitect` plans the run; and every unfinished node's model, image
  inputs and ceiling before the first node call, including on resume.
  `Synthesizer.check_merge_support()` performs the synthesis check.
- Accounting results name the unit of their integer amounts. A durable
  result's `workflow_accounting` has `"unit": "nanoUSD"`, and the `cost`
  block of Jobs snapshots, run lists and inspection output has
  `"unit": "microUSD"`. Stored rows and receipts are unchanged. The
  [cost guardrails guide](docs/budgets.md#money-units) lists each ledger's
  unit and how amounts convert between them: exactly for confirmed charges,
  up for reservations and unknown exposure, and down for budgets.
- `Sentinel.cost_contains_token_estimates` reports whether the total prices
  any call's tokens at the blended `cost_per_token` rate, which
  `cost_contains_estimates` does not cover. `Sentinel.token_estimate_nodes`
  names those calls' nodes and `Sentinel.restore()` accepts them. Ordinary
  checkpoints record them under `budget.token_estimate_nodes`, still at
  version 4, so a resumed run keeps the flag; a checkpoint without the key
  restores none, and one whose value is not a list of node IDs stops resume
  with `BudgetValidationError`. Existing flags keep their meaning.
- `AnthropicProvider`, `OpenAIProvider` and `GeminiProvider` accept
  `missing_usage="unknown"`. A reply without usage, or with an unset token
  count, then marks its cost unknown (`cost_usd_unknown`) instead of counting
  the call as a known charge for zero tokens: the ledger keeps a reservation
  held for the call as its estimated charge, and without one the run reports
  `cost_is_complete=False`. The default, `"zero"`, keeps the current
  behavior. A Gemini image's per-call ceiling or per-image cost is never
  affected.
- `smythe.validation` holds `ValidationReport`, `ValidationFinding`,
  `FindingSeverity` and their `hard_finding`/`advisory_finding` helpers
  without importing Pillow; `smythe.assets` re-exports the same classes. A
  report has an `outcome` of `passed`, `rejected` or `error`, derived from its
  hard findings unless the validator states it, and an `error` report never
  passes: as a `CallableVerifier` result it fails the gate with the reason
  `validator error` and any findings it carries. A report can name its
  `validator` with a `ValidatorIdentity`: name, version and the SHA-256 of
  the validator's canonical JSON configuration. Reports, findings and
  identities convert with `to_dict()` and `from_dict()`. See
  [validation reports](docs/verifier.md#validation-reports).
- Delivery acceptance types in `smythe.acceptance`, reported apart from
  execution and cost. `AcceptancePolicy` is advisory by default; a strict
  policy names its final validators, whether a passed gate may accept output
  identical to its target's result, and the decisions for an exhausted gate
  (`rejected` or `needs_review`) and an invalid verdict (`indeterminate` or
  `needs_review`). Its `to_dict()` lists validator identities, never code,
  and `sha256` identifies it. `DeliveryAcceptance` is a versioned record of
  one decision (`not_evaluated`, `accepted`, `rejected`, `needs_review` or
  `indeterminate`) bound to the SHA-256 of the delivered bytes, the synthesis
  strategy, source nodes and generations, the task and policy hashes and each
  validator's outcome; `from_dict()` rejects unknown record versions and
  decisions. A validator is local code with a `ValidatorIdentity` and a
  synchronous `validate(subject)` returning a `ValidationReport`;
  registration refuses providers, classes, coroutine functions, awaitables
  and an async `validate`, including an object whose `__call__` is async.
  Runs do not evaluate a policy, and `Swarm` takes none. See
  [delivery acceptance records](docs/verifier.md#delivery-acceptance-records).
  and an async `validate`. See [delivery acceptance records](docs/verifier.md#delivery-acceptance-records).
- `SwarmResult.acceptance` and `SwarmResult.rejected_output`, appended after
  `timing` so positional construction is unchanged, a read-only
  `SwarmResult.accepted` that is true only for an `accepted` record, and
  `SwarmResult.to_dict()`, which returns the result as JSON data with the
  record under `delivery_acceptance`. A run without a strict acceptance
  policy returns `acceptance=None` and its output as before. `PlannerMemory`
  records a result's decision as `acceptance`, and `"not_evaluated"` without
  a record.
- Checkpoint version 5: a state with a delivery acceptance record keeps it in
  a top-level `delivery_acceptance` block, and `build_state(...,
  delivery_acceptance=record)` writes it. Every version 5 checkpoint names
  the state features its reader must understand in a top-level `requires`
  list, here `["delivery_acceptance"]`; a reader refuses a missing list or a
  feature it does not know, finished or not, and requires the record exactly
  when it is listed. Ordinary runs still write version
  4, and this build reads versions 1 through 5. Versions 1 to 4 never report
  a record. Resuming a completed version 5 checkpoint checks the saved output
  against the record's digest and returns the record; output the record did
  not accept is returned as `rejected_output` with an empty `output`. That
  includes a `not_evaluated` record, whose output is checked only when the
  record binds a digest and is never delivered. An unfinished version 5
  checkpoint is refused before any provider call. See
  [delivery acceptance](docs/checkpoint-format.md#delivery-acceptance-version-5).
- Jobs record each completed call's delivery decision. The built-in artifact
  check, `smythe.jobs.artifact-spec` version 1, reports each finding with a
  code such as `artifact_undecodable`, `mime_type_mismatch` or
  `dimensions_mismatch` and its expected and observed values. The call's
  `call_completed` or `artifact_rejected` event stores the decision, every
  artifact's path, SHA-256 and size, and the check's reports under
  `delivery_acceptance`, in the transaction that settles the call's cost.
  `SQLiteRunStore.complete_call(delivery_acceptance=...)` refuses a decision
  that disagrees with `accepted`, names another attempt or omits an artifact.
  Attempt errors, operation statuses and the Jobs schema are unchanged. See
  [delivery acceptance](docs/jobs.md#delivery-acceptance).
- Jobs manifests can name registered validators in an operation's
  `artifact.validators`, by name and version only (`ValidatorReferenceV1`).
  `preflight_job(..., validators=[...])` and `JobRunner(..., validators=[...])`
  register trusted local validator objects; preflight refuses a name that is
  not registered at that version before any run exists, and a runner refuses
  a plan that names a validator it lacks. The plan records each resolved
  identity, so a changed validator configuration invalidates the approval
  and refuses resume. Each validator runs after the built-in check on every
  decodable artifact's exact bytes and its reports are recorded with the
  decision. A validator that raises or returns anything but its own
  `ValidationReport` has outcome `error` and makes the decision
  `indeterminate`: the call settles at its cost and the operation is
  `rejected`, never an unknown outcome. Manifests without validators keep
  their manifest hash, plan hash, operation IDs and approval tokens. See
  [registered validators](docs/jobs.md#registered-validators).
- `jobs inspect` reports each attempt's recorded delivery decision, and for
  each operation a projected `delivery_acceptance` and a derived
  `delivery_eligibility`: eligible only with an `accepted` record, the
  validators the plan requires now, and every accepted artifact verified on
  disk; otherwise it lists reasons such as `artifact_changed` or
  `validator_changed`. Attempts recorded by earlier releases are projected as
  a legacy built-in decision with the validator unrecorded, never as accepted
  or not evaluated. `JobRunner.resume` and `reroll` recheck the SHA-256 of
  every accepted artifact and return the same fields, so a corrupted accepted
  file is reported and ineligible. Eligibility is never written back, and
  recorded statuses and `accepted` flags are unchanged. The HTML report shows
  it in a Delivery column. See
  [delivery eligibility](docs/jobs.md#delivery-eligibility).
  not accept is returned as `rejected_output` with an empty `output`. See
  [delivery acceptance](docs/checkpoint-format.md#delivery-acceptance-version-5).
- Strict delivery acceptance on ordinary runs. `Swarm(acceptance=...)` and
  `Swarm.from_yaml(..., acceptance=...)` take an `AcceptancePolicy`; the
  default `None` and an advisory policy keep current behavior. A strict
  policy decides after synthesis, on exactly the delivered text: every final
  validator runs once on its bytes with no provider call, every gate's final
  verdict is read, and the record binds the output digest, synthesis
  strategy, source nodes and generations, fallback flag, task and policy
  hashes and validator outcomes. A final rejection is terminal. Without final
  validators a passed gate accepts only `DELIVERABLE` output that is exactly
  the node it judged, and only a gate you supplied counts: a strict `Swarm`
  marks each gate of a plan that an architect other than a
  `DeterministicArchitect` or `ConstrainedArchitect` generates with
  `generated_gate`, and such a gate, like one a supervisor revision added,
  accepts only with `AcceptancePolicy(allow_generated_gates=True)`, which is
  off by default and part of the policy hash only when set; other synthesis
  strategies, the deliverable fallback
  and steps after a gate need a final validator, and artifact deliverables
  are not accepted. Under a strict policy an exhausted gate's last verdict is
  read and recorded (`rejected` by default), a gate with
  the node it judged; other synthesis strategies, the deliverable fallback
  and steps after a gate need a final validator, and a delivered node's files
  need a passed gate on that node. Under a strict policy an exhausted gate's
  last verdict is read and recorded (`rejected` by default), a gate with
  `max_regenerations=0` records each judge attempt's verdict under
  `advisory_verdict`, and an unreadable or non-bool verdict, a raising
  verifier, or a skipped or failed judge leaves acceptance `indeterminate` by
  default without spending a regeneration or stopping the run. Only verdicts
  the run's judges produce decide it: a fresh strict run refuses, before any
  provider call, a graph in which a judge already completed, whatever record
  or reply it holds. A result that is not accepted
  returns `output=""` with the text in `rejected_output`. Every checkpoint of
  a strict run is version 5 with its record, an unfinished one resumes only
  under the same policy, and resuming a completed one rechecks the output
  digest, task, source generations and policy, giving `indeterminate` on any
  mismatch. A strict `Swarm` resuming a completed checkpoint that has no
  record returns its text only as `rejected_output`, with `acceptance=None`,
  and logs how to obtain a decision. The trace gains an `acceptance:delivery` span. `Verdict.invalid`
  marks a verdict that could not be read,
  `smythe.acceptance.evaluate_delivery()` makes the decision outside a run,
  `smythe.synthesizer.deliverable_sources()` names the nodes `DELIVERABLE`
  synthesis returns, and executors take a keyword-only `acceptance`. See
  [strict delivery acceptance](docs/verifier.md#strict-delivery-acceptance).
- Autotune assessments state a three-way verdict read from both endpoints of
  the primary paired interval: `improved` when the lower endpoint exceeds
  `min_improvement`, `worse` when the upper endpoint is below zero, and
  `inconclusive` otherwise. `PromotionAssessment.verdict`,
  `OptimizationResult.verdict` and
  `smythe.optimize.statistics.improvement_verdict()` expose it; decisions
  record it for each paired stage and the campaign, and a decision made in
  development records `inconclusive`; and
  `smythe optimize concurrency --json` reports it beside `promoted`, with
  unchanged exit codes. Promotion is unchanged. HTML reports show each verdict,
  and decisions recorded earlier show one derived from their saved interval
  endpoints, labeled as derived. See the
  [three-way verdict](docs/optimize.md#confirmation-and-holdout-protocol).
- Preview contract types for external actions in `smythe.actions`, which
  `import smythe` does not load. `ActionIntent` binds an operation, a
  tenant-scoped `ActionTarget`, JSON parameters, the accepted content's
  `ContentRef` and the connector's name and version under contract
  `smythe.action.v1`; its `digest()` is stable under key reordering and
  changes with any field. `action_key()` derives one key from the journal,
  run, slot and sequence that every attempt of an action reuses.
  `ConnectorProfile` declares deduplication, lookups, a visibility bound,
  compensation, nanoUSD ceilings and readback limits with exact types.
  `WriteConnector` and `ApprovalVerifier` are synchronous host protocols with
  `ConnectorReceipt`, `ObservedObject`, `Principal`, `ConnectorRefused`,
  `ConnectorAmbiguous` and `ApprovalRejected`; `check_connector()` and
  `check_verifier()` refuse providers, classes, coroutine functions and
  awaitables, and async protocol methods: `submit()`, `lookup()` and `read()`
  on a connector, whose `profile` must be a `ConnectorProfile`, and `verify()`
  and `still_authorized()` on a verifier, whose `identity` must be a name
  without control characters. Other methods are not checked. Nothing
  stores or dispatches actions yet. See [external actions](docs/actions.md).
- Version 2 workflow journals have four tables for external actions:
  `workflow_actions`, `workflow_action_attempts`, `workflow_action_approvals`
  and the append-only `workflow_action_records`, which nothing writes yet.
  Action money is a separate `external_actions` accounting scope that never
  enters a run's balances. `SQLiteWorkflowStore.audit()` also checks a run's
  actions: intent digests, action keys, states, verdicts and record kinds,
  approvals bound to their action's intent, and balances equal to the sums
  of each action's attempts and records. `inspect_run()` adds an `actions`
  block only for a run that has actions, without parameters, approval
  assertions, receipts, record payloads or error text. See
  [external action records](docs/workflow-accounting.md#external-action-records).
- Jobs operations have a semantic input digest.
  `smythe.jobs.operation_input_digest()` hashes the provider kind and its
  adapter version (`smythe.jobs.providers.ADAPTER_VERSIONS`), the model,
  options, prompt and artifact spec, and each attachment's SHA-256 and MIME
  type in order, the type read from a fixed table of image extensions so
  that the digest is the same on every platform. It leaves out the template
  key, count, ordinal, profile name, attempt limit, cost ceilings and
  attachment paths, so every sample of a template shares it. `jobs inspect`
  adds `input_digest` to each operation, computed from the stored plan, and
  prints it; the HTML report shows it in an Input digest column. It is null
  where it cannot be computed, including when the stored attachment
  fingerprints are unreadable or an operation's stored attachments are not an
  array, which the function refuses with `ValueError`.
  `SQLiteRunStore.plan_attachments()` reads a run's recorded attachment
  fingerprints. The digest is never stored: plan hashes, operation IDs,
  approval tokens, exports and the Jobs schema are unchanged. See
  [operation identity](docs/jobs.md#operation-identity-and-input-digest).
- Strict runs check their acceptance obligations before they spend.
  `smythe.acceptance.validate_acceptance_obligations(graph, policy)` lists
  every reason a strict policy could not decide a graph's output: a gate
  that verifies itself, a node not in the graph or a node it does not depend
  on, a gate with `FailurePolicy.SKIP`, and, without final validators,
  delivered output that is not exactly one node judged by a gate that
  counts. A strict `Swarm` refuses such a graph with `ValueError` before any
  provider call, and a generated plan before any node runs; without final
  validators or `allow_generated_gates` it refuses before planning a run
  whose architects all generate their gates. A supervisor revision that
  would break an obligation the graph met, such as a step added after the
  gated node with no final validator to check it, is refused and traced as
  `revision_rejected`, and the run continues on its plan. Runs without a
  strict policy are unchanged. See
  [acceptance obligations](docs/verifier.md#acceptance-obligations).
- `examples/15_strict_acceptance.py` runs offline with scripted
  `OfflineProvider` replies: a draft passes its gate, the final step drops a
  required section, and a strict policy with a local final validator
  rejects the returned text and keeps it as `rejected_output`. Without the
  validator the same strict run is refused before any provider call.
- Built-in final validators `TextValidator` and `JSONValidator` in
  `smythe.validators`, also exported from `smythe`. Each interprets a
  specification given as JSON data with versioned code, and its identity
  binds its name, its code version and the SHA-256 of the specification with
  every default filled in. `TextValidator` checks length bounds and required
  Markdown sections, in order and with body text by default, so a strict
  policy rejects a final step that returns only its own increment; a section
  written as a heading line, such as `## Sources`, drops its `#` marker when
  the validator is built.
  `JSONValidator` requires strict JSON within a size bound and runs checks at
  JSON paths: `present`, `equals`, `number` within a tolerance, `nonempty`,
  `object`, `length`, `keys`, `required_keys` and `citations`. Both run
  locally with no provider call, reject output that fails a check, including
  bytes that are not UTF-8, and refuse an invalid specification with
  `ValueError`. See
  [built-in final validators](docs/verifier.md#built-in-final-validators).
- `smythe.assets.validate_image_bytes(data, spec, ...)` validates one
  in-memory snapshot of image bytes: the bytes hashed for a finishing receipt
  are the bytes decoded. A relative logo or font path resolves only against
  an explicit `base_dir`, never the working directory, and is otherwise
  reported unavailable. See
  [validation reports](docs/verifier.md#validation-reports).
- Ordinary runs record each artifact file's `sha256` and `size_bytes` in
  `node.metadata["artifacts"]` beside its path and MIME type. Under a strict
  acceptance policy a regenerated node writes its files under names that
  carry its generation (`draw_g1_00.png`; the first generation keeps
  `draw_00.png`) and never replaces an existing file: a name that is taken,
  for example after an interrupted run, gets a numbered suffix
  (`draw_00-2.png`). This publication needs hard links: on a filesystem
  without them, such as FAT or exFAT, the node fails with an `OSError` that
  names the cause and asks for an `artifact_dir` on a filesystem with hard
  links. Runs without a strict policy keep their file names and
  still replace a regenerated node's files. See
  [artifacts and jobs](docs/architecture.md#artifacts-and-jobs).
- Under a strict acceptance policy a regeneration keeps a bounded record of
  each generation it discards in the node's `rejected_generations` metadata:
  the generation, status, judge and reason, the result's SHA-256 and size, an
  excerpt of at most 1,024 bytes, a sidecar file holding a longer result in
  full when the run has an `artifact_dir`, and the generation's artifact
  records. All records add at most 64 KiB to a checkpoint
  (`REJECTED_GENERATIONS_MAX_BYTES`); past that a record drops its excerpt,
  and then it is not kept. Runs without a strict policy still clear rejected
  generations, and a new durable workflow refuses a graph that carries these
  records. See [rejected generations](docs/verifier.md#rejected-generations).
- Strict delivery acceptance covers files that delivered nodes wrote. Their
  bytes are re-hashed against the `sha256` and `size_bytes` recorded with
  them when the run decides, and they count as checked when a passed gate
  judged their node at its current generation and the policy allows gate
  acceptance; final validators still check only the text. The record lists
  each verified file under `artifacts` (`AcceptanceArtifact`: node, file
  name, SHA-256, size and MIME type), a key a text-only record leaves out. A
  file that is not hashed, missing or changed, or that no passed gate covers,
  gives `on_invalid`, and resuming a completed run re-hashes every bound file.
  `smythe.acceptance.delivered_artifacts(node)` performs the file check. See
  [files a delivered node wrote](docs/verifier.md#strict-delivery-acceptance).
- `design_verifier(require_artifacts=True)` fails a judged node that wrote no
  files; the default still passes it. The returned verifier's `identity` is a
  `ValidatorIdentity` named `smythe.design` at
  `smythe.design.DESIGN_VERIFIER_VERSION`, whose configuration digest covers
  the palette, dimensions, `include_advisory`, `require_artifacts` and the
  detector thresholds. See
  [gating without a model](docs/verifier.md#gating-without-a-model).
- `smythe runs inspect` and `smythe runs list` read saved runs without
  changing them: ordinary checkpoints from a `FileCheckpointStore` directory
  (`--checkpoints`, default `~/.smythe/checkpoints`), or durable runs from a
  workflow journal opened read-only (`--journal`), so a version 1 journal is
  never upgraded. No provider is constructed. `inspect` reports execution
  state, the delivery acceptance decision with its reasons, output digest and
  validator identities and outcomes, the digest and size of withheld output
  (never its text), cost completeness and unknown exposure, timing, and the
  next action; `--json` prints one JSON document. A decided record whose saved
  output, task or source generations no longer match is reported as
  `indeterminate`, as resume would return it. Durable runs under a strict
  policy report the decision their version 2 checkpoint holds, which must
  also match the decision the journal applied. `inspect` exits with 10 when a
  completed run's output was `rejected` or `needs_review` and 11 when the
  decision is `indeterminate`; an accepted run, or one without a record such
  as a version 4 checkpoint, exits 0. An unknown run exits 6 and an unreadable
  store or run 7; `list` reports each run it cannot read, such as an entry it
  cannot open or JSON nested too deeply, with its error and continues.
  `SwarmResult.to_dict()` adds the same `execution`,
  `delivery`, `unknown_exposure` and `next_action` blocks. See
  [Inspect saved runs](docs/runs.md).
- Gate validators in durable text workflows. `LocalOnly(factory, identity,
  version, role="validator")` declares a registered validator, and a durable
  `Swarm` takes the declaration as `verifier`: each gate's verdict is that
  validator's report on the exact text of the node it judges, and a
  validator that raises or cannot decide ends the gate without a verdict,
  spending no regeneration and buying no further judge call, with or without
  a strict policy. The declaration refuses a `Provider`, coroutine function or
  awaitable factory, and a built validator that is a provider, awaitable or
  class, lacks a synchronous `validate()` or names another identity or
  version. The recipe records `"verifier": "validator-v1"` and the
  validator's name, version and configuration digest, so resuming with
  another validator raises `WorkflowConflictError` before any call; recipes
  without one are unchanged. An ordinary `Swarm` refuses a `LocalOnly`
  verifier with `TypeError`. See
  [gate validators in durable workflows](docs/verifier.md#gate-validators-in-durable-workflows).
- Strict delivery acceptance in durable text workflows. A `Swarm` with a
  `run_store` now takes a strict `AcceptancePolicy` instead of refusing it,
  and `WorkflowRuntime` takes it as `acceptance`. The recipe records the
  policy, so a run resumes only under it, and gates read with
  `TokenVerifier` record the verifier identity `token-v2`
  (`smythe.verifier.TOKEN_VERIFIER_IDENTITY`). The same gates count and the
  same obligations hold as on ordinary runs: the saved planning result marks
  each gate of a generated plan `generated_gate` (a `LocalOnly` architect's
  plan counts as yours), a run whose architects all generate their gates is
  refused before the planning call when the policy has no final validator or
  `allow_generated_gates`, and the acceptance obligations are checked for a
  supplied graph before its run is created, for a generated or edited plan
  before its first node runs, and for each supervisor revision before it
  applies. After synthesis an `acceptance` journal operation makes the
  decision on the exact output, and the completed checkpoint consumes it
  with the output in one transaction: a run stopped before the decision
  commits decides again on resume, and a committed decision replays with no
  validator run or provider call. Such runs, and runs with a gate validator, write checkpoint
  version 2 with a `delivery_acceptance` block and key synthesis after a
  regeneration as `synthesis/<digest>`, a short digest of every node id and
  its generation, so two graph states whose generations add up to the same
  number never share a synthesis. Resuming a completed run rechecks the
  output digest, task, source generations and journal decision, giving
  `indeterminate` on any mismatch, and results follow the strict output
  rule. A verification receipt left pending, including one from a release
  before 0.8.1, is read on resume with the current rules and never passes
  unread. Recipes and checkpoints without a strict policy or gate validator
  are unchanged, including `"verifier": "token-v1"`. See
  [strict acceptance in durable workflows](docs/verifier.md#strict-acceptance-in-durable-workflows).
- `smythe.optimize.EvaluatorManifest` derives an Autotune `evaluator_hash`
  from the evaluator's code digest, the task-pack manifest digest of each
  split, validator identities, judge configuration and price snapshot
  version. Key and validator order do not change the hash and empty fields
  are left out. Because the holdout identity includes the evaluator hash, a
  changed held-out task pack or validator defines a new holdout. Runners still
  accept a raw evaluator hash. See
  [evaluator manifests](docs/optimize.md#evaluator-manifests).
- `OptimizationRunner` takes `task_pools`, the number of distinct tasks each
  split's evaluator draws from. `TrialContext.task_index` then names the
  task for each trial: the incumbent and challenger share it on a seed, no
  two seeds of a split repeat it, and it is reproducible on resume while
  holdout tasks follow the campaign's sealed seeds. A plan with more
  repetitions than a split's pool is refused before the campaign is created.
  Declared pools join the plan hash; plans that declare none keep theirs.
  See [finite task pools](docs/optimize.md#finite-task-pools).
- `OptimizationRunner` takes `max_holdout_draws`, the number of campaigns
  that may draw from one holdout identity's finite task pack, whatever
  policy they test. Every campaign that prepared a holdout trial under the
  identity counts, incumbent trials and uncapped campaigns included. Once
  the cap is reached a new campaign is refused with `HoldoutAlreadyUsedError`
  before it is created, and `ExperimentLedger.prepare_trial()` rechecks a
  `max_holdout_draws` cap in the transaction that records a campaign's first
  holdout trial, so racing campaigns cannot both take the last draw.
  `ExperimentLedger.holdout_draw_campaigns()` lists the campaigns that drew.
  The cap joins the plan hash only when set and is not part of the holdout
  identity, so changing it never reopens a used holdout; the ledger schema is
  unchanged.
- `OptimizationRunner.run()` takes `references`, sealed comparison policies
  such as a predecessor or a simpler alternative, and a `reference_rule`.
  References skip development and run beside the incumbent and the selected
  challenger on the same confirmation and holdout seeds, and the challenger is
  assessed against each. Under `"must_not_be_worse"` a stage also fails when a
  reference is better on the primary objective; the default `"report"` only
  records the comparisons. A reference is never promoted, and its holdout
  trials draw from a finite holdout pool but are not holdout use, so a policy
  that already used its holdout can still be a baseline. Decisions record
  `reference_rule` and a `references` list with each comparison and verdict,
  `OptimizationResult.references` returns them, HTML reports add a table of
  every arm, and `smythe optimize concurrency --reference N` adds reference
  concurrencies. References join the plan hash only when declared, so two-arm
  plans keep their hashes; the ledger schema, contract hash and holdout
  identity are unchanged. Earlier releases count reference holdout trials as
  holdout use and refuse to export a report for a decision with references.
  See [reference arms](docs/optimize.md#reference-arms).
- Recorded read steps in durable text workflows. A `ReadProfile` declares one
  versioned read: an HTTP `GET` of a URL whose path or query placeholders take
  a step's checked input, a decoder and its version, whether a read can be
  charged (`metered`, with its per-call price as `ceiling_nanousd`) and
  whether it is safe to send again (`repeat_safe`), and limits on response
  bytes, time, redirects and hosts. A read stops at its time limit however
  slowly the source sends, and once connected, as soon as its step stops
  waiting. Loopback, private and other non-public addresses are refused
  before the request and at each redirect unless `allow_private_networks` is
  set, and the request connects to the address that was checked.
  `Swarm(run_store=..., read_tools=[...])` runs a node whose
  `metadata["read_step"]` names a profile and its input, as
  `ReadProfile.step()` builds, without a model call. The journal records each
  read as a `read_tool` call in the `tool` phase, quoted at its declared
  ceiling and dispatched once, and saves the response before settling and
  decoding it, so a saved response or text is reused on resume without
  reading again. A free read costs nothing; a metered read is charged its
  price for a 200 response and otherwise keeps unknown exposure. A rejected
  read raises `ReadToolError` for the node's failure policy. A free,
  repeat-safe read whose response was lost is read again as a new call; any
  other read without a response blocks the run and is never sent again. Only
  graphs and templates you write declare read steps, never a generated plan,
  and an ordinary run refuses one before any node call, in a caller's graph,
  an architect's or template's plan, or a saved checkpoint. The recipe records
  `read_tools` only when a run declares profiles, so other recipes keep their
  identity. See
  [recorded read steps](docs/workflow-accounting.md#recorded-read-steps).
- Immutable context snapshots in durable text workflows. A `Swarm` with a
  `run_store` takes `context_sources`, a sequence of `ContextSource` (an id,
  text and optional version), and its new runs record `context_assembly: 1`
  in their recipe and the sources once, as revision 0, in a `context` journal
  operation keyed `context-sources/<revision>`. Before each node call the run
  records a `ContextSnapshot` as a `context` operation keyed
  `context/<node>/<generation>`: the task's goal, constraints and acceptance
  criteria, the step, the sources with their versions and digests, the
  digest of each dependency output with the calls that produced it, the
  agent's capability tags and the assembly version. The call's trigger names
  the snapshot id, checkpoints hold only digests, and the prompt is rendered
  from the snapshot with the task's context data, sources, dependency
  outputs and tool results in delimited untrusted-data blocks. A resumed or
  retried call uses the retained snapshot after checking the task and
  dependency outputs against it, so sources changed while a run was paused do
  not reach it; `Swarm.refresh_context` records new sources as the next
  revision and makes every node pending at its next generation for `resume`
  to re-evaluate. Such runs key synthesis by node generation. Executors take
  a keyword-only `context_assembler` that selects this rendering, and
  `smythe.context` reads the records back. Recipes without
  `context_assembly` keep their prompts and records byte for byte. See
  [context snapshots](docs/tasks.md#context-snapshots).
- Source evidence. `smythe.evidence` adds versioned JSON records:
  `SourceRecord` (a source's id, version or digest, and its retained text, a
  reference, or a tombstone that keeps only its identity, digest, reason and
  time), `ClaimRecord` (a claim, supporting and contradicting passages quoted
  from named sources, scope, unresolved questions and a proposed conclusion)
  and `SourceEvidence`. `from_dict()` rejects unknown versions and keys. An
  artifact carries its records in one `source-evidence` fenced block.
  `SourceEvidenceValidator`, a final validator whose identity pins each
  source's id, version, digest and state, rejects output whose evidence
  states no claim, cites a source the run was not given, states another
  version or digest, quotes a passage that is not verbatim in its source,
  cites a source it does not list, or lists a contradiction the artifact's
  body leaves out. A
  passage from a tombstoned source cannot be checked, which gives an `error`
  outcome and never acceptance. Whether a passage supports its claim stays a
  judgement for a gate. A durable run whose strict policy has such a
  validator journals the evidence its output carries as a `source_evidence`
  operation keyed `source-evidence/<sha256>`, and
  `smythe.evidence.load_source_evidence` reads it back without source text
  unless `include_text=True`. In a durable run with context assembly, a
  validator that checks other sources than the run's current source set,
  such as those before `Swarm.refresh_context`, does not run: its outcome is
  `error`, the decision follows `on_invalid` (`indeterminate` by default)
  and the reason names the differing sources; `evaluate_delivery` takes a
  keyword-only `unusable` mapping of validator identities to the reason each
  cannot check the output. See [source evidence](docs/source-evidence.md).
- Failure classes. Every exception a node attempt raises has one class:
  `input_error`, `authority_denied`, `quality_defect`, `provider_rejected`,
  `transient`, `unknown_outcome`, `terminal_accounting`, `persistence` or
  `workflow`; `validator_error` and `security_stop` are named, but no
  exception has them. Each failed attempt's trace span records its class as
  `metadata["failure_class"]`, which `Tracer.summary()` leaves out and
  checkpoints do not store. The read-only
  `ProviderRequestRejectedError.failure_class` is `authority_denied` for HTTP
  401 or 403, `input_error` for 404 or 413 and `transient` for 429. Both
  executors decide which failures stop a run, whatever the node's failure
  policy, through one shared classifier, and they retry, skip and stop on
  exactly the failures they did before. See
  [failure classes](docs/execution.md#failure-classes).
- `ProviderResponseError.stop_reason` names why a native
  `OpenAIResponsesProvider` or `AnthropicMessagesProvider` response ended
  unfinished, with the values the SDK providers use: `max_tokens` or
  `model_context_window_exceeded` for a truncation, `refusal`,
  `content_filter`, or `incomplete` for any other incomplete response,
  including a Messages `pause_turn`. It is `None` for other unusable
  responses and for errors raised elsewhere. The error's message is
  unchanged, so durable runs still match the text their journals saved for
  rejected calls. See [execution policies](docs/execution.md).
- `Swarm.reevaluate(execution_id, acceptance=policy)` and
  `Swarm.areevaluate()` record a new delivery acceptance decision for a
  completed run's saved output, for example under a fixed validator. The
  strict policy, by default the `Swarm`'s own, runs its final validators on
  the saved bytes and reads every gate's recorded verdict; no node runs, no
  provider is called, and costs, the budget, retry and regeneration counters
  and node generations are left as saved. Only a completed run with a decided
  record is re-evaluated, and only while its saved output still matches the
  record's digest and its task, source generations and delivered files are
  unchanged; otherwise nothing is written, so changed output never becomes
  accepted. Each decision has a sequence, 0 for the run's own and one more
  for each re-evaluation, so the same bytes under an unchanged policy get a
  new decision. An ordinary checkpoint keeps the new record as its
  `delivery_acceptance` and the last 16 it superseded under
  `delivery_acceptance_history`, which `smythe.checkpoint` reads back. A
  durable run accepts a recipe that differs only in its acceptance policy and
  gate verifier, journals the decision as a new `acceptance` operation keyed
  `acceptance/<sequence>/<digest>` and applies it with a new completed
  checkpoint, leaving calls, charges and the recipe unchanged;
  `smythe.workflow.current_decision()` and `decision_history()` read the
  decisions back. Resume returns the current decision and never makes a new
  one: a strict `Swarm` resuming a completed run without a record, or one
  decided under another policy, behaves as before, and a durable run still
  resumes only under the policy its recipe records. `smythe runs inspect`
  adds a `decisions` block with the current decision's sequence and the
  decisions before it, and its exit codes follow the current decision. See
  [re-evaluation](docs/verifier.md#re-evaluation).
- Durable text workflows can publish their delivered output as a local
  document. `Swarm(run_store=..., renderer=MarkdownRenderer())`, or a
  deterministic renderer declared with `LocalOnly(..., role="renderer")`,
  renders the synthesized text after synthesis, and the document's bytes are
  published once into `artifact_dir/<run id>/`, which an owner marker binds
  to the run, without replacing any file. The recipe records the renderer's
  identity and version only when there is one. The document is the run's
  output, so a strict policy's final validators check those bytes and its
  decision binds their SHA-256; a strict policy with a renderer needs a final
  validator. Each publication is a journaled `publication` operation holding
  a `PublishedArtifact` record, and a process lost after publishing reuses
  its file. `smythe.workflow_artifact.current_artifact()` returns a run's
  current publication, the decision that covers it and whether it is
  eligible for delivery. A missing or changed file is not: resuming the run
  gives an `indeterminate` decision, and `smythe runs inspect` reports the
  `artifact` block, exits with code 11 and names the next action
  `republish`. `Swarm.republish()` publishes the saved document again under
  a free name, which becomes current, with no provider call and the same
  decision; re-evaluation needs an intact document, so it is refused until
  then. See [published documents](docs/workflow-accounting.md#published-documents).
- An opt-in `RetryPolicy` makes the executor the only retry owner of an
  ordinary run's node calls; `Swarm(retry_policy=...)` and both executors
  take one. A `RETRY` node then calls again only after a failure whose class
  the policy lists (`transient` by default, so never after a 401, 403, 404 or
  413), waits as long as a transient status error's `Retry-After` asks or
  else a jittered backoff, never longer than `backoff_max_s`, and stops at
  `deadline_s`, which a node's attempts and waits share: each attempt's
  timeout shrinks to the time left, and the node fails with
  `RetryDeadlineError`, failure class `deadline`; the trace records that
  refused attempt as a span with outcome `timed_out`, class `deadline` and no
  call. A run under a policy refuses, before routing, planning or any node
  call, every provider it can call that does not declare
  `ProviderFeatures.sdk_retries` as `0`, so no SDK retries a call unseen:
  its own, a `WhiteRabbit` classifier's, that of each `LLMArchitect` or
  `ConstrainedArchitect` that could plan it, including one built from
  `planning_provider`, and an `LLMSupervisor`'s when `max_revisions` lets it
  review; `Executor` and `AsyncExecutor` check the supervisor's with their
  own. Built-in providers declare it, and `AnthropicProvider` takes
  `max_retries`, passed to the SDK only when set. Durable runs refuse the
  policy, and checkpoints record it only with `persist_attempts`. Executors
  also take `sleep` and `rng`, which replace the retry wait and its jitter.
  Without a policy, retries are unchanged. See
  [retry policy](docs/execution.md#retry-policy).
- `RetryPolicy(persist_attempts=True)` keeps a node's allowance of attempts
  across crashes and resumes. Each attempt is saved as spent before its call,
  whatever `checkpoint_every_n_nodes` is, and checkpoints record the policy
  and each node's attempts in its current `execution_generation` under
  `control.retry_policy` and `control.attempts_used`. Such checkpoints are
  version 5 and list `attempt_allowance` in `requires`, so builds that read
  only version 4, or that do not know the feature, refuse them instead of
  refilling the allowance. Resume starts each node from the attempts its
  generation used, so a `RETRY` node makes at most `1 + max_retries` calls in
  total, and a node with none left fails without a call; a regenerated node
  gets a fresh allowance. An unfinished run resumes only under an equal
  policy, which `smythe runs inspect` names in its resume action. Executors
  take `attempts_used=` for their first run and report the `attempts_used`
  property, and `smythe.checkpoint.attempt_allowance_from_state()` reads the
  record. Other runs keep writing version 4. See
  [persisted attempts](docs/execution.md#persisted-attempts).
  policy, and checkpoints do not record it. Executors also take `sleep` and
  `rng`, which replace the retry wait and its jitter. Without a policy,
  retries are unchanged. See [retry policy](docs/execution.md#retry-policy).
- Under a `RetryPolicy`, an ordinary run classifies the native providers'
  failures as a durable journal does. A 401, 403, 404, 413 or 429 response
  from `OpenAIResponsesProvider` or `AnthropicMessagesProvider` whose body is
  only the provider's plain error object settles at zero cost and raises
  `ProviderRequestRejectedError`, whose class follows its status, so the
  default policy retries a native 429 with exact totals. A 400, another
  status or a body with usage still leaves the bill unknown and stops the
  run. A refused or `max_tokens` reply with a known charge raises the new
  `NativeOutputDefectError`, class `quality_defect`, after recording that
  charge once, when the policy lists `quality_defect`; otherwise it still
  stops the run. Neither leaves an unresolved marker, so the run resumes like
  any other failure. Without a policy, native failures still stop the run.
  See [retry policy](docs/execution.md#retry-policy).
- `jobs stop --drain-timeout-s SECONDS` bounds a pause's drain. That long
  after the request, the worker cancels each call still awaiting its provider
  and records it as `unknown_outcome`, keeping its full per-call ceiling as
  exposure; a call whose response has arrived settles normally.
  `SQLiteRunStore.request_pause()` and `stop_job()` take `drain_timeout_s`
  (0-3600), the earliest deadline of an open request holds until resume
  clears it, and `SQLiteRunStore.drain_time_remaining()` reads it. The
  worker reads it at least once a second while calls run; a read that fails,
  for example on a locked database, leaves the run and any deadline already
  read unchanged. The `pause_requested` event records it as
  `drain_deadline_ns`, so the Jobs schema stays at version 4. Without it, a
  pause waits for every admitted call as before. See
  [stop and resume](docs/jobs.md#stop-and-resume-safely);
  the guide's [current boundaries](docs/jobs.md#current-boundaries) now state
  that Jobs do not read a response's `stop_reason`.
- A durable run with context assembly can remove a source.
  `Swarm.remove_source(execution_id, source_id, reason)` and
  `aremove_source()` journal its tombstone, a `SourceRecord` that keeps the
  source's id, version and SHA-256, the reason and the time and none of its
  text, as a `source_tombstone` operation keyed `source-tombstone/<sha256>`
  that a new checkpoint applies, with no call. The run then never uses the
  text again, and nothing rebuilds it from the journal's copy:
  `load_context_sources` and `load_context_snapshot` raise the new
  `smythe.context.SourceRemovedError` for a record that holds it,
  `load_source_evidence` marks the source `removed` and returns no text for
  it, an unfinished run resumes only after `refresh_context` gives it other
  sources, and a completed run's earlier decision becomes `indeterminate`
  until `reevaluate` decides again under a source evidence validator that
  holds the tombstone. A decision binds the keys of the tombstones the run
  held when it was made, so a re-evaluation stopped before a removal is
  decided again rather than finished. `smythe.evidence.load_source_tombstones`
  reads a run's tombstones. Earlier journal records and published documents
  keep what they hold. See [remove a source](docs/tasks.md#remove-a-source).
- `smythe runs inspect` reports what a durable research run was given, read
  and claimed, without source text, read bodies, quotes or output: a
  `context` block (the context-assembly and snapshot versions, the current
  sources by id, version, digest and size, removed sources, each node's
  snapshot id and the run's tombstones), `tool_calls` (each journaled
  read's node, attempt, profile, URL, states, charge, HTTP status, response
  digest and the size and digest of its decoded text), `evidence` (the
  delivered source evidence: each cited source's availability, and each
  claim's passage counts, cited sources, scope and unresolved questions)
  and `issues` (unresolved questions, unavailable sources, failed reads and
  final validators that did not pass). Ordinary runs report null for the
  first three. See [research runs](docs/runs.md#research-runs).
- [`examples/16_source_to_artifact.py`](examples/16_source_to_artifact.py)
  runs a durable research workflow offline and without keys. It plans one
  journaled read step per bundled source, reads the sources from a server it
  starts on the loopback address, stops part way as a lost process would,
  resumes without reading a saved source again, publishes a brief carrying
  its source evidence, accepts it under a strict policy with
  `SourceEvidenceValidator` and `TextValidator`, and prints the accepted
  document's path, the decision, the evidence and the
  `smythe runs inspect` report. `tests/test_examples_smoke.py` runs it.
- Routing decision records. `WhiteRabbit.decide()` and `adecide()` return the
  chosen architect with a `RoutingDecision`: the tier, the method
  (`classifier`, `fallback_no_classifier` or `fallback_unmatched`), the tiers
  offered, reasons, the routing policy version, the matched reply only as its
  normalized token, and the classifier call's token usage and reported cost;
  `route()` and `aroute()` still return the architect. A `Swarm` with a
  `WhiteRabbit` router records the decision as `graph.planning["routing"]`,
  where the new optional `ExecutionGraph.planning` holds informational
  records keyed by namespace. Ordinary checkpoints keep it and stay version
  4, a graph without records is saved as before, and the classifier's usage
  is recorded without being charged to the run's budget. A durable run's
  decision names the journal calls of its `routing` phase instead of copying
  their amounts, and its `planning` operation result carries the decision
  under `routing`; resume restores `graph.planning` from the saved plan
  without classifying again. The classifier prompt and durable recipes are
  unchanged. See [routing decisions](docs/architecture.md#routing-decisions).
- `LineagePolicy` opts a durable run into artifact lineage:
  `Swarm(run_store=..., acceptance=<strict policy>, lineage=LineagePolicy())`,
  `Swarm.from_yaml(..., lineage=...)` and `WorkflowRuntime` take one. It
  declares an `access_scope` label (declared, not authenticated, and compared
  exactly), the `reuse` switch and what `undeclared_coverage` calls for
  (`review` or `regenerate`). Without a `run_store` or a strict policy it is
  refused before any routing, planning or node call. The recipe records it
  under `lineage` only when configured, so other recipes keep their bytes,
  and plan, execute, resume and re-evaluation compare it exactly: a run
  resumed under another policy, or without its own, raises
  `WorkflowConflictError`. A new durable graph cannot carry `lineage_reuse`
  node metadata. `ArtifactLineage` is the versioned record of a run's
  accepted artifact, delivery decision and each node's inputs as ids,
  versions and digests, with no source text; its reader rejects unknown
  versions and keys. Runs do not write lineage records yet. See
  [artifact lineage](docs/lineage.md).
- `Swarm(reserve_for_checks=True)` holds allowance for a run's verifiers and
  its `LLM_MERGE` synthesis so other nodes cannot spend it. Under
  `max_budget_usd`, before any node runs, the run holds what each pending
  verifier and the merge will reserve; another node whose reservation would
  reach into that allowance is refused with `SentinelAlert` before its call.
  Each verifier and the merge converts its own hold into its reservation, a
  regeneration holds again for the verifiers it resets before their work
  reruns, and a run releases leftover holds after synthesis or when it fails.
  Holds never count as spend and are not saved: resume holds again, and
  checkpoints stay at version 4. `Sentinel.hold()`, `release_hold()`,
  `holds` and `reserve(..., from_hold=True)` expose the same allowance, and
  `Sentinel.check()` treats held allowance as spent. Durable runs refuse the
  option. Without it, reservations are unchanged. See
  [held allowance](docs/budgets.md#held-allowance-for-checks).
- `Synthesizer(retain_contradictions=True)` keeps the contradictions that
  source evidence lists. `DELIVERABLE` synthesis appends a deterministic
  `Contradictions` section for each contradicting passage that the delivered
  nodes' claim records list and their text does not quote, citing both
  sides exactly, so strict acceptance's `contradictions` check passes on the
  exact output; text that quotes every one is delivered unchanged.
  `LLM_MERGE` appends `CONTRADICTION_NOTE` to the merge prompt; a strict
  policy with `SourceEvidenceValidator` still rejects a merge that drops a
  listed contradiction. `MERGE_SYSTEM_PROMPT` is unchanged. `CONCATENATE` and `STRUCTURED` refuse
  the option. A durable recipe names it in its synthesizer description only
  when set, so other recipes keep their identity.
  `smythe.evidence.render_contradictions()` writes the section. See
  [keep contradictions through synthesis](docs/source-evidence.md#keep-contradictions-through-synthesis).
- An authorized operator can resolve output that a strict policy left
  `needs_review` or `indeterminate`. `Swarm.reevaluate(execution_id,
  operator=OperatorDecision(...))`, and `WorkflowRuntime.reevaluate(run_id,
  operator=...)`, take the operator's `accepted` or `rejected`, the SHA-256 of
  the evidence they decided on and an assertion that the host's
  `ApprovalVerifier` must accept for exactly `operator_decision_sha256()` of
  the run, the current decision's sequence and record, the decision and the
  evidence. A forged, expired or misbound assertion raises the verifier's
  `ApprovalRejected` and nothing is written, and an `accepted` or `rejected`
  decision stands. No node runs, no provider is called, and costs, budget and
  counters stay as saved. The decision is a new delivery acceptance record,
  version 2, whose `operator` block names the verified principal, the
  verifier and the evidence digest; the earlier record is kept. An ordinary
  checkpoint lists `operator_decision` in `requires` while it holds such a
  record, and a durable run journals it as its next `acceptance` operation.
  Readers that know only version 1 records, or only the
  `delivery_acceptance` feature, refuse it. A policy's records stay version
  1, and checkpoint versions are unchanged. See
  [operator decisions](docs/verifier.md#operator-decisions).
- `jobs adopt RUN_ID OPERATION --reason TEXT` accepts the bytes an
  unknown-outcome attempt already published, without a provider call. A
  response's bytes are published before its call settles, so a process that
  stopped in between can leave the whole response on disk. The latest
  attempt's directory must hold exactly the files of the artifact count the
  profile requests and nothing else, and they must pass the built-in artifact
  check and the plan's registered validators again; other bytes or operation
  states are refused with exit code 6 and leave the operation unchanged. The
  operation becomes `succeeded` with that attempt accepted and no attempt
  used, while the attempt and its call keep their unknown outcome and the
  call its exposure, so no charge is settled. An `unknown_outcome_adopted`
  event records the reason, each artifact's SHA-256 and the delivery
  decision, and `jobs inspect` and its HTML report show the adoption beside
  the kept exposure. In Python, `JobRunner.adopt()` records the verified
  bytes with `SQLiteRunStore.adopt_published_attempt()`; the Jobs schema
  stays at version 4. See
  [adopt published bytes](docs/jobs.md#adopt-published-bytes).

### Changed

- An ordinary `Swarm` without a per-call cost ceiling that its provider
  requires under `max_budget_usd` raises `BudgetEstimateRequired` before
  routing or planning when `execute(Task)` or `execute_async(Task)` has
  `LLMArchitect` plan the run or an `LLM_MERGE` synthesis needs the
  ceiling, and otherwise before the first node call. It used to be raised
  when a node or the synthesis reserved its budget, after planning and any
  earlier nodes had been paid for. `plan()` and `aplan()` still return a
  generated plan, so a caller can set each node's
  `metadata["estimated_cost_usd"]` and then execute the graph. Its `node_id`
  is `None` when a generated plan's model is refused before planning, and
  the message then names that alternative, and `"__synthesis__"`
  (`smythe.graph.SYNTHESIS_NODE_ID`) when an `LLM_MERGE` synthesis is
  refused.
- An ordinary `Swarm` raises `ValueError` before any provider call for a
  `tool_runtime` with `OfflineProvider`, `OpenAIImageProvider` or
  `AnthropicMessagesProvider`, for `attach_dep_artifacts` on
  `OpenAIImageProvider`, `OpenAIResponsesProvider` or
  `AnthropicMessagesProvider`, and for a node model those providers do not
  accept. These failed when a node made its call, after planning and earlier
  nodes. Such a run used to succeed when no node was offered a tool or no
  dependency produced an image; it is now refused.
- A durable run's `WorkflowBindingError` names the argument it refuses:
  `run_store` and `verifier`, with the type that was passed, `memory`,
  `tool_runtime` or `checkpoint_store`. One message used to cover memory,
  tools and the checkpoint store. The exception is still a `ValueError`.
- Elapsed-time records for ordinary runs. `SwarmResult.timing` reports a
  call's elapsed time on a monotonic clock beside its summed node time, the
  time spent in routing, planning, recovery, execution, verification,
  supervision, synthesis, checkpoint writes and waiting, and waits for retry
  backoff, admission and supervisor reviews. Routing and planning are
  measured only when `execute(Task)` plans in the same call. Checkpoints keep
  one segment per call in a new top-level `timing` block and stay at version
  4; a resumed run adds its own segment rather than comparing clocks across
  processes. Trace spans record their attempt index, phase and outcome, and
  `Tracer.timing()` and `Tracer(clock=..., wall_clock=...)` expose the
  measurements and clocks. `Tracer.summary()` and `SwarmResult.trace` keep
  their keys.
- `PlannerMemory` records each run's elapsed time beside its summed node
  time. New `ExecutionOutcome` fields `elapsed_ms`, `latency_complete`,
  `resumed`, `cost_is_complete`, `cost_contains_estimates` and `acceptance`
  default to `None`, meaning not recorded, so existing history recalls
  unchanged; new records set `acceptance` to `"not_evaluated"`. Recall skips
  rows whose new fields are malformed. `total_duration_ms` and the planner
  prompt's "Summed node time" are unchanged.
- Durable `run_store` results report `SwarmResult.timing` from the journal's
  wall-clock event times, which relate calls in different processes: one
  segment per lease session, the call's elapsed time, the wall time since the
  run was created, and whether a session ended without releasing its lease.
  `SQLiteWorkflowStore.lease_sessions()` returns the same session bounds.
- A durable run's `WorkflowBindingError` names the argument it refuses:
  `run_store` and `verifier`, with the type that was passed, `memory`,
  `tool_runtime` or `checkpoint_store`. One message used to cover memory,
  tools and the checkpoint store. The exception is still a `ValueError`.
- Trace span durations come from a monotonic clock, so a wall-clock
  adjustment during an attempt no longer distorts a span's `duration_ms` or
  `PlannerMemory`'s summed node time. `start_time` and `end_time` remain
  wall-clock seconds.
- The package audit's size limits are 800,000 bytes for the wheel and
  2,000,000 bytes for the compressed source archive, up from 400,000 and
  1,000,000, to fit delivery acceptance, recorded tools and context,
  governed external actions, artifact lineage, the evaluation additions and
  their tests.
- **Durable workflow journals upgrade to format version 2, which Smythe
  0.9.x and earlier cannot open, even read-only. Back up journals before
  upgrading.** Version 2 adds the `tool` call phase to `CallKey` and
  `CallScope` and the external action tables, and a journal must hold
  exactly its version's tables. Opening a version 1 journal with
  `SQLiteWorkflowStore(path)` upgrades it in one transaction that creates
  those tables, empty, and changes the recorded version, so every run, call,
  balance, reservation, evidence row and the store identity are kept; an
  interrupted upgrade rolls back. While any run holds an unexpired lease the upgrade raises
  `WorkflowLeaseError`, so stop workers from earlier releases first.
  `SQLiteWorkflowStore(path, read_only=True)` reads a version 1 journal
  without upgrading it, and the new `store_version` property reports the
  version. A store open for writing raises `WorkflowCorruptionError` and
  writes nothing once another release changes the journal's version. See
  [journal upgrades](docs/workflow-accounting.md#journal-upgrades).
- A Jobs approval token creates one run. `jobs run` with a token that already
  created a run in the same database exits with code 4 and names that run
  instead of starting another run with the full ceiling. `--new-run`, or
  `allow_approval_reuse=True` on `SQLiteRunStore.create_run` and
  `JobRunner.start`, starts another run on purpose, and that run's
  `run_created` event records how many earlier runs used the token and the
  five most recent run IDs under `approval_reuse`.
  Token derivation, plan hashes and the Jobs schema are unchanged.
- `ValidationReport` raises `TypeError` when it is constructed with a finding
  that is not a `ValidationFinding`, and keeps its findings as a tuple. It
  used to accept any value, failing only when `passed`, `hard_findings` or
  `advisories` read the finding, and kept a list of findings as a list, so a
  report built from a list did not equal one built from the same tuple.
- A Jobs response whose artifact bytes cannot be decoded, or exceed the size,
  pixel, frame or count limits, is a rejected attempt settled at the cost the
  provider reported, and those bytes are not written. It used to become an
  `unknown_outcome` that held the call's whole ceiling as unresolved exposure
  and needed a duplicate-spend acknowledgment to reroll. Invalid provider
  accounting still makes the call an unknown outcome, and attempts recorded
  earlier are unchanged.
- The Autotune runner is version 4. A decision made after confirmation keeps
  every challenger's development score under `development_scores`, not only
  the selected one, and the HTML report shows them all. The runner version is
  part of the plan hash, so plan hashes and automatic campaign IDs
  (`optimization_v4_…`) change: the same command without `--campaign-id`
  starts a new campaign, and an explicit `--campaign-id` created by an earlier
  release exits with code 8 (campaign binding drift). Holdout identities
  exclude the runner version, so a used holdout stays used; contracts, contract
  hashes and the ledger schema are unchanged.
- `design_verifier` checks its palette when it is built: a colour that is not
  six hexadecimal digits after an optional `#` raises `ValueError` from
  `design_verifier(...)` instead of from the first verdict. The gate reads
  each artifact file once, decodes those bytes, and fails a file whose bytes
  no longer match the `sha256` its record carries with a `changed-artifact`
  finding.

### Fixed

- Jobs commands that read a detached worker's receipts, such as `jobs attach`
  and `jobs status`, retry for up to a second when Windows briefly denies
  opening a receipt that the worker is replacing, instead of failing.
- Nine retained evidence archives no longer carry local user paths: the Astra
  pilot and main-study archives with their offline and CI verification logs,
  the Fable pilot and Fable Code archives, and the 5,000-operation Jobs
  bundle. Paths are replaced with placeholders such as `<checkout>` and
  `<temp>`, and frozen test sources use the placeholder API keys the live
  tests use. Every digest, run identifier, approval token and call-evidence
  binding derived from the changed bytes was recomputed, the Astra and Jobs
  charts cite the new digests, and measured values are unchanged. Each results
  folder adds `path-redaction-20260930.json`, which maps original digests to
  their replacements.
- `python -m benchmarks.verify_astra_publication` reproduces the committed
  Astra main-study analysis exactly on Python 3.12 and later again. Python
  3.12's `sum()` compensates float rounding, which changed the last digits of
  the arm cost totals; the analysis now adds those values left to right, as
  earlier Pythons did. A test recomputes the committed analysis from the
  archive and requires an exact match.
- The durable workflow guide lists `AnthropicMessagesProvider` with
  `claude-fable-5-1` among its supported providers instead of calling that
  support planned. The verification guide states that a gate node makes its
  own provider call even when a `CallableVerifier` computes the verdict. The
  architecture guide no longer says that `PlannerMemory` keeps distilled
  templates or that ordinary artifact records carry hashes, and the Jobs guide
  states that the per-attempt idempotency key is a local correlation key sent
  to no provider. The MCP guide states that a serial run checks the budget cap
  after each billed tool-loop turn, so the turn that crosses the cap is paid
  for.
- `CallableVerifier` fails closed. Only `True`, `False`, a `Verdict` whose
  `passed` is a bool, or an object with a bool `passed` attribute, such as a
  `ValidationReport`, decides a gate. Any other result used to count as a pass
  when truthy, so the string `"FAIL"`, a non-empty list of findings, a
  `ValidationReport` with hard findings, and an `async def` function's
  unawaited coroutine all passed. Such results now fail, and the verdict's
  reason names the returned type; the coroutine is closed. A function that
  returned a truthy non-bool to pass, such as `1` or a NumPy boolean, must
  now return `bool(...)`.
- A gate passes only when its verifier's verdict has `passed` exactly `True`.
  The executor tested truthiness, so a custom verifier's
  `Verdict(passed="false")` passed. Any other value is now a failed verdict,
  and its reason states that `passed` must be a bool.
- A custom verifier whose failed `Verdict` has no text reason, such as
  `Verdict(passed=False, reason=None)`, sends its target back for another
  try. The run used to stop with `VerificationRecoveryError` because the
  saved regeneration intent required a text reason; a missing reason is now
  empty and any other value is described by its type.
- A failed attempt's trace span now has status `failed` instead of `running`.
  Serial and parallel execution recorded every attempt that raised as still
  running, whether the node then retried, was skipped or stopped the run, and
  serial execution did the same for terminal errors such as a budget alert.
  The node itself still counts as running between retries, and the keys of
  `Tracer.summary()` are unchanged.
- `PlannerMemory` takes each node's `duration_ms` from that node's final
  attempt, so a retried node no longer reports the time of its first, failed
  attempt. `total_duration_ms` still sums every recorded span, and planner
  prompts still label it "Summed node time".
- A Jobs attempt started by a reroll records the reroll's reason. The reason
  was kept only in the `reroll_queued` event, so every attempt's `reason` was
  empty. `jobs inspect` now shows it on that attempt in text, JSON and the
  HTML report, and JSON status and export include it. First attempts and
  safe retries after recovery keep an empty reason, attempts recorded earlier
  are not changed, and the Jobs database stays at version 4.
- Executors accept only a finite, non-negative number for `retry_backoff_s`
  and raise `ValueError` otherwise, as durable workflows already did. `True`
  used to act as a one-second backoff, and NaN or infinity failed a serial
  retry or stalled an async retry indefinitely.
- `Swarm` without a `run_store` rejects an invalid `max_concurrency`,
  `max_revisions` or `retry_backoff_s` with `ValueError` when it is
  constructed, using the executor's rules. Such a value used to surface only
  when the executor was built, after the router's classifier and the planner
  had already been called. Durable runs already rejected these values before
  routing or planning. A serial `Swarm` (`parallel=False`) with
  `max_concurrency=0` used to run because serial execution ignores that value;
  it is now rejected too.
- `OpenAIImageProvider` raises `ValueError` for a request with image
  attachments, such as a node with `attach_dep_artifacts=True` whose
  dependencies produced images, instead of sending, and paying for, the
  request without them. The message matches the one Jobs already gives when
  it refuses OpenAI image attachments.
- `ConstrainedArchitect` copies its template list when it is constructed and
  raises `ValueError` for duplicate template names. It used to keep the
  caller's list, so changing that list later changed the menu shown to the
  model but not the templates it could select, and a selection from the
  changed menu failed and was retried as a new paid call. With duplicate
  names, the menu listed every template, but only the last one with a name
  could be built.
- `distill_template` keeps verification gates and execution policies. A
  distilled node now carries its `verifies` target, `max_regenerations`,
  `failure_policy`, `max_retries`, `timeout_s`, `attach_dep_artifacts` and
  metadata `role`. Previously a distilled judge stopped gating, so its
  target was never regenerated and `DELIVERABLE` synthesis returned the
  verdict instead of the approved output. Distilling a graph in which a node
  verifies a missing node or itself raises `DistillationError`.
- Jobs preflight rejects an artifact the profile cannot produce, so
  `jobs validate`, `jobs plan` and `jobs run` fail before a run is created.
  An `offline` profile emits 1x1 PNGs and now requires `image/png` with no
  dimensions or 1x1; another MIME type or size used to pass preflight and
  then fail every attempt. A `gemini_image` profile now requires a PNG,
  JPEG, GIF or WebP artifact; a manifest declaring, for example,
  `application/pdf` used to be approved and then had every paid image
  rejected.
- A durable workflow journal stays readable after a release adds a price
  version or changes the supported models. Every saved call was checked
  against the current price version, model list and request rules, so after
  such a change `lookup_call`, `inspect_run`, `audit` and recovery raised
  `WorkflowCorruptionError` for every existing run. A saved call is now
  verified by recomputing its stored binding, needs only a registered price
  version that prices its model, and is quoted and settled at that version's
  rates; its stored receipt is unchanged. Price tables are registered by
  version in `smythe.pricing.PRICE_TABLES` and
  `smythe.pricing_anthropic.PRICE_TABLES`, and the providers' model lists
  come from the current table. A new call still needs the current version.
  A retirement date in the current table stops new calls to that model and
  the sending of saved calls to it that were never sent; calls already sent
  still settle and replay.
- A durable run saved before a release that adds a price version or a
  supported model resumes. The saved recipe embeds each provider's price
  version and model list, and resume compared the recipe exactly, so it
  raised `WorkflowConflictError`; a call prepared before the release raised
  the same error when its node ran again. A saved provider description may
  now name any registered price version and a different model list, while
  every other recipe field still compares exactly. Saved calls replay and
  decode at the price version they recorded, including calls not yet sent
  to a model that has not retired, and new calls use the current version.
  Recipes keep their bytes.
- Saving a durable checkpoint no longer audits the whole journal. Each save
  called `inspect_run`, which verified every call, evidence row and earlier
  checkpoint and loaded every event, so checkpoint cost grew with the run's
  history and a long run's total grew quadratically or worse. To find the
  calls of regenerated nodes, a save now reads only the keys of settled
  calls, through the new `SQLiteWorkflowStore.superseded_node_calls`, and
  reads nothing until a node has been regenerated. It consumes the same
  calls as before, and each plan, execute or resume entry still audits the
  journal once when it starts.
- Jobs money amounts no longer depend on the caller's `decimal` context.
  `normalize_usd`, `usd_to_micros`, `micros_to_usd` and the USD strings in
  plans and approvals used the thread's context, so a lower precision, such
  as `decimal.getcontext().prec = 10`, rejected a valid amount like
  `9223372036854.775807` as unable to "fit exact microdollar precision" or
  raised `decimal.InvalidOperation`. Values and error messages under Python's
  default context are unchanged.
- A native Messages receipt's `cost_usd` and the `ceiling_usd` of an
  `AnthropicMessagesProvider.quote()` are exact whatever the caller's
  `decimal` context. A lower precision rounded them, for example a receipt to
  `0.0025350` instead of `0.002535000` and a quote to `0.43382` instead of
  `0.433820000`, so settling the same saved response in two contexts could
  produce receipts with different bytes. Receipts and quotes under the
  default context are unchanged.
- A durable run can complete after a call was reserved but never dispatched
  and the run did not send it again. Such a reservation, left by a process
  killed between reserving a call and claiming its dispatch, or by a retried
  attempt whose dispatch claim failed, was never released, so it held part
  of the allowance and saving the completed checkpoint raised
  `WorkflowStateError`. A resumed run whose regeneration moved that call's
  node to a later generation could therefore never finish. The run now
  releases such a call when a checkpoint records the regeneration that
  supersedes its generation, including on resume, and releases any other
  undispatched reservation when it completes. A reserved call of a node's
  current generation is still sent under its reservation when the node runs
  again, and a call whose dispatch was claimed is never released. Each
  release returns the reservation to the allowance and records a
  `call_released` event, which `release_before_dispatch` now records too.
  The new `SQLiteWorkflowStore.release_superseded_calls` and
  `release_reserved_calls` perform these releases.
- In a gated graph, the node a judge verifies is told it is the final step,
  not the judge. Terminal detection counted the judge's edge to that node, so
  in a draft-to-judge graph the judge's prompt said its output was the
  deliverable while `DELIVERABLE` synthesis returned the draft, whose prompt
  lacked the instruction. Prompts now follow the same rule as that synthesis.
  This changes the prompts of gated graphs on ordinary runs, so benchmark
  results for gated graphs from before and after this change are not directly
  comparable. Executors take a keyword-only `prompt_assembly`, where `1`
  restores the earlier placement. Durable text workflows keep the earlier
  placement, so runs in flight resume with the requests they saved; no public
  setting opts them into the new one yet, and a saved recipe whose
  `prompt_assembly` version this release does not support is refused. See
  [the final step](docs/verifier.md#the-final-step).
- `smythe.assets.validate_image` reads the file once and checks that snapshot.
  It used to hash the file for a finishing receipt and then open it again to
  decode it, so bytes replaced between the two reads could pass against a
  receipt they did not match.
- A node with `attach_dep_artifacts` reads each dependency image from its
  path when it calls, and used to send whatever the file held then, so an
  image replaced after it was recorded, for example while a run was paused,
  reached the model in place of the bytes the dependency wrote. An image
  whose bytes no longer match the `sha256` in its record now raises
  `smythe.executor_base.ArtifactIntegrityError` before the call. Records
  without a digest, written before digests were recorded, attach as before.
- Provider and service error text no longer keeps account identifiers where
  Smythe stores it. A failed or skipped node's result, which checkpoints save
  and dependents receive, its trace span error and log line, the native
  response error marker in node metadata, Jobs call and attempt errors, and
  Autotune trial errors now replace organization and project ids and keys
  in known provider formats with placeholders such as `org-[redacted]`, as
  journaled read and source-evidence reasons already did. Raised exceptions
  are unchanged. Durable decode rejections keep their text, which journals
  compare byte for byte and which names no account. See
  [failure classes](docs/execution.md#failure-classes).

## [0.9.1] - 2026-09-29

This patch release fails a Gemini image node that receives no image, moves
the Gemini image benchmarks and examples to `gemini-3.1-flash-image` before
`gemini-2.5-flash-image` shuts down on October 2, 2026, and keeps provider
account identifiers and local paths out of benchmark records. The Python API
is unchanged. See the [release guide](docs/release-0.9.1.md).

### Added

- Diagnostic benchmark records from September 26, 2026: second runs of the
  live transparent and SVG Noumenon lanes, a zero-latency Noumenon profile, a
  192-glyph Gemini concurrency sweep up to 128 with repeats at 32 and 64,
  Gemini smoke runs on both image models, and a repeat of the image
  concurrency sweep with three k=25 runs. The reports label them diagnostic.

### Changed

- The jobs scale campaign refuses an evidence folder inside the home folder,
  because its journal and records bind absolute paths, and replaces the home
  folder with `~` in worker logs before any record hashes them.
  `SMYTHE_ALLOW_PRIVATE_EVIDENCE_PATHS=1` lifts the check for records that
  will never be published.
- The Noumenon and image concurrency benchmarks replace the home folder with
  `~` in the records they write. `benchmarks/artifact_records.py` adds
  `redact_local_paths`, `scrub_record` and `evidence_directory` for new
  harnesses.
- Gemini image benchmarks and examples default to `gemini-3.1-flash-image`,
  recording Google's $0.067 per 1K image, because `gemini-2.5-flash-image`
  shuts down on October 2, 2026. Committed records keep the model they ran.

### Fixed

- `GeminiProvider` reports an image request that finishes without an image
  as `incomplete`, so the node fails and a `RETRY` policy can recover it,
  instead of completing with no artifact. Gemini can end such a request
  normally with only an empty text part.
- The Noumenon benchmark redacts account identifiers, such as the
  organization named in a provider's rate-limit error, from the records it
  writes. `redact_account_identifiers` in `benchmarks/artifact_records.py`
  covers OpenAI organization, project and key formats and Google API key and
  project formats.
- A repository test fails when any tracked file contains an identifier or key
  in one of those formats.
- Retained benchmark records and test fixtures no longer carry local user
  paths or identifier-shaped placeholder values. The asset-suite records use
  repository-relative artifact paths. The September 7 jobs-scale pilots and
  Windows detached-worker qualification replace local paths with
  placeholders, recompute the digests their archive and qualification
  records pin, and describe the change in their READMEs; the pilots' 25
  byte-identical accepted PNGs are kept as one file.

## [0.9.0] - 2026-09-26

This release narrows the repository to the framework and its evidence. The
glyph fan-out example is now Noumenon and was re-measured under its
unchanged protocol; the screensaver apps and Repo Doctor moved to their own
repositories; and retired benchmark records moved into an archive pinned by
SHA-256. The Python API changes only by addition. See the
[release guide](docs/release-0.9.0.md).

### Added

- `OpenAIImageProvider(background=...)` accepts `"auto"` (default),
  `"opaque"` or `"transparent"`. It is sent only when not `"auto"`, so
  existing requests are unchanged; a transparent background requires PNG or
  WebP output.
- Noumenon benchmark lanes for transparent PNGs and SVG conversion.
  `--background transparent` keeps the provider's alpha, asks for a
  transparent background in the prompt, gates every tile on an objective
  transparency check, and refuses models without transparent output before
  any call. `--vectorize` traces every accepted tile into an even-odd SVG in
  pure Python and Pillow and accepts it only when it rasterizes back to the
  source mask at an IoU of at least 0.98.
- `tools/evidence_archive.py` builds a deterministic archive of retired
  benchmark evidence from git history and verifies it against a committed
  pointer, every member's SHA-256 and, optionally, the recorded commit.
  [`benchmarks/archive/`](benchmarks/archive/README.md) pins the first archive:
  the first glyph fan-out campaign, the browser renderer timing study,
  superseded glyph partitions, review records and the withdrawn native package
  receipts, read from `v0.8.2`.

### Changed

- **The glyph fan-out example is now Noumenon.** The harness is
  `benchmarks/run_noumenon.py` (assets in `benchmarks/noumenon_assets.py`), its
  records are `benchmarks/results/noumenon_*.json`, its report is
  [`benchmarks/noumenon_benchmark.md`](benchmarks/noumenon_benchmark.md), and its
  charts are `assets/benchmarks/noumenon_scaling.svg` and `assets/noumenon/`.
- **The Noumenon benchmark was re-run.** The four-width realistic-latency sweep
  and the default 250 ms profile ran again from a clean checkout under the
  unchanged protocol. At 192 nodes, concurrency 64 now measures 52.32×
  (21.7 s against 1,133.3 s serially); the superseded August records,
  which measured 56.21×, are in the evidence archive. Live GPT Image lanes
  recorded 192 transparent PNGs and their SVG conversions.
- **The screensaver apps moved to the
  [Noumenon repository](https://github.com/petehottelet/noumenon)**: the web
  explorer, the Windows, macOS and Linux ports, the catalog exporters, their
  tests and the screensaver workflow. The glyph generator, its contours and the
  192-glyph catalog stay in Smythe under `benchmarks/noumenon/`, where the SVG
  workflow benchmark imports them.
- A live Noumenon run that halts, for example on a provider rate limit,
  records what its completed calls charged and marks the total incomplete,
  instead of recording no cost.
- Benchmark environment snapshots mark a checkout `dirty` only when tracked
  files differ from the recorded revision, and count untracked files
  separately in `untracked_files`.
- Released changelog sections 0.1.0 through 0.8.1 moved, unchanged, to
  [`CHANGELOG-ARCHIVE.md`](CHANGELOG-ARCHIVE.md).

### Removed

- **Repo Doctor moved to its own repository,
  [repodoctor](https://github.com/petehottelet/repodoctor).** The skill, its
  archive builder `tools/skill_archive.py`, its tests and its distribution
  workflow left Smythe, and Smythe releases no longer attach the skill ZIP.

## [0.8.2] - 2026-09-25

This correctness release closes gaps in the execution envelope: rejected
durable plans that stopped a run for good, verification gates a supervisor
could revise away, refused provider replies counted as output, and image
frames decoded before they were bounded. Some fixes tighten rules that 0.8.1
accepted; each is listed under **Changed**. The
[upgrade guide](docs/release-0.8.2.md) lists what to check before upgrading.

### Fixed

- **A durable run repairs a generated plan it cannot execute instead of
  stopping for good.** A schema-valid plan that set `attach_dep_artifacts`, or
  broke the run's `WorkflowGraphPolicy`, was rejected only after its planning
  call was journaled, so every resume replayed it and failed. `LLMArchitect`
  and `ConstrainedArchitect` now check each plan against the run's graph rules
  inside their retry loop and send the reason back to the model; each repair is
  a new journaled call. Resuming a run that 0.8.1 stopped this way asks for a
  repair.
- A durable plan with a node id the journal cannot key (over 256 characters, or
  with a control character) is repaired before planning is saved, instead of
  failing at the node's first call on every resume.
- In a durable run, a revision that adds a node under the id of a node with
  journaled calls is rejected and traced. It raised `WorkflowConflictError` on
  every resume, or replayed the old node's result when the request matched.
- **A supervisor revision can no longer drop a verification gate or cut it off
  from the node it judges.** A revision that dropped a verifier let the output
  it judged through unverified, and one that rewired the verifier off its target
  let a parallel judge finish first, so its FAIL verdict was discarded.
  `ExecutionGraph.apply_revision` now rejects a revision that drops a verifier
  or the node it judges, or that leaves the verifier no longer depending on
  that node, directly or through the nodes between them. The rule applies to
  every supervisor; a rejected revision is traced and the run continues. A
  revision can still add steps after a gated node, and the gate does not judge
  their output.
- **Generated plans may contain at most one gating node, which must list the
  node it verifies in `depends_on`.** Every node could be a gate: an 8-node plan
  with seven gates on one node made 78 node calls. A gate without that edge
  could run before its target and have its verdict discarded. Such plans are
  retried with the problem named.
- A gate inside a `ConstrainedArchitect` template judges its own renamed
  target. Composition prefixed node ids but not `verifies`, so the gate's
  verdict was always discarded.
- `LLMSupervisor` limits how many revision-added nodes a run's graph may hold
  with `max_total_added_nodes` (default 8, the generated-plan node limit).
  `max_added_nodes` bounded only one revision, so five revisions could append 15
  nodes. Nodes a revision adds carry `"added_by_revision": true` in their
  metadata, so resuming does not reset the count. An added node's label may be
  at most 500 characters.
- A node can no longer use the id `__synthesis__`, the key under which
  `LLM_MERGE` synthesis books its charge; a node with that id shared the
  synthesis entry in the budget breakdown.
- **Jobs artifact inspection bounds every animation frame, not only the
  first.** A GIF frame that extends past the logical screen grew the canvas
  during decoding: a 74-byte GIF was accepted at 10×10 after a 718 MB decode at
  9001×9001. The canvas every GIF frame needs is now checked from the bytes
  before Pillow parses them, and each frame, including each picture of a
  multi-picture JPEG, against the per-image and aggregate pixel limits before
  it is decoded. Pillow reads on past two GIF block terminators where the
  format ends a block, so a frame hidden after one escaped the check; a GIF
  with bytes after such a terminator is refused as undecodable. The GIF check
  also covers `validate_image`, `finish_image` and the design checks, where
  Pillow allocated an oversized first frame while opening the file.
- WebP canvases are checked before libwebp allocates them. A 40-byte lossy
  WebP, or a 26-byte lossless one, declaring 16383×16383 made libwebp allocate
  about 2 GB. This covers Jobs inspection, `finish_image`, `validate_image` and
  the design checks, which now open a path once, so the bytes checked are the
  bytes decoded.
- Jobs artifact inspection no longer changes the process's warning filters.
  Escalating `DecompressionBombWarning` inside `warnings.catch_warnings()` is
  not thread-safe, so overlapping inspections could leave the escalation
  installed for the whole process. Every undecodable image, including a GIF on
  which Pillow raises `EOFError`, `IndexError` or `struct.error`, is reported as
  `ArtifactInspectionError`.
- Distilled templates ignore a model-supplied param named `task` or `params`
  instead of raising `TypeError`, which cost a paid planner retry.
- **Refused and filtered replies no longer count as output.** With the
  Anthropic, OpenAI and Gemini SDK providers, an Anthropic `refusal`, an OpenAI
  `content_filter` finish, or any Gemini finish except `STOP`, `MAX_TOKENS` or an
  unspecified one (such as `SAFETY`, `RECITATION` or `MALFORMED_FUNCTION_CALL`)
  completed its node with any partial text, and a filtered OpenAI turn ran its
  tool calls. So did a Gemini response to a blocked prompt (no candidates and a
  `prompt_feedback.block_reason`) and an OpenAI message with `refusal` set.
  The node now fails with `OutputRefusedError` after its cost is recorded, no
  tool call from that turn runs, and its failure policy applies. A refused plan
  is retried with the stop reason named, a refused supervisor review applies no
  revision, and a refused `LLM_MERGE` synthesis fails the run.
- Twelve exceptions whose constructors take more than a message could not be
  unpickled or copied intact. Most, including `SentinelAlert`,
  `BudgetReconciliationError`, `ProviderRequestRejectedError` and
  `ResponseQuoteError`, raised `TypeError`; `WorkerStartupInterrupted` raised
  `ValueError`; `OutputTruncatedError` came back with its message wrapped twice
  and `AssetPreflightError` with its message split into characters. They now
  survive `pickle`, `copy.copy` and `copy.deepcopy` with their message and
  fields, including across process boundaries.

### Changed

- A generated node `timeout_s` must be at least 60 seconds
  (`smythe.loader.MODEL_PLAN_MIN_TIMEOUT_S`); a shorter one is retried. A timeout
  cancels calls already sent, so their spend bought nothing, and a durable run
  blocked on unknown billing. Developer-written Python and YAML graphs are
  unchanged.
- `ConstrainedArchitect` limits the composed graph to `max_nodes` (default 64),
  checked after each builder call; a selection over the limit is retried. A
  66-byte reply could compose 200,001 nodes. Builders receive model-chosen
  params and must bound them.
- In a durable run, a generated plan that the graph policy or the plain-text
  rule rejects now costs up to `max_retries` repair calls, and planning that
  still fails raises `ArchitectError` instead of `WorkflowBindingError`.
- In a durable run, a node id must be at most 256 characters with no control
  characters (`smythe.workflow.MAX_NODE_ID_CHARS`). A caller-built graph, or a
  resumed saved graph, with another id raises `WorkflowBindingError` before any
  node runs.
- Any supervisor revision, including one from a custom `Supervisor`, that drops
  a verification gate or its target, cuts the gate off from its target, or adds
  a node with the id `__synthesis__` is rejected and traced as
  `revision_rejected`.
- `ExecutionGraph.validate()` raises `ValueError` for the node id
  `__synthesis__` in any graph, including when resuming a checkpoint or durable
  run that contains one.
- Jobs reports an animated GIF whose frames extend past its logical screen at
  the size of its enlarged canvas, and counts every frame at that size toward
  the aggregate pixel limit.
- A GIF with bytes after a block terminator that Pillow would read past is
  refused by Jobs inspection, `validate_image`, `finish_image` and the design
  checks.
- `OpenAIProvider` and `GeminiProvider` report the stop reasons `refusal`
  (OpenAI only), `content_filter` and (Gemini only) `incomplete` where they
  reported `end_turn` or `tool_use`; see the [execution guide](docs/execution.md). Stop
  reasons outside `TRUNCATED_STOP_REASONS` and `REFUSED_STOP_REASONS`,
  including any a custom provider returns, still count as complete.
  `OutputTruncatedError` now subclasses `IncompleteOutputError`.

### Added

- `IncompleteOutputError` and `OutputRefusedError`, exported from `smythe`, and
  `smythe.provider.REFUSED_STOP_REASONS`.
- `LLMSupervisor(max_total_added_nodes=...)` and
  `ConstrainedArchitect(max_nodes=...)`.
- `smythe.loader.MODEL_PLAN_MAX_GATES` and `MODEL_PLAN_MIN_TIMEOUT_S`;
  `smythe.graph.SYNTHESIS_NODE_ID` and `REVISION_ADDED_KEY`;
  `smythe.workflow.MAX_NODE_ID_CHARS`.

[Unreleased]: https://github.com/petehottelet/smythe/compare/v0.9.1...HEAD
[0.9.1]: https://github.com/petehottelet/smythe/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/petehottelet/smythe/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/petehottelet/smythe/compare/v0.8.1...v0.8.2
