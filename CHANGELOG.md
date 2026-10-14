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
  `ConnectorAmbiguous` and `ApprovalRejected`; a verifier's
  `verify(assertion, subject_sha256, now_ns)` checks the digest an assertion
  approves; `check_connector()` and
  `check_verifier()` refuse providers, classes, coroutine functions and
  awaitables, and async protocol methods: `submit()`, `lookup()` and `read()`
  on a connector, whose `profile` must be a `ConnectorProfile`, and `verify()`
  and `still_authorized()` on a verifier, whose `identity` must be a name
  without control characters. Other methods are not checked. See
  [external actions](docs/actions.md).
- Version 2 workflow journals have four tables for external actions:
  `workflow_actions`, `workflow_action_attempts`, `workflow_action_approvals`
  and the append-only `workflow_action_records`.
  Action money is a separate `external_actions` accounting scope that never
  enters a run's balances. `SQLiteWorkflowStore.audit()` also checks a run's
  actions: intent digests, action keys, states, verdicts and record kinds,
  approvals bound to their action's intent, and balances equal to the sums
  of each action's attempts and records. `inspect_run()` adds an `actions`
  block only for a run that has actions, without parameters, approval
  assertions, revocation reasons, receipts, record payloads or error text. See
  [external action records](docs/workflow-accounting.md#external-action-records).
- Prepared external actions and host-verified approvals in `smythe.actions`.
  `prepare_action(store, run_id, *, slot, intent, profile)` records a
  logical action only for a completed durable run whose current delivery
  decision is `accepted`, including an operator's, and only for content with
  the exact SHA-256 and size of the accepted output, or of the published
  document, whose file must still hold it. Advisory and earlier runs have no
  recorded decision and are refused with `ActionNotAuthorized`. The same
  intent again returns the slot's action; another intent raises
  `WorkflowConflictError` while it is unfinished, and becomes the next
  sequence with its own key once it is observed, reconciled or canceled.
  `approve()` records an approval only when the host's verifier accepts the
  assertion for exactly the action's `PreparedAction.approval_sha256`, a
  versioned digest (`smythe.action-approval.v1`) of its action key and
  intent digest that approvers sign in place of the bare intent digest, the
  principal belongs to the intent's tenant, the nanoUSD ceiling covers the
  action's attempts and readback, and the expiry is later than the journal's
  clock; an assertion approves one action, never an identical intent in
  another journal, and is recorded once per journal.
  `reject()` cancels an undispatched action for good under a verified
  principal, `revoke()` is a fail-safe control that needs no principal and
  stores its reason redacted, and `check_authority()` verifies the newest
  approval again and checks expiry, revocation, consumption, the intent
  digest and that the run's content is still accepted, intact and not
  unavailable after a source removal. None needs a run lease. See
  [approvals](docs/actions.md#approvals).
- A durable dispatch journal for external actions in
  `smythe.workflow_actions`, fenced by the run's lease. `begin_attempt()`
  reserves the profile's write ceiling after checking the content as
  `check_authority()` does and, in the same transaction, that no security
  stop holds the run, the action is not dispatched, the approval is still its
  newest, unrevoked, unconsumed and unexpired, fewer than `max_attempts`
  attempts were dispatched and the spend fits the approval's ceiling.
  `claim_dispatch()` checks again and commits the attempt as dispatched,
  with the lease epoch, before anything is sent, returning an
  `ActionDispatchPermit`; an attempt no longer authorized is released.
  `record_receipt()` keeps the receipt with the external object's identity,
  confirms its cost or keeps the write ceiling as unknown exposure, and
  consumes the approval; `record_refused()` gives the reservation back and
  leaves the approval for the next attempt; `record_ambiguous()` keeps the
  write ceiling as unknown exposure. Connector error text is stored
  redacted. A permit from a superseded lease session, or for an attempt
  already settled, changes nothing and is kept as a `late_receipt` record.
  A late receipt that shows a commit of an attempt found not executed
  appends a `contradicted_absence` finding, returns an action that was
  `not_dispatched` again to `outcome_unknown`, releasing an unclaimed
  attempt, and keeps any empty lookup or operator decision from finding any
  attempt of that action not executed, so it is not dispatched again.
  `recover_actions()` releases prepared attempts and turns dispatched ones
  into `outcome_unknown` without resending or releasing exposure, and
  `reject()` releases an attempt prepared but not yet claimed. A run's
  model-call latches do not block actions. The audit also checks that at
  most one attempt is in flight and that a refused attempt never consumes an
  approval. See [the dispatch journal](docs/actions.md#dispatch-journal).
- `ActionRunner` in `smythe.actions` dispatches a run's prepared actions
  through a connector, serially and under the run's lease, as the only retry
  owner. `dispatch()` recovers what a stopped worker left in flight, checks
  the approval and begins an attempt, reads the payload once and refuses
  bytes that are not the bound content before the claim, then submits once
  with the action's key and records the receipt, refusal or ambiguous
  outcome; a receipt for another key or target is ambiguous. A receipt alone
  never makes an action observed: `readback()` reads the receipt's object at
  most `readback_reads` times per attempt, at the declared interval, and
  records each read with its cost and a `smythe.action-readback` validation
  report. Only the receipt's target holding the bound digest, the action's
  key where exposed and at least the receipt's version makes the action
  `observed`; a contradiction makes it `compensation_required`, and reads
  that decide nothing leave it dispatched with `reconcile` next. Nothing is
  resubmitted. `record_notice()` keeps a relayed service callback as an
  untrusted record that can only prompt a readback, once per payload digest
  and at most 64 per action; each further notice is only counted, by its
  digest, in an `action_notice_dropped` event. A connector whose profile
  differs from the action's, or that retries on its own, is refused. See
  [the action runner](docs/actions.md#action-runner).
- `ActionRunner.reconcile()` settles an external action whose outcome a
  dispatch left unknown, only as its connector profile declares. With
  `lookup_by_key` it looks the action's key up, at most `readback_reads`
  times per attempt, and reads back the receipt a lookup finds, making the
  action `reconciled` or `compensation_required`. An empty lookup never
  causes a submission, and shows that the write was not made only once a
  declared `visibility_bound_s` plus the connector's timeout has passed since
  the dispatch while the key is still inside its window; the action is then
  `not_dispatched` again under its unconsumed approval once no other
  attempt's outcome is unknown. That proof assumes
  each request leaves within the timeout of its claim: only a deduplicating
  service protects against a worker that stalls past its lease before its
  request leaves. A deduplicating
  service without lookups gets one same-key retry within `max_attempts`,
  only while `dedupe_window_s` from the first dispatch, less the timeout, is
  open and the approval and content still hold; a refused retry leaves the
  outcome unknown. Anything else, including a connector with neither
  declaration, records an `unresolved` finding with `operator_review` next
  and sends nothing. Lookups and reads, including the readback of an attempt
  dispatched after a `not_executed` finding, are charged against the ceiling
  of the attempt's approval and never made past it, and the uncertain write
  keeps its unknown exposure until a receipt reports its cost.
  `smythe.workflow_actions` gains `record_lookup()`, `claim_retry()` and
  `record_finding()`. See [reconciliation](docs/actions.md#reconciliation).
- Revoked authority takes precedence over external action dispatch.
  `stop_actions(store, run_id, reason)` in `smythe.actions` is a fail-safe
  control that needs no principal or lease: it revokes every approval of the
  run's actions, storing the reason redacted, and records an
  `actions_stopped` event. No resume, recovery, lease or reopened journal
  clears it; only a new approval reopens an action. A submission already
  claimed still records its outcome, and reads and lookups that settle it go
  on. `ActionRunner.dispatch_pending(run_id)` dispatches a run's waiting
  actions serially under one lease, checking each action's authority again
  before its claim and in the claim's transaction, so a revocation during a
  submission refuses every later claim; a refused action is returned with
  its `ActionProgress.refusal`. Action dispatch takes no drain policy.
  Revoking never changes an action's state, and `inspect_run()` shows each
  revoked approval's `revocation`: `before_dispatch`, `after_dispatch` or
  `after_effect`. See [revocation](docs/actions.md#revocation).
- `prepare_compensation(store, action_id, *, operation, parameters=None)` in
  `smythe.actions` records a compensation as a new logical action. For an
  action that is `observed`, `reconciled` or `compensation_required`, whose
  connector profile declares `compensation="new_action"`, it writes the
  operation and parameters, without content, to the object the action's
  receipt named and its readback read. `PreparedAction.compensates_action_id`
  names the action it undoes, and its slot, `<slot>/<sequence>/compensation`,
  gives it its own key. It needs its own approval, since the original's binds
  another action, and makes its own attempts with their own receipt
  and readback; the original's records never change. A connector without
  compensation refuses it with `ActionStateError`, and the action stays
  `compensation_required` with its unknown exposure. A security stop refuses
  it as it refuses every action. See
  [compensation](docs/actions.md#compensation).
- Operator decisions on external actions in `smythe.actions`.
  `resolve(store, action_id, decision, *, assertion, verifier, evidence=None,
  note)` records `adopt_observed`, `not_executed` or `abandon` on an action
  whose outcome reconciliation left open, only when the host's verifier
  accepts the assertion for exactly `decision_sha256()`, which binds the
  action, its state and attempts, the decision and the evidence, from a
  principal of the intent's tenant; a security stop refuses every decision.
  `adopt_observed` needs a readback that showed the bound content, or
  evidence, and makes the action `reconciled` without a submission.
  `not_executed` needs evidence and returns the action to `not_dispatched`
  with duplicate risk: its earlier approval no longer authorizes a dispatch,
  and only `approve(..., duplicate_risk=True)` with an assertion for
  `duplicate_risk_sha256()`, bound to the attempts the decision found, does.
  `abandon` closes the action in the new `unresolved` state. `ActionEvidence`
  is kept as a SHA-256 and a redacted summary of at most 500 characters,
  never its bytes. Decisions only append `decision` and `evidence` records:
  no attempt, approval or balance changes, so unknown exposure stays.
  `inspect_actions(store, run_id)` adds each action's duplicate risk,
  contradiction and next action and lists the unresolved actions, including
  one whose late receipt shows a `second_effect`, which needs compensation,
  and one not yet settled by a read after a `contradicted_absence`.
  `smythe runs inspect` reports them
  in an `actions` block and exits with 12 when the decision is `accepted` but
  an external action is unresolved; codes 10 and 11 take precedence.
  `smythe runs list` adds each run's `unresolved_actions` count and marks a
  run that has any in its table, with unchanged exit codes.
  `smythe runs revoke-action` is the lease-free `revoke()` fail-safe, and
  never upgrades a version 1 journal. See
  [operator decisions](docs/actions.md#operator-decisions) and
  [external actions in runs inspect](docs/runs.md#external-actions).
- [`examples/19_governed_publication.py`](examples/19_governed_publication.py)
  delivers a strict durable run's accepted document to an in-memory
  publishing service, offline and without keys. It prepares the action,
  approves it through a local HMAC approval host, loses the first
  submission's answer after the service commits it, asserts that
  `smythe runs inspect` exits with 12, reconciles the action from a lookup
  and a readback with one write on the service, and prints its inspection,
  which then exits with 0. `tests/test_examples_smoke.py` runs it. The
  [external actions guide](docs/actions.md) says where actions run, what a
  connector must declare and which guarantees the tests check, and the
  [support matrix](docs/support-matrix.md#external-actions) lists external
  actions as supported only on completed durable text workflows whose strict
  delivery acceptance is `accepted`, and unsupported on ordinary runs and
  Jobs, with the test behind each row. `tests/test_action_conformance.py`
  races two workers, in threads and across processes, to one dispatch claim
  and one write; relays receipts and notices out of order without moving an
  action back; and sweeps deduplication, lookup, visibility and key-window
  declarations, including no visibility bound, after a lost answer, a lost
  request or a request that reaches the service only as the connector's
  `timeout_s` ends, to show that no second write follows without proven
  absence or live deduplication.
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
- `OptimizationRunner` refuses a contract whose mutable fields include
  `acceptance`, `validators`, `verifier`, `evaluator`, `harness`, `budget`,
  `spend`, `ceilings`, `capabilities`, `tools`, `permissions`, `authority` or
  `holdout`, or a path under one, in any letter case, with an
  `OptimizationError` raised before it reads or writes the ledger. Contract
  parsing is unchanged, so stored contracts still load for inspection and
  holdout accounting, but a stored campaign that declares such a field no
  longer runs or resumes. See
  [experiment contracts and budgets](docs/optimize.md#experiment-contracts-and-budgets).
- `OptimizationRunner(single_change=True)` requires every challenger to
  differ from the incumbent in exactly one policy path and refuses a plan
  with a challenger that changes more before the campaign is created.
  References are not limited. The setting joins the plan hash only when set,
  so default plan hashes and automatic campaign IDs are unchanged, and an
  explicit campaign ID created with the other setting is refused as campaign
  binding drift. See
  [experiment contracts and budgets](docs/optimize.md#experiment-contracts-and-budgets).
- `smythe.optimize.PolicyRegistry` records which policy version is in effect
  for each dotted target, in its own SQLite file
  (`~/.smythe/policy-registry.sqlite3` by default, schema version 1) that
  never writes an Autotune ledger. Each target has an append-only history of
  `baseline`, `activate`, `rollback` and `retire` events numbered in order.
  `resolve()` returns the `ActiveVersion` in effect, or `None` after
  retirement; `history()` lists every `ActivationEvent`; `rollback()`
  restores an earlier baseline or activate entry as a new event; and
  `retire()` leaves no policy in effect. `record_review()` records an
  approving or rejecting `PolicyReview` of an Autotune campaign's terminal
  decision after validating the campaign's decision inventory, and never
  writes the ledger; a ledger path is opened read-only, so a version 3
  ledger is not migrated. The review names the campaign by ID, plan hash and
  holdout commitment and the decision by ID and payload SHA-256, copies the
  decision candidate's and the incumbent's policies, and records a declared,
  unauthenticated reviewer. Only a promoted decision can be approved, and
  rejecting reviews are kept. `activate()` puts an approving review's policy
  in effect when every policy path lies under the target: a target's first
  activation records the campaign incumbent's policy as its baseline, and a
  later one requires that incumbent to be in effect. Each change names the
  head it expects, and a stale `expected_seq` raises `RegistryConflictError`
  without writing, so of two writers on the same head exactly one succeeds.
  Triggers refuse every update and delete, an insert that would replace or
  remove a row (as `INSERT OR REPLACE` does, by rowid, review, event or any
  other unique key) or skip a target's next number, and inserts from another
  schema version; a writable open refuses a database holding other tables,
  and a read-only open requires a closed, checkpointed file. The ledger
  schema and `smythe optimize inspect --json` output are unchanged. See
  [policy registry](docs/optimize.md#policy-registry).
- Policy registry commands in the CLI. `smythe optimize review CAMPAIGN_ID
  --verdict approve|reject --reviewer NAME` records a review of a campaign's
  terminal decision, with `--evidence FILE` recorded only as the file's
  SHA-256 in `evidence_sha256`. `smythe optimize activate TARGET REVIEW_ID`,
  `rollback TARGET --to-seq SEQ` and `retire TARGET` each name the head they
  expect with `--expected-seq` and record an `--actor` and `--reason`; they
  print the target and each appended event's seq and policy, need an
  existing registry and never edit configuration. `smythe optimize history
  TARGET` opens the ledger and registry read-only and reports the target's
  activations and every campaign whose candidates set the target, with its
  decision and recorded three-way verdict, reviews and cost; confirmed,
  reserved and unknown micro-USD are summed exactly across those campaigns
  as optimization cost, never netted against inference savings, and unknown
  trials are counted so a zero-ceiling unknown outcome still shows.
  Campaigns an activation names that the ledger does not hold are listed.
  `collect_target_report(ledger, registry, target)` in
  `smythe.optimize.inspection` returns the same report, and
  `ExperimentLedger.list_campaigns()` lists every campaign read-only. Each
  command takes `--registry` (default `~/.smythe/policy-registry.sqlite3`)
  and `--json`; invalid input exits with code 2 before any file is opened,
  and a refused review or change or a stale `--expected-seq` exits with
  code 8, records nothing and creates no registry. The ledger schema,
  campaign report version 1 and `smythe optimize inspect --json` output are
  unchanged. See
  [review and change targets from the CLI](docs/optimize.md#review-and-change-targets-from-the-cli)
  and [cross-campaign target report](docs/optimize.md#cross-campaign-target-report).
- `smythe.optimize.workflow_eval.workflow_trial_evaluator(runtime_factory,
  task_for, measure)` runs each Autotune trial as one durable text workflow.
  The run's budget is the trial's ceiling, its id derives from the trial key
  (`trial_run_id()`), and the trial's cost is the journal's confirmed nanoUSD
  rounded up to whole micro-USD. A run that leaves reserved or unknown spend,
  or an unknown call, or that fails, already exists or cannot be measured,
  raises `WorkflowTrialError` with fixed text, so the runner records the
  trial as unknown and stops without storing provider text; the runner
  records an outcome that breaks its contract as an invalid outcome instead.
  A revoked `run_control` is refused before the run is created, while a plan
  that only the run refuses, such as one no gate could accept under a strict
  acceptance policy, leaves its run in the journal with nothing spent.
  Nothing is retried, and a trial's run is never run again. See
  [durable workflow trials](docs/optimize.md#durable-workflow-trials).
- Bounded exploration. `smythe.optimize.explore(runner, registry, proposer,
  incumbent, budget)` runs one `OptimizationRunner` campaign per round on a
  proposer program's challengers against a fixed incumbent, and never reviews
  or activates anything. `ExplorationBudget` stops it at the first promotion,
  after `max_non_promotions` consecutive campaigns that did not promote, at
  `max_campaigns`, or before a campaign whose full required reservation would
  bring the exploration's campaign exposure over
  `max_total_exposure_microusd` or a proposer call whose ceiling would pass
  `max_proposer_microusd`; `ExplorationResult` names the stop.
  `SubprocessProposer(argv, ceiling_microusd=..., timeout_s=...)` exchanges
  one JSON object each way over standard input and output, and kills the
  program at its deadline. Of the caller's environment the program inherits
  only what a process needs to start, such as `PATH`, `SYSTEMROOT`, `TEMP`,
  `HOME`, `LANG` and a Python interpreter's `PYTHONPATH`, so no credential
  reaches it whatever its name, plus the variables `env=` passes; it never
  receives the ledger or its path. It sees
  `proposer_view(ledger, campaign_ids)`: contracts, candidates, development
  and confirmation evidence, and only whether the holdout passed. Proposals
  that set a reserved path, leave the contract's allowlist or rules, break
  `single_change` or a cap, or repeat a campaign or a used holdout are
  refused before the ledger is written. Each proposer charge is reserved at
  the ceiling before the program starts, settled as it declares, or left
  unknown at the ceiling when it fails or is killed, in new append-only
  exploration tables of the policy registry's schema version 1, which
  `PolicyRegistry.exploration()` reports. No model-backed proposer ships. See
  [bounded exploration](docs/optimize.md#bounded-exploration).
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
  `transient`, `unknown_outcome`, `terminal_accounting`, `persistence`,
  `workflow` or `security_stop`; `validator_error` is named, but no
  exception has it. Each failed attempt's trace span records its class as
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
  decisions back. Each decision a policy makes, the run's own included,
  binds the SHA-256 of every gate's status and final receipt
  (`gate_receipts_sha256`): everything its verdict is read from, including
  the target and judge generations it is bound to. A gate verdict or binding
  changed in the journal afterwards, or a decision whose inputs no longer
  match its key, is refused by re-evaluation, with nothing written, and gives
  `indeterminate` on resume and inspection, each with its own reason. Resume
  returns the current decision and never makes a new one: a strict `Swarm`
  resuming a completed run without a record, or one decided under another
  policy, behaves as before, and a durable run still resumes only under the
  policy its recipe records. `smythe runs inspect` adds a `decisions` block
  with the current decision's sequence and the decisions before it, and its
  exit codes follow the current decision. See
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
  decision (a lineage run's is rendered and decided again); re-evaluation
  needs an intact document, so it is refused until then. See
  [published documents](docs/workflow-accounting.md#published-documents).
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
  `max_retries`, passed to the SDK only when set. Ordinary checkpoints record
  the policy only with `persist_attempts`; durable runs record it in their
  recipe, as described below. Executors also take `sleep` and `rng`, which
  replace the retry wait and its jitter. Without a policy, retries are
  unchanged. See
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
- `RetryPolicy(retain_uncertain_dispatch=True)` keeps the exposure of an
  ordinary run's node call that was sent and never answered. When the node's
  timeout, the policy's deadline or a cancellation, such as a `HALT`
  sibling's failure or a run stopping on an interrupt, ends a call from
  `AnthropicProvider`, `OpenAIProvider`, `OpenAIImageProvider`,
  `GeminiProvider` or a custom provider before it answered, its reservation
  becomes the node's recorded cost, flagged `cost_usd_unknown` and
  `cost_usd_is_estimate`, instead of being released, so the run reports
  `cost_is_complete=False` and version 4 checkpoints and resume keep the
  amount and flags. `Sentinel.retain_unknown()` makes the transfer, also for
  a verifier's reservation converted from a hold. Such a timeout raises the
  new `UncertainDispatchError`, a `TimeoutError` of class `unknown_outcome`,
  which the policy retries only when it lists that class; a retry reserves
  again beside the kept exposure. A failure before the call is sent, or an
  error the provider raises itself, still releases the reservation, native
  providers keep their own settlement, and without the option nothing
  changes. See [calls without an answer](docs/execution.md#calls-without-an-answer).
- Durable runs take a `RetryPolicy`. `Swarm(run_store=..., retry_policy=...)`
  retries a node only after a failure whose class the policy lists, so a 429
  rejection is retried and a 401, 403, 404 or 413 is not, and a refused or
  truncated reply, whose known charge the journal keeps, only when it lists
  `quality_defect`. Each node starts from the attempts its current generation
  has used, counted from the journal's rejected calls on every invocation, so
  a crash or resume continues its `1 + max_retries` attempts instead of
  refilling them, and a node with none left fails without a call. A node
  whose latest call is a refused, truncated or unusable reply enters that
  attempt again, so the journal replays the reply without a new call and the
  policy decides as it did before the resume. The recipe records the policy
  under `retry_policy`, without `persist_attempts`, so a run resumes under
  the policy it started with: a `Swarm` without one uses the saved policy,
  and a different policy, or one for a run that started without, raises
  `WorkflowConflictError` before any call. Recipes without a policy keep
  their bytes and their retries. The journal keeps no response headers, so a
  durable run waits its backoff, never a `Retry-After`; every provider it
  can call must declare `sdk_retries=0`, and a policy with
  `retain_uncertain_dispatch` is refused, since the journal keeps every sent
  call's exposure itself.
  `SQLiteWorkflowStore.node_generation_calls()` lists a node generation's
  settled calls, and executors take `attempts_spent=`, which journal-managed
  calls need under a policy. See
  [durable retry policy](docs/workflow-accounting.md#retry-policy).
- `Swarm(termination=TerminationPolicy(drain_timeout_s=...))`, or
  `termination=` on `AsyncExecutor`, drains the nodes in flight when a
  failure stops a run or the budget refuses a ready node at admission: no
  queued node starts and no retry is sent, each node in flight runs its
  current attempt on, tool turns included, for up to `drain_timeout_s`, and
  each node that finishes keeps and checkpoints its result and cost, without
  supervision or regeneration. A node whose attempt fails during the drain,
  or that is waiting to retry, returns to pending with its unused
  reservation released, so a resume runs it again. Those still in flight
  are then cancelled as before and the
  original error is raised. A cancelled call keeps its exposure as without
  a policy: a native provider's unresolved bill stays terminal, and another
  provider's call keeps its reservation as unknown cost only under
  `RetryPolicy(retain_uncertain_dispatch=True)`. A cancellation, an
  interrupt and a failure of class `security_stop` never drain, and the
  serial executor has nothing in flight to drain.
  `AsyncExecutor.drain_record` and a failed
  checkpoint's `control.drain` name the nodes the drain settled or
  cancelled and whether its deadline ended it; the record is informational,
  so checkpoints stay version 4. The time spent is the new `drain` wait in
  `SwarmResult.timing` and `Tracer.timing()`. Without a policy, a failure
  still cancels the nodes in flight at once. See
  [draining in-flight work](docs/execution.md#draining-in-flight-work).
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
  sources, a read step whose response is the removed text is settled as a
  rejected read with the removal as its reason and never delivers it, and a
  completed run's earlier decision becomes `indeterminate` until
  `reevaluate` decides again under a source evidence validator that holds
  the tombstone; any other policy's decision on output delivered from the
  removed source is `indeterminate`, and an operator can reject that output
  but accepts it only through a `Swarm` whose policy holds the tombstone,
  or `reevaluate` raises `WorkflowStateError` and writes nothing. A decision
  binds the keys of the tombstones the run held when it was made, so a
  re-evaluation stopped before a removal is decided again rather than
  finished. `smythe.evidence.load_source_tombstones` reads a run's
  tombstones. Earlier journal records and published documents keep what they
  hold. See [remove a source](docs/tasks.md#remove-a-source).
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
  runs a durable research workflow offline and without keys. It plans a
  journaled read step for each of three bundled sources, reads them from a
  server it starts on the loopback address, stops part way as a lost process
  would, resumes without reading a saved source again, publishes a brief
  carrying its source evidence, accepts it under a strict policy with
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
  unchanged. `smythe runs inspect` reports the decision as a `routing` block,
  with the name of the rule that chose the tier and the classifier's usage or
  journal calls, and as a `Routing:` line, for ordinary and durable runs and
  never with task text. See
  [routing decisions](docs/architecture.md#routing-decisions) and
  [routing in runs inspect](docs/runs.md#routing-decisions).
- Declarative routing rules and a single-worker tier.
  `WhiteRabbit(single=..., rules=[...])` takes a single-worker architect, such
  as `SimpleArchitect`, which the classifier is offered as `single`, and
  `RoutingRule` values evaluated in order on the submitted Task before any
  classifier call: the first that matches chooses its tier, and the decision
  records the method `rule` with the rule's name and reason. Conditions read
  only the goal (`goal_any`, `goal_none`), the number of constraints and
  acceptance criteria (`max_constraints`, `max_done_when`), and context keys
  and sizes (`context_keys`, `min_items`), never model output; Task data
  cannot add or change a rule. A rule naming a tier the router does not
  configure raises `ValueError` when the router is built, and without a
  matching rule routing is unchanged. Such a router records routing policy
  version 2. A durable recipe adds `single`, `rules` and
  `routing_policy_version` only when they are configured, so a router without
  them keeps its classifier prompt and recipe bytes; resuming with other rules
  raises `WorkflowConflictError`, and a planned run replays its saved
  decision. See [routing rules](docs/architecture.md#routing-rules).
- `LineagePolicy` opts a durable run into artifact lineage:
  `Swarm(run_store=..., acceptance=<strict policy>, lineage=LineagePolicy())`,
  `Swarm.from_yaml(..., lineage=...)` and `WorkflowRuntime` take one. It
  declares an `access_scope` label (declared, not authenticated, and compared
  exactly), the `reuse` switch and what `undeclared_coverage` calls for
  (`review` or `regenerate`). Without a `run_store` or a strict policy it is
  refused before any routing, planning or node call. The recipe records it
  under `lineage` only when configured, so other recipes keep their bytes.
  Plan, execute and resume compare it exactly, and re-evaluation, which
  reuses nothing, in all but `reuse`: a run resumed under another policy, or
  without its own, raises `WorkflowConflictError`. A new durable graph cannot
  carry `lineage_reuse` node metadata. `ArtifactLineage` is the versioned
  record of a run's accepted artifact, delivery decision and each node's
  inputs as ids, versions and digests, with no source text; its reader
  rejects unknown versions and keys, blank ids, a node that lists a source
  reference twice, in any key order, and a node listed before a node it
  depends on, which also rules out dependency cycles. Records hash by their
  digest. A run with a policy writes its record once when it completes, as
  a `lineage` journal operation that the completed checkpoint applies with
  its decision: `smythe.lineage.load_lineage()` reads it, and `inspect_run()`
  and `list_runs()` never return it. Each read step names its read profile
  and the source version and digest it read, each model node or gate the
  context snapshot and sources it was given and its agent's digest, each
  gate the gate verifier and gate validators that judged it, and the
  synthesis its synthesizer; the record also binds the published bytes'
  digest and the decision. A node whose inputs are not all known, such as a
  model node without a context snapshot, is `undeclared` and makes the
  coverage `partial`. A process lost before the record commits writes it on
  resume without a call, one lost after it replays it, and resuming a
  completed run whose record no longer matches its journal gives
  `indeterminate`. Lineage runs refuse, before any call, node and source ids
  their record cannot hold: blank ids and ids with DEL. See
  [artifact lineage](docs/lineage.md), and the
  [support matrix](docs/support-matrix.md#lineage-and-repair) for what each
  execution path supports.
- Reuse decisions for artifact lineage. `smythe.reuse.evaluate(record,
  state)` compares an `ArtifactLineage` record, or None, with a
  `CurrentState` (a developer's zero-cost `SourceCatalog`, the validator
  identities, the acceptance policy digest, the gate verifier and gate
  validators, tool, component, renderer, context assembly and agent
  identities, optional task changes, published-document integrity and the
  `LineagePolicy`) and returns an `ItemDecision`: `reuse`,
  `recheck`, `rerender`, `regenerate` or `review` for the item and each
  node, with stable reason codes and the scope, recipe, policy, validator
  set and source versions any kept result is kept under. A source counts as
  current only when its id, version and digest all match, so equal bytes
  under a new version regenerate; an unavailable or withdrawn source needs
  review; a changed tool, decoder, component, context assembly, agent or
  task regenerates the node and everything that depends on it, while other
  branches are reused; a missing or changed document with current inputs is
  rendered again, but a renderer that changed, was added to a run that
  published no document, or was removed regenerates the synthesis in a
  successor that keeps every other node (`renderer_changed`); a changed
  validator set or policy calls for a
  recheck; a changed gate verifier or gate validator regenerates the gate and the
  nodes it judged (`gate_validator_changed`), never rechecking their
  recorded verdicts, and under a gate validator a gate recorded without its
  verifier follows `undeclared_coverage` (`gate_validator_unknown`); a
  decision other than `accepted` needs review; and a missing record or
  undeclared coverage follows `undeclared_coverage`. Gates and the nodes
  they judge are never reused in an item that computes anything again,
  `reuse=False` turns reuse and rendering again into regeneration, and a
  record in another access scope is invisible before any digest is
  compared. A state part left at its default is compared as it is, so a
  state without a renderer regenerates every published document's
  synthesis; `smythe.lineage.renderer_identity()` names a configured
  renderer as a record does. A catalog that declares itself metered is
  refused. The evaluator reads no store or file and its output is
  deterministic JSON data. See
  [reuse decisions](docs/lineage.md#reuse-decisions).
- Read-only repair plans for artifact lineage.
  `Swarm.plan_repair(execution_ids=None, *, sources)` evaluates each run in
  the Swarm's access scope, by default the heads of the archive, against
  what a new run would use now and returns a `RepairPlan`: each run's
  decision, derived status and predicted work (nodes reused and regenerated
  by kind, read calls, model steps, publications, acceptance evaluations,
  new runs, whether charges may be incurred and whether an allowance is
  required), the catalog's answers, and a timestamp-free `plan_sha256` that
  a later apply must match. Planning opens the journal again read-only,
  takes no lease, writes nothing, calls no provider, reads no source and
  describes each source once; a run that has not completed needs `review`
  (`run_in_progress`, or `run_failed` for a run that failed or is blocked),
  and a run in another scope is not found.
  `archive_heads(store, scope)` follows predecessor links: a completed
  successor whose decision is accepted supersedes its predecessor, and one
  whose decision is not supersedes nothing and is not a head.
  `smythe.reuse.derived_status()` computes the lifecycle status `current`,
  `stale`, `invalid`, `restricted` or `superseded`, which is never stored.
  `smythe runs inspect` adds a `lineage` block to durable lineage runs, in
  JSON and text: coverage, source references, predecessor and successors,
  the published document's integrity and the derived status, with no
  source text and no new exit code. See
  [plan a repair](docs/lineage.md#plan-a-repair) and
  [lineage in runs inspect](docs/runs.md#lineage).
- In-place repair of completed lineage runs. `Swarm.republish()` on a run
  with a lineage policy whose published document is missing or changed
  renders the saved synthesis again, publishes it as a new file named for
  its publication (`document-p1.md`) without replacing any file, and makes a
  new decision on those bytes under the Swarm's policy, bound to that
  publication, so identical bytes are decided again rather than copied. It
  needs the saved synthesis to have the digest its lineage records and is
  refused for an intact document, a changed renderer or synthesizer, and a
  name already holding other bytes. Every new decision on a lineage run,
  from republication, `Swarm.reevaluate()` or an operator, journals a
  lineage record keyed `lineage/<sequence>` that names the decision, the
  current publication and the deciding validators and supersedes the
  earlier record, applied in the same checkpoint, so a later repair plan
  finds nothing left to recheck or rerender. A policy's new decision on a
  lineage run with gates reads their recorded verdicts only under the gate
  verifier that judged them: re-evaluation and republication under another
  raise `WorkflowStateError` and write nothing. No node runs, nothing is read
  or called, earlier publications, decisions and records stay in the
  journal, and a process lost before the checkpoint is finished by the same
  call again. `smythe runs inspect` lists superseded publications under
  `artifact.superseded`. See
  [repair a run in place](docs/lineage.md#repair-a-run-in-place).
- Reuse operations for durable successor runs. A successor of a completed
  lineage run, which applying a repair plan starts, runs the predecessor's
  task and graph again as a new run, but each node it keeps completes first
  from a `reuse` journal operation, keyed `reuse/<node id>`, with the
  predecessor's text byte for byte, so one changed source runs only its
  read, its extraction and what depends on them, and the rerun nodes get the
  prompts a full recomputation would. Each
  operation names the predecessor's store, run and lineage digest, the
  node's generation and result digest, the digests of its lineage record and
  source evidence, its reason codes and the repair plan's digest; it carries
  no calls and no cost, and one checkpoint consumes it. Only a completed run
  in the same store and access scope, whose lineage record has the named
  digest, is reused from; a node whose saved result no longer has its
  recorded digest, whose inputs are undeclared or used a removed source, a
  gate and the nodes it judges or depends on, a node depending on one that
  runs again, and a graph the successor's graph policy, read profiles,
  models or agents cannot run are refused before the successor is created
  or any call is made. A resumed successor replays its reuse operations
  without reading the predecessor's results. Refreshing an unfinished successor's
  sources, even one lost before it saved a kept node, runs its kept nodes
  again, and a kept result naming a source removed from the successor is
  never used. Its lineage names its `predecessor` and `repair`, each kept
  node keeps its recorded inputs with `reused_from`, and inspection never
  returns kept text. A new durable graph cannot carry a `repair` planning
  record. See [reused results](docs/workflow-accounting.md#reused-results).
- Applying a repair plan. `Swarm.repair(plan, *, plan_sha256,
  max_total_budget_usd, reason, sources)` (and `arepair()`) makes the plan
  again for the same runs and refuses it with `WorkflowConflictError`
  unless it still has the approved digest. It needs a durable `run_store`
  and a lineage policy, refuses a configured supervisor, and admits the new
  runs only when their number times `max_budget_usd` fits
  `max_total_budget_usd`, refusing an unbounded per-run budget when any may
  incur charges (`WorkflowBudgetError`); every refusal comes before any run
  is created or call made. Items are applied in plan order, each once:
  `recheck` decides the run again and `rerender` publishes its document
  again, both binding the plan's digest and reason, `regenerate` starts a
  successor run that keeps the nodes the plan reuses and leaves the
  repaired run's rows untouched, and `review` is reported. The first item
  that raises or is not accepted stops the repair without a retry. A
  `RepairResult` lists each item's outcome, decision, successor and work
  (new runs, read and model calls, reused nodes, publications and
  decisions). A successor that has not completed is a repair in progress:
  it is no archive head, and a plan gives the run it repairs `review`
  (`repair_in_progress`), naming it in `repairs_in_progress`, until
  ordinary resume finishes it. Under `LineagePolicy(reuse=False)` a
  regenerated item keeps no result and runs every node, and its document is
  byte for byte a new run's, while a `recheck` item is still decided again.
  See [apply a repair plan](docs/lineage.md#apply-a-repair-plan).
- Source restriction for artifact lineage.
  `smythe.lineage.restrict_source(store, source_id, *, version=None,
  scope=None, reason, purge=False)` withdraws a source's text from every
  completed lineage run whose record names it, and from every successor in
  progress whose saved graph or reuse operations name a result of its
  predecessor derived from it, optionally only one version or one access
  scope, compared before any digest. Completed runs come first. Each run
  journals the same `source_tombstone` record that `remove_source` writes
  (identity, digest, reason and the journal clock's time, no text) under its
  lease, and a copy of its latest checkpoint applies it. `purge=True` deletes
  each file the run published and records its path, SHA-256, reason and
  time as an `artifact_purge` operation (`purged_artifacts()` reads them);
  tombstones and purge records always remain. A run restricted this way,
  or by a source removed from a lineage run
  (`smythe.lineage.restricted_sources()`), never delivers its output again:
  completed resume, inspection and document eligibility give
  `indeterminate`, external actions refuse it as unavailable, and
  `reevaluate`, `republish` and reuse in a successor refuse it. A repair
  plan needs `review` for it while the catalog describes the withdrawn
  text and allows `regenerate` from another version; its derived status
  stays `restricted`. A successor holds its predecessor's tombstones, so it
  is refused a withdrawn text as a context source, and a read step that
  reads one again is refused it; it takes those journaled since when it
  resumes and before it decides. After a restriction made while it is in
  progress, resuming it refuses a reused result the text reached, whether
  or not a checkpoint saved it, before any node runs, and a successor
  running meanwhile completes `indeterminate`. `smythe runs inspect` lists a
  lineage run's tombstones and purged files. Text copied into journal
  checkpoints and provider responses is not scrubbed. See
  [restrict a source](docs/lineage.md#restrict-a-source).
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
- `jobs reroll RUN_ID OPERATION --repair-integrity --reason TEXT` regenerates
  a succeeded operation whose accepted file was deleted, truncated or
  overwritten. Under the run's lease, the runner hashes every file of the
  accepted attempt again and queues the operation only when one is `missing`
  or `changed` and the rest verify, so a verified success still cannot be
  rerolled. Verified bytes, an unreadable or unsafe file and an exhausted
  attempt allowance refuse the repair with exit code 6, and a budget overrun
  with exit code 5. The run's artifact directory is claimed first, so one that
  cannot be claimed also refuses it; a refused repair queues nothing. The
  repair uses the run's original approval and attempt allowance. An
  `integrity_repair_queued` event records the reason and each file's finding,
  and the new attempt, linked to the one it replaces, records the reason. The
  operation keeps naming its accepted attempt until a new attempt is accepted,
  even when the repair attempt is rejected; artifact rows and their `accepted`
  flags never change, and `resume` and `reroll` hash only each operation's
  current accepted attempt again. In Python, `JobRunner.reroll(...,
  repair_integrity=True)` records the findings with
  `SQLiteRunStore.queue_integrity_repair()`; the Jobs schema stays at
  version 4. See
  [repair corrupted accepted outputs](docs/jobs.md#repair-corrupted-accepted-outputs).
- `jobs recheck RUN_ID OPERATION... --reason TEXT` decides succeeded
  operations' accepted files again under the current checks, without a
  provider call. Delivery eligibility uses the latest decision made under
  the validators an operation requires now, the call's own or a
  recheck's, so a registered validator's new version or configuration, or a
  newer built-in artifact check, reports `validator_changed`: a recheck is
  required. `JobRunner.recheck()` runs this release's built-in check and the
  validators registered under the names the plan gives, at any version, so
  it works where `resume` and `reroll` are refused. Under the run's lease it
  hashes every accepted file again, reads each operation's files once into
  memory, and records a `delivery_acceptance_reevaluated` event with the
  reason, outcome, each validator's result, the findings and each file's
  SHA-256 through `SQLiteRunStore.record_reevaluation()`. Statuses, accepted
  attempts and `accepted` flags never change, so a rejecting recheck leaves
  the operation succeeded but not eligible. A missing or changed file,
  another state or another worker's lease refuses it with exit code 6 and
  nothing recorded. `jobs inspect` lists each attempt's rechecks as
  `delivery_reevaluations` and the HTML report shows them;
  `inspect_job(..., validators=[...])` judges eligibility under the
  validators registered now. The Jobs schema stays at version 4. See
  [recheck accepted outputs](docs/jobs.md#recheck-accepted-outputs).
- `jobs reroll RUN_ID [OPERATION...] --dry-run` previews what each named
  operation, or every operation, calls for, before anything is dispatched.
  It opens the Jobs database read-only, takes no lease, writes nothing,
  calls no provider and exits 0. Each operation gets `reuse`, `recheck`,
  `regenerate` or `review` with sorted reason codes from the reuse
  decisions' vocabulary, `smythe.reuse.REASONS`, whose `outcome_unknown`
  (`review`) marks an operation whose provider call may have completed; its
  remaining attempts; and `new_plan_required` when its regeneration needs a
  new plan and approval: it sends an attachment whose bytes changed since
  the plan, which the preview fingerprints again, has no attempt left, or
  the run's budget overran. The document also gives each attachment's
  status, the run's remaining budget and a count per action; a `regenerate`
  without `new_plan_required` for a missing or changed accepted file is what
  an integrity repair admits. Nothing is repaired automatically, and no
  other run's artifacts are offered, even for the same input digest.
  Without `--dry-run`, `jobs reroll` still needs operations and `--reason`;
  its operation keys may come before, between or after its options. In
  Python, `JobRunner.repair_preview(run_id, operation_keys=None)` judges
  under that runner's validators. See
  [preview repairs](docs/jobs.md#preview-repairs).
- Contribution lint. `lint_contributions(graph, registry=None)` lists the
  groups of parallel siblings that could only repeat each other's work: the
  same dependency inputs, label, agent, model, required capabilities,
  verification target, read step and image inputs. With a registry, agents
  compare by persona, capabilities and MCP servers. A `Swarm` with a
  `WhiteRabbit` router, ordinary or durable, records the groups of each new
  plan as `graph.planning["coordination"]`; checkpoints and durable replay
  keep the record. The lint is advisory and never changes a plan, and a plan
  it does not flag records nothing. The generated-plan schema is unchanged.
  [`examples/17_contribution_templates.py`](examples/17_contribution_templates.py)
  defines `SubGraphTemplate` values whose workers each contribute distinct
  evidence, capabilities, responsibilities or checkable outputs, and runs
  offline. See [contribution lint](docs/architecture.md#contribution-lint).
- Opt-in claim exchange between workers. A node in a graph you write that
  sets `metadata["exchange"] = "claims"` is given, for each dependency whose
  output carries a `source-evidence` block with a claim, a compact record in
  place of that output: every claim with its supporting and contradicting
  passages, scope, unresolved questions and conclusion, and each source's id,
  version and digest without its text. Other dependencies, and the prompts
  of nodes that do not opt in, are unchanged.
  `smythe.evidence.claim_exchange(output)` makes the record. A durable run's
  journaled request holds the compact records, so a resumed call is given
  the same bytes; its context snapshot records their digests, and artifact
  lineage declares the node's inputs. Another `exchange` value is refused
  before any node call, and a generated plan cannot set the key. See
  [claim exchange](docs/architecture.md#claim-exchange).
- A supervisor revision records what it changed. Its trace span's `revision`
  record adds the completed results the run `retained`, the delivered results
  it `invalidated` because a new step now follows them, the
  `preserved_gates`, `revisions_remaining_after`, the revision-added nodes in
  the graph (`added_total`) and the review's `trigger`; the earlier keys and
  `summary()`'s keys are unchanged. Ordinary checkpoints keep the latest 32
  decisions, applied or refused, in `control.revision_log`, which resume
  extends: node ids only, at most 16 per list, and the supervisor's reason
  and a refusal's detail redacted and cut to 200 characters. The log is
  informational, so checkpoints stay version 4 and a run without decisions
  saves no key. A durable run's journal is unchanged; its log is derived
  when read from its `supervision` operations
  (`smythe.workflow.revision_log()`). `smythe runs inspect` reports either
  as `revision_log`. See [revision log](docs/supervisor.md#revision-log).
- Model profiles route an ordinary run's nodes to developer-approved
  providers and models. `Swarm(profiles=...)`, `Swarm.from_yaml()` and both
  executors take `ModelProfile(name, provider, model, max_cost_per_call_usd,
  requires=...)`. A node whose `metadata["profile"]` names one calls that
  provider with that model and reserves the profile's ceiling, unless it
  sets `estimated_cost_usd`; held checks and persisted attempts use the same
  ceiling. Developer graphs, YAML, architects, templates and the nodes a
  developer `Supervisor` adds can name a profile; a generated plan and an
  `LLMSupervisor` proposal cannot. Other nodes and synthesis call the run's
  provider. Before routing, planning or any node call, a run refuses with
  `NoAllowedProfileError` (a `ValueError`, failure class `input_error`) a
  profile whose provider does not declare `True` each feature it `requires`
  (`tools`, `attachments`, `image_output`, `exact_accounting`), does not
  support its model or prices the model as retired, or cannot call tools in
  a run with a `tool_runtime`, and a node that names a profile the run does
  not allow or sets another model than its profile's; a revision that adds
  such a node is refused before it applies. While revisions remain, a
  model-less `LLMSupervisor`, which reviews with the reviewed node's model,
  is refused the same way when a profile calls another provider than its
  own. Under a `RetryPolicy` each profile's provider must declare
  `sdk_retries=0`. `ProviderFeatures.exact_accounting` is new: the native
  providers declare `True`, the other built-in providers `False`.
  Checkpoints of a run with a profiled node are version 5 and list
  `model_profiles` in `requires`, so older readers refuse them instead of
  calling the run's provider, and resume routes the nodes to the resuming
  Swarm's profiles of the same names. Durable runs refuse profiles. See
  [model profiles](docs/execution.md#model-profiles).
- `Swarm(capacity_policy=CapacityPolicy(...))` limits the calls each provider
  type and model takes on an ordinary run, and both executors take
  `capacity_policy=`. `CapacityPolicy.limits` maps a model, or a
  `(provider type name, model)` pair, to a `CapacityLimit` with
  `max_in_flight` and `min_interval_s`; a field it leaves unset comes from the
  new `Provider.capacity_hint(model)`, which returns None by default. The
  async executor runs at most `max_in_flight` nodes of a model at once and
  passes over a ready node whose model has no free slot, still ready and with
  nothing reserved, to admit other ready work, so a full limit never stops the
  run; a budget reservation that fails still does. Every node call, including
  tool turns and retries, and an `LLM_MERGE` synthesis take turns in the order
  they ask, start at least `min_interval_s` after the previous call that
  started, and wait out the cooldown that a transient status error's
  `Retry-After` starts, at most `cooldown_max_s`. A call that a timeout or a
  cancellation stops while it waits takes no turn.
  `round_robin=True` admits ready nodes in turn across branches, so one wide
  fan-out cannot hold back another's work. Timing reports these waits as
  `waiting_ms["capacity"]`, a key every timing record now has. A hint that is
  not valid is refused before any call. Durable runs refuse the policy, and
  checkpoints do not record it. Without a policy, admission and calls are
  unchanged. See [provider capacity](docs/execution.md#provider-capacity).
- A module entry in `tests/distribution_profile.json` can be a glob pattern,
  so one entry excludes a family of repository-only test modules from the
  source-archive profile. Every entry must still name an existing module.
- An offline selector comparison harness,
  `benchmarks/run_selector_comparison.py`. Four arms (a single worker, the
  fixed pipeline, a generated plan, and adaptive selection, in which a
  `WhiteRabbit` router applies developer routing rules and then a scripted
  classifier) run on the task-shape suite and on the Astra pack's development
  tasks with their sources. One strict acceptance policy per family decides
  every arm, arms run in a seeded counterbalanced order with two attempts per
  cell, every failed attempt is recorded and charged, and every adaptive
  attempt is charged its selector cost and routing time. An Autotune campaign
  then compares the generated plan (incumbent) with adaptive selection
  (challenger), with the other two arms as references, under an evaluator
  hash from an `EvaluatorManifest` and a `no_new_failures` regression gate,
  which fails when a pinned reference output is not accepted or fails the
  answer key, or a pinned planted defect is accepted. A held-out pack is
  loaded only by path from outside the repository, verified against its
  manifest. `benchmarks/offline_arms.py` is the shared runner for offline
  arm harnesses, and `make_swarm` in `benchmarks/harness.py` builds
  the `adaptive_selection` arm. The committed record,
  `benchmarks/results/selector_comparison_offline.json`, is labeled offline
  mechanics, not claimable, with zero API calls: fixtures rank no arm and
  change no default. See
  [selector comparison](benchmarks/README.md#selector-comparison-offline-mechanics).
- `EvidenceTriggeredSupervisor(inner, triggers, on_trigger=None)` reviews only
  when a `ReviewTriggers` check fires after a completed node; otherwise it
  returns no change without any call. Triggers read the graph and the node's
  recorded state: the stages `review_after`, `fan_in` and `final`, and the
  anomalies `failed_or_skipped`, `verifier_rejected_or_exhausted`,
  `acceptance_not_accepted`, `contradiction_records`, `incomplete_output`,
  `empty_or_short_output` (`min_output_chars`), `cost_over_estimate` and
  `tool_errors` (`min_tool_errors`). A developer `on_trigger` can return a
  `Revision` with no model call; an inner `LLMSupervisor` reviews with the
  fired triggers and their evidence ahead of its unchanged prompt. Revisions
  keep gate protection and strict acceptance obligations. Each triggered review
  adds a `review_triggered` trace span, and a decision's `trigger` names the
  triggers. Durable recipes record the triggers and replay journaled reviews;
  they refuse `on_trigger`. Retry-owner and model-profile checks apply to a
  wrapped `LLMSupervisor`. A node's tool calls that return errors are counted
  in `metadata["tool_errors"]`. See
  [evidence-triggered review](docs/supervisor.md#evidence-triggered-review).
- `ModelProfile(fallback=...)` names another of the run's profiles for a
  node's retries. After an attempt on the profile fails with a `transient`,
  `provider_rejected` or `quality_defect` failure, the node's next attempt
  calls the fallback instead: a new attempt within the node's allowance,
  persisted attempts included, that waits the retry's backoff, reserves its
  own profile's ceiling and, under a capacity policy, takes that provider
  type and model's turn and in-flight slot. Fallbacks can form a chain. A
  retry after any other failure, `unknown_outcome` included, calls the same
  profile again. A node timeout that ends a sent, unanswered call is
  `unknown_outcome` only under `RetryPolicy(retain_uncertain_dispatch=True)`;
  without it the timeout is `transient`, and its retry calls the fallback
  while the first call may still run and be charged. Each attempt's span on
  `Tracer.spans` records its `profile` and, after a fallback,
  `fallback_from`; `summary()` keeps its keys. A fallback that is
  not one of the run's profiles, requires fewer features than the profile it
  serves or loops back is refused when the Swarm or executor is built, and
  each unfinished node is checked against every profile it may fall back to
  before any call. Checkpoints are unchanged, and a resumed node's first
  attempt calls the profile it names. See
  [profile fallback](docs/execution.md#profile-fallback).
- An offline trigger ablation harness, `benchmarks/run_trigger_ablation.py`.
  Three arms (routine `LLMSupervisor` review on its default schedule,
  `EvidenceTriggeredSupervisor` around the same reviewer with the default
  `ReviewTriggers`, and no supervisor) run the same generated plans on four
  clean fixture tasks and five that each plant an anomaly of a kind the
  default triggers detect, under one strict acceptance policy with one
  attempt per cell. A scripted reviewer decides each review from the run
  state, never from trigger evidence. Task by task, the record keeps each
  arm's supervision calls and their declared cost, the triggers that fired,
  the revisions applied and the accepted outcomes. An Autotune campaign then
  compares routine review (incumbent) with triggered review (challenger),
  with no supervision as a reference arm, minimizing supervision calls while
  the accepted-outcome rate must not regress. `run_cell` in
  `benchmarks/offline_arms.py` takes an `attempt_runner`, so a harness with
  arms of its own reuses the shared runner, with `execute_attempt` and
  `attempt_record`. The committed record,
  `benchmarks/results/trigger_ablation_offline.json`, is labeled offline
  mechanics, not claimable, with zero API calls: fixtures show no measured
  benefit and change no default. Its campaign's holdout is labeled an
  offline stand-in, not an untouched holdout, and its `decision_status`
  states that the engine's decision, kept as it ran, is mechanics only and
  promotes nothing. See
  [trigger ablation](benchmarks/README.md#trigger-ablation-offline-mechanics).
- Security stops. `RunControl.revoke(reason_code)`, on a control given to
  `Swarm(run_control=...)` or to either executor, stops runs at once and for
  good: `plan()`, `execute()` and `resume()` refuse to start under a revoked
  control, and a run in progress admits no node and sends no call after it,
  tool turns, supervisor reviews and synthesis included. The async executor
  cancels the nodes in flight at once and never drains; a revocation during
  a drain ends it without waiting for the deadline, and one during a pending
  supervisor review cancels the review with the nodes in flight, in ordinary
  and durable runs alike. A call that was sent and not answered keeps its
  reservation as unknown cost, even without
  `retain_uncertain_dispatch`. A `SecurityStop` that a tool runtime raises,
  alone or in an exception group, stops the run the same way instead of
  going back to the model. `SecurityStop.reason_code` is a short developer
  code (1-64 letters, digits, `.`, `_`, `:` or `-`), and its failure class,
  `security_stop`, is terminal. An ordinary run's checkpoints record the
  stop as `control.security_stop` in version 5, listing `security_stop` in
  `requires`, and resume and re-evaluation raise its `SecurityStop`. A
  durable run is blocked in its journal with `blocked_reason`
  `security_stop` and a `security_stop` event, which no latch or method
  clears; `SQLiteWorkflowStore.security_stop()` stops any run, a completed
  one too, without a lease, and `load_security_stop()` reads the stop.
  Resuming, executing, re-evaluating or republishing a stopped durable run,
  reusing its results and preparing or approving an external action for it
  are refused, and `smythe runs inspect` shows the stop with the next action
  `rerun`. See [security stops](docs/execution.md#security-stops).
- Durable runs drain their in-flight calls under
  `Swarm(run_store=store, termination=TerminationPolicy(drain_timeout_s=...))`.
  When a failure stops the run, such as a `HALT` node's zero-cost rejection,
  nothing more is admitted or sent, and the requests already sent may finish
  until the deadline: a node answered in time settles its known charge and
  keeps its result, the failed checkpoint consumes its call, and resume never
  calls it again, so the run is not blocked. A node that had not sent its
  request, such as one still counting its input tokens or a read step
  resolving its destination, returns to pending without reserving or sending
  it, so a drain never adds unknown exposure. A request still unanswered at
  the deadline is cancelled and keeps unknown exposure at its quoted
  ceiling, as without a policy, and a security stop still cancels at once.
  The recipe records the policy under `termination` only when one is
  configured, so existing recipes keep their bytes, and a run keeps its
  policy: resuming without one uses the saved policy, and a different one
  raises `WorkflowConflictError`. Unknown model-call exposure still has no
  reconciliation path; a run it blocks needs a new run. See
  [draining in-flight calls](docs/workflow-accounting.md#draining-in-flight-calls).
- [`examples/18_archive_maintenance.py`](examples/18_archive_maintenance.py)
  maintains an archive of four lineage runs offline and without keys. Two of
  its bundled sources are shared between runs. After one shared source gets
  a new version, one published brief is damaged and one final validator's
  version is bumped, it prints the repair plan and applies it: the runs that
  read the revised source regenerate only what depends on it, the damaged
  brief is published again and the untouched run is decided again. It then
  computes the four runs again in full in a second journal and prints the
  reads, model steps, local steps, publications, acceptance evaluations and
  new runs each took, with byte and decision equality per run. Example 16
  now reads its sources through the shared `examples/_source_library.py`,
  which also serves later versions of a bundled source.
  `tests/test_examples_smoke.py` runs the example.
- An offline policy ablation harness, `benchmarks/run_policy_ablations.py`.
  Each ablation runs a baseline arm and a policy arm built from one
  configuration that differ only in the policy under test, and the record
  lists the keys each case's arms differ in: drain on ordinary runs
  (`TerminationPolicy` on `AsyncExecutor`, on a manual clock) and on durable
  runs (on the shared fake Responses transport), `CapacityPolicy` admission
  against a fixture provider's quota (on a manual clock, then in the
  provider-concurrency simulator), `Swarm(reserve_for_checks=True)` (one
  node in flight at a time, so each arm's calls and spend repeat exactly),
  and `Synthesizer(retain_contradictions=True)` under strict source-evidence
  acceptance. Every arm keeps each failed, cancelled, refused or rejected
  call or attempt, and groups the failures that share a cause as correlated
  errors. A matched-graph scheduler overhead test runs one fixed pipeline
  under an ordinary Swarm, a durable Swarm and a minimal reference asyncio
  executor on the same fake transport, requires byte-identical outputs, and
  reports each executor's overhead with checkpoint and journal store time
  apart; the durable overhead includes the declared latency of its
  input-token count calls, which the record lists as
  `extra_wire_latency_ms`. The committed record,
  `benchmarks/results/policy_ablations_offline.json`, is labeled offline
  mechanics, not claimable, with zero API calls; it lists each policy as
  experimental and off by default, and its overhead times are
  machine-specific. See
  [policy ablations](benchmarks/README.md#policy-ablations-offline-mechanics).
- An offline template evaluation harness, `benchmarks/template_eval.py`. A
  template version is an Autotune policy for its registry target,
  `planner.template.report`: an arm is a `ConstrainedArchitect` over the
  catalog record of the version the policy names, with an `OfflineProvider`
  scripting its selection, or `single_agent`, a `SimpleArchitect`. The
  predecessor version is the incumbent, the candidate the challenger, and
  `single_agent` a sealed reference arm under `must_not_be_worse`. One trial
  is one pass over a split's tasks from a task pack generated in Python and
  pinned by SHA-256; the holdout pack holds a task family development and
  confirmation do not, and only a holdout trial loads it. A strict acceptance
  policy per family judges every arm. Trials record accepted and attempted
  tasks, attributable micro-USD cost and planned nodes, and gate on an
  unchanged policy, complete accounting, no template on an excluded family
  and a nonzero accepted count. The evaluator manifest binds the harness
  file's digest, each split's pinned pack and acceptance policies, the
  validator identities and the fixture price, and cost per accepted outcome
  is reported as a ratio of sums, apart from the campaign's own spend.
  Fixture campaigns promote a version that recovers planted step failures and
  decline one that drops a section only the held-out family needs (verdict
  `worse`), one that loses to the single worker and one applied to an
  excluded family. The harness is repository-only and not packaged. The
  committed record, `benchmarks/results/template_eval_offline.json`, is
  labeled offline mechanics, not claimable, with zero API calls: fixtures
  rank no template, activate nothing and change no default. See
  [template evaluation](benchmarks/README.md#template-evaluation-offline-mechanics).
- `jobs review RUN_ID [--limit N] [--offset N]` lists only what in a Jobs
  run needs an operator's decision, and the command that carries each
  decision out. Items come from states the run already records: a budget
  overrun, invalid provider accounting, a stop request, a failed detached
  worker or an expired lease (the states `jobs attach` exits 7 for), an
  unknown outcome with its kept exposure, a rejected or failed operation
  with or without an attempt left, a missing, changed, unreadable or unsafe
  accepted file, and a succeeded operation that is not eligible for
  delivery. Each item names what it holds back (admission, completion or
  delivery), its subject, the recorded status, a redacted error excerpt of
  at most 300 characters without local directories, the decision needed and
  `resolve_with`: `jobs adopt`, `reroll`, `reroll --acknowledge-unknown`,
  `reroll --repair-integrity`, `recheck` or `resume` command lines that
  parse as given, without `--store`. Its `item_id` hashes the item's kind,
  run, subject and recorded state, so it stays the same across reviews
  until that state changes. Blocking kinds come first, then operation keys;
  operations are paged, and accepted files are hashed within the 64 MiB
  inspection limit. The printed output names the operations a page holds by
  position, quotes each command for the local shell (cmd.exe's rules on
  Windows) and reminds you of `--store` whenever it prints a command. The
  review opens the database read-only, never writes to or migrates it
  (SQLite may create the `-wal` and `-shm` files of a WAL-mode database, as
  for `jobs inspect`), constructs no provider and works on schema-2 to
  schema-4 databases. It exits 0 whether or not anything needs a decision,
  and 130 without partial output when interrupted. The versioned JSON packet
  (`review_version: 1`) holds no local paths. In Python,
  `smythe.jobs.inspection.review_job()` returns it. See
  [review what needs attention](docs/jobs.md#review-what-needs-attention).
- `runs review [RUN_ID] [--limit N] [--offset N]` lists only what in a saved
  run, or in a page of the newest runs, needs a decision: ordinary
  checkpoints by when each was last written, or a durable journal's runs by
  creation, opened read-only, so a version 1 journal is never upgraded.
  Each run is read once, as `runs inspect` reads it without a lineage run's
  successors, and its items come from that projection: a security stop; a
  durable run blocked by a budget overrun, unknown exposure or conflicting
  evidence, with each unknown call's id and nanoUSD; a failed run; a strict
  run whose nodes completed without a committed decision, which resume
  decides again; a read sent without a saved response; a source withdrawn
  by a restriction, or removed or not held; a damaged published document; a
  `rejected`, `needs_review` or `indeterminate` decision with its validators
  and reasons; and external actions waiting for an approval or with an open
  outcome. A run without an acceptance record has no decision item, and a
  page of runs reports one it cannot read as `unreadable`. Items have the
  Jobs review's schema and stable IDs, with the Python entry point that
  resolves each, or the `runs revoke-action` command that stops further
  writes; the review exits 0 whether or not anything needs a decision.
  `smythe.review_run()` returns the packet and, given a strict policy, adds
  a `policy_changed` item for a completed run decided under another one.
  `runs inspect` adds `unknown_cost_calls` to a durable run's `accounting`.
  See [review what needs a decision](docs/runs.md#review-what-needs-a-decision).
- Versioned template records. `distill_record(graph, *, name,
  description=None, registry=None, require_success=True, source_task=None)`
  returns a `TemplateRecord`: the distilled blueprint as JSON data (type
  `smythe.template_record`, version 1) with optional applicability,
  prerequisites (capabilities, skills, tools and validator identities),
  expected outputs, up to 16 failure examples by digest, tests, source runs,
  a parent version and a review. A record never carries MCP servers, models,
  environment, URLs, tool grants, acceptance policies, budgets, evaluator
  configuration or paths, and `from_dict()` refuses unknown keys and
  versions. A blueprint whose dependencies form a cycle, and text without a
  UTF-8 form, such as a lone surrogate, raise `TemplateRecordError`, or
  `DistillationError` when distilling. `template_version` is the digest of
  the fields that decide what a template builds and `record_sha256` that of
  the whole record, by which records compare and hash. `to_template()`
  builds the template `distill_template` returns, which is now built on it,
  and `local_only()` declares that builder for a durable run as
  `template:<name>` at the template version, so the recipe pins the exact
  template. Labels, personas and the description are copied verbatim; the
  review lists labels that repeat the source task's goal, and the text that
  holds a rooted or home path, by node and field, and leaves them unchanged.
  `TemplateCatalog(root)` keeps records in append-only,
  owner-only files, `records/<record_sha256>.json` and
  `versions/<template_version>`, published without replacement and never
  deleted, and `get()` checks each file against its digest. Its
  `layout.json` records the file layout, version 1, and a catalog in
  another layout is refused; one without the file is read as layout 1.
  `structure_sha256(graph)` digests a graph's shape without its labels,
  results or history. `distill_template` raises `DistillationError` for a
  graph that cannot be written as a record, such as one whose node depends on
  a node the graph does not contain. See
  [learning from runs](docs/learning.md).
- `PlannerMemory` records verification and acceptance evidence. A run whose
  graph holds a gate's verification receipt, or whose result has a delivery
  acceptance record or rejected output, also records its `execution_id`, its
  graph's `structure_sha256`, one `verification` entry per gate receipt
  (`gate`, `target`, `state`, `passed`, `regenerations_used`) and, for
  rejected output, `rejected_output_sha256`, a digest that never holds the
  text. `passed` is true only when the receipt records a pass, so history
  shows an exhausted gate, or one whose target did not complete, as not
  passed, while `success` still means every node completed. Other runs write
  the same keys as before, recall skips rows whose new fields are malformed,
  and planning prompts are unchanged. See
  [evidence in planner history](docs/learning.md#evidence-in-planner-history).
- Opt-in failure recording in planner history. With
  `PlannerMemory(record_failures=True)`, a `Swarm` run that raises during
  execution, synthesis or acceptance, including a resumed run, is recorded
  before its exception propagates, with `failure_category` (its failure
  class), `failure_type` (the exception's class name) and, for provider and
  service errors only, a `failure_excerpt` of at most 200 characters with
  account identifiers redacted, the home folder written `~` in any spelling
  of its separators, including a repr's doubled backslashes, and other
  absolute paths `<path>`, whole when quoted and, unquoted, with the words
  after them up to the end of the line, a quote, a URL, an HTTP route or a
  word ending in a colon or semicolon, since the rest of a name with a space
  cannot be told from them. URL paths and HTTP routes stay. `success` keeps
  its meaning, and planning prompts keep their format and show each failure
  row's outcome as `failure`, even for a run whose nodes all completed
  before synthesis failed. Interrupts, exits and cancellation are never
  recorded, and a history write that fails is logged without replacing the
  run's exception.
  `PlannerMemory.record_failure()` writes such a row directly. The default
  `PlannerMemory`, and memory objects that implement only `record()`, record
  nothing on failure. See [failed runs](docs/learning.md#failed-runs).
- Template records are validated as untrusted candidates.
  `validate_template_record(record, *, parent=None, limits=None)` checks a
  `TemplateRecord` or its JSON data and raises `TemplateValidationError`,
  naming the node and field, for a key that would grant authority at any
  depth (`mcp_servers`, `model`, `env`, `command`, `url`, `tools`,
  `allowed_tools`, `acceptance`, `evaluator` or `budget`; the record's own
  prerequisite `tools` and source-run `acceptance` grant nothing and are
  allowed); for a blueprint that breaks
  the generated-plan limits, or the developer's `TemplateLimits`, including
  a gate that does not depend on the node it verifies; and, compared with a
  parent, for a dropped or retargeted gate, lowered `max_regenerations`, a
  gate or target relaxed to `skip`, a removed expected-output validator or
  prerequisite, a dropped `proven_with`, or a task family the parent's
  applicability excludes that it no longer excludes.
  `TemplateCatalog.put(record, *, parent=None, limits=None)` validates
  first, by default against the stored record of the candidate's
  `parent_version`, writes nothing for a refused record, and also accepts a
  record's JSON data. A new version of a name the catalog
  already holds must name one of its stored versions as `parent_version`,
  so it is always compared with a stored parent; only a name's first
  version may name none. A prerequisite at another version counts as
  removed, so a candidate learned after a prerequisite of each stored
  version of its name changed is stored under a new name.
  `distill_template` and templates built in code are not validated. See
  [validate a candidate record](docs/learning.md#validate-a-candidate-record).
- Template versions are activated and rolled back through the policy
  registry. A template's target is `planner.template.<name>`
  (`template_target(name)`), whose policy maps it to a template version, so
  a reviewed Autotune decision activates a version and a rollback restores
  the previous one. `TemplateCatalog.active_templates(registry, names)`
  builds the version of each name in effect for ordinary runs, leaving out
  a retired template, and `active_local_templates(registry, names)` declares
  them with `record.local_only()` for durable runs, whose recipes pin them.
  A `ConstrainedArchitect` keeps the menu it was built with, and a rollback
  writes only the registry, so running work keeps its versions.
  `pinned_templates(config, catalog)` rebuilds the versions a saved durable
  recipe pinned, leaving the recipe unchanged, to resume the run; resuming
  with another version still raises `WorkflowConflictError`. A plan composed
  from templates built from records, including `distill_template`'s, lists
  each such selection's name, index and template version in
  `graph.planning["templates"]`, never in node metadata; templates written
  in code record nothing. See
  [activate and roll back templates](docs/learning.md#activate-and-roll-back-templates).
- Candidate templates from repeated accepted structure.
  `repeated_structures(memory, *, min_accepted)` groups planner history by
  graph structure digest and returns the structures that at least
  `min_accepted` runs whose output was accepted share, counting each
  execution id once; `success` alone never counts, so a completed run whose
  output was rejected does not. `candidate_from_graph(graph, *, rows, name,
  min_accepted=1, ...)` distills the graph into a `TemplateRecord` whose
  source runs are those accepted runs and whose failure examples keep, for
  each failed, rejected, `needs_review` or `indeterminate` run of the
  structure, only a category and an evidence digest, never text, and raises
  `DistillationError` below `min_accepted`. Nothing is stored or activated.
  `PlannerMemory.iter_outcomes()` reads each valid history row, oldest
  first, without writing. See
  [promote repeated accepted structure](docs/learning.md#promote-repeated-accepted-structure).
- Deterministic templates. `TemplateRecord.as_architect()` returns a
  `TemplateArchitect`, a `DeterministicArchitect` that builds the record's
  graph for every task with no planning call: the graph a
  `ConstrainedArchitect` composes when it selects the template once, with
  the same node ids and `graph.planning["templates"]` entry. Its
  `local_only()` declares it for durable runs as `LocalOnly(role="architect")`
  with identity `template:<name>` and the template version, so the recipe
  pins the exact template. See
  [build a template without planning](docs/learning.md#build-a-template-without-planning).
- Template applicability. `TemplateEnvironment.capture(registry=...,
  tool_runtime=..., validators=..., models=...)` snapshots the agents'
  effective capabilities, every version each skill is listed at, such as
  Agent Skills digests, from `Registry.skill_provider` by default,
  `LocalToolRuntime.describe()`, including each runtime in
  `CompositeToolRuntime.runtimes`, MCP server allowlists, acceptance
  validator identities and model ids. A skill provider that raises for an
  agent is logged and sets the snapshot's `skills_complete` to false.
  `candidate_from_graph(..., environment=...)` records what the candidate
  relied on as prerequisites and in a new optional `proven_with` section,
  part of the template version, that holds only digests of the model ids
  and MCP allowlists, and refuses a snapshot whose skills could not all be
  read. `check_applicability(record, environment)` returns an
  `ApplicabilityReport(valid, reasons, details)` with the reason codes
  `capability_removed`, `skill_removed`, `skill_changed` (a skill not
  listed at exactly the recorded versions), `skills_unreadable` (a record
  that names a skill, in a snapshot whose skills could not all be read),
  `tool_removed`, `tool_changed`, `validator_set_changed`, `model_changed`
  and `mcp_allowlist_changed`. `TemplateCatalog.active_templates()` and
  `active_local_templates()` take `environment=` to leave out a version in
  effect that no longer applies, logging why, or with `strict=True` raise
  `TemplateApplicabilityError`; nothing is deactivated, and pinned durable
  runs are unchanged. See [applicability](docs/learning.md#applicability).
- Agent Skills directories as a skill source.
  `AgentSkillsProvider({agent_id: [root, ...]})` reads each
  `<root>/<name>/SKILL.md` when it is built and lists each skill as a
  `SkillRef` whose version is the `sha256:` digest of the file with CRLF read
  as LF; its metadata holds the description, directory name and declared
  `allowed-tools` and version, never a path or the instructions. Only the
  frontmatter is parsed, with PyYAML's safe loader; `parse_skill_md()` refuses
  an invalid name, a name that differs from its directory, a missing or
  overlong description, frontmatter that YAML cannot load, that is not a
  mapping or that uses YAML aliases (refused before they expand), and files
  over 64 KiB; the provider refuses one name in two directories and a
  `SKILL.md` that resolves outside its root. Each refusal is an
  `AgentSkillError` whose message stays short however large the refused
  value. A skill adds only a capability tag: declared tools are never
  granted, and its instructions, which `instructions(name)` returns after
  checking the digest, never enter a prompt. `refresh()` discovers again and
  returns the changed names. Scripts are never run or imported.
  `Registry.static_snapshot()` returns a `STATIC_ONLY` registry of each
  agent's current effective capabilities, which durable runs accept in place
  of live hydration. See [Skills](docs/skills.md).
- Python-backed skills and local tools. `LocalTool(spec, handler, version,
  timeout_s=None)` runs a Python handler for a model's tool call, and
  `LocalToolRuntime(tools, agents={agent_id: [tool_name, ...]})` offers each
  tool only to the agent ids listed, never by name or capability tag, so an
  agent a generated plan creates under the same name is offered none.
  `python_skill(skill, handler, input_schema=..., version=...)` makes a tool
  named `skill.<name>` described by an Agent Skill's description, without
  reading the skill's directory. A synchronous handler runs in a thread of its
  own and an `async` one on the event loop. Each call runs once: an
  exception, a result that is not text or an expired `timeout_s` returns to
  the model as an error result, while `BudgetValidationError`,
  `ProviderResponseError`, `WorkflowError` and `SecurityStop` stop the node.
  A handler that returns a coroutine instead of awaiting it gets an error
  result, and the coroutine is closed unrun, so it never warns.
  Result text is cut to 65,536 characters by default, and `describe()` lists
  each tool's name, version and input-schema digest in the form template
  records use. A tool name that contains `__`, or `_` before `.`, is refused,
  since a provider would read a call to it back as another name.
  `CompositeToolRuntime(*runtimes)` merges local and MCP tool sessions and
  refuses a tool name that two of them offer, comparing names as a provider
  receives them, so `fs.read` and `fs__read` clash. Local tools serve
  ordinary runs; durable text workflows still refuse every `tool_runtime`.
  See [Python-backed skills](docs/skills.md#python-backed-skills) and
  [`examples/20_agent_skills.py`](examples/20_agent_skills.py).
- [`examples/21_template_lifecycle.py`](examples/21_template_lifecycle.py)
  runs a template's lifecycle offline and without keys, with an evaluator it
  defines. Three accepted runs in planner history become a candidate record,
  validated and stored as the predecessor, and a record that would grant
  itself a tool is refused. An Autotune campaign compares a candidate that
  drops a step with its predecessor and a single-worker reference arm and
  rejects it on held-out incident reviews, and the policy registry refuses
  to approve that decision. An evaluator that holds out only briefings
  promotes the same candidate, which is reviewed and activated; the next
  incident review is rejected, and a rollback restores the predecessor for
  new runs without changing the ledger, the catalog or an architect already
  built. `tests/test_examples_smoke.py` runs the example.
  `tests/test_learning_boundaries.py` checks in one place that no Autotune
  campaign or exploring proposer varies a reserved path, no template record
  grants authority or weakens its parent, and no `SKILL.md` changes the
  tools agents are offered, acceptance, the budget or evaluator data, each
  of which stays byte-identical. See
  [a template's lifecycle](docs/learning.md#a-templates-lifecycle) and
  [what learning cannot change](docs/learning.md#what-learning-cannot-change).
- Sample-size and repeated-comparison guidance for Autotune, from the paired
  Student-t promotion bound: the smallest observed improvement each number
  of pairs can promote, the pairs a stage needs to pass a given true
  improvement, and how repeated campaigns, and proposers that see
  confirmation results, raise the chance of promoting a challenger with no
  real effect. An evaluator that returns the same metrics on every seed
  promotes with three pairs and a zero-width interval. See
  [choosing sample sizes](docs/optimize.md#choosing-sample-sizes).
- A [support matrix section](docs/support-matrix.md#who-owns-each-execution-responsibility)
  states who owns each execution responsibility on ordinary runs, durable
  text workflows and Jobs: admission and scheduling, logical and attempt
  identity, permission to dispatch, calling again, settling cost, keeping
  results for recovery, stopping, waits and timers, ownership and stale
  writers, recovery after process loss, observation and upgrades, with the
  test behind each cell and a procedure for upgrading with work in flight.
  Ordinary checkpoints have no ownership fence: two processes that resume
  one execution id both run its pending nodes, and the snapshot written last
  wins. A durable run's lease lasts 60 seconds and its event loop renews it
  every 15 seconds; a resume while another process holds the lease raises
  `WorkflowLeaseError` before any call, and a run whose lease lapses while
  its event loop is blocked fails with `WorkflowLeaseError` and resumes
  without sending a saved request again. `tests/test_support_matrix.py` pins
  these behaviors, and a repository test checks the section's rows, columns
  and cited tests.

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
  `CallScope`, and the journal gains the external action tables. Opening a
  version 1 journal with `SQLiteWorkflowStore(path)` upgrades it in one
  transaction that creates those tables, empty, and changes the recorded
  version, so every run, call, balance, reservation, evidence row and the
  store identity are kept, as are tables that other tools added to the
  file; an interrupted upgrade rolls back. While any run holds an unexpired
  lease the upgrade raises `WorkflowLeaseError`, so stop workers from
  earlier releases first. `SQLiteWorkflowStore(path, read_only=True)` reads
  a version 1 journal without upgrading it, and the new `store_version`
  property reports the version. A store open for writing raises
  `WorkflowCorruptionError` and writes nothing once another release changes
  the journal's version. See
  [journal upgrades](docs/workflow-accounting.md#journal-upgrades).
- A Jobs approval token creates one run. `jobs run` with a token that already
  created a run in the same database exits with code 4 and names that run
  instead of starting another run with the full ceiling. `--new-run`, or
  `allow_approval_reuse=True` on `SQLiteRunStore.create_run` and
  `JobRunner.start`, starts another run on purpose, and that run's
  `run_created` event records how many earlier runs used the token and the
  five most recent run IDs under `approval_reuse`.
  Token derivation, plan hashes and the Jobs schema are unchanged.
- `jobs inspect` text output names the operations its page holds by
  position, as `jobs review` does:
  `Operation page: operations 1-2 of 3 (limit 2)`. It used to print the
  counts as `2 of 3 (offset 0, limit 2)`, which read as page numbers. The
  JSON output is unchanged.
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
- `distill_template` raises `DistillationError` for a graph it cannot write
  as a template record, which it used to accept or fail on with another
  error: a blank template name; a blank or duplicate node id; a node that
  depends on itself, on a blank id or on a node the graph does not contain;
  dependencies that form a cycle; a timeout that is not a positive finite
  number; a negative or non-integer `max_retries` or `max_regenerations`; a
  label or metadata `role` that is not text; text without a UTF-8 form; and
  an integer timeout too large to be a float. A plan a `ConstrainedArchitect`
  composes from its templates records them under `graph.planning["templates"]`,
  so such checkpoints gain that key. See
  [distill a template](docs/learning.md#distill-a-template).
- `benchmarks/run_optimizer.py` and `benchmarks/run_banner_optimizer.py` are
  labeled diagnostic. They keep a change whose mean judge score beats the
  incumbent's by a fixed margin, with no paired bound, sealed holdout or
  review, so a change they keep is not a promotion and supports no claim. See
  [prompt-policy optimizer loops](benchmarks/README.md#prompt-policy-optimizer-loops-diagnostic).

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
- Each build of a template from `distill_template` gives its agents their own
  capability lists. They shared the template's lists, so changing one built
  agent's capabilities changed every later build.
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
