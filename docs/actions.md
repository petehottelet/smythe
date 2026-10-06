# External actions (preview)

**Preview.** `smythe.actions` defines the contract types for actions that
write accepted content to a service outside Smythe, such as publishing a post.
It [prepares actions](#prepared-actions) in a durable run's
[journal](workflow-accounting.md#external-action-records), bound to the run's
accepted content, and records [verified approvals](#approvals) of them.
`smythe.workflow_actions` keeps the [dispatch journal](#dispatch-journal) of
each attempt, and the [action runner](#action-runner) makes those attempts
through your connector, reads their results back,
[reconciles](#reconciliation) outcomes a dispatch left unknown and keeps the
service's [notices](#notices) as untrusted records. These types and functions may
change before they leave preview. `import smythe` loads neither module;
import them explicitly.

An external action has three parts that this page describes:

- an **intent**: exactly what the action writes, and where;
- an **action key**: the one key a service receives for every attempt of the
  action; and
- a **connector profile**: what the connector declares about the service it
  writes to.

A retry is another attempt of the same logical action, under the same intent
and key. A changed intent is a different action, never a retry.

## Intent identity

```python
from smythe.actions import ActionIntent, ActionTarget, ContentRef

body = b"# Launch notes\n"
intent = ActionIntent(
    "publish",
    ActionTarget("acme", "posts"),
    parameters={"title": "Launch notes", "tags": ["news"]},
    content=ContentRef.for_bytes(body, "text/markdown"),
    connector="example-posts",
    connector_version="1",
)
print(intent.digest())  # 64 lowercase hex characters
```

| Field | Meaning |
|---|---|
| `operation` | What the connector does, such as `publish` or `update`. |
| `target` | An `ActionTarget(tenant, collection, object_id=None)`. `object_id` is `None` until the service has created the object. |
| `parameters` | A JSON object, copied when the intent is made. |
| `content` | A `ContentRef(sha256, size_bytes, media_type)` for the accepted bytes the action delivers, or `None`. `sha256` is plain lowercase hex. |
| `connector`, `connector_version` | The name and version of the connector profile. |
| `contract_version` | `smythe.action.v1`. |

`digest()` is the SHA-256 of the intent's canonical JSON (`to_dict()` with
sorted keys, no whitespace and UTF-8 text). Reordering keys in `parameters`
does not change it; changing any field, parameter, content digest, size,
media type, connector version or contract version does. The intent records no
attempt number, time or key, so every attempt of an action has the same
digest.

Parameters must be JSON data. NaN, infinity, bytes, tuples, sets, non-string
keys and cycles raise `ValueError`. `ActionIntent.from_dict()` reads
`to_dict()` data and raises `ValueError` for an unknown contract version, for
a missing or unknown key at any level, and for `parameters` that are not an
object, `null` included: `to_dict()` always writes an object.

Intents compare equal exactly when their canonical forms match, parameters
included, and equal intents hash alike.

## Action keys

```python
from smythe.actions import action_key

key = action_key(store.store_id, run_id, "publish", 1)
```

`action_key(store_id, run_id, slot, sequence)` names one logical action: the
journal, the run, a host-chosen slot for the action in that run, and the
action's sequence number within the slot. The key is 64 lowercase hex
characters. It does not depend on the intent or on any attempt, so it stays
the same across attempts and restarts, and a service that deduplicates by key
can recognize a repeated submission of the same action.

This differs from the Jobs idempotency key, which Jobs derives for each
attempt and transmits to no provider (see
[the dispatch journal](jobs.md#the-dispatch-journal)).

## Connector profiles

A connector declares what its service guarantees in a `ConnectorProfile`:

```python
from smythe.actions import ConnectorProfile

profile = ConnectorProfile(
    "example-posts", "1", ["publish", "update"],
    deduplicates=True, dedupe_window_s=86_400,
    lookup_by_key=True, lookup_by_object=True, visibility_bound_s=5,
    write_ceiling_nanousd=2_000_000, read_ceiling_nanousd=10_000,
    max_attempts=3, readback_reads=4, readback_interval_s=1, timeout_s=10,
)
```

| Field | Default | Meaning |
|---|---|---|
| `name`, `version` | required | The connector's identity; intents name it. |
| `operations` | required | At least one distinct operation name; stored sorted. |
| `deduplicates` | `False` | The service answers a repeated key from its earlier commit. |
| `dedupe_window_s` | `None` | How long the service remembers a key, in positive seconds; `None` declares no expiry. Only with `deduplicates=True`. |
| `lookup_by_key` | `False` | `lookup()` finds an earlier submission by its key: within `dedupe_window_s` when the service deduplicates, and without expiry when it does not. |
| `lookup_by_object` | `False` | `read()` returns an object's current bytes. |
| `visibility_bound_s` | `None` | The longest a commit can stay invisible to lookups and reads; `None` means unbounded. |
| `cancellation` | `"unsupported"` | A submitted write cannot be canceled; no other value is accepted. |
| `compensation` | `"unsupported"` | `"new_action"` when a separately approved action can undo an effect. |
| `write_ceiling_nanousd`, `read_ceiling_nanousd` | `0` | The most one write or one read can cost, in whole nanoUSD. |
| `max_attempts` | `1` | The most attempts one action may make. |
| `readback_reads` | `1` | How many reads may check a write's result. |
| `readback_interval_s` | `0` | Seconds between those reads. |
| `timeout_s` | `30` | Seconds each connector call may take. |

Every setting after `operations` is keyword-only, and values have exact
types: a bool is not a number, so `max_attempts=True` raises `ValueError`.
Seconds must be finite and non-negative, and positive for `timeout_s` and
`dedupe_window_s`. Ceilings and counts are non-negative integers, and
`max_attempts` is at least 1. `to_dict()` omits settings at their defaults, and
`ConnectorProfile.from_dict()` refuses unknown keys.

## Connectors and approval verifiers

Connectors and approval verifiers are trusted host code that you supply.
Their protocols are synchronous, because actions run outside the asynchronous
node executor and its scheduling policies never apply to writes.

A `WriteConnector` has a `profile` and three methods. Each call makes one
request and never retries:

- `submit(key, intent, payload, timeout_s)` returns a `ConnectorReceipt`
  naming the committed object, or raises `ConnectorRefused` when the service
  definitely committed nothing and `ConnectorAmbiguous` when it may have
  committed the write.
- `lookup(key, target, timeout_s)` returns the receipt of an earlier
  submission within the target's tenant, or `None`.
- `read(target, timeout_s)` returns an `ObservedObject` with the object's
  exact bytes, or `None`. `ObservedObject.content_ref` hashes those bytes
  itself.

An empty lookup or read alone does not prove that nothing was written.

An `ApprovalVerifier` authenticates approvals; Smythe ships none and does not
authenticate users. It has an `identity` string naming the verifier and its
version. `verify(assertion, intent_sha256, now_ns)` returns the `Principal`
(issuer, subject and tenant) that issued the assertion for exactly that intent
digest, or raises `ApprovalRejected`. `still_authorized(principal, intent,
now_ns)` reports whether that principal may still approve the intent. The
same `verify()` authenticates an [operator's decision](verifier.md#operator-decisions)
on output that strict delivery acceptance left for review.

`check_connector()` returns the connector's profile and `check_verifier()`
the verifier's identity. Both refuse a `Provider`, a class, a coroutine
function and an awaitable, closing a coroutine first, and check only the
protocol's methods for async code, counting an object whose `__call__` is
async as an async method:

- `check_connector()` refuses an async `submit()`, and an async `lookup()` or
  `read()` wherever one is present. `profile` must be a `ConnectorProfile`,
  so an async property is refused and its coroutine closed.
- `check_verifier()` refuses an async `verify()` or `still_authorized()`.
  `identity` must be a nonempty string of at most 512 characters without
  control characters, as other names are.

Other methods, such as an async `close()`, are not checked. The check is
structural, a caller contract rather than a sandbox: it cannot detect
synchronous host code that misbehaves.

## Action states

`ACTION_STATES` lists the states a logical action can record:

| State | Meaning |
|---|---|
| `not_dispatched` | No attempt has been sent, or the last one was refused. |
| `dispatched` | An attempt was sent, and no read has confirmed its result yet. |
| `outcome_unknown` | An attempt may or may not have taken effect. |
| `observed` | A read of the target showed the intended result. |
| `reconciled` | An uncertain outcome was resolved from the service's records. |
| `compensation_required` | The target does not hold the intended result. |
| `canceled` | The action closed before any dispatch. |

`ActionStateError` reports a transition that an action's recorded state does
not allow.

## Prepared actions

An action delivers the accepted output of one completed
[durable run](workflow-accounting.md). `prepare_action()` records it in the
run's journal before anything is sent:

```python
from smythe.actions import ActionIntent, ActionTarget, ContentRef, prepare_action
from smythe.workflow_artifact import current_artifact

document = current_artifact(store, run_id).publication.path.read_bytes()
action = prepare_action(
    store, run_id, slot="publish",
    intent=ActionIntent("publish", ActionTarget("acme", "posts"),
                        parameters={"title": "Launch notes"},
                        content=ContentRef.for_bytes(document, "text/markdown"),
                        connector=profile.name, connector_version=profile.version),
    profile=profile,
)
print(action.action_key, action.ceiling_nanousd)
```

The run must have completed under a
[strict acceptance policy](verifier.md#strict-acceptance-in-durable-workflows)
whose current decision is `accepted`, made by the policy, a re-evaluation or
an [operator](verifier.md#operator-decisions). An advisory run, or one from an
earlier release, has no recorded decision and cannot prepare actions; Smythe
never assumes one. A `rejected`, `needs_review` or `indeterminate` decision is
refused too. The intent's content must have exactly the SHA-256 and size of
the accepted output: the
[published document](workflow-accounting.md#published-documents) for a run
with a renderer, whose file must still hold those bytes, and the output text
as UTF-8 otherwise. A run that a
[security stop](workflow-accounting.md#security-stops) holds is refused
whatever its decision. The intent must name the profile's connector and one of
its operations. A refusal raises `ActionNotAuthorized` and records nothing.

`prepare_action()` returns a `PreparedAction`: its `action_id`, `run_id`,
`slot`, `sequence`, `action_key`, `intent`, `profile`, `state` and
`ceiling_nanousd`: the cost of `max_attempts` writes and `readback_reads`
reads at the profile's ceilings, which an [approval](#approvals) must cover.
No attempt, read or [reconciliation](#reconciliation) lookup is made that
would take the action's confirmed, reserved and unknown nanoUSD past its
approval's ceiling, so lookups share the budget of the reads.
`load_action(store, action_id)` reads it again.

A slot names one place an action goes in a run. Its first action has
sequence 1. Preparing the same intent and profile again returns the slot's
latest action, whatever its state, so a rejected action stays closed. Another
intent or profile raises `WorkflowConflictError` while that action is
unfinished: a changed intent is a new action, never a retry. Once it is
`observed`, `reconciled` or `canceled`, the next intent becomes the slot's
next sequence, with its own action key.

Preparing needs no run lease. Only your trusted host code calls these
functions: a generated plan, source text or model output cannot create an
action or an approval.

## Approvals

Smythe does not authenticate users; your host's
[approval verifier](#connectors-and-approval-verifiers) does. An approval
covers one logical action and at most one effect: exactly the action's intent
digest, up to a nanoUSD ceiling and until an expiry.

```python
from smythe.actions import approve, check_authority

assertion = identity_host.issue(action.intent.digest(), ...)   # your host's approval flow
approval = approve(store, action.action_id, assertion, identity_host,
                   ceiling_nanousd=action.ceiling_nanousd, expires_at_ns=expiry_ns)
assert check_authority(store, action.action_id, identity_host) == approval
```

`approve(store, action_id, assertion, verifier, *, ceiling_nanousd,
expires_at_ns)` returns an `ActionApproval`: the `approval_id`, `action_id`,
verified `principal`, `ceiling_nanousd` and `expires_at_ns`. It refuses:

- an assertion the verifier does not accept for exactly this action's intent
  digest: forged, expired, revoked or issued for another intent. The
  verifier raises `ApprovalRejected`, which propagates;
- a principal whose tenant is not the intent target's, or whom
  `still_authorized()` no longer authorizes, with `ApprovalRejected`;
- a ceiling below the action's `ceiling_nanousd`, and an expiry that is not
  later than the journal's clock, with `WorkflowValidationError`, before the
  verifier is called;
- an action that is `observed`, `reconciled`, `canceled` or
  `compensation_required`, with `ActionStateError`; and
- content the run can no longer deliver, with `ActionNotAuthorized`.

An assertion is recorded once in a journal. The same approval again returns
it; the assertion presented for any other action raises `ApprovalRejected`,
even one whose intent is identical, as when two runs accepted the same
document. Issue a new assertion for each approval. The newest approval of an
action supersedes earlier ones, so a new approval can renew or replace one.

This differs from the Jobs [approval token](jobs.md#validate-plan-and-approve),
an acknowledgment of a plan and its spend ceiling that anyone with the
manifest can compute. It names no principal and has no expiry or revocation.

`reject(store, action_id, assertion, verifier)` cancels an action that was
never dispatched. Its assertion is verified as an approval's is, and the
rejection is recorded with the principal. The action is `canceled` for good,
across restarts: preparing the same intent in its slot returns it, and it can
no longer be approved. An attempt begun in the
[dispatch journal](#dispatch-journal) but not yet claimed is released with
its reservation, since nothing was sent. Rejecting it again changes nothing.

`revoke(store, action_or_approval_id, reason)` revokes one approval, or every
approval of an action, and returns the IDs it revoked. It is a fail-safe
control: it needs no principal and no lease. The reason is stored with
account identifiers redacted. A revoked approval never authorizes the action
again, and its assertion cannot be recorded again; only a new approval
reopens the action.

`check_authority(store, action_id, verifier)` returns the approval that
authorizes the action now, or raises `ActionNotAuthorized` naming why not. It
checks that:

- the action is not `observed`, `reconciled`, `canceled` or
  `compensation_required`;
- its newest approval is not revoked or consumed, has not expired on the
  journal's clock, binds the action's intent digest, covers its ceiling and
  was recorded by this verifier;
- the verifier accepts the stored assertion again, for the same principal,
  and still authorizes that principal; and
- the content is still deliverable, as for `prepare_action()`: no security
  stop holds the run, the run's current decision is still `accepted`, the
  published file still holds the bound bytes, and no source the accepted
  output came from has been [removed](tasks.md#remove-a-source) since, and no
  source a lineage run's record names has been
  [restricted](lineage.md#restrict-a-source), either of which makes the
  content unavailable.

An approval row written into the journal by other means never authorizes
anything: the verifier must vouch for its assertion. `check_authority()` only
reads, so it works on a store opened read-only. The verifier is called
outside any journal transaction, so a check describes one moment: a
revocation made at the host, or with `revoke()`, after a check returns is
seen by the next check, not that one. Approval and the work it authorizes
can happen in different processes, so check again immediately before acting.

The journal keeps each assertion so that it can be verified again; protect
the journal accordingly. `inspect_run()` shows each approval's principal,
verifier identity, ceiling, expiry, consumption and revocation time, and
never assertions or revocation reasons.

## Dispatch journal

`smythe.workflow_actions` records each attempt of an action in the run's
journal, so that a worker that stops part way leaves a state that another
can recover. The [action runner](#action-runner) makes these calls for you;
host code that dispatches by itself makes them in this order, holding the
run's lease, which fences dispatch on the completed run:

```python
from smythe.actions import ConnectorRefused, check_authority
from smythe.workflow_actions import (
    begin_attempt, claim_dispatch, record_ambiguous, record_receipt, record_refused,
    recover_actions,
)

lease = store.acquire_lease(run_id, worker_id, ttl_s=60)
recover_actions(store, lease)  # settle what a stopped worker left in flight
approval = check_authority(store, action.action_id, identity_host)
attempt = begin_attempt(store, lease, action.action_id, approval.approval_id)
permit = claim_dispatch(store, lease, attempt["attempt_id"])
try:
    receipt = connector.submit(action.action_key, action.intent, document,
                               action.profile.timeout_s)
except ConnectorRefused as error:
    record_refused(store, permit, error)
except Exception as error:  # anything else may follow a commit
    record_ambiguous(store, permit, error)
else:
    record_receipt(store, permit, receipt)
store.release_lease(lease)
```

Attempts are numbered from 1, and every attempt sends the action's one
[key](#action-keys). An attempt is in one of these states:

| State | Meaning |
|---|---|
| `prepared` | Begun under an approval, with its write ceiling reserved. Nothing was sent. |
| `dispatched` | Its dispatch is claimed: the request may have been sent. |
| `acknowledged` | The service's receipt names the object it committed. |
| `refused` | The service definitely committed nothing. |
| `outcome_unknown` | The write may or may not have taken effect. |
| `released` | Given up before its dispatch was claimed. Nothing was sent. |

### Attempts and outcomes

`begin_attempt(store, lease, action_id, approval_id)` begins the action's
next attempt under its approval and reserves the profile's
`write_ceiling_nanousd`. It checks the content again, as `check_authority()`
does. Then, in the transaction that makes the reservation, it checks that no
security stop holds the run, that the run is completed, that the action is
`not_dispatched`, and that *approval_id* is still the action's newest
approval, unrevoked, unconsumed and unexpired. Fewer than `max_attempts`
attempts may have been dispatched; a released attempt does not count. The
action's confirmed, reserved and unknown nanoUSD plus the write ceiling must
fit the approval's ceiling, or `WorkflowBudgetError` is raised. It does not
call the verifier: call `check_authority()` first. It returns the attempt as
`inspect_run()` shows it. Beginning again under the same approval returns the
same prepared attempt, a newer approval releases an attempt prepared under an
older one, and `reject()` releases a prepared attempt as it cancels the
action.

`claim_dispatch(store, lease, attempt_id)` checks the same conditions again
and commits the attempt and its action as `dispatched`, with the lease's
epoch and the time, before anything is sent. It returns an
`ActionDispatchPermit`, whose secret token the journal keeps only as a
digest. Each attempt is claimed once. When the approval or the content no
longer authorizes the attempt, it is released with its reservation and
`ActionNotAuthorized` is raised.

The permit then records the outcome of the one submission:

- `record_receipt(store, permit, receipt)`: the attempt is `acknowledged`.
  The receipt must name the action's key, its target's tenant and collection
  and, when the intent names an object, that object; otherwise
  `WorkflowValidationError` is raised and the answer should be recorded as
  ambiguous. The receipt is kept with the external object's identity, and it
  consumes the approval. Its `cost_nanousd` is confirmed. A receipt without
  a cost confirms zero only for a profile whose write ceiling is zero;
  otherwise the reservation becomes unknown exposure. The action stays
  `dispatched`, because a receipt alone does not show that the target holds
  the intended result.
- `record_refused(store, permit, error)`: the attempt is `refused` at zero
  cost. Its reservation is given back, and the action is `not_dispatched`
  again with its approval unconsumed, so the same approval can authorize the
  next attempt. The refusal counts toward `max_attempts`. A refused
  [same-key retry](#reconciliation) leaves the action `outcome_unknown`
  instead, since it says nothing about the earlier attempt.
- `record_ambiguous(store, permit, error)`: the attempt and its action are
  `outcome_unknown`, and the reserved write ceiling becomes unknown exposure.
  No attempt begins after it, and nothing releases its exposure.

Connector error text is stored with account identifiers redacted, at most
500 characters. Recording the same outcome again returns it. An approval is
consumed by the receipt of an effect, never by a refused attempt; a consumed
approval authorizes nothing further.

### Leases and late receipts

`begin_attempt()`, `claim_dispatch()` and `recover_actions()` need the
run's current lease and raise `WorkflowLeaseError` for an expired or
superseded one. A permit records an outcome without the lease while the
run's lease epoch is still the one it was claimed under, so a worker whose
lease lapsed can still record the answer it holds until another worker
acquires the lease. After that, or once the attempt's outcome is recorded or
recovered, a permit changes nothing: its receipt or error is kept as a
`late_receipt` record, marked trusted because only the claim's permit can
write it, and `WorkflowLeaseError` or `ActionStateError` is raised. A late
receipt carries no cost, because the attempt keeps its exposure.

### Recovery

`recover_actions(store, lease)` settles the attempts that a stopped worker
left in flight. Call it once after acquiring the lease and before
dispatching: it treats every attempt in flight as abandoned. A prepared
attempt was never sent, so it is released with its reservation and does not
count toward `max_attempts`. A dispatched attempt without an outcome may have
reached the service, so it and its action become `outcome_unknown` with the
reason `recovered after dispatch`, and its write ceiling becomes unknown
exposure. Recovery audits the run's actions first and sends nothing. Neither
recovery nor a stopped worker ever releases unknown exposure;
[reconciliation](#reconciliation) resolves an unknown outcome as far as the
connector's declarations allow.

### The local transaction boundary

Each step is one local SQLite transaction in the journal. The claim commits
the `dispatched` record before the request is sent, so the journal always
knows that a request may have gone out. That commit is not a transaction
with the service: no local commit can include the remote API. A worker may
stop after the claim but before the request leaves, while the service
commits it, or after the service answers but before the outcome is recorded.
The journal looks the same in each case, `dispatched`, and recovery turns it
into `outcome_unknown` rather than sending it again. A worker that stops
before its claim leaves at most a `prepared` attempt, which recovery releases
safely, because nothing is sent without a claim.

`check_authority()` calls the host's verifier outside any journal
transaction, and `begin_attempt()` and `claim_dispatch()` read the run's
decision, sources and published file just before their write transactions.
Inside those transactions they check again what the journal alone records:
revocation, expiry, consumption, the newest approval, a security stop and the
run's completion. A `revoke()` therefore stops every later claim, while a
revocation known only to the host, or a change to the published file, after
the last check is seen by the next check. A run's `budget_overrun` and
`unknown_exposure` latches close admission to its model calls and do not
block actions, which keep their own accounting scope; a security stop does.

`inspect_run()` shows each attempt's number, approval, state, lease epoch,
receipt digest and balances, and each late receipt by its digest. It never
shows dispatch tokens, receipts or error text. Each transition adds an
`action_attempt` event, and each late receipt an `action_late_receipt`
event, carrying only identifiers and the new state.

## Action runner

`ActionRunner` dispatches a run's prepared actions through your connector,
one at a time, reads their results back and reconciles unknown outcomes:

```python
from smythe.actions import ActionRunner

runner = ActionRunner(store, connector, identity_host, lease_ttl_s=60)
progress = runner.dispatch(run_id, action.action_id)       # one submission
if progress.next_action == "readback":
    progress = runner.readback(run_id, action.action_id)   # independent reads
if progress.next_action == "reconcile":
    progress = runner.reconcile(run_id, action.action_id)  # lookups or a same-key retry
print(progress.state, progress.next_action)
```

`ActionRunner(store, connector, verifier, *, owner_id=None, lease_ttl_s=60,
sleep=time.sleep, clock=time.monotonic)` checks the connector and verifier
as `check_connector()` and `check_verifier()` do. The runner is the only
retry owner: each attempt submits once, and only
[reconciliation](#reconciliation) submits again, with the same key, to a
service that deduplicates it, within `max_attempts`. Your connector must
make one request per call, with any client retries turned off; a connector
that exposes `max_retries` must set it to 0, or `TypeError` is raised. Each
step holds the run's lease as `owner_id`, a new name by default, and
releases it when the step ends. Before submitting, and before each read or
lookup and the wait that precedes it, it renews the lease for
`lease_ttl_s` beyond the connector's timeout and that wait, up to a day, so a
wait longer than the lease does not lose it. Each step first
[recovers](#recovery) whatever a stopped worker left in flight. While
another worker's lease is live, a step raises `WorkflowLeaseError`. `sleep`
and `clock`, in seconds, space the reads and lookups. Dispatch is serial and
runs outside the asynchronous node executor, so its scheduling and drain
policies never apply to writes.

Every step refuses, with `WorkflowConflictError`, an action of another run
and a connector whose `profile` is not exactly the one the action was
prepared with.

### Dispatch

`runner.dispatch(run_id, action_id)` makes the action's next attempt:

1. It checks the approval with `check_authority()`, including a security
   stop on the run, and begins the attempt under it with `begin_attempt()`.
2. It reads the payload once: the run's current
   [published document](workflow-accounting.md#published-documents), or its
   output text as UTF-8 for a run without a renderer. The bytes must have
   exactly the SHA-256 and size of the intent's content. Otherwise the
   attempt is released, unclaimed, and `ActionNotAuthorized` is raised:
   nothing is sent.
3. It claims the dispatch and calls the connector's `submit()` once, with the
   action's key, intent and the bytes it checked, which it never reads
   again. A change to the file after the check cannot reach the service.
4. It records the outcome. A receipt is recorded with `record_receipt()`, a
   `ConnectorRefused` with `record_refused()`, and any other exception with
   `record_ambiguous()`. A receipt that names another key or target, or a
   value that is not a `ConnectorReceipt`, is recorded as ambiguous.

A refusal leaves the action `not_dispatched` under the same, unconsumed
approval, so `dispatch()` can try again while attempts remain. `dispatch()`
never sends an ambiguous outcome again; only
[reconciliation](#reconciliation) can. A receipt never shows the result by
itself: the action stays `dispatched` until a read does.

### Readback

`runner.readback(run_id, action_id)` reads the object an acknowledged attempt
committed, at the target its receipt names, with the connector's `read()`. It
makes at most the profile's `readback_reads` reads per attempt in all,
however often it is called, `readback_interval_s` apart, and only when the
profile declares `lookup_by_object`. Each read is appended as a `readback`
record, charged the profile's `read_ceiling_nanousd` as confirmed cost
against the action, and no read is made that would take the action's
confirmed, reserved and unknown nanoUSD past the ceiling of the attempt's
approval. Without an acknowledged attempt, it reads nothing.

Each read's verdict is a `ValidationReport` from the validator
`READBACK_VALIDATOR` (`smythe.action-readback`, version 1), whose path names
the object as `tenant/collection/object_id`:

| Read | Outcome | Action |
|---|---|---|
| The object is the receipt's target, holds bytes whose SHA-256 is the intent's content digest, carries the action's key when the connector exposes one, and has at least the receipt's version | `passed` | `observed`, or `reconciled` when an earlier attempt's outcome is still unknown |
| Another tenant's, collection's or object's target (`target_mismatch`), other bytes (`content_mismatch`) or another key (`key_mismatch`) | `rejected` | `compensation_required` |
| No object (`not_visible`), a failed read (`read_failed`) or a version older than the receipt's (`stale_version`) | `error` | stays `dispatched` |

The digest is recomputed from the bytes the read returned, never taken from
the service. For an intent without content, the bytes must be the canonical
JSON of its parameters. The first passed or rejected read settles the action.
Reads that decide nothing leave the attempt acknowledged but unverified; once
none are left, or none fits under the approval's ceiling, the next action is
`reconcile`. An empty read alone never shows that nothing was written, and
nothing is submitted again.

Each record holds the attempt, the read's number, the report and, for an
object, its target, content digest, size, media type, version and key; a
failed read's error text is kept with account identifiers redacted. Records
never hold the object's bytes. `inspect_run()` shows them by kind and digest,
and each read adds an `action_readback` event with the record's identifier
and the action's state.

### Reconciliation

A lost acknowledgement, or a worker that stopped after its claim, leaves an
action `outcome_unknown`: the service may or may not have committed the
write. `runner.reconcile(run_id, action_id)` settles that outcome only as far
as the connector's profile allows, and never by guessing:

```python
progress = runner.reconcile(run_id, action.action_id)
if progress.next_action == "dispatch":           # the write was shown not to have been made
    progress = runner.dispatch(run_id, action.action_id)
elif progress.next_action == "operator_review":  # the outcome stays unknown
    notify_an_operator(action.action_id)         # your host's escalation
```

| The profile declares | `reconcile()` | Results |
|---|---|---|
| `lookup_by_key` | Calls `lookup()` with the action's key, at most `readback_reads` times per attempt in all, `readback_interval_s` apart, until one finds the receipt. Reads the receipt's object back with `read()`, which needs `lookup_by_object`. Never submits. | `reconciled` when a read shows the intended result, `compensation_required` when it shows another; `not_dispatched` when an empty lookup proves the write was not made; otherwise `outcome_unknown`, with `operator_review` next. |
| `deduplicates`, without `lookup_by_key` | Submits once more with the same key, as the action's next attempt, while the key's window is open, and reads its receipt back. | `reconciled` or `compensation_required`; `outcome_unknown` after a refused or ambiguous retry, with `reconcile` next, which retries again while attempts remain; otherwise `operator_review`. |
| Neither | Sends nothing. | `outcome_unknown`, with `operator_review` next, at once. |

**An empty lookup is not proof.** An empty lookup never causes a submission.
It shows that the write was not made only when all of these hold on the
journal's clock: the profile declares `visibility_bound_s`; that bound plus
the connector's `timeout_s` has passed since the latest dispatch claim, so a
commit would be visible by now; and the key is still inside its window. The
finding is then `not_executed`, and the action returns to `not_dispatched`
under its unconsumed approval, so `dispatch()` can send it again while
attempts remain; that attempt counts toward `max_attempts`. Without a
declared bound, before the bound has passed, or once the key's window may
have closed, empty lookups leave the outcome unknown.

**The key's window** is `dedupe_window_s` from the action's first dispatch
claim, the earliest the service can have committed the key; `None` declares
no expiry. A call made now may take up to `timeout_s` to reach the service,
so the window counts as open only while now plus `timeout_s` is still inside
it. A connector should declare `lookup_by_key` only when `lookup()` finds a
key for that long: for a service that does not deduplicate, without expiry.

**A same-key retry** is made only while the key's window is open, when
attempts remain under `max_attempts` and when `check_authority()` and the
payload check pass, as for `dispatch()`. `claim_retry()` in
`smythe.workflow_actions` checks again in the journal and claims the retry
as `dispatched` before it is sent. A deduplicating service answers it from
the earlier commit, and that receipt is read back. A refused retry says
nothing about the earlier attempt, so the action stays `outcome_unknown`. A
revoked or expired approval, a security stop, changed content or a closed
window means that nothing is sent.

**Unresolved outcomes.** What reconciliation cannot settle gets an
`unresolved` finding with its reason, and the next action is
`operator_review` until another attempt begins. An acknowledged attempt
whose reads are used up, or no longer fit under the approval's ceiling,
without confirming the receipt is unresolved in the same way. Reconciling
again makes no lookup or read beyond the bounds and records no duplicate
finding.

**Records and accounting.** Each lookup is a `lookup` record holding the
attempt, the lookup's number and the receipt it found, or a failed lookup's
error text, redacted; an answer that names another key, tenant, collection
or object proves nothing. Reads are `readback` records as above. Each lookup
and read is charged the profile's `read_ceiling_nanousd` as confirmed cost
against the action, and none is made that would take the action's
confirmed, reserved and unknown nanoUSD past the ceiling of the attempt's
approval. Lookups and reads share that budget: an attempt dispatched after a
`not_executed` finding has only the room that the lookups left for its
reads, and once no read fits, its next action is `reconcile`, which records
an `unresolved` finding. Findings are `finding` records holding the
attempt, the finding, its reason, the records it rests on and the number of
attempts. Each lookup adds an `action_lookup` event and each finding an
`action_finding` event.
The ambiguous attempt keeps its write ceiling as unknown exposure, even once
the action is reconciled or found not executed, until a receipt that reports
a cost confirms it: state and accounting are separate. No attempt or record
is removed, and no confirmed cost is reduced.

Host code that reconciles by itself calls `record_lookup()`, `claim_retry()`
and `record_finding()` in `smythe.workflow_actions` under the run's lease.

### Notices

A service may report its commits with callbacks. `record_notice(store,
payload)` keeps one that your host relays:

```python
from smythe.actions import record_notice

action = record_notice(store, {"key": key, "target": {"tenant": "acme", "collection": "posts"},
                               "notice": callback_body})
if action is not None:
    runner.readback(action.run_id, action.action_id)
```

The payload is a JSON object of at most 64 KiB as canonical JSON, which names
the action key it reports as `key` and the action's target as `target`;
other fields are kept as they are. A notice whose key names an action of the
journal is kept with that action as an untrusted `notice` record, with an
`action_notice` event; a notice under any other key is not kept. It returns
the action only when the action is `dispatched` and the notice names its
tenant and collection, and None otherwise. A notice never changes an action's
state: anyone can send one, so it can only prompt a readback, which reads the
object the receipt names and decides by itself. A notice for another tenant,
for an action not yet dispatched or for one already settled prompts nothing.
It needs no lease.

### Progress

Every step returns an `ActionProgress`: the `action_id`, its `state`, its
latest `attempt` as `inspect_run()` shows it, the `report` of the step's
last read, or None, and the `next_action`:

| `next_action` | When |
|---|---|
| `dispatch` | The action is `not_dispatched` and attempts remain. |
| `readback` | An acknowledged attempt has reads left that fit under its approval's ceiling. |
| `reconcile` | The outcome is unknown, or the reads did not settle it. |
| `operator_review` | Reconciliation found it can do no more: the outcome stays unknown or unverified. |
| `compensate` | The action is `compensation_required`. |
| None | The action is settled, or has used every attempt. |

Errors from the journal's checks, such as `ActionNotAuthorized`,
`ActionStateError` and `WorkflowBudgetError`, propagate unchanged.
