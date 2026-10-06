# External actions (preview)

**Preview.** `smythe.actions` defines the contract types for actions that
write accepted content to a service outside Smythe, such as publishing a post.
It [prepares actions](#prepared-actions) in a durable run's
[journal](workflow-accounting.md#external-action-records), bound to the run's
accepted content, and records [verified approvals](#approvals) of them. Smythe
does not yet dispatch actions, and these types and functions may change
before it does. `import smythe` does not load the module; import
`smythe.actions` explicitly.

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
| `lookup_by_key` | `False` | `lookup()` finds an earlier submission by its key. |
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
| `not_dispatched` | No attempt is in flight. |
| `dispatched` | An attempt was sent and its result is not yet known. |
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
as UTF-8 otherwise. The intent must name the profile's connector and one of
its operations. A refusal raises `ActionNotAuthorized` and records nothing.

`prepare_action()` returns a `PreparedAction`: its `action_id`, `run_id`,
`slot`, `sequence`, `action_key`, `intent`, `profile`, `state` and
`ceiling_nanousd`, the most its attempts and readback may cost:
`max_attempts` writes and `readback_reads` reads at the profile's ceilings.
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
no longer be approved. Rejecting it again changes nothing.

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
- the content is still deliverable, as for `prepare_action()`: the run's
  current decision is still `accepted`, the published file still holds the
  bound bytes, and no source the accepted output came from has been
  [removed](tasks.md#remove-a-source) since, which makes the
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
