# External actions (preview)

**Preview.** `smythe.actions` defines the contract types for actions that
write accepted content to a service outside Smythe, such as publishing a post.
Smythe does not yet store, approve or dispatch actions, and these types may
change before it does. `import smythe` does not load the module; import
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
now_ns)` reports whether that principal may still approve the intent.

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
