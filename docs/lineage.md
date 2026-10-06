# Artifact lineage

Artifact lineage records how a durable run's accepted artifact was made: the
run, the published document, the delivery decision, and each node's result
and the inputs it was given, as ids, versions and SHA-256 digests. A run opts
in with a `LineagePolicy`, which also declares the access scope its results
belong to and whether they may be reused.

This release adds the policy, the recipe opt-in and the versioned record
format, `ArtifactLineage`. Completed runs do not write lineage records yet,
and nothing reuses or repairs work from them yet. It also adds
[reuse decisions](#reuse-decisions): a pure evaluator that says what a
record calls for under the current sources, tools and validators.

## Opt a run in

Lineage needs a durable `run_store` and a strict
[acceptance policy](workflow-accounting.md#strict-delivery-acceptance),
because a record binds the run's delivery decision:

```python
from smythe import (
    AcceptancePolicy, LineagePolicy, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore,
    Swarm, Task, TextValidator,
)

with SQLiteWorkflowStore("smythe-runs.db") as store:
    swarm = Swarm(
        model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
        run_store=store,
        acceptance=AcceptancePolicy(mode="strict", final_validators=[TextValidator()]),
        lineage=LineagePolicy(access_scope="research-team"),
    )
    result = swarm.execute(Task("Write the brief"))
    print(store.load_run(result.execution_id)["config"]["lineage"])
```

- **Refused early.** A `Swarm` given `lineage` without a `run_store`, or
  without a strict policy (none, or an advisory one), raises
  `WorkflowBindingError` when it is built, before any routing, planning or
  node call. `Swarm.from_yaml(path, ..., lineage=...)` behaves the same.
- **Recorded only when configured.** The run's recipe records the policy
  under `lineage`, as `{"version": 1, "access_scope": ..., "reuse": ...,
  "undeclared_coverage": ...}`. A recipe without lineage keeps its exact
  bytes, including `"type": "text_workflow"`, `"version": 1` and its
  `verifier`, so existing runs resume as before.
- **Identity, not behavior.** Like `graph_policy`, the policy is compared
  exactly whenever a run is planned, executed, resumed or re-evaluated. A run
  never changes its access scope, reuse mode or coverage rule while in
  flight: resuming it with another policy or without its policy, or resuming
  a run without lineage under a policy, raises `WorkflowConflictError` before
  any call. The policy holds no pricing metadata, so a price release never
  changes it.
- **Configuration only.** The policy comes from the `Swarm` (or
  `WorkflowRuntime`) constructor or the `from_yaml` argument, never from a
  task, a graph, a YAML file, a generated plan or a supervisor revision. A
  new graph whose nodes carry `lineage_reuse` metadata, which marks a result
  reused from another run, is refused like other execution history.

## The policy

| Field | Default | Meaning |
|---|---|---|
| `access_scope` | `"local"` | Who may reuse the run's results; see [access scope](#access-scope). |
| `reuse` | `True` | Whether the run's recorded results may be reused. `False` turns reuse off, so affected work is computed again in full while the lineage stays readable. |
| `undeclared_coverage` | `"review"` | What a run whose lineage does not declare every node's inputs calls for: `"review"` by a person, or `"regenerate"`. |
| `version` | `1` | The policy format. |

`LineagePolicy` is a frozen, keyword-only dataclass. `to_dict()` writes all
four fields and `from_dict()` accepts exactly those keys and version 1;
`sha256` is the digest of `to_dict()` as canonical JSON. A subclass, or
anything that is not a `LineagePolicy`, is refused.

## Access scope

`access_scope` is a declared label: 1-64 ASCII letters, digits, `.`, `_`,
`:` or `-`, starting with a letter or digit, such as `research-team` or
`tenant:acme`. Smythe does not authenticate it. It is configuration, like the
store path, not a credential, and it does not control who can read the
journal file.

Scopes are compared exactly and before any digest. Results are reused only
within one scope: identical content under another scope is never reused,
because a matching hash identifies content, it does not grant access to it.
Give each tenant or team that shares a journal its own scope.

## The record

`ArtifactLineage` (exported from `smythe`) is a versioned record. Its JSON
form:

```text
{
  "version": 1,
  "run": {"store_id", "run_id", "recipe_sha256", "task_sha256"},
  "access_scope": "research-team",
  "artifact": {"relative_path", "sha256", "size_bytes", "media_type",
               "renderer": {"name", "version", "spec_sha256"}} or null,
  "acceptance": {"operation_key", "record_sha256", "decision", "policy_sha256",
                 "validator_set_sha256", "validators": [validator identities]},
  "nodes": [{"id", "generation", "kind", "depends_on", "result_sha256",
             "inputs": {"sources": [{"source_id", "version", "sha256"}],
                        "context": {"snapshot_key", "assembly_version"},
                        "tool": {"name", "version", "decoder_version"},
                        "component": {"name", "version"},
                        "agent_sha256"},
             "coverage": "declared" | "undeclared",
             "reused_from": {"run_id", "generation"}}],
  "coverage": "complete" | "partial",
  "predecessor": {"store_id", "run_id", "lineage_sha256"},
  "repair": {"plan_sha256", "reason"}
}
```

- **Ids, versions and digests only.** A record holds no source text, prompt
  or output. A source reference given with its text is refused. Digests are
  64 lowercase hexadecimal characters; `task_sha256` and a node's
  `result_sha256` may be null for a run without a task or a node without a
  result. The artifact's path is relative to the artifact directory, so the
  record holds no local path; `artifact` is null for a run that published no
  document.
- **The decision.** `acceptance.decision` is a made decision (`accepted`,
  `rejected`, `needs_review` or `indeterminate`). `validators` is the set of
  validator identities, each once, sorted by name, version and configuration
  digest, and `validator_set_sha256` must be
  `smythe.lineage.validator_set_sha256(validators)`, the digest of that list.
- **Nodes.** Nodes follow the graph's order, each id once, and depend only on
  nodes the record lists. `kind` is a short lowercase label for what produced
  the result, such as `model` or `read`. A node is `declared` only when its
  inputs name a context snapshot, a read tool or a component; `coverage` is
  `complete` when every node is declared and `partial` otherwise.
- **Optional blocks.** `context`, `tool`, `component`, `agent_sha256`,
  `reused_from`, `predecessor` and `repair` appear only when present, never
  as null. `predecessor` is another run in the same store, and every
  `reused_from` names it.
- **Strict reading.** `ArtifactLineage.from_dict()` rejects any other
  version and any missing or unknown key at every level, rather than
  guessing. `to_dict()` returns detached JSON data; the record's parts are
  read-only. `sha256` is the digest of `to_dict()` as canonical JSON, so it
  does not depend on key order.

A record grows with its nodes and their source references, not with the
run's calls or events: a 256-node record with two source references per node
stays under 256 KiB.

## Reuse decisions

`smythe.reuse.evaluate(record, state)` decides what one recorded artifact
calls for now. It compares an `ArtifactLineage` record, or None for a run
that has none, with a `CurrentState`, and returns an `ItemDecision`: one
action for the item, one for each node, the [reason codes](#reason-codes)
behind them, and the authority under which anything is kept.

```python
from smythe import CurrentState, LineagePolicy, SourceStatus
from smythe.reuse import evaluate


class LocalIndex:
    """Describes each source from a local index: no network, no cost."""

    def __init__(self, entries):
        self.entries = entries  # source id -> (version, sha256)

    def describe(self, source_id):
        if source_id not in self.entries:
            return None
        version, sha256 = self.entries[source_id]
        return SourceStatus(version=version, sha256=sha256)


state = CurrentState(
    catalog=LocalIndex(entries),
    validators=acceptance.validator_identities,  # the strict AcceptancePolicy
    policy_sha256=acceptance.sha256,
    tools={"handbook": {"version": "1", "decoder_version": "utf8-text-v1"}},
    components={"brief-writer": "2"},
    agents=agent_digests,
    integrity={run_id: "verified"},
    lineage=LineagePolicy(access_scope="research-team"),
)
decision = evaluate(record, state)
print(decision.action, decision.reasons)
```

The evaluator is a pure function. It opens no store, reads no file and
calls nothing but the catalog's `describe`; equal inputs give equal
decisions. Nothing applies a decision yet: it says what a record calls for.

### The current state

`CurrentState` is a frozen, keyword-only dataclass describing what a run
would use now. Each part is checked when it is built and kept read-only.

| Field | Default | Meaning |
|---|---|---|
| `catalog` | required | The [source catalog](#the-source-catalog). |
| `validators` | required | The final validators' `ValidatorIdentity` values, such as a strict policy's `validator_identities`. They are kept as a sorted set, so order and repeats do not matter, and `validator_set_sha256` is their digest as a record holds it. |
| `policy_sha256` | required | The acceptance policy's `sha256`. |
| `tools` | `{}` | Each read tool's name mapped to `{"version": ..., "decoder_version": ...}`. |
| `components` | `{}` | Each component's name mapped to its version. |
| `context_assembly_version` | the library's | The context assembly a run would use; None for none. |
| `agents` | none | The digests of the agent descriptions a run would use. |
| `tasks` | `{}` | Optional task changes: a recorded task's digest mapped to the digest of the task that replaces it. |
| `integrity` | `{}` | A run id mapped to its published document's integrity `status`: `verified`, `missing`, `changed`, `unsafe`, `unreadable` or `not_checked`, as [`smythe runs inspect`](runs.md#published-documents) reports it. |
| `lineage` | `LineagePolicy()` | The configured policy, whose `access_scope`, `reuse` and `undeclared_coverage` apply. |

A price release changes none of these, so it never changes a decision.

### The source catalog

A `SourceCatalog` is developer code with a synchronous
`describe(source_id)` that returns a `SourceStatus`, or None for a source it
does not know. It must be a zero-cost local read, such as a lookup in a
local index, never a network fetch or a paid call. A catalog that declares
`metered` as anything but `False` is refused with `ValueError`; a class, an
object without a synchronous `describe`, or an `async` one is refused with
`TypeError`, and so is an answer that is not a `SourceStatus` or None (a
returned coroutine is closed first).

`SourceStatus(version=None, sha256=None, available=True,
tombstone_reason=None)` is frozen and keyword-only. An available source has
the `sha256` of its current content, a `version` when the catalog versions
it, and no tombstone reason. An unavailable one may name why it was withdrawn
in `tombstone_reason`.

Each source is described once per evaluation, in the record's order. A
missing record, or one in another scope, describes none.

### Actions

From least to most conservative, the actions are:

| Action | Meaning |
|---|---|
| `reuse` | The recorded result stands. |
| `recheck` | The recorded output needs a new acceptance decision under the current validators and policy; no node runs. |
| `rerender` | The recorded synthesis needs to be rendered and published again, with a new decision on the new bytes; no node runs. |
| `regenerate` | The node runs again. For the item, a new run can keep the nodes marked `reuse`. |
| `review` | A person decides. |

`smythe.reuse.ACTIONS` lists them in this order. The item takes the most
conservative action that any node or item rule calls for, and each node the
most conservative of its own reasons.

### Rules

| Condition | Action | Reason |
|---|---|---|
| The record's `access_scope` is not the state's | none: `evaluate` returns None | |
| No lineage record | `undeclared_coverage` | `lineage_missing` |
| A source the catalog does not know, or reports unavailable | `review` | `source_unavailable` |
| A source the catalog reports withdrawn | `review` | `source_tombstoned` |
| A source whose version differs, including the same bytes under a new version | `regenerate` | `source_version_changed` |
| A source whose digest differs | `regenerate` | `source_changed` |
| A read tool whose version differs, or that is not listed | `regenerate` | `tool_changed` |
| A read tool whose decoder version differs | `regenerate` | `decoder_changed` |
| A component whose version differs, or that is not listed | `regenerate` | `component_changed` |
| A context assembly version that differs | `regenerate` | `context_assembly_changed` |
| An agent digest that is not current | `regenerate` | `agent_changed` |
| A replaced task, for every node | `regenerate` | `task_changed` |
| A node without a result, which did not complete | `regenerate` | `node_incomplete` |
| A node whose inputs are undeclared | `undeclared_coverage` | `coverage_undeclared`, and `coverage_partial` on the item |
| A node that depends on one that regenerates | `regenerate` | `dependency_changed` |
| A node that depends on one under review | `review` | `dependency_review` |
| A gate, and each node it depends on, in an item where any node regenerates or needs review | `regenerate` | `gate_pair` |
| A published document that is missing, or holds other bytes | `rerender` | `artifact_missing`, `artifact_changed` |
| A published document that cannot be checked safely, cannot be read, or has no `verified` finding | `review` | `artifact_unsafe`, `artifact_unreadable`, `artifact_unchecked` |
| A recorded decision other than `accepted` | `review` | `decision_not_accepted` |
| A changed validator set | `recheck` | `validator_set_changed` |
| A changed acceptance policy | `recheck` | `policy_changed` |
| `reuse=False`, for each node that would be kept, unless the item is a `recheck` | `regenerate` | `reuse_disabled` |

- **Scope first.** Scopes are compared before any digest. A record in
  another scope is invisible, however exactly its digests match.
- **Identity, not bytes.** A source is current only when its id, version and
  digest all match the record. Equal bytes under a new version, or a version
  where the record had none, regenerate.
- **Dependents.** A node that regenerates takes every node that depends on
  it, directly or through others, as a run does when it resets a node; a
  node under review does the same with `dependency_review`. Nodes on other
  branches keep `reuse`, so one changed source regenerates only its own
  read, its extraction and what depends on them.
- **Gates.** A node of kind `gate` is a verification gate, and its lineage
  node depends on the node it judges. When an item computes anything again,
  no gate and no node a gate depends on is reused. An item that computes
  nothing keeps its gates' results.
- **Nothing computed, item rules decide.** When no node regenerates or needs
  review, the published document, the recorded decision, and the validator
  set and policy decide, and every node takes the item's action: `reuse`,
  `recheck` or `rerender`. A run without a document needs no integrity
  finding. When any node regenerates, the item never rerenders: a changed
  premise is not fixed by rendering again.
- **Coverage.** A missing record, or an undeclared node and its dependents,
  get what the policy's `undeclared_coverage` says: `review` by default, or
  `regenerate`.
- **The reuse switch.** With `LineagePolicy(reuse=False)`, `reuse` and
  `rerender` become `regenerate` with `reuse_disabled`, so a whole item is
  computed again; `recheck` is unchanged and `review` stays `review`.

### The decision

`ItemDecision` is frozen: `action`, `reasons` (every code behind any action,
sorted), `nodes` (`NodeDecision` values with `id`, `action` and sorted
`reasons`, in the record's order) and `authority`. `to_dict()` returns JSON
data in a fixed order:

```text
{
  "action": "regenerate",
  "reasons": ["dependency_changed", "source_changed", "source_version_changed"],
  "nodes": [{"id": "read_a", "action": "reuse", "reasons": []},
            {"id": "read_b", "action": "regenerate",
             "reasons": ["source_changed", "source_version_changed"]}, ...],
  "authority": {"access_scope", "recipe_sha256", "policy_sha256",
                "validator_set_sha256", "sources": [{"source_id", "version", "sha256"}]}
}
```

`authority` names what kept results are kept under: the access scope, the
record's `recipe_sha256`, the current `policy_sha256` and
`validator_set_sha256`, and each source reference, once and sorted, that a
node with `reuse`, `recheck` or `rerender` read. A missing record has a null
`recipe_sha256` and no sources. Decisions are deterministic: the same record
and state give the same JSON, whatever order the state's validators, agents
or mappings were given in.

### Reason codes

These codes are stable. `smythe.reuse.REASONS` maps each to the action it
calls for, or None where the policy's `undeclared_coverage` decides.

| Code | Action | Meaning |
|---|---|---|
| `lineage_missing` | `undeclared_coverage` | The run has no lineage record. |
| `coverage_partial` | `undeclared_coverage` | The record does not declare every node's inputs. |
| `coverage_undeclared` | `undeclared_coverage` | This node's inputs are undeclared. |
| `source_unavailable` | `review` | A source is unknown to the catalog or unavailable. |
| `source_tombstoned` | `review` | A source was withdrawn. |
| `dependency_review` | `review` | A node this one depends on needs review. |
| `decision_not_accepted` | `review` | The recorded decision is not `accepted`. |
| `artifact_unsafe` | `review` | The published document cannot be checked safely. |
| `artifact_unreadable` | `review` | The published document cannot be read. |
| `artifact_unchecked` | `review` | The published document has no `verified` finding. |
| `source_changed` | `regenerate` | A source's digest differs. |
| `source_version_changed` | `regenerate` | A source's version differs. |
| `tool_changed` | `regenerate` | A read tool's version differs, or it is gone. |
| `decoder_changed` | `regenerate` | A read tool's decoder version differs. |
| `component_changed` | `regenerate` | A component's version differs, or it is gone. |
| `context_assembly_changed` | `regenerate` | The context assembly version differs. |
| `agent_changed` | `regenerate` | The node's agent description is not current. |
| `task_changed` | `regenerate` | The run's task was replaced. |
| `node_incomplete` | `regenerate` | The node has no result. |
| `dependency_changed` | `regenerate` | A node this one depends on regenerates. |
| `gate_pair` | `regenerate` | A gate, or a node a gate depends on, in an item that computes something again. |
| `reuse_disabled` | `regenerate` | The policy's `reuse` is off. |
| `artifact_missing` | `rerender` | The published document is missing. |
| `artifact_changed` | `rerender` | The published document holds other bytes. |
| `validator_set_changed` | `recheck` | The validator set differs. |
| `policy_changed` | `recheck` | The acceptance policy differs. |
