# Artifact lineage

Artifact lineage records how a durable run's accepted artifact was made: the
run, the published document, the delivery decision, and each node's result
and the inputs it was given, as ids, versions and SHA-256 digests. A run opts
in with a `LineagePolicy`, which also declares the access scope its results
belong to and whether they may be reused.

This release adds the policy, the recipe opt-in, the versioned record
format, `ArtifactLineage`, and the record each opted-in run
[writes when it completes](#the-record-a-run-writes), with its
[coverage](#coverage). It also adds [reuse decisions](#reuse-decisions):
a pure evaluator that says what a record calls for under the current
sources, tools and validators, [repair plans](#plan-a-repair), which
show before anything is dispatched what each run of an archive calls for
and the work that would take, [repair in place](#repair-a-run-in-place),
which decides a run again or publishes its damaged document again without
running a node, the records of a
[successor run](#reuse-in-a-successor-run) that reuses a completed run's
results, [applying a plan](#apply-a-repair-plan) under an explicit
allowance, and [source restriction](#restrict-a-source), which withdraws a
source's text from every run that used it and can delete their published
documents.

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
  exactly whenever a run is planned, executed, resumed or re-evaluated,
  except that a re-evaluation, which reuses nothing, ignores `reuse`. A run
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
  or output. A source reference given with its text is refused, and a node
  lists each reference, its id, version and digest, once. Store, run, node
  and source ids are never blank, that is, only whitespace. Digests are
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
- **Nodes.** Each id is listed once, after every node it depends on, so a
  record holds no dependency cycle, and a node depends only on nodes the
  record lists. `kind` is a short lowercase label for what produced
  the result: a run writes `read`, `model`, `gate` and `synthesis`. A node
  is `declared` only when its inputs name a context snapshot, a read tool or
  a component; `coverage` is `complete` when every node is declared and
  `partial` otherwise.
- **Optional blocks.** `context`, `tool`, `component`, `agent_sha256`,
  `reused_from`, `predecessor` and `repair` appear only when present, never
  as null. `predecessor` is another run in the same store, and every
  `reused_from` names it.
- **Strict reading.** `ArtifactLineage.from_dict()` rejects any other
  version and any missing or unknown key at every level, rather than
  guessing. `to_dict()` returns detached JSON data; the record's parts are
  read-only. `sha256` is the digest of `to_dict()` as canonical JSON, so it
  does not depend on key order. Records compare by value and hash by
  `sha256`, so they can be set members and dictionary keys.

A record grows with its nodes and their source references, not with the
run's calls or events: a 256-node record with two source references per node
stays under 256 KiB.

## The record a run writes

A run with a lineage policy writes its record once, when it completes:
after its synthesis, its [published document](workflow-accounting.md#published-documents)
and its decision, it journals the record as a `lineage` operation, and the
completed checkpoint applies all of them together. Read it with
`smythe.lineage.load_lineage`:

```python
from smythe.lineage import load_lineage

lineage = load_lineage(store, result.execution_id)
print(lineage.coverage, [(node["id"], node["kind"]) for node in lineage.nodes])
```

- **What it binds.** `run` names the store, the run, the digest of its
  recipe and of its task. `artifact` is the published document: its path
  under `artifact_dir` (`<run id>/<file name>`), the SHA-256 and size of the
  published bytes, its media type, and the renderer's name, version and the
  digest of its recipe description; it is null without a renderer.
  `acceptance` is the decision the run completed with: the key of its
  `acceptance` operation, the digest of the decision record, the decision,
  the policy digest and the policy's final validators.
- **Nodes.** The graph's nodes come first, each after the nodes it depends
  on and otherwise in graph order, at their final generations, each with the
  digest of its result:
  - A read step (`read`) names the read profile's name, version and decoder
    version, and the source its accepted read returned. Text that is exactly
    one of the run's `context_sources`, under the `source_id` the step's
    input names when it names one, is that source at its version. Other text
    read under a named `source_id` is that source with a null `version` and
    the digest of the text read.
  - A model node (`model`) or a gate (`gate`) names the context snapshot it
    was given, by its journal key (`context/<node>/<generation>`) and
    context-assembly version, the run's sources the snapshot held, and
    `agent_sha256`, the digest of its agent's registry description without
    the agent's id.
  - The synthesis (`synthesis`) comes last and depends on every node. Its id
    is the synthesis operation's key, its result the digest of the
    synthesized text, and a synthesizer that calls no model is its
    `component`: a `LocalOnly` synthesizer by its identity and version, a
    built-in one as `synthesizer/<strategy>`, version `1`, with
    `+retain_contradictions` when set.

  A label longer than 256 characters, or with control characters, is
  recorded as `sha256:` and the digest of its text. A lineage run refuses,
  before any call, node and source ids that a record cannot hold: ids that
  are only whitespace, and ids with DEL (U+007F), which the journal accepts.
- **Written once.** The record is assembled from targeted loads of the
  records it names (the synthesis, decision and publication operations, each
  node's snapshot and the read behind each read step), never from the whole
  journal, so checkpoint saves cost what they did. Its operation is keyed
  `lineage` while every node is at generation 0 and `lineage/<digest>` after
  that, following the synthesis key, so a run whose nodes ran again never
  replays an earlier graph state's record. A process lost before the record
  commits writes it on resume, without a provider call or a read; one lost
  after it commits and before the completed checkpoint replays it, never
  assembling it again.
- **Checked on resume.** Resuming a completed run assembles its current
  record again from the records its operation names; a record that is
  missing, has another digest or is not the record of the run's current
  decision and publication makes the decision `indeterminate`, and the
  output moves to `rejected_output`. Each later decision, by re-evaluation
  or republication, journals a record that
  [supersedes](#repair-a-run-in-place) the one before.
- **Summarized in inspection.** `inspect_run()`, `list_runs()` and journal
  events never return the record. `smythe runs inspect` shows a
  [summary](runs.md#lineage) by ids, versions and digests: its coverage,
  source references, predecessor and successors, the published document's
  integrity and the run's [derived status](#derived-status). Operation kinds
  are free text, so lineage needs no journal upgrade.

## Coverage

A node is `declared` only when the record knows its inputs:

- a model node or gate given a context snapshot at its final generation
  whose dependency outputs are the results the record lists for those
  dependencies, or, for a node that
  [exchanges claims](architecture.md#claim-exchange), the compact claim
  records of those results;
- a read step with an accepted read at its final generation whose source
  can be named;
- the synthesis, when its synthesizer calls no model.

Every other node is `undeclared`, which makes the run's coverage `partial`.
Model nodes of a run without `context_sources` are given no snapshot, so
such a run's lineage is partial. Coverage never changes the run's decision;
the policy's `undeclared_coverage` says what a later reuse of a partial
record calls for.

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
decisions. It only says what a record calls for;
[applying a plan](#apply-a-repair-plan) carries decisions out.

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
| `run_in_progress` | `review` | The run has not completed, so it has no record yet; only a [repair plan](#plan-a-repair) gives it. |
| `repair_in_progress` | `review` | A successor that has not completed is repairing the run; only a [repair plan](#plan-a-repair) gives it. |
| `outcome_unknown` | `review` | A Jobs provider call's outcome is unknown: it may have completed and been charged. Only the [Jobs repair preview](jobs.md#preview-repairs) gives it. |

## Plan a repair

`Swarm.plan_repair(execution_ids=None, *, sources)` shows, before anything
is dispatched, what each run's accepted artifact calls for now and what
carrying that out would take. *sources* is a
[source catalog](#the-source-catalog):

```python
with SQLiteWorkflowStore("smythe-runs.db") as store:
    # The configuration a new run would use: its policy, read profiles,
    # synthesizer and context assembly, under the archive's access scope.
    swarm = Swarm(
        model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
        run_store=store, acceptance=acceptance,
        lineage=LineagePolicy(access_scope="research-team"),
    )
    plan = swarm.plan_repair(sources=LocalIndex(entries))
    for item in plan.to_dict()["items"]:
        print(item["run_id"], item["status"], item["action"], item["work"]["new_runs"])
    print(plan.plan_sha256)
```

- **Read only.** Planning needs a durable `run_store` and a lineage policy,
  or raises `ValueError`. It opens the journal again read-only, so it takes
  no lease, writes and migrates nothing, and works on a store opened
  read-only or while a worker holds a run; the journal's database file is
  unchanged. It calls no provider and reads no source: the catalog's
  `describe` is called once per source for the whole plan, and the catalog
  is checked as the evaluator checks it.
- **Which runs.** By default, every [head](#archive-heads) in the Swarm's
  access scope, in journal order. `execution_ids` names runs instead; each
  is planned once, in the order given, and a run in another scope, a run
  without lineage and an unknown id each raise `WorkflowNotFoundError`, so
  another scope's runs stay invisible. A single string is refused with
  `TypeError`.
- **The current state.** Each run is [evaluated](#reuse-decisions) against
  what a new run of this Swarm would use: its strict policy's final
  validators and digest, its read profiles' names, versions and decoder
  versions, its synthesizer, its context assembly version and the agents
  its registry defines. An agent that a run recorded and the registry does
  not define, such as one assigned when the run was planned, stays
  current. Each completed run's published document is checked through
  `current_artifact`, the accessor resume and inspection use.
- **Runs in progress.** A run that has not completed has no record yet. Its
  item needs `review` with `run_in_progress`, and has no status, no nodes
  and no work.
- **Repairs in progress.** A run that a successor which has not completed
  is repairing, such as one a lost process left, needs `review` with
  `repair_in_progress` added to its decision's reasons, so the plan never
  starts that repair again. Its item keeps its status and nodes, predicts
  no work, and names the successors in `repairs_in_progress`; resuming one
  finishes the repair.

`smythe.reuse.repair_plan(store, recipe, catalog, run_ids=None)` does the
same for a `WorkflowRuntime`'s `recipe` and a store the caller opened.

### Archive heads

`archive_heads(store, scope)` (exported from `smythe`) returns the ids of
the runs in access scope `scope` that nothing supersedes, in journal order.
A run is in a scope when its recipe records a `LineagePolicy` with that
`access_scope`; runs in other scopes and runs without lineage are never
listed. Links come from the records: a run's `predecessor` names the run it
repairs.

- A completed successor whose current decision is `accepted` supersedes its
  predecessor, which is no longer a head.
- A successor whose decision is not `accepted` supersedes nothing and is not
  a head itself, so its predecessor stays the head.
- A successor that has not completed, which its saved graph's `repair`
  record links to its predecessor, is a repair in progress: it supersedes
  nothing and is not a head, and resumes as any durable run does.
- Any other run that has not completed has no record yet, so it supersedes
  nothing and is a head.
- A run whose record or decision cannot be read links nothing.

### The plan

`RepairPlan` (exported from `smythe`) is frozen. `to_dict()` returns:

```text
{
  "version": 1,
  "store_id": "...",
  "access_scope": "research-team",
  "sources": [{"source_id", "version", "sha256", "available", "tombstoned"}],
  "items": [{"run_id", "lineage_sha256", "status", "action", "reasons",
             "nodes": [{"id", "action", "reasons"}], "authority": {...},
             "work": {"nodes_reused": {kind: count}, "nodes_regenerated": {kind: count},
                      "read_calls", "model_steps", "publications",
                      "acceptance_evaluations", "new_runs",
                      "may_incur_charges", "allowance_required"},
             "repairs_in_progress": [run ids]}],
  "plan_sha256": "..."
}
```

- **Items.** One per planned run. `action`, `reasons`, `nodes` and
  `authority` are the evaluator's [decision](#the-decision);
  `lineage_sha256` names the record it was made from (null without one) and
  `status` is the run's [derived status](#derived-status).
  `repairs_in_progress` appears only on a run that a
  [repair in progress](#plan-a-repair) holds.
- **Sources.** What the catalog said about each source the planned records
  name, sorted by id. A source the catalog does not know is `available:
  false` with a null `version` and `sha256`.
- **The digest.** `plan_sha256` is the SHA-256 of the rest of `to_dict()` as
  canonical JSON, which holds no time and no text. Planning the same
  archive under the same state gives the same digest, and any changed
  decision or catalog answer changes it, including the same bytes under a
  new version. It is a staleness guard that applying a plan must match; it
  authorizes nothing.

### Predicted work

`work` predicts what carrying out an item takes, from its record alone;
nothing is called to predict it.

| Action | Predicted work |
|---|---|
| `reuse` | Every node is kept (`nodes_reused`); nothing else |
| `recheck` | Every node is kept; one acceptance evaluation |
| `rerender` | Every node is kept; one publication and one acceptance evaluation |
| `regenerate` | One new run that keeps the nodes marked `reuse` and runs the others (`nodes_regenerated`): a read call for each read step it runs, a model step for each model node and gate and for a synthesis without a local component; one publication when the run published a document; one acceptance evaluation |
| `review` | Nothing: a person decides |

`may_incur_charges` is true for a new run that makes a model step, a read
whose profile is not declared `metered=False` now, or whose record is
missing. `allowance_required` is true for every item that starts a new run,
which needs admission under an explicit allowance. The counts are what the
record shows; retries and regenerations within a run can add calls.

### Derived status

A run's lifecycle status is computed from what is found for it each time it
is needed and is never stored. `smythe.reuse.derived_status(reasons, *,
superseded=False, sources_checked=True)` gives the first that applies:

| Status | When |
|---|---|
| `superseded` | A completed successor whose current decision is `accepted` names the run as its predecessor |
| `restricted` | A source the run used was withdrawn: the catalog reports it tombstoned, or a tombstone in the run's journal withdraws a text its record names (see [restrict a source](#restrict-a-source)) |
| `invalid` | The published document is missing, changed, unsafe or unreadable, or the decision is not `accepted` |
| `current` | Nothing calls for work: the item is reused as it is |
| `stale` | Anything else does: an input, validator or policy changed, coverage is partial or a source is unknown |

A run that has not completed has no status. `smythe runs inspect` has no
source catalog, which is application code, so it never reports `current`
or `stale`: without one of the first three statuses, it reports none.
`smythe.reuse.STATUSES` lists the five.

## Repair a run in place

The `recheck` and `rerender` actions repair a completed run where it is. No
node runs, no source is read and no provider is called; the budget, retry
and regeneration counters stay as they are; and every earlier publication,
decision and lineage record stays in the journal.

- **Recheck.** A changed validator never inherits the decision made under
  the old one. `swarm.reevaluate(run_id, acceptance=policy)`
  [decides the saved document again](verifier.md#re-evaluation) under the
  new policy, as a new decision.
- **Rerender.** When the published document is missing or changed,
  `swarm.republish(run_id)` renders the run's saved synthesis again with
  the run's renderer, publishes the bytes as a new file named for the new
  publication (`document-p1.md`, then `document-p2.md`), and decides those
  bytes again under the `Swarm`'s policy. The damaged file stays as it is.

```python
swarm.republish(run_id)                      # the document was deleted or edited
swarm.reevaluate(run_id, acceptance=policy)  # a validator's version changed
```

- **Checked first.** Rerendering needs a completed run whose recipe differs
  from the `Swarm`'s only in its acceptance policy and gate verifier: a
  changed renderer or synthesizer raises `WorkflowConflictError`, because a
  document made another way is regenerated in a successor run. The saved
  decision must still hold, the saved synthesis must have the digest the
  run's lineage records, and the lineage must match its journal; otherwise
  `WorkflowStateError` is raised. A document whose file still holds its
  bytes is refused as having nothing to repair, and one that cannot be
  checked safely or read is left for review. A refusal writes nothing.
- **Never replaced.** The new file is published without replacing any file.
  A file under its name that already holds the same bytes, such as one a
  lost process published, is reused; one holding other bytes is refused.
- **Never copied.** The new bytes usually equal the earlier ones, and they
  are decided again anyway. The decision is a new `acceptance` operation
  whose inputs bind the new publication's sequence, so it is never the
  decision made for an earlier publication of the same bytes.
- **Lineage superseded.** Every new decision on a lineage run, whether a
  re-evaluation, an [operator's decision](verifier.md#operator-decisions) or
  a republication, journals a record keyed `lineage/<sequence>` after the
  decision's sequence (`lineage/<digest>/<sequence>` for a run whose nodes
  ran again). It names the new decision, the current publication and the
  validators of the policy that decided; an operator's decision keeps those
  of the decision it resolved. The operation's inputs name `supersedes`, the
  digest of the record before it. One completed checkpoint applies the
  publication, the decision and the record together, and `load_lineage`
  returns the new record, so a plan made afterwards finds nothing left to
  recheck or rerender.
- **The plan's reason.** A repair [applied from a plan](#apply-a-repair-plan)
  binds `repair`, the plan's `plan_sha256` and its reason, into the inputs of
  the publication, the decision and the lineage record. A run without a
  lineage policy records none and republishes as before: the same bytes
  under a free name, with its decision unchanged.
- **Recovery.** A process lost before the completed checkpoint leaves the
  run as it was, its damaged document still current. Calling again with the
  same arguments finishes the same publication and decision, reusing a file
  already published, rather than adding others.

[`smythe runs inspect`](runs.md#published-documents) lists the publications
a republication superseded under `artifact.superseded`.

## Reuse in a successor run

A successor of a completed run is a new run of the same task and graph that
keeps some of the predecessor's node results, each through a `reuse` journal
operation, and runs every other node again
([reused results](workflow-accounting.md#reused-results)). It reuses only
from a run in the same store and the same access scope, compared before any
digest, and only nodes the predecessor's record declares, whose saved
results still have their recorded digests and whose sources were not removed.
Gates, the nodes they judge or depend on, and nodes that depend on a node
that runs again are never kept. The successor holds its predecessor's source
tombstones from its first checkpoint, so text
[withdrawn](#restrict-a-source) from the predecessor is withdrawn from it
too. [Applying a plan](#apply-a-repair-plan) starts one for each
`regenerate` item.

When the successor completes, its record shows what it kept:

- `predecessor` names the predecessor's `store_id`, `run_id` and the
  `lineage_sha256` of the record the reuse was checked against;
- `repair` holds the `plan_sha256` of the repair plan and its `reason`;
- each kept node copies the `kind`, `inputs` and `coverage` the predecessor
  recorded for it, has the same `result_sha256`, and adds `reused_from`: the
  predecessor's `run_id` and the node's `generation` there. Its own
  `generation` and `depends_on` are the successor's.

Nodes that ran again are recorded from the successor's own snapshots and
reads. A kept node is always `declared`, because a node whose inputs are
undeclared is never kept.

## Apply a repair plan

`Swarm.repair(plan, *, plan_sha256, max_total_budget_usd, reason, sources)`
applies a [plan](#plan-a-repair) that a person reviewed, item by item, and
returns a `RepairResult`; `await swarm.arepair(...)` does the same in a
running event loop. *plan_sha256* is the digest the plan was approved by,
*max_total_budget_usd* the allowance for the new runs it starts, *reason*
(1-256 characters without control characters) why it is applied, and
*sources* the same [source catalog](#the-source-catalog) the plan was made
with:

```python
plan = swarm.plan_repair(sources=LocalIndex(entries))
approved = plan.plan_sha256  # after a person has reviewed plan.to_dict()
result = swarm.repair(plan, plan_sha256=approved, max_total_budget_usd=5.0,
                      reason="The handbook has a new edition",
                      sources=LocalIndex(entries))
for item in result.to_dict()["items"]:
    print(item["run_id"], item["action"], item["outcome"], item.get("successor_run_id"))
```

- **Never stale.** The plan is made again for the same runs, from the
  catalog's answers now, and must still have `plan_sha256`. A source, run,
  document or configuration that changed after the preview changes the
  digest, and the plan is refused with `WorkflowConflictError`; make and
  review a new one. *plan_sha256* must also be the given plan's own digest
  (`ValueError`).
- **Durable only.** Repair needs a durable `run_store` and a lineage policy,
  or raises `ValueError`, so the ordinary path, which keeps no lineage, is
  refused. A configured `supervisor` is refused with `WorkflowBindingError`,
  because a revision could change the graph a successor keeps results in.
- **An explicit allowance.** Each `regenerate` item starts one new run under
  the Swarm's `max_budget_usd`, which the run's journal enforces. Their
  number times `max_budget_usd` must not exceed *max_total_budget_usd*, and
  an unbounded `max_budget_usd` is refused when any of them
  [may incur charges](#predicted-work); both raise `WorkflowBudgetError`.
  Items that repair a run in place call nothing and need no allowance, so
  `max_total_budget_usd=0` applies a plan without new runs.
- **Checked first.** Every refusal above comes before anything is written,
  read or called, and before any run is created.

Items are applied in plan order, each once:

| Action | What applying it does |
|---|---|
| `reuse` | Nothing. |
| `recheck` | [Decides the run again](#repair-a-run-in-place) under the Swarm's policy, binding the plan's digest and reason. |
| `rerender` | [Publishes the run's document again](#repair-a-run-in-place) and decides the new bytes, binding the plan's digest and reason. |
| `regenerate` | Starts a [successor run](#reuse-in-a-successor-run) of the run's task and graph that keeps each node the plan marks `reuse` and runs the others, and records the plan's digest and reason in its lineage. |
| `review` | Nothing: it is reported for a person. |

- **One retry owner.** An item whose step raises, or ends with a decision
  other than `accepted`, stops the repair. It is never retried, and every
  later item is reported `unapplied`. Retries happen only inside each
  successor run, under its own node and retry policies.
- **Earlier versions stay.** A successor never changes the run it repairs:
  that run's calls, balances, events, checkpoints and published document
  stay as they were, and each successor's calls and costs are its own. When
  the successor completes with an `accepted` decision it
  [supersedes](#archive-heads) the run; one whose decision is not accepted
  supersedes nothing.

### The result

`RepairResult` (exported from `smythe`) is frozen. `to_dict()` returns:

```text
{
  "plan_sha256": "...",
  "items": [{"run_id", "action", "outcome", "decision", "successor_run_id", "error",
             "work": {"new_runs", "read_calls", "model_calls", "reused_nodes",
                      "publications", "acceptance_evaluations"}}],
  "successor_run_ids": [run ids],
  "decisions": {run id: delivery acceptance record},
  "work": {the items' work, summed}
}
```

| Outcome | Meaning |
|---|---|
| `kept` | A `reuse` item. |
| `reported` | A `review` item, left for a person. |
| `applied` | The item's step ended with an `accepted` decision. |
| `failed` | The step raised, named in `error` as the exception's type and message, or ended with another decision. |
| `unapplied` | An item after a failure. |

An item that decided names its `decision`, and `decisions` maps the run that
holds each decision, the run itself or its successor, to its
`DeliveryAcceptance`. A `regenerate` item names the `successor_run_id` of
the run it created, even one that then failed. `work` counts what each item
added to the journal: the new runs it started, and the read calls, model
calls, `reuse` operations, publications and acceptance decisions of the run
it repaired in place or the successor it started. Offline runs cost nothing,
so these counts, not money, compare a selective repair with computing again
in full.

### A repair in progress

A successor that a lost process left, or that failed, is a repair in
progress: it is not an [archive head](#archive-heads), and a plan made
afterwards gives the run it repairs `review` (`repair_in_progress`) and names
the successor in `repairs_in_progress`, so applying a plan never starts that
repair again. Resume the successor like any durable run with
`swarm.resume(successor_run_id)`: the reuse operations it committed are
replayed rather than run again, and the nodes, reads and model calls it
completed are not repeated. Once it completes with an `accepted` decision,
it is the head.

### The reuse switch

`LineagePolicy(reuse=False)`, under the same access scope, turns reuse off
for new work without touching what was recorded. Its plans
[regenerate](#rules) every item that is not a `recheck` or a `review`, with
`reuse_disabled`, so applying one computes each such item again in full: no
`reuse` operation is written, every node of each successor runs, and the
documents are byte for byte those of a new run. A `recheck` item is decided
again as before, since a decision reuses nothing; rendering a document again
from its saved synthesis is reuse, so `republish` refuses a run recorded
under another `reuse`. Every lineage record stays readable, and turning reuse
on again makes later plans keep results again.

## Restrict a source

When a source's retention ends, or access to it is revoked, withdraw its
text from every run that used it with `smythe.lineage.restrict_source`:

```python
from smythe.lineage import restrict_source

with SQLiteWorkflowStore("smythe-runs.db") as store:
    restricted = restrict_source(store, "handbook", scope="research-team",
                                 reason="withdrawn by its owner", purge=True)
    for item in restricted:
        print(item["run_id"], [record["sha256"] for record in item["tombstones"]],
              item["purged"])
```

- **Which runs.** Every completed run whose recipe records a lineage policy
  and whose applied record names the source as any node's input, in journal
  order. `version` limits it to the references recorded with that version.
  `scope` limits it to the runs in that access scope, compared before any
  record or digest is read, so a run in another scope is never touched
  however exactly its digests match; without it, every scope is covered.
- **What it records.** Each such run journals a tombstone of each withdrawn
  text under the run's lease: the same `SourceRecord` and
  `source_tombstone` operation that
  [removing a source](tasks.md#remove-a-source) writes, with the source's id,
  version and SHA-256, the reason (1-500 characters) and `removed_at` from
  the journal's clock (the store's `clock_ns`), and none of the text. A copy
  of the run's completed checkpoint applies it; nothing is called. A text a
  run already holds a tombstone of keeps its first one, so calling
  `restrict_source` again, for example after another process released a run
  whose lease it held (`WorkflowLeaseError`), finishes a restriction without
  changing what it recorded.
- **Purge.** With `purge=True`, every file the run published is deleted,
  never through a symbolic link or junction, and an `artifact_purge`
  operation, keyed `artifact-purge/<publication sequence>`, records its path
  under the artifact directory, the SHA-256 its publication recorded, the
  reason and the time; `smythe.lineage.purged_artifacts(store, run_id)`
  reads them. Accepted documents are deleted only by an explicit purge, and
  tombstones and purge records always remain.
- **What it returns.** One entry per restricted run: its `run_id`, the
  `tombstones` of the withdrawn text as JSON records, and its `purged`
  files.
- **Refusals.** A blank or overlong source id or version, a reason that is
  not 1-500 characters without control characters, a scope that is not an
  access-scope label and a `purge` that is not `True` or `False` raise
  `WorkflowBindingError`. A run in scope whose record cannot be read raises
  `WorkflowStateError` before anything is written, and so does a journal
  opened read-only.

### A restricted run

`smythe.lineage.restricted_sources(store, run_id)` returns the tombstones
that restrict a run: each withdraws a text the run's record names, whether
`restrict_source` or `remove_source` journaled it. While any does, whatever
decision follows it:

- **No delivery.** Resuming the completed run, inspecting it and its
  published document's eligibility give `indeterminate`, naming the
  withdrawn source, and the output moves to `rejected_output`.
  `reevaluate` refuses the run, so no new decision, by a policy or an
  operator, delivers it again.
- **No reuse.** A successor never reuses a node whose inputs name a
  withdrawn text, nor a node that depends on one.
- **No republication.** `republish` refuses the run, so a purged document is
  never published again.
- **No external delivery.** Republication outside the journal goes through
  the [governed action path](actions.md), whose `prepare_action`, `approve`
  and `check_authority` refuse the content as unavailable.
- **Inspection.** [`smythe runs inspect`](runs.md#lineage) shows the derived
  status `restricted`, each tombstone by source id, version, digest, reason
  and time, and each purged file, without text.
- **Repair plans.** A text withdrawn from a run is never available to it: a
  [plan](#plan-a-repair) describes the source as withdrawn for that run, so
  the run needs `review` (`source_tombstoned`) while the catalog still
  describes that text. Once the catalog describes another text, such as a
  new version, the run can `regenerate`; its status stays `restricted`.
- **Successors.** A successor holds its predecessor's tombstones from its
  first checkpoint. One given such a text as a context source is refused
  before it is created, and one that reads such a text again is restricted
  when it completes.

### Limits

- **Scope is declared, not authenticated.** Restricting one
  [access scope](#access-scope) leaves the runs under every other label as
  they are; the label says whose runs these are, not who may read the
  journal.
- **Copies in the journal.** A tombstone is not an erasure. Text that the
  journal's checkpoints, context snapshots, saved requests and provider or
  read responses copied, including a completed run's output, is not
  scrubbed: those records are bound by their digests. Protect or delete the
  journal itself when those copies must go.
- **Runs in progress.** Only a run that has completed with a record is
  restricted. Call `restrict_source` again after a run in progress
  completes.
- **Files elsewhere.** Only the files a run published into its artifact
  directory are purged, never copies made elsewhere.
