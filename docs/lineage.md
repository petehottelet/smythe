# Artifact lineage

Artifact lineage records how a durable run's accepted artifact was made: the
run, the published document, the delivery decision, and each node's result
and the inputs it was given, as ids, versions and SHA-256 digests. A run opts
in with a `LineagePolicy`, which also declares the access scope its results
belong to and whether they may be reused.

This release adds the policy, the recipe opt-in and the versioned record
format, `ArtifactLineage`. Completed runs do not write lineage records yet,
and nothing reuses or repairs work from them yet.

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
