# Learning from runs

Smythe learns from finished runs in two reviewable ways. `PlannerMemory`
recalls the outcomes of earlier runs into later planning prompts (see the
[learning loop](architecture.md#learning-loop)). Distillation turns a graph
that ran into a template that `ConstrainedArchitect` can offer as a menu item,
so a proven shape is selected instead of planned again.

This page covers the evidence planner history keeps, failed runs,
distillation, template records, their versions, and the template catalog. A
record is data: building a template from it, storing it and pinning it in a
durable run never changes the base planner, and nothing is promoted or
activated automatically.

## Evidence in planner history

A history row's `success` means only that every node completed, so a run
whose gate ran out of regenerations, or whose output a strict policy
rejected, still counts as a success. `PlannerMemory` therefore records the
run's verification and acceptance evidence beside it. A run whose graph holds
a gate's verification receipt, or whose result has a
[delivery acceptance record](verifier.md#delivery-acceptance-records) or
rejected output, also records:

| Field | Value |
|---|---|
| `execution_id` | the run's execution id |
| `structure_sha256` | the [structure digest](#structure-digests) of the run's graph |
| `verification` | one `{gate, target, state, passed, regenerations_used}` entry per gate with a receipt |
| `rejected_output_sha256` | the SHA-256 of the result's `rejected_output`, never its text |

- `passed` is true only when the gate's receipt records a pass. An exhausted
  gate, or one whose target did not complete, is recorded with `passed`
  false. Gates with `max_regenerations=0` hold no receipt and have no entry.
- `acceptance` is the decision of the result's record: a strict run that
  rejected its output records `rejected` beside `success` true.
- Any other run, such as one without gates or a strict policy, writes exactly
  the keys it wrote before. Rows stay small, because recall reads the whole
  history file.
- Recall skips a row whose evidence fields are malformed. Rows written before
  these fields existed recall with them `None`.
- Planning prompts do not change: they still show each recalled run's goal,
  topology, cost, summed node time and outcome.

The execution id, structure digest and acceptance decision are what a
template record's [`source_runs`](#template-records) name.
[Durable runs](support-matrix.md#durable-text-workflow-components) still
refuse `PlannerMemory`.

## Failed runs

By default `PlannerMemory` records only runs that finish. With
`record_failures=True`, a `Swarm` also records a run that raises during
execution, synthesis or acceptance, including a resumed run, before the
exception reaches the caller:

```python
from smythe import PlannerMemory, Swarm

swarm = Swarm(memory=PlannerMemory(record_failures=True))
```

A failure row records:

| Field | Value |
|---|---|
| `failure_category` | the exception's [failure class](execution.md#failure-classes), such as `terminal_accounting` for a budget overrun, `provider_rejected` for a refusal or `quality_defect` for the tool-loop limit |
| `failure_type` | the exception's class name, such as `OutputRefusedError` |
| `failure_excerpt` | for a provider or service error only, such as an SDK status error or a refusal: at most 200 characters of its message, with account identifiers redacted, the home folder written `~` and any other absolute path `<path>` |

- Other errors keep no message, because it can quote task text or local
  data. No row holds a provider's raw response.
- The row also records the run's `execution_id`, `structure_sha256`, any
  gate `verification` entries, its cost and timing so far, and `acceptance`
  `not_evaluated`. A resumed run records `resumed` true.
- `success` keeps its meaning: a run whose synthesis failed after every node
  completed records `success` true beside its `failure_category`. Planning
  prompts keep their format and show every failure row's outcome as
  `failure`, including such a run.
- A keyboard interrupt, `SystemExit` or a cancellation is never recorded, nor
  is a run that fails during planning or that executes a graph without a
  task.
- When the row cannot be written, the error is logged as a warning on the
  `smythe.swarm` logger and the run's own exception is raised unchanged.
- Recall returns failure rows and skips one whose failure fields are
  malformed.

`record_failure(task, graph, exc, *, execution_id=None, timing=None,
trace=None, budget=None)` writes a failure row directly. `timing` and
`trace` take the forms of `SwarmResult.timing` and `SwarmResult.trace`, and
`budget` is the run's `Sentinel`. A memory object of your own is called on
failure only when its `record_failures` is `True` and it has a
`record_failure` method; one that implements only `record()` is unaffected.

## Distill a template

`distill_template(graph, *, name, description=None, registry=None,
require_success=True)` returns a `SubGraphTemplate`:

```python
from smythe import ConstrainedArchitect, distill_template

template = distill_template(result.graph, name="diligence-review", registry=swarm.registry)
architect = ConstrainedArchitect(provider=provider, templates=[template])
```

What carries over is structure: node ids, labels, dependency edges, required
capabilities, verification gates (`verifies` and `max_regenerations`), each
node's failure policy, retries, timeout and `attach_dep_artifacts`, its
metadata `role`, and, with a registry, each agent's name, persona and
capabilities. Results, statuses, costs, models and all other metadata stay
behind. Each build returns new pending nodes and a new registry, and points
each label at the new task: a `{goal}` placeholder is replaced by the new
goal, and any other label gets `Apply this step to: <goal>` appended.

`require_success` (the default) refuses a graph with a node that did not
complete. `DistillationError` is raised for an empty graph, a gate that
verifies itself or a node the graph does not contain, and a graph that
cannot be written as a [template record](#template-records): for example
one with a blank or duplicate node id, a dependency on a node the graph
does not contain, a dependency cycle, a timeout that is not a positive
finite number, a negative or non-integer retry or regeneration count, a
label or `role` that is not text, or text without a UTF-8 form.

## Template records

`distill_record(graph, *, name, description=None, registry=None,
require_success=True, source_task=None)` takes the same arguments and
returns a `TemplateRecord`. `record.to_template()` is the template
`distill_template` returns, and `distill_template` is built on it.

A record's JSON form:

```text
{
  "type": "smythe.template_record",
  "version": 1,
  "name": "diligence-review",
  "description": "...",
  "blueprint": [{"id", "label", "depends_on", "capabilities",
                 "profile": {"name", "persona", "capabilities"} or null,
                 "verifies", "max_regenerations",
                 "failure_policy": "halt" | "skip" | "retry",
                 "max_retries", "timeout_s", "attach_dep_artifacts",
                 "metadata": {"role"}}],
  "applicability": {"task_families", "keywords", "required_context_keys",
                    "excluded_task_families"},
  "prerequisites": {"capabilities", "skills": [{"name", "version"}],
                    "tools": [{"name", "version", "schema_sha256"}],
                    "validators": [validator identities]},
  "expected_outputs": [{"node", "description", "validators"}],
  "failure_examples": [{"category", "evidence_sha256"}],
  "tests": [{"name", "sha256"}],
  "source_runs": [{"execution_id", "structure_sha256", "acceptance"}],
  "parent_version": "<template version>",
  "review": {"label_findings": [{"node", "contains_goal", "words"}],
             "path_findings": [{"node", "field"}]}
}
```

- **Instructions and structure only.** Labels and personas are the record's
  instructions. A record never carries MCP servers, models, environment
  variables, URLs, tool grants, acceptance policies, budgets, evaluator
  configuration or filesystem paths: no key holds them, and every unknown
  key is refused. Free text, the description, labels and personas, is copied
  verbatim from the graph and its agents, so it can still hold a path or the
  source task's subject; the [review](#review) flags both. `prerequisites` names what a template expects to find,
  such as a tool by name, version and schema digest; it grants nothing.
  Validators appear as `ValidatorIdentity` data (name, version and
  configuration digest), never as code or configuration.
- **Optional sections.** Everything after `blueprint` is optional and is
  omitted when empty, never written as null or an empty list. Lists inside
  `applicability` and `prerequisites` are sets: they are sorted and hold each
  item once. A record keeps at most 16 `failure_examples`, each a category
  and the SHA-256 of its evidence, never the evidence itself. `source_runs`
  name the runs a template was learned from by execution id, graph
  [structure digest](#structure-digests) and acceptance decision.
- **Consistency.** Node ids are unique; `depends_on`, `verifies`, expected
  outputs and review findings name blueprint nodes; a node neither depends on
  nor verifies itself, and `depends_on` edges form no cycle; and a record is
  never its own parent. Digests are 64 lowercase hexadecimal characters.
  Text must have a UTF-8 form, so a string holding a lone surrogate such as
  `"\ud800"`, which no digest could cover, raises `TemplateRecordError`, and
  `DistillationError` when distilling.
- **Strict reading.** `TemplateRecord.from_dict()` refuses another `type`, any
  version other than 1, and any missing or unknown key at every level.
  `to_dict()` returns detached JSON data; the record's parts are read-only.
  A record written by `to_dict()` reads back to the same canonical bytes.

## Versions and digests

A record has two SHA-256 digests of canonical JSON (sorted keys, no
whitespace, UTF-8):

- `template_version` covers what decides what the template builds: the type,
  version, name, description, blueprint, applicability, prerequisites and
  expected outputs. Any change to a gate, policy, edge, label, persona or
  applicability gives a new version. Key order does not change it, and
  neither do `failure_examples`, `tests`, `source_runs`, `parent_version` or
  `review`.
- `record_sha256` covers the whole record. Records are equal, and hash
  alike, exactly when their `record_sha256` is equal, so a timeout of `90`
  and one of `90.0` make different records.

`parent_version` names the template version a record was derived from.

### Structure digests

`structure_sha256(graph)` (also in `smythe.graph`) is the SHA-256 of a
graph's shape: each node's id, `depends_on`, `verifies`, required
capabilities, failure policy, retries, timeout, tool-iteration cap,
`attach_dep_artifacts`, `max_regenerations` and metadata `role`, in id order.
Labels, agents, results, statuses and other metadata are left out, so two
runs of one template on different goals share a structure digest.

## Review

`distill_record` copies the description, labels and personas verbatim, so
they can keep what the source run was about. Its review lists that text for
a reviewer:

- **Labels that repeat the source goal.** A distilled label can keep the
  subject of the task it was learned from, such as "Research the European
  market for solar chargers". `distill_record` compares each label with the
  source goal: `source_task`'s goal, or by default the goal of the graph's
  own `task`. A label that contains the goal, compared without case and with
  runs of whitespace collapsed, or that shares two or more of its words of
  four or more letters, is listed in `review["label_findings"]` with its
  node, whether it contains the whole goal, and the shared words.
- **Rooted or home paths.** The description, each label and each persona
  that holds a path starting at a drive letter (`C:\`), a UNC share
  (`\\server`), a leading slash (`/srv/...`), `~` or `~user`, `$HOME` or
  `%USERPROFILE%` is listed in `review["path_findings"]` by its node (`null`
  for the description) and field (`description`, `label` or `persona`). A
  finding says where a path is, never what it is. A slash inside a word or a
  URL, as in `PASS/FAIL` or `https://...`, and a relative path such as
  `./src` are not flagged.

Each list is omitted when it is empty. The review only flags text: it is
not changed, and each build points labels at the new task as before. The
review is not part of `template_version`.

## The template catalog

`TemplateCatalog(root)` keeps records as append-only files:

```text
<root>/
  records/<record_sha256>.json   a record's canonical JSON
  versions/<template_version>    the record_sha256 of that version's record
```

```python
from smythe import TemplateCatalog, distill_record

catalog = TemplateCatalog("templates")
version = catalog.put(distill_record(result.graph, name="diligence-review"))
template = catalog.get(version).to_template()
catalog.versions("diligence-review")  # every stored version of that name
```

- `put(record)` returns the record's template version. Storing a record the
  catalog already holds writes nothing. A catalog holds one record per
  version: a different record for a stored version, for example one with
  other source runs, raises `TemplateRecordError` and the stored record is
  kept.
- `get(version)` reads the record again and checks it: the record file's
  SHA-256 must equal its name, and the record must have the version that
  names it. An edited file, including one whose JSON is merely reformatted,
  raises `TemplateRecordError`, as does a version file whose record is
  missing; a version the catalog does not hold raises `KeyError`.
- `versions(name)` lists the stored versions of the records with that name,
  in digest order, checking each.
- Files are published whole through a flushed temporary file linked into
  place without replacing an existing file, and nothing is ever deleted. A
  record is published before the version file that names it, so a process
  killed during `put` leaves at most a whole record that no version names
  yet; `get` and `versions` do not see it, and the next `put` completes it.
  A missing root, `records` or `versions` folder is created owner-only
  (0700), and files are owner-only (0600), where the platform supports
  permission bits. Publication needs a filesystem with hard links, such as
  NTFS, APFS or ext4.

## Durable runs

A durable run (`run_store=SQLiteWorkflowStore(...)`) accepts only template
builders declared with `LocalOnly(role="template_builder")`.
`record.local_only()` is that declaration, with identity `template:<name>`
and the record's `template_version` as its version:

```python
from smythe import ConstrainedArchitect, SubGraphTemplate

record = catalog.get(version)
template = SubGraphTemplate(record.name, record.description, record.local_only())
architect = ConstrainedArchitect(provider, [template], model="gpt-6-astra")
```

The run's recipe records the identity and version, so it pins the exact
template. Resuming the run with a template of another version raises
`WorkflowConflictError`; resume with the record the catalog holds for the
saved version instead.

## Limits

- Records hold structure from model output. This release checks their format
  and consistency, not their limits: it does not yet compare a record's
  gates, retries and regenerations with the model-plan rules or with the
  record it came from. Build templates only from records you trust.
- A catalog has no activation, review or rollback state. Choosing which
  version a planner offers is the caller's decision.
