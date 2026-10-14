# Learning from runs

Smythe learns from finished runs in two reviewable ways. `PlannerMemory`
recalls the outcomes of earlier runs into later planning prompts (see the
[learning loop](architecture.md#learning-loop)). Distillation turns a graph
that ran into a template that `ConstrainedArchitect` can offer as a menu item,
so a proven shape is selected instead of planned again.

This page covers the evidence planner history keeps, failed runs,
distillation, template records, their versions, promoting structure that
accepted runs repeat to a candidate record, validation of records as
untrusted candidates, the template catalog, building a template without a
planning call, activating, rolling back and pinning template versions,
when a template stops applying because what its runs relied on changed, a
template's lifecycle from record to rollback, and what learning cannot
change. A record is data: building a template from it,
storing it, activating it and pinning it in a durable run never changes the
base planner, and nothing is promoted or activated automatically.

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
template record's [`source_runs`](#template-records) name, and what
[promotion](#promote-repeated-accepted-structure) counts.
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

# agents is the Registry the run's Swarm(registry=agents) used.
template = distill_template(result.graph, name="diligence-review", registry=agents)
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
  "proven_with": {"models_sha256", "mcp_allowlists_sha256"},
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
  configuration digest), never as code or configuration. `proven_with` holds
  only the digests of the model ids and MCP allowlists that a candidate's
  runs used, to tell when they [change](#applicability).
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
  version, name, description, blueprint, applicability, prerequisites,
  `proven_with` and expected outputs. Any change to a gate, policy, edge, label, persona or
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

## Promote repeated accepted structure

A shape whose runs keep being accepted can become a candidate template, so
it is selected instead of planned again.
`repeated_structures(memory, *, min_accepted)` reads planner history and
returns each [structure digest](#structure-digests) that at least
`min_accepted` accepted runs share, with the first row of each such run.
History holds no graphs, so build the candidate from a graph of that
structure, such as one run's `result.graph`:

```python
from smythe import (
    PlannerMemory, TemplateCatalog, candidate_from_graph, repeated_structures, structure_sha256,
)

memory = PlannerMemory()
if structure_sha256(result.graph) in repeated_structures(memory, min_accepted=3):
    record = candidate_from_graph(result.graph, rows=memory.iter_outcomes(), name="memo-review",
                                  min_accepted=3, registry=agents)
    version = TemplateCatalog("templates").put(record)
```

- **Accepted, not successful.** A run counts only when its row records
  `acceptance` `accepted`, which a
  [strict acceptance policy](verifier.md#strict-delivery-acceptance) decides,
  an `execution_id` and a `structure_sha256`, and no `failure_category`.
  `success` never counts, so a completed run whose output was rejected, or
  that no policy evaluated, does not. Each execution id counts once.
- **Source runs.** `candidate_from_graph(graph, *, rows, name,
  min_accepted=1, description=None, registry=None, require_success=True,
  source_task=None, environment=None)` distills `graph` as `distill_record`
  does, keeps its [review](#review), and lists each accepted run of the
  graph's structure in `rows` as a source run. With fewer than
  `min_accepted` such runs it raises `DistillationError`. Given the
  `environment` the runs used, it also records what they relied on, so the
  template stops being offered when that changes (see
  [applicability](#applicability)).
- **Failure examples.** Each other row of that structure that has a
  `failure_category`, or whose `acceptance` is `rejected`, `needs_review` or
  `indeterminate`, becomes a failure example: that category, or else the
  decision, and `evidence_sha256`, the SHA-256 of the row's `execution_id`,
  `structure_sha256`, `acceptance`, `verification`,
  `rejected_output_sha256`, `failure_category` and `failure_type` as
  canonical JSON, each when set. A completed run that no policy evaluated
  is not an example. No text from a row is kept, neither its goal nor a
  failure excerpt. Rows with equal evidence give one example, and the
  latest 16 are kept.
- **A candidate only.** Nothing is stored or activated. The catalog
  [validates](#validate-a-candidate-record) a candidate when it stores it,
  and a version is put in effect only through a
  [reviewed decision](#activate-and-roll-back-templates).

`PlannerMemory.iter_outcomes()` returns the history rows, oldest first. It
reads the file once, when it is called, never writes it, and skips the rows
recall skips.

## Validate a candidate record

A record's structure can come from model output, and a record read from a
file or another process is untrusted data.
`validate_template_record(record, *, parent=None, limits=None)` checks a
`TemplateRecord`, or its JSON data, as such a candidate and returns the
record. `TemplateCatalog.put` runs it before it writes anything.

```python
import json

from smythe import TemplateLimits, validate_template_record

record = validate_template_record(json.loads(text))  # generated-plan limits
record = validate_template_record(data, parent=catalog.get(version))
record = validate_template_record(data, limits=TemplateLimits(max_nodes=12, max_gates=2))
```

A refused candidate raises `TemplateValidationError`, a `TemplateRecordError`
whose `node` and `field` attributes name the blueprint node, or `None`, and
the field at fault, such as `"max_retries"` or `"profile.mcp_servers"`. A
candidate is refused when:

- **It would carry authority.** A key named `mcp_servers`, `model`, `env`,
  `command`, `url`, `tools`, `allowed_tools`, `acceptance`, `evaluator` or
  `budget` is refused at any depth: on a node, its agent profile or
  metadata, an expected output, a validator or the record itself. The
  record's own fields with these names grant nothing and are allowed: the
  `tools` list in `prerequisites`, which names tools a template expects,
  and the `acceptance` decision each source run received. Data that
  `TemplateRecord.from_dict()` refuses is refused too, with `node` and
  `field` set to `None`.
- **It breaks the limits.** By default these are the limits on a generated
  plan, `TemplateLimits()`, taken from `smythe.loader.MODEL_PLAN_*`: at
  most 8 nodes and 5 levels, at most 3 retries and 2 regenerations per
  node, at most 1 gate, and no timeout under 60 seconds. Node ids must be 1
  to 64 letters, digits, `-` or `_`; labels and agent names must not be
  blank; and each gate must list the node it verifies in `depends_on`, so
  it cannot judge that node before it runs. The blueprint, written as a
  generated plan, must also pass every check
  `build_graph_from_model_output` makes. Pass `TemplateLimits(...)` with
  other values only for a record whose structure you wrote or reviewed.
- **It guarantees less than its parent.** Given a `parent`, each of the
  parent's gates must still verify the same node; its `max_regenerations`
  must not be lowered; neither a gate nor its target may be relaxed to
  `failure_policy` `skip`, under which a failing judge or target is passed
  over and the run continues without a verdict; and no validator of an
  expected output and no prerequisite of the parent may be removed. A
  prerequisite at another version counts as removed. A candidate's
  `parent_version`, when set, must be the parent's version.

Validation checks structure and authority, not instructions: a label or
persona that tells a judge to pass everything is for a reviewer to catch.
`distill_template`, `distill_record` and templates a developer builds in
code are not validated, because developer code is trusted.

## The template catalog

`TemplateCatalog(root)` keeps records as append-only files:

```text
<root>/
  layout.json                    {"layout":1,"type":"smythe.template_catalog"}
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

- `put(record, *, parent=None, limits=None)` [validates](#validate-a-candidate-record)
  the record, a `TemplateRecord` or its JSON data, under `limits` and
  against `parent` or, without one, against the record the catalog holds
  for the candidate's `parent_version`, when it holds one. A refused record
  raises `TemplateValidationError` and writes nothing. `put` returns the
  record's template version. Storing a record the catalog already holds
  writes no record or version file. A catalog holds
  one record per version: a different record for a stored version, for
  example one with other source runs, raises `TemplateRecordError` and the
  stored record is kept. Two processes that store the same record at once
  both succeed.
- `layout.json` records the catalog's file layout
  (`smythe.templates.TEMPLATE_CATALOG_LAYOUT`, now 1), so a later layout
  change is detected rather than misread. A catalog without the file was
  written before it existed and is read as layout 1; its next `put` adds the
  file. `put`, `get` and `versions` raise `TemplateRecordError` for a catalog
  in another layout and write nothing.
- `get(version)` reads the record again and checks it: the record file's
  SHA-256 must equal its name, and the record must have the version that
  names it. An edited file, including one whose JSON is merely reformatted,
  raises `TemplateRecordError`, as does a version file whose record is
  missing; a version the catalog does not hold raises `KeyError`.
- `versions(name)` lists the stored versions of the records with that name,
  in digest order, checking each.
- Files are published whole through a flushed temporary file linked into
  place without replacing an existing file, and nothing is ever deleted. A
  record is published before the version file that names it, and the layout
  file last, so a process killed during `put` leaves at most a whole record
  that no version names yet; `get` and `versions` do not see it, and the
  next `put` completes it.
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
saved version instead, which [`pinned_templates`](#resume-a-pinned-run)
rebuilds.

## Build a template without planning

A template that suits every task a run receives needs no selection call.
`record.as_architect()` returns a `TemplateArchitect`
(`smythe.templates.TemplateArchitect`), a `DeterministicArchitect` that
builds the record's graph in local code:

```python
from smythe import Swarm

swarm = Swarm(model="gpt-6-astra", architect=catalog.get(version).as_architect())
```

- Each plan is the graph a `ConstrainedArchitect` composes when its model
  selects the template once: node ids prefixed `<name>-0-`, labels pointed
  at the task, a new registry of the record's agents, and
  `graph.planning["templates"]` naming the record's version. Planning makes
  no provider call.
- A [strict acceptance policy](verifier.md#strict-delivery-acceptance)
  counts its gates as gates you supplied, as it does those of a
  `ConstrainedArchitect`'s templates.
- A durable run takes `record.as_architect().local_only()`, a
  `LocalOnly(role="architect")` declaration with identity `template:<name>`
  and the record's `template_version`, so the recipe pins the exact
  template, and resuming with another version raises
  `WorkflowConflictError`. Resume with the record the catalog holds for the
  saved version.

## Activate and roll back templates

The catalog keeps every version, and the
[policy registry](optimize.md#policy-registry) records which one new runs
use. A template's target is `planner.template.<name>`, which
`template_target(name)` returns, and its policy maps the target to a
template version:

```text
{"planner.template.report": "<template version>"}
```

A template whose name is not one dotted segment, a letter followed by up to
63 letters, digits, `-` or `_`, has no target: `template_target` raises
`ValueError`.

A version is put in effect only through a reviewed Autotune decision. The
campaign's contract makes the target its mutable field, with the template
versions as string choices, such as `MutableFieldRule("string",
choices=(v1, v2))`, and its evaluator builds each candidate's template from
the catalog. An approving review of a promoted decision is then activated
for the target, and the first activation records the incumbent version as
the target's baseline:

```python
from smythe import ConstrainedArchitect, TemplateCatalog, template_target
from smythe.optimize import PolicyRegistry

catalog = TemplateCatalog("templates")
target = template_target("report")  # "planner.template.report"
with PolicyRegistry("policy-registry.sqlite3") as registry:
    review = registry.record_review("optimize.sqlite3", campaign_id,
                                    reviewer="report-owner", verdict="approve")
    registry.activate(target, review.review_id, expected_seq=0,
                      actor="release-operator", reason="approved for new reports")
    architect = ConstrainedArchitect(provider, catalog.active_templates(registry, ["report"]))
```

- `catalog.active_templates(registry, names)` returns `record.to_template()`
  for the version of each name in effect, in the order of `names`, for an
  ordinary run. It reads the registry, which may be opened read-only, and
  never writes it. A `ConstrainedArchitect` copies its menu when it is
  built, so an architect keeps offering the versions it was built with;
  build a new one, for example for each run, to offer the versions in effect.
- `catalog.active_local_templates(registry, names)` returns the same
  templates with `record.local_only()` as their builders, for a
  [durable run](#durable-runs), whose recipe then pins those versions.
- A retired template has no version in effect and is left out. A name whose
  target has no history, and a version the catalog does not hold, raise
  `KeyError`. A policy that holds anything besides the target's version, or
  that names a version of another template, raises `TemplateRecordError`.

A rollback appends an event that restores an earlier version (see
[resolve, roll back and retire](optimize.md#resolve-roll-back-and-retire)):

```python
with PolicyRegistry("policy-registry.sqlite3") as registry:
    registry.rollback(target, to_seq=1, expected_seq=len(registry.history(target)),
                      actor="release-operator", reason="v2 drops the conclusions")
```

New architects and new durable runs then resolve the restored version, and
the history keeps the baseline, activation and rollback events. A rollback
writes only the registry. The catalog, the Autotune ledger, durable
journals, architects already built and running work are unchanged, and the
ledger's decisions and trials stay readable through a read-only open.

### Resume a pinned run

A durable run keeps the versions its recipe pinned, whatever is activated or
rolled back later. Resuming it with a template of another version, such as
the one now in effect, raises `WorkflowConflictError` and changes nothing.
`pinned_templates(config, catalog)` rebuilds the pinned versions:

```python
from smythe import ConstrainedArchitect, Swarm, pinned_templates

config = store.load_run(run_id)["config"]
architect = ConstrainedArchitect(provider, pinned_templates(config, catalog), model="gpt-6-astra")
swarm = Swarm(model="gpt-6-astra", provider=provider, run_store=store, architect=architect)
result = swarm.resume(run_id)
```

`config` is the saved recipe, whose architect is a `ConstrainedArchitect`,
or that architect's description within it, such as a router tier's
`config["components"]["router"]["constrained"]`. Each template keeps the
recipe's name and description, and its builder is `record.local_only()` for
the recorded version, so the rebuilt recipe matches the saved one, and so
does its `config_sha256`, and the run finishes on the versions it started
with. A version the catalog
does not hold raises `KeyError`. A template whose builder is not a record's
declaration, such as one written in code, raises `TemplateRecordError`.

### Which versions a plan used

When a `ConstrainedArchitect` composes a plan from templates built from
records, by `to_template()`, `local_only()` or `distill_template`, the
graph's `planning["templates"]` lists each such selection as
`{"name", "index", "version"}`: the template's menu name, its position among
the selections, which also prefixes its node ids, and its template version.
Templates written in code add nothing, so their plans keep `planning` as
before, and node metadata never carries a version. Like the
[routing decision](architecture.md#routing-decisions), the record is
informational: checkpoints and durable plans save it with the graph.

## Applicability

A template learned from accepted runs relied on what those runs used: tools
at their versions and input schemas, capabilities agents offered, skills at
their digests, acceptance validators, models and MCP server allowlists.
When one of them changes, that evidence no longer covers the template.
`TemplateEnvironment.capture(*, registry=None, skill_provider=None,
tool_runtime=None, validators=(), models=())` snapshots what a run would use
now:

| Part | Captured from |
|---|---|
| `capabilities` | each registry agent's effective capabilities |
| `skills` | each skill with a version that `skill_provider`, by default the registry's own, lists for an agent: for an [Agent Skill](skills.md#agent-skills-directories), the `sha256:` digest of its `SKILL.md` |
| `tools` | `describe()` of a `LocalToolRuntime`, or of each one a `CompositeToolRuntime` holds: name, version and `schema_sha256` |
| `mcp_allowlists` | the `allowed_tools` of each MCP server on an agent's profile, merged by server name; `None`, every tool, wins |
| `validators` | acceptance validators or their identities, such as `AcceptancePolicy.validator_identities` |
| `models` | the model ids a run would use |

`TemplateEnvironment(...)` takes the same parts directly, and every part is
read-only. Given the environment its runs used,
`candidate_from_graph(..., environment=environment)` records what the
candidate relied on: as `prerequisites`, the capabilities that nodes without
an agent profile require and an agent offered, and every skill, tool and
validator; as `proven_with`, the SHA-256 of the sorted model ids and of the
MCP allowlists as canonical JSON, never the ids or servers themselves.

`check_applicability(record, environment)` returns an
`ApplicabilityReport(valid, reasons, details)`. `reasons` are the sorted
codes of `smythe.templates.APPLICABILITY_REASONS` behind any problem, each
once, and `details` one sentence per problem:

| Reason | When |
|---|---|
| `capability_removed` | no agent offers a prerequisite capability |
| `skill_removed`, `skill_changed` | a prerequisite skill is missing, or at another version |
| `tool_removed`, `tool_changed` | a prerequisite tool is missing, or has another version or input schema digest |
| `validator_set_changed` | a validator identity that the prerequisites or an expected output name is not in use: removed, or at another version or configuration |
| `model_changed` | the run's model ids are not those `proven_with` records |
| `mcp_allowlist_changed` | the run's MCP allowlists are not those `proven_with` records |

Prerequisites are what a template needs, so an environment that adds a
capability, skill, tool or validator leaves it applicable, while
`proven_with` must match exactly. A record with no prerequisites, no
`proven_with` and no validators named in its expected outputs applies
anywhere.

Pass `environment=` to `active_templates` or `active_local_templates` to
offer only the versions in effect that still apply:

```python
from smythe import TemplateEnvironment

environment = TemplateEnvironment.capture(
    registry=agents, tool_runtime=tool_runtime,
    validators=policy.validator_identities, models=["gpt-6-astra"])
templates = catalog.active_templates(registry, ["memo-review"], environment=environment)
```

A version that does not apply is left out, and a warning on the
`smythe.templates` logger names it and gives the details. With
`strict=True` it raises `TemplateApplicabilityError` instead, whose `name`,
`version` and `report` say which version and why. Nothing is deactivated:
the policy registry, the catalog and durable runs that pinned the version
are unchanged, and [`pinned_templates`](#resume-a-pinned-run) still rebuilds
it to resume them. To offer the template again, build a candidate from runs
in the new environment and activate it through a reviewed decision.

## A template's lifecycle

Each step above is a separate, explicit operation.
[`examples/21_template_lifecycle.py`](../examples/21_template_lifecycle.py)
runs them in order, offline and without keys:

1. **Record.** Strict runs record their acceptance decision and structure
   digest in planner history. `repeated_structures` finds a structure that
   accepted runs repeat, and `candidate_from_graph` distills it into a record
   that names those runs.
2. **Validate and store.** `TemplateCatalog.put` validates each candidate,
   whether distilled, written by hand or proposed by a model, against its
   parent before it stores it. A candidate whose step grants itself a tool is
   refused and writes nothing.
3. **Evaluate.** An [Autotune campaign](optimize.md#evaluate-template-versions)
   makes the template's target its mutable field. The version in effect, or
   the predecessor, is the incumbent and the candidate the challenger, and a
   simpler arm, such as a single worker, is a
   [reference](optimize.md#reference-arms) under `must_not_be_worse`. A task
   family that only the holdout holds tests what development and confirmation
   did not.
4. **Review and activate.** A reviewer approves a promoted decision, and
   activation puts the candidate in effect for new runs. A decision that was
   not promoted cannot be approved.
5. **Roll back.** A rollback puts the previous version back in effect for new
   runs and writes only the registry. Running work keeps its versions.

Validation and evaluation catch different things. Validation reads structure
and authority: in the example, a candidate that drops its Timeline step
passes, because its parent declares no check that needs one. Its campaign
rejects it, since it is better on the briefings of development and
confirmation and `worse` on the incident reviews held out. Evaluation is in
turn only as good as its held-out tasks: a second evaluator whose holdout
holds only briefings promotes the same candidate. Once a reviewer approves it
and it is activated, the next incident review is rejected for its missing
section, and a rollback restores the predecessor. A review approves a
decision, not a template, so check what the evaluator held out before
approving one.

The example's fixtures show the mechanics only. Its evaluator is
deterministic, so each decision rests on three identical pairs; read
[choosing sample sizes](optimize.md#choosing-sample-sizes) before sizing a
real campaign.

## What learning cannot change

Template records, Agent Skills and Autotune campaigns change what new work
is built from, and none of them can change how work is judged, what it may
spend or what it may do:

| What learns | What it can change | What it cannot change |
|---|---|---|
| A template record | The structure and instructions a plan is built from | The tools, MCP servers, models or environment agents get, acceptance, budgets or evaluation: no record holds them, and a candidate that checks less than its parent is [refused](#validate-a-candidate-record) |
| An Agent Skill | A capability tag that routes work to an agent | The tools an agent is offered, acceptance, the budget, prompts or evaluator data, whatever its `SKILL.md` declares ([skills](skills.md#what-a-skill-can-and-cannot-change)) |
| An Autotune campaign | A policy under its target | A [reserved path](optimize.md#experiment-contracts-and-budgets), such as acceptance, validators, the evaluator, budgets, tools or the holdout, whether its contract or an exploring proposer names it |

`tests/test_learning_boundaries.py` checks the three in one place: after
each refused or harmless attempt, the evaluator's data and identity, the
acceptance policy, the budget and the tools each agent is offered are
byte-identical to what they were before, as are the ledger, catalog and
skill files involved.

## Limits

- Validation compares a candidate with one parent, not with the whole
  lineage, and a record whose parent the catalog does not hold is checked
  against the limits only. It does not read the instructions in labels and
  personas.
- The catalog has no activation, review or rollback state; the policy
  registry holds it. An architect offers the templates it was given, so
  only a menu built by `active_templates` or `active_local_templates`
  follows the registry. Reviews are declared, not authenticated.
- Applicability compares a record with the environment you capture, not
  with what a run does. Nothing re-evaluates a template whose environment
  changed; that takes new runs and a new candidate.
- A promotion is only as good as the evaluator behind it: an evaluator whose
  held-out tasks miss what a candidate breaks promotes it. Template
  campaigns have run only on offline fixtures, so no template version has
  been shown to do better than its predecessor on real tasks.
