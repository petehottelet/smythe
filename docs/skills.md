# Skills

A skill is a capability tag that an agent's inventory lends it. The registry
asks a skill provider which skills each agent has, maps their names to tags,
and matches a node's `required_capabilities` against them. The planner sees
the same tags in its agent inventory. A skill routes work to an agent; it
never changes what that agent may do. A [Python-backed skill](#python-backed-skills)
is a tool you write and register for the agents you name.

## Sources

| Source | Where skills come from | Version |
|---|---|---|
| `AgentSkillsProvider` | [Agent Skills](https://agentskills.io) directories, `<root>/<name>/SKILL.md`, read once when the provider is built | `sha256:` digest of the file |
| `MCPSkillProvider` | The agent's MCP tools: allowlisted names, or names `prefetch()` discovered ([MCP](mcp.md#planner-awareness-and-assignment)) | None |
| `OpenClawSkillProvider` | The OpenClaw SDK's skill inventory (`pip install smythe[openclaw]`) | As OpenClaw reports it |
| A custom provider | Any object with `list_agent_skills(agent_id) -> list[SkillRef]` | Yours |

```python
from smythe import AgentSkillsProvider
from smythe.agent import Agent, AgentProfile
from smythe.registry import Registry

provider = AgentSkillsProvider({"writer": ["skills/team", "skills/writer"]})
registry = Registry(skill_provider=provider)
registry.register(Agent(AgentProfile("Writer", capabilities=["writing"]), id="writer"))
```

The registry combines provider tags with each profile's own capabilities by
`CapabilityHydrationMode`: `MERGE` (the default) takes both, `REPLACE` takes
only the provider's, and `STATIC_ONLY` never asks the provider. It caches each
agent's tags for `capability_cache_ttl_seconds` (300 by default);
`refresh_agent_capabilities(agent_id)` and `refresh_all_capabilities()` clear
the cache. A provider that raises leaves the agent with its static
capabilities, and the registry logs a warning.

## Agent Skills directories

`AgentSkillsProvider` takes skills roots per agent id, so register those
agents with explicit `id`s. Each root holds one directory per skill, and a
directory is a skill when it holds a file named exactly `SKILL.md`:

```markdown
---
name: release-notes
description: Write release notes from merged changes.
allowed-tools: Bash(git:*) Read
---
# Release notes

Collect merged changes and group them by area.
```

Only the frontmatter is read as YAML, with PyYAML's safe loader, and it must
be a mapping that loads: invalid YAML, nesting deeper than Python can follow
and a value such as a date that does not exist are refused. YAML aliases are
refused before anything is built, since through nested lists or merge keys a
few hundred bytes could load as more entries than memory holds. The file may
hold at most 64 KiB. These rules raise `AgentSkillError`, a `ValueError` that
names the skill directory and the field and quotes a refused value only when
it is short text:

- `name` is 1 to 64 lowercase letters, digits and single hyphens, neither
  starting nor ending with a hyphen, and equals its directory name.
- `description` is required, at most 1024 characters.
- `license` is text and `compatibility` is text of at most 500 characters.
  `metadata` maps text to text.
- `allowed-tools` is space-separated text or a list of text, and `version` is
  recorded as text. Both are recorded only.
- A `SKILL.md` that resolves outside its root through a link is refused, and
  so is one name defined by two directories, in one agent's roots or in two
  agents' roots.

Other frontmatter keys are kept, by name only, in `ignored_fields`.
`parse_skill_md(data, skill_id=...)` applies the same rules to bytes you
already hold, and returns the `AgentSkill` record.

Discovery is eager: the provider reads every root when it is built and keeps
a frozen snapshot. Its `SkillRef`s carry `version`, the `sha256:` digest of
the file with CRLF line endings read as LF, and `source="agent-skills"`.
Their metadata holds the `description`, the directory name as `skill_id`,
`declared_allowed_tools` and `declared_version`, and never a path or the
instructions. `provider.skill(name)` returns the full `AgentSkill`.

Nothing changes until you call `refresh()`. It discovers again and returns
the sorted names of skills added, removed, edited or listed for other agents;
a discovery that fails raises and keeps the previous snapshot. Clear the
registry's cache afterwards with `refresh_all_capabilities()`.

`provider.instructions(name)` returns the instruction body and its SHA-256.
It reads the file again and raises `AgentSkillError` if the file no longer
has the digest discovery recorded. `Swarm`, its executors and its planners
never call it, so the instructions reach a prompt only if your code puts
them there.

Discovery reads `SKILL.md` files and nothing else. It never runs a skill's
`scripts/`, never imports from a skill directory and makes no network
request.

## What a skill can and cannot change

A skill can add a capability tag to an agent, through the registry's
capability mapper. That tag can route a node with `required_capabilities` to
the agent, and it appears in the planner's agent inventory.

A skill cannot:

- grant a tool. `allowed-tools` is recorded, never granted: an agent's
  `mcp_servers` and the tools a `ToolRuntime` offers its nodes stay as you
  configured them;
- change an `AcceptancePolicy`, a gate, a validator or the budget `Sentinel`'s
  ceiling;
- put its instructions or description into a prompt. Node prompts are
  byte-identical with and without skills, and the planner's prompt differs
  only by the tags it lists. Only a Python-backed skill that you register
  sends its description to the model, as its tool's description, and only to
  the agents you list.

A tag can still route a node to an agent that already holds tools, so give
tool-bearing agents only the skills you would trust with those tools.

## Python-backed skills

A skill's instructions tell a model what to do; a Python-backed skill does it
in your code. `python_skill()` makes a `LocalTool` of a discovered skill and a
handler you write, and a `LocalToolRuntime` offers that tool only to the
agent ids you list:

```python
from smythe import LocalToolRuntime, Swarm, python_skill

def collect_changes(arguments: dict) -> str:
    return changes_since(arguments["since"])  # your code

notes = python_skill(
    provider.skill("release-notes"), collect_changes, version="1",
    input_schema={"type": "object", "properties": {"since": {"type": "string"}},
                  "required": ["since"]},
)
runtime = LocalToolRuntime([notes], agents={"writer": [notes.name]})
swarm = Swarm(registry=registry, tool_runtime=runtime, ...)
```

The tool is named `skill.<name>`, so its skill's name may have at most 57
characters, and is described by the skill's `description`; `input_schema`,
`version` and an optional `timeout_s` are yours. `python_skill()` reads
nothing from the skill's directory: it never runs `scripts/` or imports from
the directory, and the skill's declared `allowed-tools` grant nothing.
`LocalTool(spec, handler, version, timeout_s=None)` makes a tool of any
`ToolSpec` the same way. [Example 20](../examples/20_agent_skills.py) runs
one offline.

Who is offered a tool:

- `agents` maps agent ids to tool names, and a node is offered a tool only
  when its agent's id is listed for it. Names never count: a generated plan
  names its agents freely, and an agent it creates under one of your agents'
  names gets a new id and no tools. A node without an agent gets none.
- Capability tags route work and grant nothing. A skill's tag can route a
  node to an agent that holds tools, but an agent not listed in `agents` is
  offered no tool, whatever tags or skills it has.
- The planner's agent inventory lists MCP allowlists, not local tools; route
  work to a tool-bearing agent with its capability tags or with `agent_id` in
  graphs you write.

How a call runs:

- The handler gets a copy of the arguments the model sent, which are
  untrusted input, and returns the result text. A synchronous handler runs
  in a thread of its own, so it never blocks the event loop; an `async`
  handler runs on the loop.
- An exception becomes an error result for the model, and so does a result
  that is not text. `BudgetValidationError`, `ProviderResponseError`,
  `WorkflowError` and `SecurityStop` stop the node instead, as they do from
  any tool runtime.
- With `timeout_s`, a call that has not returned in time becomes an error
  result. An `async` handler is cancelled. A thread cannot be: a synchronous
  handler keeps running until it returns, though the run does not wait for
  it, and stops wherever it is if the process exits first.
- A result's text and an error's message are cut to `max_result_chars`
  characters (65,536 by default), with a note of how many more there were.
- The model's tool loop is the only retry owner: each call runs its handler
  once, and the model decides what to do with an error. Calls are not
  journaled, so a retried node or a resumed run may call a handler again
  ([side effects are at-least-once](mcp.md#side-effects-are-at-least-once)).
- What a tool returns reaches the model as untrusted data, like any tool
  result ([threat model](mcp.md#threat-model)).

`CompositeToolRuntime(runtime, MCPToolRuntime())` offers a node the tools of
every runtime it merges, each call going to the runtime that offered the
tool. A tool name may come from one runtime only: local runtimes that define
the same name are refused when composed, and sessions that offer the same
name raise `ValueError` when the node's session opens, before its first
provider call.

`runtime.describe()` lists each tool's `name`, `version` and `schema_sha256`,
the SHA-256 of its input schema as canonical JSON, in the form a
[template record](learning.md#template-records) lists the tools it needs. A
changed schema changes the digest; change `version` when the handler's
behavior changes.

Local tools serve ordinary runs. Durable text workflows refuse every
`tool_runtime` before any call, and Jobs have no tools
([support matrix](support-matrix.md#skill-sources)).

## Durable runs

A durable text workflow (`Swarm(run_store=...)`) records its registry in the
run's recipe and refuses live hydration: a registry with a skill provider and
`MERGE` or `REPLACE` mode raises `WorkflowBindingError` ("Journaled workflows
require static registry capabilities") before any call.
`Registry.static_snapshot()` returns a new `STATIC_ONLY` registry, with no
provider, whose agents hold the capabilities `effective_capabilities()`
reports for them now; call `refresh_all_capabilities()` first to skip the
registry's cache:

```python
swarm = Swarm(run_store=store, registry=registry.static_snapshot(), ...)
```

The recipe pins the capability tags, never skill versions or instructions,
which never enter a prompt. Resuming with a snapshot whose tags differ, for
example after a skill is renamed, added or removed, raises
`WorkflowConflictError`; an edit that keeps the tags resumes as before.
