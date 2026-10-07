# Skills

A skill is a capability tag that an agent's inventory lends it. The registry
asks a skill provider which skills each agent has, maps their names to tags,
and matches a node's `required_capabilities` against them. The planner sees
the same tags in its agent inventory. A skill routes work to an agent; it
never changes what that agent may do.

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

Only the frontmatter is read as YAML, with `yaml.safe_load`, and it must be a
mapping. The file may hold at most 64 KiB. These rules raise
`AgentSkillError`, a `ValueError` that names the skill directory and the
field:

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
  only by the tags it lists.

A tag can still route a node to an agent that already holds tools, so give
tool-bearing agents only the skills you would trust with those tools.

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
