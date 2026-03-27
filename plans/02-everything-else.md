# Plan 2 — Everything Else

Architecture evolution, DX improvements, and closing gaps between the README vision and the implementation.

---

## Phase 1: Three-Tier Planner Architecture

The current planner hierarchy has a "missing middle":

```
Planner (ABC)
├── SimplePlanner  — single-node stub (too simple)
└── LLMPlanner     — full autonomous (too expensive for known workflows)
```

Implement the three-tier planner system:

```
Planner (ABC)
├── DeterministicPlanner  — pure Python, zero LLM cost, user-defined DAGs
│   └── SimplePlanner becomes a trivial subclass
├── ConstrainedPlanner    — LLM chooses from a menu of pre-built sub-graph templates
└── LLMPlanner            — full autonomous decomposition (existing, unchanged)
```

### 1a. DeterministicPlanner (smythe/planner.py)

A base class for users who know their DAG shape. Replaces `SimplePlanner` as the "fast path."

```python
class DeterministicPlanner(Planner):
    """Override plan() with pure Python logic. Zero LLM cost."""

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        raise NotImplementedError("Subclass and implement plan()")

class SimplePlanner(DeterministicPlanner):
    """Single-node fallback — trivial deterministic planner."""
    def plan(self, task):
        node = Node(label=task.goal)
        graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
        graph.validate()
        return graph, Registry()
```

Users build custom planners:

```python
class SiteBuilder(DeterministicPlanner):
    def __init__(self, num_pages: int = 5):
        self.num_pages = num_pages

    def plan(self, task):
        pages = [Node(id=f"page-{i}", label=f"Write page {i}") for i in range(self.num_pages)]
        review = Node(id="review", label="Review all pages", depends_on=[p.id for p in pages])
        graph = ExecutionGraph(
            topology=[Topology.FORK_JOIN, Topology.SERIAL],
            nodes=[*pages, review],
        )
        graph.validate()
        registry = Registry()
        return graph, registry
```

**Tests:**
- `test_deterministic_planner_subclass` — custom planner returns expected graph
- `test_simple_planner_is_deterministic` — verify `SimplePlanner` still works
- `test_deterministic_planner_no_llm_cost` — verify no provider calls made

### 1b. ConstrainedPlanner (new file: smythe/constrained_planner.py)

The LLM selects from a menu of pre-built sub-graph templates. Templates are defined as Python objects or loaded from YAML.

```python
@dataclass
class SubGraphTemplate:
    """A reusable DAG fragment the constrained planner can select."""
    name: str
    description: str
    builder: Callable[[Task], tuple[list[Node], Registry]]

class ConstrainedPlanner(Planner):
    """LLM selects and composes from a fixed menu of sub-graph templates."""

    def __init__(
        self,
        provider: Provider,
        templates: list[SubGraphTemplate],
        model: str = "claude-mythos",
    ) -> None:
        self._provider = provider
        self._templates = templates
        self._model = model
```

The LLM receives a prompt listing templates by `name` + `description` and returns a JSON array of template names with optional parameters. The planner builds the graph from the selected templates.

Key design constraints:
- The LLM **cannot invent nodes** — it can only choose from templates.
- Templates are validated at registration time — structural errors are caught before execution.
- A cheaper/faster model is appropriate since the decision space is small.

**New file: smythe/constrained_prompts.py**
- `CONSTRAINED_SYSTEM_PROMPT` — lists available templates, output schema (JSON array of template selections)
- `build_constrained_user_prompt(task, templates)` — assembles menu + task description

**Tests:**
- `test_constrained_planner_selects_template` — mock provider returns template name, verify correct graph built
- `test_constrained_planner_rejects_unknown_template` — LLM returns nonexistent template name, verify error
- `test_constrained_planner_composes_multiple` — LLM selects 2 templates, verify they're composed into one graph

### 1c. PlannerRouter (new file: smythe/router.py)

Two routing strategies:

**Explicit routing** (already supported via `planner=` argument to `Swarm`):

```python
swarm = Swarm(planner=SiteBuilder(num_pages=5))
```

**Classifier routing** (new):

```python
class PlannerRouter:
    """Routes tasks to the appropriate planner tier based on classification."""

    def __init__(
        self,
        *,
        deterministic: dict[str, DeterministicPlanner] | None = None,
        constrained: ConstrainedPlanner | None = None,
        autonomous: LLMPlanner | None = None,
        classifier_provider: Provider | None = None,
        classifier_model: str = "gemini-2.0-flash",
    ) -> None: ...

    def route(self, task: Task) -> Planner:
        """Classify task and return the appropriate planner."""
```

The classifier is a lightweight LLM call that returns one of: `deterministic:<key>`, `constrained`, or `autonomous`. If no classifier provider is set, defaults to autonomous.

**Wire into Swarm:**
- Add `router: PlannerRouter | None = None` to `Swarm.__init__`
- When `router` is set and no explicit `planner` is provided, `plan()` calls `self._router.route(task)` to get the planner dynamically

**Tests:**
- `test_router_explicit` — deterministic planner keyed by name, verify correct one chosen
- `test_router_classifier` — mock classifier returns "constrained", verify constrained planner used
- `test_router_fallback` — no classifier, verify autonomous used

### 1d. Update exports (__init__.py)

Export `DeterministicPlanner`, `ConstrainedPlanner`, `SubGraphTemplate`, `PlannerRouter`.

---

## Phase 2: Executor Consolidation

`Executor` and `AsyncExecutor` share ~90% identical code: `_build_system_prompt`, `_build_user_prompt`, `_node_by_id`. They diverge only in how they drive the loop (sync/serial vs async/gather).

### Changes

**New file: smythe/executor_base.py**

Extract shared logic into `ExecutorBase`:

```python
class ExecutorBase:
    def __init__(self, provider, registry, tracer, budget=None): ...

    @staticmethod
    def _build_system_prompt(agent): ...

    @staticmethod
    def _build_user_prompt(node, dep_results): ...

    @staticmethod
    def _node_by_id(node_id, graph): ...

    def _gather_dep_results(self, node, graph): ...
```

Update `Executor(ExecutorBase)` and `AsyncExecutor(ExecutorBase)` to inherit.

**Tests:**
- All existing executor tests continue to pass unchanged

---

## Phase 3: Node Failure Handling

Currently, any node failure in either executor raises and aborts the entire graph. The README promises "deterministic guardrails" but there's no way to say "continue executing even if this branch fails."

### Changes

**smythe/graph.py**

Add `on_failure` field to `Node`:

```python
class FailurePolicy(Enum):
    HALT = "halt"       # default: stop the whole graph
    SKIP = "skip"       # mark dependents as SKIPPED, continue other branches
    RETRY = "retry"     # retry up to N times before failing

@dataclass
class Node:
    ...
    on_failure: FailurePolicy = FailurePolicy.HALT
    max_retries: int = 0
```

**smythe/executor.py + async_executor.py**

In `_execute_node`, instead of re-raising on failure:
- If `HALT`: re-raise (current behavior)
- If `SKIP`: mark node as FAILED, mark all transitive dependents as SKIPPED, continue
- If `RETRY`: retry up to `node.max_retries` times, then fall back to HALT or SKIP

**smythe/loader.py**

Support `on_failure` and `max_retries` in YAML node definitions.

**Tests:**
- `test_skip_policy_continues_other_branches` — 2 parallel branches, one fails with SKIP, other completes
- `test_retry_policy_retries_then_fails` — node fails twice, succeeds on third try
- `test_dependents_skipped_on_failure` — node fails with SKIP, its dependents are SKIPPED

---

## Phase 4: Synthesizer Intelligence

The current `Synthesizer` just concatenates results with `\n\n`. The README calls synthesis a "first-class tier."

### Changes

**smythe/synthesizer.py**

Add strategy-based synthesis:

```python
class SynthesisStrategy(Enum):
    CONCATENATE = "concatenate"    # current behavior (default)
    LLM_MERGE = "llm_merge"       # LLM merges results into coherent output
    STRUCTURED = "structured"     # JSON merge of structured outputs

class Synthesizer:
    def __init__(
        self,
        strategy: SynthesisStrategy = SynthesisStrategy.CONCATENATE,
        provider: Provider | None = None,
        model: str | None = None,
    ): ...
```

- `CONCATENATE`: current behavior, no change.
- `LLM_MERGE`: send all node results to the LLM with a merge prompt, get a coherent output.
- `STRUCTURED`: parse each result as JSON and deep-merge.

**smythe/swarm.py**

When `Swarm` auto-constructs the synthesizer, pass the provider so `LLM_MERGE` is available:

```python
self._synthesizer = synthesizer or Synthesizer(provider=self._provider, model=self.model)
```

**Tests:**
- `test_concatenate_strategy` — existing behavior
- `test_llm_merge_strategy` — mock provider, verify merge prompt sent
- `test_structured_strategy` — two JSON results, verify deep merge

---

## Phase 5: Capability-Aware Agent Assignment

The `Registry.assign()` method creates generic agents for unassigned nodes. The README says agents have "capability profiles" and the framework "learns which agents are best suited to which work."

### Changes

**smythe/registry.py**

Add capability matching to `assign()`:

```python
def assign(self, graph: ExecutionGraph) -> ExecutionGraph:
    for node in graph.nodes:
        if node.agent_id is None:
            match = self._best_match(node)
            if match:
                node.agent_id = match.id
            else:
                agent = Agent(profile=AgentProfile(name=f"agent-{node.id}"))
                self.register(agent)
                node.agent_id = agent.id
    return graph

def _best_match(self, node: Node) -> Agent | None:
    """Find the agent whose capabilities best overlap the node's requirements."""
    required = set(node.metadata.get("capabilities", []))
    if not required:
        return None
    best, best_score = None, 0
    for agent in self._agents.values():
        overlap = len(required & set(agent.profile.capabilities))
        if overlap > best_score:
            best, best_score = agent, overlap
    return best
```

**smythe/loader.py + prompts.py**

Allow nodes to declare `capabilities` in YAML and LLM output, used for matching:

```yaml
nodes:
  - id: research
    label: "Research the topic"
    capabilities: ["web-search", "summarization"]
```

**Tests:**
- `test_assign_matches_by_capability` — agent with matching capabilities is assigned
- `test_assign_fallback_creates_generic` — no matching agent, generic created
- `test_assign_skips_already_assigned` — node with agent_id is left alone

---

## Phase 6: README Update

Once the above phases are implemented, update `Readme.md`:

1. **Current Status** section — update "What exists" to reflect actual implemented features
2. **Architecture** — add a section showing the three-tier planner hierarchy
3. **Configuration** — document `FailurePolicy`, `SynthesisStrategy`, `PlannerRouter`
4. **Examples** — add a DeterministicPlanner example and a ConstrainedPlanner example

---

## Execution Order

| Phase | Dependencies | Estimated Effort |
|-------|-------------|-----------------|
| 1a. DeterministicPlanner | None | Small |
| 1b. ConstrainedPlanner | 1a | Medium |
| 1c. PlannerRouter | 1a, 1b | Medium |
| 1d. Update exports | 1a-1c | Trivial |
| 2. Executor consolidation | None (parallel with Phase 1) | Small |
| 3. Node failure handling | 2 | Medium |
| 4. Synthesizer intelligence | None (parallel) | Medium |
| 5. Capability-aware assignment | None (parallel) | Small |
| 6. README update | All above | Small |

Phases 1, 2, 4, and 5 are independent and can be built in parallel. Phase 3 depends on 2 (executor consolidation first). Phase 6 comes last.
