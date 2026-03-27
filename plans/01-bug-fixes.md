# Plan 1 — Bug Fixes

Fix 7 concrete bugs across planner, budget, memory, provider, loader, and swarm.

---

## Bug 1: Nested event loop crash in async planning path

**Severity: Critical**

`LLMPlanner.plan()` calls `asyncio.run()` (planner.py line 73). When `Swarm(parallel=True).execute(task)` is used, the call chain is:

```
execute() → asyncio.run(execute_async()) → self.plan(task) → LLMPlanner.plan() → asyncio.run(...)
```

The inner `asyncio.run()` raises `RuntimeError: cannot be called from a running event loop`.

### Changes

**smythe/planner.py**

1. Add `async def aplan()` to the `Planner` ABC with a default implementation that delegates to `plan()`:

```python
class Planner(ABC):
    @abstractmethod
    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]: ...

    async def aplan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        return self.plan(task)
```

2. In `LLMPlanner`, move the core planning loop into `aplan()`, using `await self._provider.complete(...)` directly instead of `asyncio.run()`. Make the sync `plan()` a thin wrapper:

```python
def plan(self, task):
    return asyncio.run(self.aplan(task))

async def aplan(self, task):
    history = self._get_history(task)
    user_prompt = build_user_prompt(task, history)
    last_error = None
    for attempt in range(1 + self._max_retries):
        prompt = ...  # same logic
        result = await self._provider.complete(
            PLANNING_SYSTEM_PROMPT, prompt, model=self._planning_model
        )
        try:
            data = self._extract_json(result.text)
            graph, registry = build_graph_from_dict(data)
            graph.estimated_cost_usd = self._estimate_cost(graph)
            return graph, registry
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            continue
    raise PlanningError(...)
```

**smythe/swarm.py**

3. Add `async def aplan()` that mirrors `plan()` but calls `await self._planner.aplan(task)`.
4. Update `execute_async()` to call `await self.aplan(task)` instead of `self.plan(task)`.

### Tests

- `test_llm_planner_aplan` — call `aplan()` from an `async` test function
- `test_execute_parallel_with_llm_planner` — `Swarm(parallel=True)` with a mock planning provider, end-to-end

---

## Bug 2: Budget overspend in parallel waves

**Severity: High**

In `async_executor.py` lines 45-49, budget `check()` is called for all ready nodes *before* `asyncio.gather()`. All pass because `_spent` is still $0. Then all execute concurrently, each calling `record()` afterward — potentially exceeding the cap.

Example: budget=$0.01, 3 ready nodes each cost $0.005. All 3 pass check, all 3 run, total becomes $0.015.

### Changes

**smythe/budget.py**

1. Add `_reservations: dict[str, float]` to `__init__`.
2. Add `reserve(node_id, estimated_cost)`:
   - If `_spent + estimated_cost > max_budget_usd`, raise `BudgetExhaustedError`.
   - Otherwise, add to `_reservations` and increment `_spent`.
3. Update `record()` to undo the reservation before recording the actual cost:
   ```python
   def record(self, node_id, result):
       reserved = self._reservations.pop(node_id, 0.0)
       self._spent -= reserved
       cost = result.total_tokens * self.cost_per_token
       self._node_costs[node_id] = cost
       self._spent += cost
       return cost
   ```

**smythe/async_executor.py**

4. Add `estimated_tokens_per_node: int = 2000` to `__init__`.
5. Replace `check()` loop with `reserve()` loop using estimated cost.

Serial `executor.py` keeps `check()` — no concurrency issue there.

### Tests

- `test_budget_reserve_prevents_overspend` — 3-node graph, budget for ~2 nodes, verify `BudgetExhaustedError` on third
- `test_budget_record_replaces_reservation` — reserve then record, verify total reflects actual not estimated
- Verify existing `test_executor_halts_on_budget` still passes

---

## Bug 3: Retry prompt drops task context

**Severity: High**

On retry (planner.py line 72), only `RETRY_PROMPT` is sent — a generic "try again" message. The original task goal, constraints, and history are completely lost.

### Changes

**smythe/planner.py** (inside the `aplan` loop after Bug 1 fix)

Concatenate the original prompt with retry context:

```python
if attempt == 0:
    prompt = user_prompt
else:
    prompt = (
        user_prompt
        + "\n\n---\n\n"
        + f"Your previous response could not be parsed: {last_error}\n\n"
        + RETRY_PROMPT
    )
```

### Tests

- Update `test_plan_retries_on_malformed_json` — capture the retry prompt and assert it contains the original goal text

---

## Bug 4: Memory recall crashes on corrupt JSONL lines

**Severity: Medium**

`PlannerMemory.recall()` (memory.py line 116) catches `json.JSONDecodeError` but not `TypeError`/`KeyError` from constructing `ExecutionOutcome` when required fields are missing.

### Changes

**smythe/memory.py**

Widen the exception handling:

```python
try:
    data = json.loads(line)
    outcome = ExecutionOutcome(**{
        k_: data[k_]
        for k_ in ExecutionOutcome.__dataclass_fields__
        if k_ in data
    })
except (json.JSONDecodeError, TypeError, KeyError):
    continue
```

### Tests

- `test_recall_skips_corrupt_lines` — write `{"partial": true}` plus a valid line, verify only valid one returns

---

## Bug 5: Provider response parsing fragility

**Severity: Medium**

- `AnthropicProvider` (provider.py line 64): `response.content[0].text` crashes on empty content or non-text blocks.
- `OpenAIProvider` (provider.py line 101): `response.choices[0].message.content` crashes if choices is empty.

### Changes

**smythe/provider.py**

```python
# AnthropicProvider.complete()
text_blocks = [b for b in response.content if hasattr(b, "text")]
text = text_blocks[0].text if text_blocks else ""

# OpenAIProvider.complete()
text = ""
if response.choices:
    text = response.choices[0].message.content or ""
```

### Tests

No new tests (API-dependent edge cases). The guards prevent crashes on edge cases the SDKs could produce.

---

## Bug 6: `from_yaml()` stores graph but provides no ergonomic way to run it

**Severity: Medium**

`Swarm.from_yaml()` (swarm.py line 204) stores the loaded graph as `instance._yaml_graph`, but users must manually extract it. The docstring says "Call execute() with the loaded graph" but doesn't make it easy.

### Changes

**smythe/swarm.py**

1. Initialize `self._yaml_graph: ExecutionGraph | None = None` in `__init__`.
2. Allow `execute()` with no arguments when a YAML graph is loaded:

```python
def execute(self, task_or_graph: Task | ExecutionGraph | None = None) -> SwarmResult:
    if task_or_graph is None:
        if self._yaml_graph is not None:
            task_or_graph = self._yaml_graph
        else:
            raise ValueError(
                "No task or graph provided, and no YAML graph loaded"
            )
    ...
```

### Tests

- `test_from_yaml_execute_no_args` — load YAML, call `execute()`, verify it runs the loaded graph

---

## Bug 7: Loader crashes on malformed node entries

**Severity: Low**

If YAML `nodes` contains a non-dict (e.g. bare string), `entry.get("id")` raises `AttributeError`.

### Changes

**smythe/loader.py**

Add type validation at the start of `build_graph_from_dict`:

```python
nodes_raw = data.get("nodes", [])
if not isinstance(nodes_raw, list):
    raise ValueError(f"'nodes' must be a list, got {type(nodes_raw).__name__}")

for i, entry in enumerate(nodes_raw):
    if not isinstance(entry, dict):
        raise ValueError(f"Node at index {i} must be a mapping, got {type(entry).__name__}")
```

### Tests

- `test_load_non_dict_node_raises` — YAML with bare string in nodes list
- `test_load_non_list_nodes_raises` — YAML where `nodes:` is a string

---

## Execution order

1. Bug 1 (async planner) — critical, blocks parallel execution
2. Bug 2 (budget reservation) — high, financial correctness
3. Bug 3 (retry context) — high, planner reliability
4. Bug 4 (memory corrupt) — medium, resilience
5. Bug 5 (provider guards) — medium, resilience
6. Bug 6 (from_yaml execute) — medium, ergonomics
7. Bug 7 (loader validation) — low, defensive coding
8. Run full test suite
