# Python 0.10 quickstart

> Historical Python reference. The active project is Rust-only. Run this guide's commands from [source revision 12f1470](https://github.com/petehottelet/smythe/tree/12f14704beff344228fffffce40d995daa1f7cce); see the [native guide](rust.md) for current behavior.

These examples use the existing Python implementation. For the Rust alpha,
see the [Rust quickstart](../README.md#quickstart) and [native guide](rust.md).
Model planning and paid provider integrations are currently Python-only.

## Quickstart

Python 3.11+:

```bash
pip install smythe
```

Plan, run, and recover a workflow with no API key:

```python
from smythe import OfflineProvider, SQLiteWorkflowStore, Swarm, Task

# The plan a model would generate. OfflineProvider returns it to the planner
# and echoes each step, so everything below runs without an API key.
plan = {
    "topology": ["fork_join"],
    "nodes": [
        {"id": "sqlite", "label": "Assess SQLite"},
        {"id": "postgres", "label": "Assess PostgreSQL"},
        {"id": "duckdb", "label": "Assess DuckDB"},
        {"id": "pick", "label": "Recommend one database",
         "depends_on": ["sqlite", "postgres", "duckdb"]},
    ],
}

with SQLiteWorkflowStore("smythe-runs.db") as store:
    swarm = Swarm(
        provider=OfflineProvider(plan=plan),
        run_store=store,
        max_budget_usd=1.00,
        parallel=True,
    )
    graph = swarm.plan(Task("Compare SQLite, PostgreSQL, and DuckDB for a local analytics app."))
    print(graph)  # inspect the generated graph before anything runs

    result = swarm.execute(graph)
    print(result.output)

    replay = swarm.resume(result.execution_id)  # replayed from the journal, no new calls
    print(replay.output == result.output)
```

`print(graph)` shows the graph before anything runs:

```text
TaskGraph(topology="fork-join")
├─ fork (parallel):
│   ├─ agent-sqlite: Assess SQLite
│   ├─ agent-postgres: Assess PostgreSQL
│   └─ agent-duckdb: Assess DuckDB
└─ join: agent-pick: Recommend one database
```

The three assessments run in parallel under the $1 budget, and the SQLite
journal records every call, so `resume` returns the finished run without
calling the provider again.

### With a real model

Install a provider extra and set its API key. With `OPENAI_API_KEY`,
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
writes the plan and the answers:

```bash
pip install "smythe[openai]"
```

```python
from smythe import OpenAIResponsesProvider, SQLiteWorkflowStore, Swarm, Task

with SQLiteWorkflowStore("smythe-runs.db") as store:
    swarm = Swarm(
        model="gpt-6-astra",
        provider=OpenAIResponsesProvider(
            reasoning_effort="medium",
            max_output_tokens=8192,
        ),
        run_store=store,
        max_budget_usd=50.00,
        parallel=True,
        max_concurrency=8,
    )
    graph = swarm.plan(Task(
        goal="Compare SQLite, PostgreSQL, and DuckDB for a local analytics app.",
        constraints=["Stay under 400 words", "Recommend one database"],
    ))
    print(graph)
    result = swarm.execute(graph)
    print(result.output)
```

With `ANTHROPIC_API_KEY`,
[Claude Fable 5.1](https://platform.claude.com/docs/en/about-claude/models/overview)
does the same through the native Claude Messages provider:

```bash
pip install "smythe[anthropic]"
```

```python
from smythe import AnthropicMessagesProvider, SQLiteWorkflowStore, Swarm, Task

with SQLiteWorkflowStore("smythe-runs.db") as store:
    swarm = Swarm(
        model="claude-fable-5-1",
        provider=AnthropicMessagesProvider(
            reasoning_effort="medium",
            max_output_tokens=8192,
        ),
        run_store=store,
        max_budget_usd=50.00,
        parallel=True,
        max_concurrency=8,
    )
    graph = swarm.plan(Task(
        goal="Compare SQLite, PostgreSQL, and DuckDB for a local analytics app.",
        constraints=["Stay under 400 words", "Recommend one database"],
    ))
    print(graph)
    result = swarm.execute(graph)
    print(result.output)
```

Both examples make paid API calls under a **$50 run allowance**. The SQLite
ledger accounts for planning and execution, reserves requests before dispatch,
and retains responses for recovery. See [budget scope](budgets.md),
[durable text workflows](workflow-accounting.md) and
[native Claude Messages](anthropic-messages.md), which covers the
`claude-fable-5-1` text workflows it supports. Gemini installs the same way,
with `smythe[gemini]`; durable run stores accept the OpenAI Responses and Claude
Messages providers. [More examples](../examples/README.md).
