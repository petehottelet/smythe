<div align="center">
  <img src="assets/wordmark.svg" alt="SMYTHE" width="340">
  <p><em>Turn a goal into an inspectable agent workflow. Run it with budgets and recovery.</em></p>
  <p>
    <a href="https://pypi.org/project/smythe/"><img src="assets/badges/pypi.svg" alt="PyPI release"></a>
    <a href="https://github.com/petehottelet/smythe/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush"><img src="https://github.com/petehottelet/smythe/actions/workflows/ci.yml/badge.svg?branch=main&amp;event=push" alt="CI status on main"></a>
    <img src="assets/badges/python.svg" alt="Python 3.11, 3.12, and 3.13">
    <a href="LICENSE"><img src="assets/badges/license.svg" alt="License: MIT"></a>
  </p>
  <p>
    <a href="#quickstart">Quickstart</a> ·
    <a href="#how-it-works">How it works</a> ·
    <a href="#measured-results">Measured results</a> ·
    <a href="docs/index.md">Documentation</a>
  </p>
</div>

**Smythe is a Python framework that plans and runs agent workflows.** Give it a
goal, inspect the generated task graph, and execute independent work in parallel.
Set spending and concurrency limits, verify outputs, and recover saved work after
an interruption.

Use it for research pipelines, document production, and artifact generation
where you need to see what will run and account for what happened.

<p align="center">
  <img src="assets/glyph_rain/glyph-rain-loop.gif" alt="Animated rain showing only the 192 revised Smythe glyphs" width="900">
</p>
<p align="center"><em><a href="screensaver/README.md">Glyph Rain</a>: its glyph catalog is compiled, validated, and exported by a Smythe workflow (<a href="benchmarks/svg_v2_results.md">benchmark</a>).</em></p>

## Quickstart

Python 3.11+. Install the released library with OpenAI support:

```bash
pip install "smythe[openai]==0.8.0"
```

Set `OPENAI_API_KEY`, then plan and execute with
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra):

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
        max_budget_usd=5.00,
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

This makes paid API calls under a **$5 run allowance**. The SQLite ledger
accounts for planning and execution, reserves requests before dispatch, and
retains responses for recovery. See [budget scope](docs/budgets.md) and
[durable text workflows](docs/workflow-accounting.md).

To try planning, execution, and recovery with **no API calls**, run the explicit
offline example from a source checkout:

```bash
git clone https://github.com/petehottelet/smythe.git
cd smythe
pip install -e .
python examples/14_durable_text_workflow.py
```

The example uses fixture responses and verifies that resuming produces the same
output. [More examples](examples/README.md).

## How it works

**The graph defines the work.** Smythe generates a directed acyclic graph for
the task, including dependencies and agent assignments. Inspect or export it
before execution. Use approved templates or a graph you write yourself when
the workflow is already known.

**The execution envelope governs the run.** Budgets, bounded concurrency,
verification, traces, artifacts, and recovery apply as the graph executes.
Durable Jobs add manifest approval, attempt history, selective rerolls, and
local HTML reports.

<p align="center">
  <img src="assets/diligence_pipeline.svg" alt="Acquisition diligence: parallel specialists, synthesis, adversarial review, and a final memo" width="900">
</p>

The [acquisition-diligence example](examples/acquisition_diligence/) shows
three specialists feeding an editor, a red-team review, and a final decision
memo. Its saved graph, trace, and expected output make the workflow inspectable.

| You need to… | Smythe provides |
|---|---|
| Adapt the workflow to the task | Generated graphs, approved templates, and deterministic planning |
| Control spending and parallel work | Request reservations and bounded concurrency |
| Recover interrupted work | Checkpoints, native response replay, and durable job journals |
| Check the deliverable | Output verification and artifact receipts |
| Understand a run | Graph exports, traces, costs, and inspection reports |

[Architecture](docs/architecture.md) · [Execution](docs/execution.md) ·
[Jobs](docs/jobs.md) · [Verification](docs/verifier.md) · [MCP tools](docs/mcp.md).

## Measured results

Each result links to its method and retained records.

| Study | Recorded result | Scope |
|---|---|---|
| [Framework comparison](benchmarks/README.md#corrected-framework-head-to-head-langgraph-and-crewai-2026-07-12) | 77% fewer mean tokens and 28% less mean wall time than CrewAI | Five tasks, three repetitions; matched executor and fixed pipeline; blind judging |
| [Interruption and recovery](benchmarks/durability_benchmark.md) | 8 repeated dispatches versus LangGraph's 32 | Three matched hard-kill trials with 64 operations |
| [SVG catalog workflow](benchmarks/svg_v2_results.md) | 256 SVGs in 8.06 seconds median; 2.20× the serial baseline | Local compilation, validation, and export of authored designs; no API calls |

The [200-workflow Astra/Sol study](benchmarks/astra_findings.md)
also reports the limits of generated plans: they increased mean time in both
models on its ten synthetic tasks. The frozen rule accepted 191/200 workflows;
human review accepted all eight disputed available answers. One missing usage
receipt limits affected exact cost comparisons. All outcomes remain published.

[All benchmarks, charts, and evidence status](benchmarks/README.md).

## Project status

**Smythe 0.8.0** is the current library release. See the
[release notes and upgrade guide](docs/release-0.8.0.md). The API is pre-1.0;
minor releases may change it. Later source changes appear in the
[changelog](CHANGELOG.md#unreleased).

Next priorities are complete-deliverable checks, broader external-task
benchmarks, and separately controlled Astra scheduler and framework studies.
See the [roadmap](ROADMAP.md) for status and acceptance criteria.

The [Glyph Rain screensaver](screensaver/README.md) is an artifact-generation
showcase with source builds and a [web explorer](screensaver/svg-preview/README.md).
Precompiled screensaver distribution is paused.

[Documentation](docs/index.md) · [Contributing](CONTRIBUTING.md) ·
[Releases](https://github.com/petehottelet/smythe/releases) ·
[Security](SECURITY.md) · [MIT license](LICENSE).
