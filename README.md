<div align="center">
  <img src="assets/wordmark.svg" alt="SMYTHE" width="340">

  <p><em>task-based agent swarms with dynamic parallelization, routing, and execution topology.</em></p>

  <p>
    <a href="https://pypi.org/project/smythe/"><img src="assets/badges/pypi.svg" alt="PyPI v0.6.0"></a>
    <a href="https://github.com/petehottelet/smythe/actions/workflows/ci.yml"><img src="assets/badges/ci.svg" alt="CI checks"></a>
    <img src="assets/badges/python.svg" alt="Python 3.11, 3.12, and 3.13">
    <a href="LICENSE"><img src="assets/badges/license.svg" alt="License: MIT"></a>
  </p>

  <p>
    <a href="#glyph-rain">Glyph Rain</a> ·
    <a href="#benchmarks">Benchmarks</a> ·
    <a href="#quickstart">Quickstart</a> ·
    <a href="#why-smythe">Why Smythe</a> ·
    <a href="docs/index.md">Documentation</a>
  </p>
</div>

**Give Smythe a goal. It generates an inspectable task graph and executes it
with bounded concurrency, execution budgets, verification, traces, and recovery.**

The graph defines the work; the durable execution envelope governs the run.
Glyph Rain shows both ideas in one visible result.

**Measured against CrewAI on the matched framework suite: 77% fewer tokens
and 28% less wall time.** Five tasks, three repetitions, the same executor
model and pipeline, with blind cross-vendor judging.

## Glyph Rain

One goal becomes 192 independent glyph tasks and a working screensaver. The
original vector glyphs fall in luminous green streams across three depth
layers, using the same catalog on web, Windows, macOS, and Linux.

<p align="center">
  <img src="assets/glyph_rain/glyph-rain-screenshot.png" alt="Glyph Rain with heavy green glyphs, varied glow, and descending streams at three depths" width="900">
</p>

**Download:** [Windows `.scr`](https://github.com/petehottelet/smythe/raw/refs/heads/main/screensaver/dist/SmytheGlyphRain.scr) ·
[macOS universal `.zip`](https://github.com/petehottelet/smythe/raw/refs/heads/main/screensaver/dist/GlyphRain-macos-universal.zip) ·
[Linux x86-64 `.tar.gz`](https://github.com/petehottelet/smythe/raw/refs/heads/main/screensaver/dist/SmytheGlyphRain-linux-x86_64.tar.gz) ·
[Web and native source](screensaver/) ·
[192-glyph atlas](assets/glyph_rain/glyph-atlas.png) ·
[256-glyph atlas](benchmarks/partitions/glyph_256/assets/glyph-atlas.png)

Compiled downloads passed native rendering and motion checks on Windows,
Apple Silicon, Intel Mac, and Ubuntu 22.04/24.04.
[Checksums and verification](screensaver/README.md#native-verification).
macOS uses an ad-hoc signature; Linux requires X11.

Each tile is normalized, checked for dimensions and uniqueness, and hashed
before assembly. [Run, build, and customize Glyph Rain](screensaver/README.md).

The [next-version plan](docs/glyph-rain-plan.md) specifies original SVG glyphs,
reference-based styling, and arrow-key exploration through a 3D field.

## Benchmarks

The glyph workload measures parallel artifact generation. Separate matched
suites measure recovery, framework overhead, and generated plans. Each result
links to its protocol and committed records.

### Glyph generation and scaling

**192 verified glyphs in 20.5 seconds.** At concurrency 64, the glyph workload
ran **56.2× faster** than serial execution. This is a controlled offline
measurement with 5.8 seconds of simulated provider latency per call.
All measured 64-, 128-, 192-, and 256-node runs produced complete sets of valid,
unique tiles. This measures artifact generation, not screensaver frame rate.

<p align="center">
  <img src="assets/benchmarks/glyph_scaling.svg" alt="Controlled offline glyph generation at four graph widths, with all tiles valid and unique at every measured concurrency" width="900">
</p>

[Glyph protocol and records](benchmarks/glyph_screensaver_benchmark.md).

### Recovery after interruption

A separate matched durability test measures work repeated after a hard kill.
Smythe repeated **8 calls versus LangGraph's 32**, a **75% reduction**, across
three repetitions. [Recovery protocol](benchmarks/durability_benchmark.md).

### Framework efficiency

The framework suite compares orchestration on a fixed pipeline. Smythe used
**77% fewer tokens and 28% less wall time than CrewAI** across five tasks and
three repetitions per framework. All runs use the same executor model and
three-stage pipeline, with blind cross-vendor judging.

<p align="center">
  <img src="assets/benchmarks/framework_callouts.svg" alt="Smythe uses 77 percent fewer mean tokens and 28 percent less mean wall time than CrewAI on the matched fixed-pipeline suite" width="900">
</p>

<p align="center">
  <img src="assets/benchmarks/framework_comparison.svg" alt="Smythe, LangGraph, and CrewAI: observed blind quality, mean token counts, and mean wall time across 15 runs per framework" width="900">
</p>

Smythe also recorded **6% less mean wall time than LangGraph**. Its observed
quality score was **9.73/10**, versus 9.53 for both comparisons. These are
suite results; token counts describe model usage, not invoice savings.
[Protocol and records](benchmarks/README.md#corrected-framework-head-to-head-langgraph-and-crewai-2026-07-12).

### Generated execution topology

The task-shape suite compares generated plans with a fixed pipeline across
five task shapes. Smythe recorded **14% less wall time**, including planning.
It used one node for a simple transformation and an average of 5.3 for parallel
research. Observed quality averaged 9.47/10 versus 9.33/10, within measured
judge variation.

<p align="center">
  <img src="assets/benchmarks/shape_efficiency.svg" alt="Generated plans adapt node count to the task and reduce mean wall time by 14 percent, including planning, across the task-shape suite" width="900">
</p>

[Task-shape protocol and records](benchmarks/shape_suite.md).

Charts are generated from committed records. The [benchmark index](benchmarks/README.md)
documents each comparison, its scope, and its evidence status.

## Why Smythe

| Generated execution topology | Durable execution envelope |
|---|---|
| Generate a DAG from the goal with `LLMArchitect` | Bound active calls with `max_concurrency` |
| Select approved templates with `ConstrainedArchitect` | Reserve execution spend before dispatch |
| Build exact workflows with `DeterministicArchitect` | Save node results and resume from checkpoints |
| Inspect and export plans before execution | Validate artifacts and gate results |
| Reuse successful graphs as templates | Trace calls, costs, failures, and revisions |

Agents use MCP tools, generate images, and pass artifacts to downstream nodes.
Durable Jobs add manifest validation, plan approvals, an attempt journal,
selective rerolls, and portable exports.

[Architecture](docs/architecture.md) · [Jobs and CLI](docs/jobs.md) ·
[MCP](docs/mcp.md) · [Verification](docs/verifier.md) ·
[All guides and examples](docs/index.md).

## Quickstart

Python 3.11+. Install the provider used below:

```bash
pip install "smythe[openai]"
```

Set `OPENAI_API_KEY`, then generate and inspect a plan:

```python
from smythe import Swarm, Task

swarm = Swarm(
    model="gpt-5.4-mini",
    max_budget_usd=0.50,
    parallel=True,
    max_concurrency=8,
)

task = Task(
    goal="Compare SQLite, PostgreSQL, and DuckDB for a local analytics app.",
    constraints=["Keep the comparison under 400 words"],
    done_when=["Explain the tradeoffs and recommend one database"],
)

graph = swarm.plan(task)
print(graph)

result = swarm.execute(graph)
print(result.output)
print(f"execution cost: ${result.total_cost_usd:.4f}")
```

The budget covers execution and synthesis; planning calls are separate.
Anthropic and Gemini use the `smythe[anthropic]` and `smythe[gemini]` extras.

Try the complete acquisition-diligence workflow without an API key:

```bash
git clone https://github.com/petehottelet/smythe.git
cd smythe
pip install -e ".[dev]"
python examples/acquisition_diligence/run.py
```

Three specialists work in parallel, an editor assembles their findings, a red
team challenges the draft, and a final node writes the decision memo.
[Graph, trace, and expected output](examples/acquisition_diligence/).

## Coming soon

- **Complete workflow cost accounting:** include planning and supervision in
  one spend ledger, then publish repeated cost comparisons from native usage.
- **Runtime hardening:** synchronize verification with active descendants,
  enforce serial halt behavior, reject invalid cost inputs, and carry the full
  task context through planning, execution, and resume.
- **Operator tools:** inspect runs, detach long jobs, and approve durable pauses.
- **Glyph Rain exploration:** independently drawn SVG glyphs and a navigable 3D
  field, specified in the [style and implementation plan](docs/glyph-rain-plan.md).
- **Native distribution:** notarized macOS downloads and native Wayland integration.
- **Broader evidence:** larger stress tests, repeated live glyph sweeps, and
  human-calibrated quality comparisons with saved outputs and judge reasoning.

[Specifications and priorities](ROADMAP.md#coming-soon) ·
[Repository review](docs/project-review-2026-09-06.md).

Smythe is pre-1.0; minor releases may change APIs.
[Release history](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) ·
[Security](SECURITY.md) · [MIT license](LICENSE).
