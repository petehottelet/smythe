<div align="center">
  <img src="assets/wordmark.svg" alt="SMYTHE" width="340">

  <p><em>task-based agent swarms with dynamic parallelization, routing, and execution topology.</em></p>

  <p>
    <a href="https://pypi.org/project/smythe/"><img src="assets/badges/pypi.svg" alt="PyPI v0.7.0"></a>
    <a href="https://github.com/petehottelet/smythe/actions/workflows/ci.yml"><img src="assets/badges/ci.svg" alt="CI checks"></a>
    <img src="assets/badges/python.svg" alt="Python 3.11, 3.12, and 3.13">
    <a href="LICENSE"><img src="assets/badges/license.svg" alt="License: MIT"></a>
  </p>

  <p>
    <a href="#benchmark">Benchmark</a> ·
    <a href="#process">Process</a> ·
    <a href="#quickstart">Quickstart</a> ·
    <a href="#why-smythe">Why Smythe</a> ·
    <a href="docs/index.md">Documentation</a>
  </p>
</div>

**Give Smythe a goal. It generates an inspectable task graph and executes it
with bounded concurrency, execution budgets, verification, traces, and recovery.** The graph defines the work; the durable execution envelope governs the run.

**Measured against CrewAI on the matched framework suite: 77% fewer tokens
and 28% less wall time.** Five tasks, three repetitions, the same executor
model and pipeline, with blind cross-vendor judging.

## Benchmark

Example task: One workflow creates and validates 192 original SVG glyphs.

<p align="center">
  <img src="screensaver/svg-preview/preview.png" alt="Classic code rain with the reference character set and occasional original Smythe glyphs" width="900">
</p>

**Explore:** [Run the web explorer](screensaver/svg-preview/README.md) ·
[Complete 192-glyph contact sheet](benchmarks/partitions/glyph_svg_v1/catalog/contact-sheet.png) ·
[56 reference glyphs](screensaver/svg-preview/reference/contact-sheet.png) ·
[24-glyph calibration sheet](benchmarks/partitions/glyph_svg_v1/catalog/calibration-sheet.png) ·
[Individual SVGs and manifest](benchmarks/partitions/glyph_svg_v1/catalog/).
In 3D mode, arrow keys move through the field; Space pauses, R resets the view, and F enters
fullscreen. Touch controls are included.
The [current browser review](benchmarks/partitions/glyph_rain_reference_v1/performance-preview-review-20260907.json)
binds this screenshot and passes 31 rendering and interaction checks.
A separate ten-minute travel and resize check passed 57 cycles.
[Rendering measurements and stability report](benchmarks/renderer_performance_20260907_results.md).

**Native downloads:** [Windows `.scr`](https://github.com/petehottelet/smythe/releases/download/v0.7.0/SmytheGlyphRain.scr) ·
[macOS universal `.zip`](https://github.com/petehottelet/smythe/releases/download/v0.7.0/GlyphRain-macos-universal.zip) ·
[Linux x86-64 `.tar.gz`](https://github.com/petehottelet/smythe/releases/download/v0.7.0/SmytheGlyphRain-linux-x86_64.tar.gz) ·
[v0.7.0 release and checksums](https://github.com/petehottelet/smythe/releases/tag/v0.7.0) ·
[Native source and setup](screensaver/README.md).

The compiled packages passed rendering and motion checks on Windows,
Apple Silicon, Intel Mac, and Ubuntu 22.04/24.04.
[Checksums and verification](screensaver/README.md#native-verification).
macOS uses an ad-hoc signature; Linux requires X11.
The native savers use layered trails; the web explorer provides the REGL effect,
3D navigation, and pixel settings.

The web renderer adapts [m8e](https://github.com/m8e/matrix-rain),
a fork of Rezmason. Native packages also include the reference artwork. Smythe's 192 added screensaver shapes have
independently authored contours from a [measured style brief](docs/glyph-rain-plan.md).
[Credits, licenses, and artwork provenance](screensaver/README.md#credits-and-references).

## Process

The glyph workload measures parallel artifact generation. Separate matched
suites measure recovery, framework overhead, and generated plans. Each result
links to its protocol and committed records.

### Original SVG generation

**192 original SVG glyphs in 4.03 seconds median**, including generation,
complete validation, and assembly. The best tested configuration used eight
process workers and ran **2.95× faster** than process execution at concurrency 1.
All 30 workflows delivered complete, accepted catalogs with identical hashes.

The new workflow measures fresh contour construction, full style and
distinctness checks, and delivery of the SVG catalog and raster atlas.
It runs locally through Smythe from a calibrated procedural grammar, with no
simulated delay and **$0 provider API charges**. Hardware and design work are
outside that API-cost figure.

<p align="center">
  <img src="assets/benchmarks/svg_workflow.svg" alt="Complete SVG workflow times across thread and process configurations, with all repetitions and a measured stage breakdown" width="900">
</p>

[Workflow results and protocol](benchmarks/svg_glyph_benchmark.md) ·
[Raw record](benchmarks/results/glyph_svg_v1.json).

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

### Jobs at 5,000 operations

**5,000 accepted artifacts after a hard process kill and recovery.** Safe
resume completed 2,500 pending operations and preserved the 2,492 already
accepted outputs. Eight interrupted operations required explicit rerolls;
their original unknown call records remain in the ledger. No accepted
operation was reissued; resuming the completed job made
zero new calls.

<p align="center">
  <img src="assets/benchmarks/jobs_scale.svg" alt="One offline Jobs campaign: 2,492 accepted after the kill, 4,992 after safe resume, and 5,000 after eight explicit rerolls" width="900">
</p>

One Windows campaign, concurrency eight, identical 1×1 PNG fixtures, and
**$0 provider API charges**. This tests durable recovery on the frozen
schema-v3 runtime; glyph generation, model quality, and the later schema-v4
operator features have separate evidence.
[Results, complete archive, and independent reconciliation](benchmarks/jobs_scale_5000_20260907_results.md).

### Recovery after interruption

A separate matched durability test measures work repeated after a hard kill.
Smythe repeated **8 calls versus LangGraph's 32**, a **75% reduction**, across
three repetitions. [Recovery protocol](benchmarks/durability_benchmark.md).

<p align="center">
  <img src="assets/benchmarks/recovery.svg" alt="Three matched interruption tests: Smythe repeated 8 dispatches and LangGraph repeated 32 in every repetition; both finished all 64 operations" width="900">
</p>

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
| Select approved templates with `ConstrainedArchitect` | Reserve supported text-workflow phases in one `run_store` ledger |
| Build exact workflows with `DeterministicArchitect` | Save node results and resume from checkpoints |
| Inspect and export plans with their complete task | Validate artifacts and recover verification decisions |
| Reuse successful graphs as templates | Trace calls, costs, failures, and revisions |

Agents use MCP tools, generate images, and pass artifacts to downstream nodes.
Durable Jobs add manifest validation, plan approvals, an attempt journal,
selective rerolls, detached workers on supported hosts, durable pauses, read-only inspection,
and portable exports. Inspect
prompts, responses, costs, and artifact receipts in a local HTML report.
Lease epochs reject stale-worker journal writes after ownership changes.
Persistent artifact namespaces and exclusive file publication preserve accepted
outputs across custom run IDs and shared output directories.
File checkpoints flush complete snapshots before atomic publication, using
independent temporary files for separate store instances.
Iterative graph traversal passes [5,000-node dependency-chain checks](docs/execution.md#deep-graphs),
including complete offline serial execution and atomic revision validation.
[Saved graph policies](docs/workflow-accounting.md#freeze-graph-limits) bound node
count, execution models, retries, and regeneration across planning and recovery.

[Architecture](docs/architecture.md) · [Task handoffs](docs/tasks.md) · [Jobs and CLI](docs/jobs.md) ·
[Failure policies](docs/execution.md) · [Cost guardrails](docs/budgets.md) ·
[MCP](docs/mcp.md) · [Verification](docs/verifier.md) ·
[Native Astra and Sol Responses](docs/openai-responses.md) ·
[Durable text accounting](docs/workflow-accounting.md) ·
[All guides and examples](docs/index.md).

## Quickstart

Python 3.11+. Install Smythe 0.7.0 with the provider used below:

```bash
pip install "smythe[openai]==0.7.0"
```

Set `OPENAI_API_KEY`, then generate and inspect a text-only plan with
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
    task = Task(
        goal="Compare SQLite, PostgreSQL, and DuckDB for a local analytics app.",
        constraints=[
            "Keep the comparison under 400 words",
            "Explain the tradeoffs and recommend one database",
        ],
    )
    graph = swarm.plan(task)
    print(graph)

    result = swarm.execute(graph)
    print(result.output)
```

This example makes paid API calls under a **$5 run allowance**. The
[Responses provider](docs/openai-responses.md) supplies native usage receipts
and model-specific prices; the [SQLite workflow ledger](docs/workflow-accounting.md)
includes planning and execution, reserves each request before dispatch, and
replays saved responses locally during recovery. Up to eight execution nodes
run concurrently. Anthropic and Gemini use the `smythe[anthropic]` and
`smythe[gemini]` extras outside this managed text-workflow path.

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

- **Native exploration:** bring the web exposure pipeline, camera controls,
  and settings to Windows, macOS, and Linux, then verify them against the
  [implementation plan](docs/glyph-rain-plan.md).
- **Native distribution:** notarized macOS downloads and native Wayland integration.
- **Renderer performance:** meet the 1080p frame-interval target with the new
  glyphs, then verify visible presentation and GPU timing. The
  [six-session headless study](benchmarks/renderer_performance_20260907_results.md)
  measured 56.21–56.24 draws/second and retained every result; its pacing target
  was not met.
- **Broader evidence:** bounded paid scale trials, repeated live glyph sweeps, and
  human-calibrated quality comparisons with saved outputs and judge reasoning.
- **Astra benchmarks:** matched model and orchestration comparisons with full
  usage accounting and blind quality scoring. The 13-task pack and schedules
  for 12 pilot and 200 main workflows are prepared. The
  [pilot runner](benchmarks/astra_runtime.md) binds spending allocations,
  source hashes, and recoverable trial identities. Live execution awaits the
  campaign allowance and [calibration gates](benchmarks/astra_benchmark_plan.md).

[Specifications and priorities](ROADMAP.md#coming-soon) ·
[Repository review](docs/project-review-2026-09-06.md).

Smythe is pre-1.0; minor releases may change APIs.
[Release verification](docs/release-0.7.0.md) ·
[Release history](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) ·
[Security](SECURITY.md) · [MIT license](LICENSE).
