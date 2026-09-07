# Smythe examples

Examples 01–05, 08–14, and the acquisition-diligence example run offline
out of the box. A built-in `DemoProvider` returns deterministic responses so
you can inspect planning, fan-out, budgets, recovery, and synthesis without an
API key. Examples 06 and 07 are live MCP integration tours. Run examples from
the repository root:

```bash
pip install -e .            # from the repo root
python examples/01_quickstart_yaml.py
```

The durable artifact-job example uses the installed CLI. It is completely
offline and its approval token is bound to the exact manifest and budget:

```bash
smythe jobs validate examples/12_jobs_manifest.yaml
smythe jobs plan examples/12_jobs_manifest.yaml
# Copy the printed approval token into the next command.
smythe jobs run examples/12_jobs_manifest.yaml --approve approve_v1_...
smythe jobs list
smythe jobs inspect RUN_ID --out job-report.html
```

| Example | What it shows |
|---|---|
| [GPT-6 Astra quickstart](../README.md#quickstart) | Generate, inspect, and execute a text-only task graph with `gpt-6-astra` using the published package. |
| [Native Astra/Sol Responses](../docs/openai-responses.md) | Count and quote an exact request before generation, inspect native token prices, and retain function-tool continuation and failed-response receipts. Requires the current repository checkout. |
| [Astra campaign preparation](../benchmarks/astra_benchmark_plan.md#prepared-experiment) | Validate 13 original task/source packs and reproduce the 12-pilot/200-main schedule locally, with no API calls. Factual checks remain separate from quality scoring. |
| [Glyph Rain](../screensaver/README.md) | A 192-node artifact workflow with verified native downloads using 56 reference and 192 original SVG shapes, mixed 90/10. The [web explorer](../screensaver/svg-preview/README.md) adapts the MIT reference renderer and base artwork, mixes in 10% original glyphs, and adds Matrix green rain, presets, VT323 pixel controls, and a Trajan Bold outline logo. Browser interaction checks pass; performance remains unmeasured. [Fresh-generation measurements](../benchmarks/svg_glyph_benchmark.md) cover the original catalog; [controlled 64–256-node scaling](../benchmarks/glyph_screensaver_benchmark.md) has its own protocol. |
| [01_quickstart_yaml.py](01_quickstart_yaml.py) | Load a declarative YAML DAG ([01_pipeline.yaml](01_pipeline.yaml)) with [halt, retry, and skip policies](../docs/execution.md) and per-node timeouts, execute it in parallel. Iterative graph traversal also supports [deep dependency chains](../docs/execution.md#deep-graphs). |
| [02_dynamic_planning.py](02_dynamic_planning.py) | The `LLMArchitect` designs the execution graph from the task itself. Inspect it, then execute with the [complete task snapshot](../docs/tasks.md). |
| [03_parallel_budget.py](03_parallel_budget.py) | Eight-node broadcast under a USD budget cap with `max_concurrency=3`, a per-node cost breakdown, and [strict cost guardrails](../docs/budgets.md). |
| [04_resume_after_crash.py](04_resume_after_crash.py) | Durable execution: resume preserves completed nodes and their costs. [Verification recovery](../docs/verifier.md#recovery-and-concurrent-work) also completes pending rejection and regeneration decisions before dispatch. |
| [05_mcp_filesystem.py](05_mcp_filesystem.py) | MCP tool use, fully offline: an agent reads real files through a bundled MCP server ([mcp_file_server.py](mcp_file_server.py)) via the bounded tool loop. Needs `pip install smythe[mcp]`. |
| [06_mcp_github.py](06_mcp_github.py) | The real GitHub MCP server with a mandatory tool allowlist and `env_passthrough` for the token. Env-gated: needs `GITHUB_PERSONAL_ACCESS_TOKEN`, an LLM key, and npx. |
| [07_mcp_saas.py](07_mcp_saas.py) | Any SaaS MCP server over streamable HTTP (Linear, Notion, ...), configured entirely by environment variables. Env-gated. |
| [08_learning_loop.py](08_learning_loop.py) | The learning loop, end to end: run 1's outcome is recorded by `PlannerMemory`, recalled into run 2's planning prompt, and the Architect returns a leaner plan (8 nodes → 3). |
| [09_image_generation.py](09_image_generation.py) | Parallel image artifacts, per-image cost accounting, and deterministic offline image fixtures; real mode uses Gemini. |
| [10_gpt_image_generation.py](10_gpt_image_generation.py) | Three GPT Image requests fan out concurrently through the dedicated `OpenAIImageProvider`; offline mode remains free and deterministic. |
| [11_vision_judge.py](11_vision_judge.py) | Select-from-N curation: parallel ad candidates judged by an art-director node that sees the actual images (`attach_dep_artifacts=True`). In its first real run the judge caught a spelling error in a generated ad. |
| [12_jobs_manifest.yaml](12_jobs_manifest.yaml) | A zero-cost, four-operation artifact job for the installed `smythe jobs` validate, plan, run, list, inspect, status, resume, reroll, and export workflow. Lease epochs fence worker writes; inspect shows attempt history, costs, and artifact integrity in a self-contained HTML report. |
| [13_adaptive_supervision.py](13_adaptive_supervision.py) | A plan that corrects itself: research surfaces a conflict, and the supervisor inserts a reconciliation step ahead of the write step instead of letting the contradiction reach the deliverable. |
| [14_durable_text_workflow.py](14_durable_text_workflow.py) | Plan and execute one journaled text run, inspect exact API-cost balances, then recover its cached output. [Native Astra version, saved graph limits, and scope](../docs/workflow-accounting.md). |
| [acquisition_diligence/](acquisition_diligence/) | **Complete acquisition-diligence example.** Task intake → generated `fork-join → adversarial → serial` topology → parallel specialists → red-team review → final memo, with committed expected artifacts (graph, trace, memo). |
