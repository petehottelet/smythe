# Smythe examples

Examples 01–05, 08–20 and the acquisition-diligence example run offline
out of the box. A built-in `DemoProvider` returns deterministic responses so
you can inspect planning, fan-out, budgets, recovery, and synthesis without an
API key. Examples 06 and 07 are live MCP integration tours. These examples
target [Smythe 0.9.1](https://github.com/petehottelet/smythe/releases/tag/v0.9.1)
and use files from the repository. Run them from its root:

```bash
pip install -e .            # from the repo root
python examples/14_durable_text_workflow.py
```

Examples 14, 15, 16, 17, 18, 19 and 20 explicitly select fixture responses even when API keys are set.
Example 14 demonstrates planning, execution and recovery with zero provider calls.
Examples 16 and 18 read their bundled sources from a server they start on the
loopback address and make no other network request. Example 19 publishes to an
in-memory service it defines and makes no network request. Example 20 uses a
scripted provider that calls tools, which `OfflineProvider` does not.
The [0.9.1 release guide](../docs/release-0.9.1.md) covers upgrading from 0.9.0,
the [0.9.0 guide](../docs/release-0.9.0.md) upgrading from 0.8.2, the
[0.8.2 guide](../docs/release-0.8.2.md) upgrading from 0.8.1, the
[0.8.1 guide](../docs/release-0.8.1.md) upgrading from 0.8.0, and the
[0.8.0 guide](../docs/release-0.8.0.md) the low-level Autotune migration.

The durable artifact-job example uses the installed CLI. Install its optional
dependencies with `pip install -e ".[jobs]"` from this checkout, or
`pip install "smythe[jobs]==0.9.1"` for the release package. The example is
completely offline and its approval token is bound to the exact manifest and
budget:

```bash
smythe jobs validate examples/12_jobs_manifest.yaml
smythe jobs plan examples/12_jobs_manifest.yaml
# Copy the printed approval token into the next command.
smythe jobs run examples/12_jobs_manifest.yaml --approve approve_v1_...
smythe jobs list
smythe jobs inspect RUN_ID --out job-report.html
smythe jobs review RUN_ID
```

Each approval token creates one run: repeating `jobs run` with the same token
is refused with exit code 4. Use `jobs resume RUN_ID` to continue the run, or
add `--new-run` to start another one.

For long jobs on supported hosts, add `--detach` to `run` or `resume` so the launcher can exit,
follow with `jobs attach`,
and use `jobs stop` to save a pause request while admitted calls drain.
Windows hosts must permit process breakaway; refused starts retain the run
for foreground resume without authorizing a worker.
[Operator commands and recovery semantics](../docs/jobs.md#detached-execution-and-attachment).
Use the saved `artifact_directory` or exported `artifact_root` to locate files;
new runs receive persistent namespaces independent of their public run IDs.
The [5,000-operation recovery report](../benchmarks/jobs_scale_5000_20260907_results.md)
retains a full offline fixture campaign on the earlier schema-v3 runtime;
it verifies safe resume and explicit unknown rerolls separately from the
current operator commands and schema-v4 namespaces.

The repository's [Autotune concurrency command](../docs/optimize.md#run-the-offline-campaign)
remains offline. Its [campaign ownership](../docs/optimize.md#campaign-ownership)
update acquires and renews a lease automatically; the CLI needs no new flags.
Low-level ledger callers must pass explicit lease tokens. These changes shipped
in 0.8.0 and are part of the package pinned above.

Export a completed local campaign with the report command:

```bash
smythe optimize concurrency --ledger local-experiment.sqlite3 --json
# Copy campaign_id from the output.
smythe optimize inspect CAMPAIGN_ID --ledger local-experiment.sqlite3 --out report.html
```

The first command runs the built-in simulator with zero provider calls. The
second exports its saved evidence; open `report.html` in a browser. It shows
decisions, comparison intervals, policies, trial details, and exact costs.
Use a new output filename for each export.
[Report limits and evidence scope](../docs/optimize.md#export-a-campaign-report).

Review the decision, put its policy in effect in a local policy registry, and
report the target across campaigns:

```bash
smythe optimize review CAMPAIGN_ID --ledger local-experiment.sqlite3 \
    --registry local-registry.sqlite3 --verdict approve --reviewer NAME
# Copy the printed review ID.
smythe optimize activate max_concurrency REVIEW_ID --registry local-registry.sqlite3 \
    --expected-seq 0 --actor NAME --reason "approved for new runs"
smythe optimize history max_concurrency --ledger local-experiment.sqlite3 \
    --registry local-registry.sqlite3
```

Activation never edits configuration; `rollback` and `retire` change the
target the same way, each naming the head seq it expects. `history` changes
neither file and sums what the campaigns cost apart from any inference
savings.
[Registry commands and exit codes](../docs/optimize.md#review-and-change-targets-from-the-cli).

| Example | What it shows |
|---|---|
| [Packages and typing](../docs/distribution.md) | Build and test a slim source package and verify typed sync/async consumers against an installed wheel. |
| [Current materials](../docs/current-materials.md) | Revised 192/256 contact sheets, detail sheets, animation, still image, native source exports, and each benchmark's evidence status. |
| [GPT-6 Astra and Claude Fable 5.1 quickstarts](../README.md#with-a-real-model) | Generate, inspect, and execute a text-only task graph through native Responses or native Claude Messages, with a $50 run allowance, concurrency eight, and planning included in the SQLite ledger. |
| [Native Astra/Sol Responses](../docs/openai-responses.md) | Smythe counts and quotes an exact request before generation, records native token prices, and retains function-tool continuation and failed-response receipts. |
| [Astra campaign preparation](../benchmarks/astra_benchmark_plan.md#prepared-experiment) | Validate 13 original task/source packs and reproduce the 12-pilot/200-main schedule locally, with no API calls. Factual checks remain separate from quality scoring. |
| [Glyph contour study](../benchmarks/noumenon/catalog/README.md) | Inspect all 192 v2 glyphs, their small-size sheets, and individual SVGs. Current previews and native source exports use v2. [Matched 192/256 workflow measurements](../benchmarks/svg_v2_results.md) cover compilation, validation and export; the live catalog remains 192. |
| [Astra findings and reproduction](../benchmarks/astra_findings.md) | Read the completed comparison and independently check archived members, native accounting, human-output bindings and both charts without provider calls. |
| [Astra campaign runners and results](../benchmarks/results/astra_20260913_main/README.md) | Inspect all 200 matched outcomes (191 automatically accepted), both native ledgers, blind judgments and the full retained cost reserve. Human review accepts all eight flagged answers at 4/4; the original automatic scores remain unchanged. Reproduce the analysis without paid calls; [runner guide](../benchmarks/astra_runtime.md). |
| [Planner-only instructions](../docs/workflow-accounting.md#freeze-graph-limits) | Bind graph policy through `LLMArchitect.planning_instructions`, with answer requirements kept in `Task.constraints`. |
| [Fable 5.1 study](../benchmarks/fable_51_benchmark_plan.md) | [Native and Code pilots](../benchmarks/results/fable_20260914_pilot/README.md) and the [ten-task Code Workflow study](../benchmarks/results/fable_code_20260914/README.md) have complete evidence. Human pilot ratings gate the 100 native main workflows and matched comparison. |
| [Native Claude Messages](../docs/anthropic-messages.md) | Fable text workflows with exact cache billing, retained raw responses, and local recovery. |
| [Noumenon fan-out](../benchmarks/noumenon_benchmark.md) | Smythe's parallel-processing example: a 192-node artifact workflow in which each glyph is an independent task, validated objectively and assembled into an atlas, still, GIF and HTML page. [Controlled 64–256-node scaling](../benchmarks/noumenon_benchmark.md#width-scaling-from-64-to-256-nodes) has its own protocol; [current 192/256 workflow measurements](../benchmarks/svg_v2_results.md) cover the authored v2 method. The [Noumenon screensaver](https://github.com/petehottelet/noumenon) renders the catalog with a web explorer and source-only native ports. |
| [01_quickstart_yaml.py](01_quickstart_yaml.py) | Load a declarative YAML DAG ([01_pipeline.yaml](01_pipeline.yaml)) with [halt, retry, and skip policies](../docs/execution.md) and per-node timeouts, execute it in parallel. Iterative graph traversal also supports [deep dependency chains](../docs/execution.md#deep-graphs). |
| [02_dynamic_planning.py](02_dynamic_planning.py) | The `LLMArchitect` designs the execution graph from the task itself. Inspect it, then execute with the [complete task snapshot](../docs/tasks.md). |
| [03_parallel_budget.py](03_parallel_budget.py) | Eight-node broadcast under a USD budget cap with `max_concurrency=3`, a per-node cost breakdown, and [strict cost guardrails](../docs/budgets.md). |
| [04_resume_after_crash.py](04_resume_after_crash.py) | Resume preserves completed nodes and their costs. [File checkpoints](../docs/checkpoint-format.md) flush complete snapshots and use independent temporary files for atomic publication. [Verification recovery](../docs/verifier.md#recovery-and-concurrent-work) completes pending rejection and regeneration decisions before dispatch. |
| [05_mcp_filesystem.py](05_mcp_filesystem.py) | MCP tool use, fully offline: an agent reads real files through a bundled MCP server ([mcp_file_server.py](mcp_file_server.py)) via the bounded tool loop. Needs `pip install "smythe[mcp]==0.9.1"`. |
| [06_mcp_github.py](06_mcp_github.py) | The real GitHub MCP server with a mandatory tool allowlist and `env_passthrough` for the token. Env-gated: needs `GITHUB_PERSONAL_ACCESS_TOKEN`, an LLM key, and npx. |
| [07_mcp_saas.py](07_mcp_saas.py) | Any SaaS MCP server over streamable HTTP (Linear, Notion, ...), configured entirely by environment variables. Env-gated. |
| [08_learning_loop.py](08_learning_loop.py) | Record run 1 in `PlannerMemory` and recall it into run 2's planning prompt; the Architect returns a leaner plan (8 nodes → 3). The [history validation](../docs/architecture.md#learning-loop) skips malformed recalled fields and labels summed node time. |
| [09_image_generation.py](09_image_generation.py) | Parallel image artifacts, per-image cost accounting, and deterministic offline image fixtures; real mode uses Gemini. |
| [10_gpt_image_generation.py](10_gpt_image_generation.py) | Three GPT Image requests fan out concurrently through the dedicated `OpenAIImageProvider`; offline mode remains free and deterministic. |
| [11_vision_judge.py](11_vision_judge.py) | Select-from-N curation: parallel ad candidates judged by an art-director node that sees the actual images (`attach_dep_artifacts=True`). In its first real run the judge caught a spelling error in a generated ad. |
| [12_jobs_manifest.yaml](12_jobs_manifest.yaml) | A zero-cost, four-operation artifact job for the installed `smythe jobs` validate, plan, run, list, inspect, status, attach, stop, resume, reroll, and export workflow. Lease epochs fence worker writes; inspect shows attempt history, costs, and artifact integrity in a self-contained HTML report. |
| [13_adaptive_supervision.py](13_adaptive_supervision.py) | A plan that corrects itself: research surfaces a conflict, and the supervisor inserts a reconciliation step ahead of the write step instead of letting the contradiction reach the deliverable. |
| [14_durable_text_workflow.py](14_durable_text_workflow.py) | Plan and execute one journaled text run, inspect exact API-cost balances, then recover its cached output. [Native Astra version, saved graph limits, and scope](../docs/workflow-accounting.md). |
| [15_strict_acceptance.py](15_strict_acceptance.py) | A draft passes its gate, then a later step drops a required section. Without a policy the defective brief is the output; a [strict acceptance policy](../docs/verifier.md#strict-delivery-acceptance) checks the exact text returned with a local final validator, rejects it and keeps it as `rejected_output`. Without a final validator the strict run is refused before any provider call. |
| [16_source_to_artifact.py](16_source_to_artifact.py) | A durable research run from [bundled sources](source_to_artifact/sources/) to an accepted document: plan the reads, fetch each source with a [journaled read step](../docs/workflow-accounting.md#recorded-read-steps), stop the run part way as a lost process would, resume without reading a saved source again, publish the brief with its [source evidence](../docs/source-evidence.md), accept it under a strict policy, and [inspect](../docs/runs.md#research-runs) the document, reads, evidence and open questions. |
| [17_contribution_templates.py](17_contribution_templates.py) | `SubGraphTemplate` values whose parallel workers each contribute something distinct: one extractor per source, one review by specialists with their own personas and capabilities, and documents that can each be checked alone. A `WhiteRabbit` routes the task to a `ConstrainedArchitect` that composes them, and the [contribution lint](../docs/architecture.md#contribution-lint) flags nothing in that plan. The lint records a repeated fan-out of identical workers under `graph.planning["coordination"]` and leaves the plan unchanged. |
| [18_archive_maintenance.py](18_archive_maintenance.py) | Maintain an archive of four accepted briefs with [artifact lineage](../docs/lineage.md). One shared source gets a new version, one published brief is damaged and one final validator's version is bumped. A [repair plan](../docs/lineage.md#plan-a-repair) shows before anything runs that the two briefs reading the revised source regenerate only its read, its extraction and the brief, that the damaged brief is published again and that the untouched one is decided again. [Applying it](../docs/lineage.md#apply-a-repair-plan) does that under an explicit allowance, and the results each successor kept are inspected apart from the conclusions that changed. A full recomputation in a second journal publishes the same bytes with the same decisions, and a table compares the reads, model steps, local steps, publications, acceptance evaluations and new runs each took. |
| [19_governed_publication.py](19_governed_publication.py) | An accepted document delivered to another service as a governed [external action](../docs/actions.md): a strict durable run publishes a release announcement, the journal prepares an action bound to its exact bytes, an approval verified by a local approval host authorizes it, and the first submission commits but loses its answer. `smythe runs inspect` exits with 12 until [reconciliation](../docs/actions.md#reconciliation) finds the key, reads the post back and reconciles the action with one write on the service. |
| [20_agent_skills.py](20_agent_skills.py) | An [Agent Skill](../docs/skills.md) whose `SKILL.md` asks for Bash and Write lends two agents a capability tag and is granted nothing. [`python_skill()`](../docs/skills.md#python-backed-skills) makes it a tool backed by a Python function, and a `LocalToolRuntime` offers that tool to the developer agent's id alone: a generated plan routes both steps through the tag, the model calls the tool in the developer's step, and the editor, holding the same tag, is offered no tool. The tool's name, version and schema digest are printed for records. |
| [acquisition_diligence/](acquisition_diligence/) | **Complete acquisition-diligence example.** Task intake → generated `fork-join → adversarial → serial` topology → parallel specialists → red-team review → final memo, with committed expected artifacts (graph, trace, memo). |
