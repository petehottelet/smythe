# Smythe examples

Every example runs offline out of the box — a built-in `DemoProvider` returns canned responses so you can see the machinery (planning, fan-out, budget enforcement, synthesis) without an API key. Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` and the same script runs against a real model. Run any example from the repo root:

```bash
pip install -e .            # from the repo root
python examples/01_quickstart_yaml.py
```

| Example | What it shows |
|---|---|
| [01_quickstart_yaml.py](01_quickstart_yaml.py) | Load a declarative YAML DAG ([01_pipeline.yaml](01_pipeline.yaml)) with failure policies and per-node timeouts, execute it in parallel. |
| [02_dynamic_planning.py](02_dynamic_planning.py) | The `LLMArchitect` designs the execution graph from the task itself. Inspect the plan before executing it. |
| [03_parallel_budget.py](03_parallel_budget.py) | Eight-node broadcast under a USD budget cap with `max_concurrency=3`, plus the per-node cost breakdown. |
| [04_resume_after_crash.py](04_resume_after_crash.py) | Durable execution: the provider dies mid-run, the checkpoint survives, and `swarm.resume()` finishes the job without re-running completed nodes. |
| [05_mcp_filesystem.py](05_mcp_filesystem.py) | MCP tool use, fully offline: an agent reads real files through a bundled MCP server ([mcp_file_server.py](mcp_file_server.py)) via the bounded tool loop. Needs `pip install smythe[mcp]`. |
| [06_mcp_github.py](06_mcp_github.py) | The real GitHub MCP server with a mandatory tool allowlist and `env_passthrough` for the token. Env-gated: needs `GITHUB_PERSONAL_ACCESS_TOKEN`, an LLM key, and npx. |
| [07_mcp_saas.py](07_mcp_saas.py) | Any SaaS MCP server over streamable HTTP (Linear, Notion, ...), configured entirely by environment variables. Env-gated. |
| [08_learning_loop.py](08_learning_loop.py) | The learning loop, end to end: run 1's outcome is recorded by `PlannerMemory`, recalled into run 2's planning prompt, and the Architect returns a leaner plan (8 nodes → 3). |
| [acquisition_diligence/](acquisition_diligence/) | **The flagship demo.** Task intake → generated `fork-join → adversarial → serial` topology → parallel specialists → red-team review → final memo, with committed expected artifacts (graph, trace, memo). |
