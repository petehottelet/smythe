# Smythe examples

Every example runs offline out of the box — a built-in `DemoProvider` returns canned responses so you can see the machinery (planning, fan-out, budget enforcement, synthesis) without an API key. Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` and the same script runs against a real model.

```bash
pip install -e .            # from the repo root
python examples/01_quickstart_yaml.py
```

| Example | What it shows |
|---|---|
| [01_quickstart_yaml.py](01_quickstart_yaml.py) | Load a declarative YAML DAG ([01_pipeline.yaml](01_pipeline.yaml)) with failure policies and per-node timeouts, execute it in parallel. |
| [02_dynamic_planning.py](02_dynamic_planning.py) | The `LLMArchitect` designs the execution graph from the task itself. Inspect the plan before executing it. |
| [03_parallel_budget.py](03_parallel_budget.py) | Eight-node broadcast under a USD budget cap with `max_concurrency=3`, plus the per-node cost breakdown. |
