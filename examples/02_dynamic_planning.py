"""Dynamic planning: the LLMArchitect decides the topology, not you.

    python examples/02_dynamic_planning.py

You hand the Swarm a goal; the Architect returns an execution graph you
can inspect before committing to execution. Runs offline with a canned
plan; set an API key to watch a real model design the DAG.
"""

from _providers import pick_provider

from smythe import Swarm, Task

provider, model = pick_provider()

swarm = Swarm(provider=provider, model=model, max_budget_usd=1.00)

task = Task(
    goal=(
        "Produce a competitive brief on portable solar phone chargers: "
        "market landscape, top competitors, and a one-page summary."
    ),
    constraints=["Keep the final brief under 400 words"],
)

# Plan first — the graph is an artifact you can inspect (or reject).
graph = swarm.plan(task)
print("=== Planned execution graph ===")
print(graph)

# Then execute the plan you just saw.
result = swarm.execute(graph)
print("\n=== Final output ===")
print(result.output)
print(f"\nTotal cost: ${result.total_cost_usd:.4f}")
