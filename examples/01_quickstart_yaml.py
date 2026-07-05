"""Quickstart: load a YAML-defined DAG and execute it.

    python examples/01_quickstart_yaml.py

Runs offline with canned responses; set an API key for real output
(see _providers.py).
"""

from pathlib import Path

from _providers import pick_provider

from smythe import Swarm

provider, model = pick_provider()

pipeline = Path(__file__).with_name("01_pipeline.yaml")
swarm = Swarm.from_yaml(str(pipeline), provider=provider, model=model, parallel=True)

result = swarm.execute()

print("=== Final output ===")
print(result.output)
print(f"\nNodes executed: {len(result.graph.nodes)}")
print(f"Total cost: ${result.total_cost_usd:.4f}")
