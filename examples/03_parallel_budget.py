"""Budget-capped parallel execution with a concurrency limit.

    python examples/03_parallel_budget.py

Eight independent nodes fan out under a USD budget cap. The Sentinel
reserves estimated cost before each wave so concurrent nodes can't
collectively overshoot, and max_concurrency keeps at most three
provider calls in flight at once.
"""

from _providers import pick_provider

from smythe import Swarm
from smythe.graph import ExecutionGraph, Node, Topology

provider, model = pick_provider()

TOPICS = [
    "battery chemistry", "solar cell efficiency", "waterproofing",
    "USB-C power delivery", "thermal management", "drop resistance",
    "manufacturing cost", "recyclability",
]

nodes = [
    Node(id=f"topic-{i}", label=f"Write two sentences about {topic} in portable chargers")
    for i, topic in enumerate(TOPICS)
]
graph = ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)

swarm = Swarm(
    provider=provider,
    model=model,
    parallel=True,
    max_concurrency=3,   # at most 3 provider calls in flight
    max_budget_usd=0.50,  # hard USD cap, enforced by reservation
)

result = swarm.execute(graph)

print("=== Per-node cost breakdown ===")
for node in result.graph.nodes:
    cost = node.metadata.get("cost_usd", 0.0)
    print(f"  {node.id:<10} {node.status.value:<10} ${cost:.5f}")
print(f"\nTotal cost: ${result.total_cost_usd:.4f} (cap was $0.50)")
