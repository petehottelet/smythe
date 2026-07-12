"""Select-from-N: parallel image candidates, judged by a node that sees them.

    python examples/11_vision_judge.py

Three candidate images generate in parallel; an ArtDirector node with
``attach_dep_artifacts=True`` receives the actual image bytes as
multimodal input — not just file paths — and picks the strongest one.
This is the curation tier: wide fan-out occasionally produces
near-duplicates or off-brief outputs (measured in
benchmarks/image_benchmarks.md), and a judge that can see fixes that.

Offline (no keys): deterministic PNGs flow through the same attachment
path and the judge acknowledges how many images it saw.

Real mode: set GOOGLE_API_KEY. Candidates use a Gemini image model; the
judge uses a Gemini text+vision model. The sample price ceiling is
user-maintained; verify current pricing and set
``GEMINI_IMAGE_MAX_COST_PER_CALL_USD`` before a live run.
"""

import os
import sys
import tempfile
from pathlib import Path

from smythe import OfflineProvider, Swarm
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider import GeminiProvider

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

live = bool(os.environ.get("GOOGLE_API_KEY"))
if live:
    ceiling = os.environ.get("GEMINI_IMAGE_MAX_COST_PER_CALL_USD")
    if not ceiling:
        raise SystemExit(
            "Live mode requires GEMINI_IMAGE_MAX_COST_PER_CALL_USD. Verify current "
            "pricing and set an inclusive per-call ceiling."
        )
    provider = GeminiProvider(
        cost_per_image_usd=0.039,
        max_cost_per_call_usd=float(ceiling),
    )
    image_model = "gemini-2.5-flash-image"
    judge_model = "gemini-flash-lite-latest"
else:
    print("No GOOGLE_API_KEY found - running offline with OfflineProvider.")
    print("Set GOOGLE_API_KEY to generate and judge real images.\n")
    provider = OfflineProvider(artifacts_per_call=1)
    image_model = judge_model = "demo-model"

BRIEF = (
    "for 'Osiris', a portable solar-powered phone charger. Brand palette: "
    "warm amber, matte black, off-white. Premium minimalist ad design."
)

candidates = [
    Node(id=f"candidate-{i}", label=f"Generate a square social-media ad, variant {i + 1}, {BRIEF}",
         metadata={"model": image_model})
    for i in range(3)
]
judge = Node(
    id="art-director",
    label=(
        "You are the art director. The candidate ads are attached as images. "
        "Judge them against the brief: brand palette adherence, typography "
        "quality, composition, and overall polish. Name the winning candidate "
        "by number, explain why in two sentences, and note any candidate that "
        "should be rejected outright."
    ),
    depends_on=[c.id for c in candidates],
    attach_dep_artifacts=True,
    metadata={"model": judge_model},
)
graph = ExecutionGraph(
    topology=[Topology.BROADCAST_REDUCE], nodes=[*candidates, judge],
)

artifact_dir = os.environ.get("SMYTHE_ARTIFACT_DIR") or tempfile.mkdtemp(
    prefix="smythe_vision_judge_",
)

swarm = Swarm(
    provider=provider,
    model=image_model,
    parallel=True,
    max_budget_usd=0.50,
    artifact_dir=Path(artifact_dir),
)
result = swarm.execute(graph)

print("=== Candidates ===")
for node in result.graph.nodes[:-1]:
    for record in node.metadata.get("artifacts", []):
        print(f"  {node.id}: {record['path']}")
print("\n=== Art director's verdict ===")
print(result.graph.nodes[-1].result)
print(f"\nTotal cost: ${result.total_cost_usd:.4f}")
