"""Parallel image generation with per-image cost accounting.

    python examples/09_image_generation.py

Three ImageAgent nodes fan out in parallel, each generating one launch
asset for the same product. Generated images are persisted under an
artifact directory and their paths recorded on each node — the bytes
never enter checkpoints or planner memory.

Offline (no keys): OfflineProvider attaches a deterministic 1x1 PNG per
call, so the full artifact pipeline runs in CI for free.

Real mode: set GOOGLE_API_KEY to generate actual images with a Gemini
image model ("Nano Banana"). Image models bill per image, not per
token — cost_per_image_usd prices each call so the Sentinel's budget
cap stays honest.
"""

import os
import sys
import tempfile
from pathlib import Path

from smythe import OfflineProvider, Swarm
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider import GeminiProvider

# ExecutionGraph.__str__ and the tracer render with characters that
# legacy cp1252 Windows consoles can't encode.
for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

if os.environ.get("GOOGLE_API_KEY"):
    provider = GeminiProvider(cost_per_image_usd=0.039)
    model = "gemini-2.5-flash-image"
else:
    print("No GOOGLE_API_KEY found - running offline with smythe's built-in OfflineProvider.")
    print("Set GOOGLE_API_KEY to generate real images with a Gemini image model.\n")
    provider = OfflineProvider(artifacts_per_call=1)
    model = "demo-image-model"

BRIEF = (
    "for the launch of 'Osiris', a portable solar-powered phone charger. "
    "Brand palette: warm amber, matte black, off-white. "
    "Style: clean product photography, natural light."
)

nodes = [
    Node(id="hero", label=f"Generate a wide hero image {BRIEF}"),
    Node(id="social", label=f"Generate a square social-media post image {BRIEF}"),
    Node(id="banner", label=f"Generate a banner image with negative space for text {BRIEF}"),
]
graph = ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)

artifact_dir = Path(tempfile.mkdtemp(prefix="smythe_image_demo_"))

swarm = Swarm(
    provider=provider,
    model=model,
    parallel=True,          # all three images generate concurrently
    max_budget_usd=0.50,    # ~$0.04/image real cost; the cap has headroom
    artifact_dir=artifact_dir,
)

result = swarm.execute(graph)

print("=== Generated assets ===")
for node in result.graph.nodes:
    cost = node.metadata.get("cost_usd", 0.0)
    print(f"  {node.id:<8} {node.status.value:<10} ${cost:.4f}")
    for record in node.metadata.get("artifacts", []):
        print(f"           -> {record['path']} ({record['mime_type']})")
print(f"\nTotal cost: ${result.total_cost_usd:.4f} (cap was $0.50)")
print(f"Artifacts written under: {artifact_dir}")
