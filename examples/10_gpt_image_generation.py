"""Three GPT Image jobs executed concurrently through Smythe.

    python examples/10_gpt_image_generation.py

Offline mode is automatic and free.  For a live run, set
``OPENAI_API_KEY`` in the process environment; the example intentionally
does not read dotenv files or print credentials. The whole-request price
ceiling is intentionally explicit: verify current provider pricing and set
``GPT_IMAGE_MAX_COST_PER_CALL_USD`` for the selected model, size, and quality.
Live mode refuses to start without that user-maintained ceiling.
"""

import os
import sys
import tempfile
from pathlib import Path

from smythe import OfflineProvider, OpenAIImageProvider, Swarm
from smythe.graph import ExecutionGraph, Node, Topology


for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")


MODEL = "gpt-image-2-2026-04-21"
OUTPUT_ONLY_COST_PER_IMAGE_USD = 0.041

if os.environ.get("OPENAI_API_KEY"):
    ceiling = os.environ.get("GPT_IMAGE_MAX_COST_PER_CALL_USD")
    if not ceiling:
        raise SystemExit(
            "Live mode requires GPT_IMAGE_MAX_COST_PER_CALL_USD. Verify current "
            "pricing for this model/size/quality and set an inclusive per-call ceiling."
        )
    provider = OpenAIImageProvider(
        size="1536x1024",
        quality="medium",
        output_format="jpeg",
        output_compression=90,
        cost_per_image_usd=OUTPUT_ONLY_COST_PER_IMAGE_USD,
        max_cost_per_call_usd=float(ceiling),
    )
    model = MODEL
else:
    print("No OPENAI_API_KEY found - running offline with OfflineProvider.")
    print("Set OPENAI_API_KEY to generate real GPT Image artifacts.\n")
    provider = OfflineProvider(artifacts_per_call=1)
    model = "demo-image-model"


COMMON = (
    "Use case: photorealistic-natural. Asset type: landscape benchmark image. "
    "Primary request: a realistic editorial photograph of the Golden Gate Bridge. "
    "Style/medium: photorealistic natural photography with physically plausible "
    "atmosphere, structural detail, water, and light. Composition/framing: 3:2 "
    "landscape, complete bridge clearly recognizable, no text. Constraints: natural "
    "color, no logos, no watermark, no fantastical architecture, no duplicate bridge. "
)

nodes = [
    Node(
        id="dawn_fog",
        label=COMMON
        + "Scene/backdrop: dawn fog flowing through the bridge towers, viewed from "
        "Battery Spencer. Lighting/mood: cool dawn with restrained warm sunlight.",
    ),
    Node(
        id="marin_day",
        label=COMMON
        + "Scene/backdrop: crisp clear afternoon viewed from the Marin Headlands, "
        "San Francisco visible beyond. Lighting/mood: clean daylight and realistic haze.",
    ),
    Node(
        id="blue_hour",
        label=COMMON
        + "Scene/backdrop: blue hour from the city approach with subtle vehicle light "
        "trails and bay reflections. Lighting/mood: cinematic but natural, not neon.",
    ),
]
graph = ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)

configured_dir = os.environ.get("SMYTHE_ARTIFACT_DIR")
artifact_dir = Path(configured_dir) if configured_dir else Path(
    tempfile.mkdtemp(prefix="smythe_gpt_image_demo_")
)

swarm = Swarm(
    provider=provider,
    model=model,
    parallel=True,
    max_concurrency=3,
    max_budget_usd=0.25,
    artifact_dir=artifact_dir,
)
result = swarm.execute(graph)

print("=== Generated assets ===")
for node in result.graph.nodes:
    cost = node.metadata.get("cost_usd", 0.0)
    print(f"  {node.id:<10} {node.status.value:<10} ${cost:.4f}")
    for record in node.metadata.get("artifacts", []):
        print(f"             -> {record['path']} ({record['mime_type']})")
print(f"\nTotal recorded cost: ${result.total_cost_usd:.4f} (cap was $0.25)")
print(f"Artifacts written under: {artifact_dir}")
