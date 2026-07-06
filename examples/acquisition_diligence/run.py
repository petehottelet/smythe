"""Flagship demo: acquisition diligence with an adversarial review tier.

    python examples/acquisition_diligence/run.py

Task intake -> Architect-generated topology -> three parallel
specialists -> join -> red-team review -> final structured memo.

Runs offline out of the box against a committed fixture plan (no keys,
no cost); the expected graph, trace, and memo live in expected/ so you
know what success looks like. Set ANTHROPIC_API_KEY, OPENAI_API_KEY,
or GOOGLE_API_KEY to watch a real model design and execute the DAG.

    python examples/acquisition_diligence/run.py --write-artifacts expected

regenerates the committed artifacts from a fixture run.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from fixtures import FixtureProvider

from smythe import Swarm, Task

# ExecutionGraph.__str__ and the tracer render with characters that
# legacy cp1252 Windows consoles can't encode.
for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")


def pick_provider():
    """Return (provider, model) — a real provider if an API key is set, else fixtures."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from smythe.provider import AnthropicProvider
        return AnthropicProvider(), "claude-mythos"
    if os.environ.get("OPENAI_API_KEY"):
        from smythe.provider import OpenAIProvider
        return OpenAIProvider(), "gpt-5.2"
    if os.environ.get("GOOGLE_API_KEY"):
        from smythe.provider import GeminiProvider
        return GeminiProvider(), "gemini-3-flash"
    print("No API key found - running offline against the committed fixture plan.")
    print("Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY for real output.\n")
    return FixtureProvider(), "fixture-model"


def sanitized_trace(result, *, fixture: bool) -> list[dict]:
    """Trace spans in graph order, with agent names instead of run-local ids.

    Fixture-mode durations reflect the OS clock's resolution, not real
    work, so they are zeroed to keep committed artifacts reproducible.
    """
    names = {n.id: n.metadata.get("agent_name") for n in result.graph.nodes}
    order = {n.id: i for i, n in enumerate(result.graph.nodes)}
    spans = sorted(result.trace, key=lambda s: order.get(s["node_id"], len(order)))
    spans = [{**s, "agent_id": names.get(s["node_id"], s["agent_id"])} for s in spans]
    if fixture:
        spans = [{**s, "duration_ms": 0.0} for s in spans]
    return spans


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--write-artifacts", metavar="DIR", default=None,
    help="write graph.mmd, trace.json, and memo.md to DIR after the run",
)
args = parser.parse_args()

provider, model = pick_provider()
fixture_mode = isinstance(provider, FixtureProvider)
swarm = Swarm(provider=provider, model=model, parallel=True, max_budget_usd=2.00)

task = Task(
    goal=(
        "Evaluate whether Acme Corp is a viable acquisition target. "
        "Analyze their financials, technical IP, and regulatory exposure, "
        "then produce a diligence memo with a go/no-go recommendation."
    ),
    constraints=[
        "Red-team every bullish claim before it reaches the memo",
        "Flag any SEC or antitrust risk factors",
        "Final output must be structured: summary, findings, risks, recommendation",
    ],
)

graph = swarm.plan(task)
print("=== The Architect's plan ===")
print(graph)

result = swarm.execute(graph)

# The memo is the graph's terminal node — whatever the Architect named it.
terminals = [n for n in result.graph.nodes if not result.graph.dependents(n.id)]
print("\n=== Final memo ===")
for node in terminals:
    print(node.result)

print("\n=== Trace ===")
for span in sanitized_trace(result, fixture=fixture_mode):
    print(f"  {span['status']:>9}  {span['agent_id']}: {span['label'][:60]}")
print(f"\nTotal cost: ${result.total_cost_usd:.4f}")

if args.write_artifacts:
    out = Path(args.write_artifacts)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.mmd").write_text(result.graph.to_mermaid() + "\n", encoding="utf-8")
    (out / "trace.json").write_text(
        json.dumps(sanitized_trace(result, fixture=fixture_mode), indent=2) + "\n",
        encoding="utf-8",
    )
    memo = "\n\n".join(str(n.result) for n in terminals)
    (out / "memo.md").write_text(memo + "\n", encoding="utf-8")
    print(f"Artifacts written to {out}")
