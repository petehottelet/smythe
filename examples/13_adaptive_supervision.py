"""A plan that corrects itself mid-run.

    python examples/13_adaptive_supervision.py

The Architect plans once, before any work exists. When the work reveals
that the plan was wrong, a supervisor revises what has not run yet —
adding the missing step, dropping work that turned out to be pointless.

This example uses a deterministic supervisor so the correction is
visible and free: the research step reports that its sources conflict,
and the supervisor inserts a reconciliation step ahead of the write
step rather than letting a contradiction flow into the deliverable.

Docs: docs/supervisor.md
"""

import sys

from smythe import OfflineProvider, Revision, Supervisor, Swarm
from smythe.graph import ExecutionGraph, Node, Topology

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")


class ReconcileConflicts(Supervisor):
    """Insert a reconciliation step when research reports a conflict."""

    async def review(self, graph, node, *, task, revisions_remaining):
        if node.id != "research" or "conflict" not in str(node.result).lower():
            return None
        return Revision(
            add_nodes=(
                Node(
                    id="reconcile",
                    label="Reconcile the conflicting sources before writing",
                    depends_on=["research"],
                ),
            ),
            # The write step now waits for reconciliation instead of
            # consuming the contradiction directly.
            rewire={"write": ("reconcile",)},
            reason="research surfaced conflicting sources",
        )


provider = OfflineProvider(
    responses=[
        "Sources conflict: two datasets disagree on the headline number.",
        "Reconciled: the later dataset supersedes the earlier one.",
        "Final brief, written from the reconciled figure.",
    ],
)

graph = ExecutionGraph(
    topology=[Topology.SERIAL],
    nodes=[
        Node(id="research", label="Research the topic"),
        Node(id="write", label="Write the brief", depends_on=["research"]),
    ],
)

print("Running offline with smythe's built-in OfflineProvider (no keys, no cost).\n")

print("=== Planned ===")
for node in graph.nodes:
    print(f"  {node.id:<10} depends on: {node.depends_on or '-'}")

swarm = Swarm(
    provider=provider,
    model="demo-model",
    parallel=True,
    supervisor=ReconcileConflicts(),
    max_revisions=1,
    artifact_dir=None,
)
result = swarm.execute(graph)

print("\n=== Executed ===")
for node in result.graph.nodes:
    print(f"  {node.id:<10} depends on: {node.depends_on or '-'}")
    print(f"             {str(node.result)[:70]}")

print("\n=== Revisions ===")
for span in result.trace:
    if "revision" in span:
        print(f"  {span['status']}: {span['label']}")
        print(f"    added={span['revision']['added']} "
              f"rewired={span['revision']['rewired']}")
