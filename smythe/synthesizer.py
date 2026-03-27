"""Synthesizer — merges outputs from parallel execution branches."""

from __future__ import annotations

from smythe.graph import ExecutionGraph, NodeStatus


class Synthesizer:
    """Combines completed node results into a coherent final output.

    The default implementation concatenates results in dependency order.
    Future versions will support per-output-type strategies (e.g. structured
    merge for JSON, narrative weaving for prose, table joins for data).
    """

    def synthesize(self, graph: ExecutionGraph) -> str:
        """Produce a single output from the completed graph."""
        completed = [
            n for n in graph.nodes
            if n.status == NodeStatus.COMPLETED and n.result is not None
        ]
        if not completed:
            return ""
        return "\n\n".join(str(n.result) for n in completed)
