"""Executor — walks the DAG and runs each node."""

from __future__ import annotations

from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.registry import Registry
from smythe.tracer import Tracer


class Executor:
    """Executes an assigned graph by walking it in dependency order.

    The default implementation runs nodes serially.  A future parallel
    executor will fan out independent nodes concurrently.
    """

    def __init__(self, registry: Registry, tracer: Tracer) -> None:
        self._registry = registry
        self._tracer = tracer

    def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Execute every node in the graph, respecting dependency order."""
        for node in self._walk(graph):
            self._execute_node(node, graph)
        return graph

    def _walk(self, graph: ExecutionGraph) -> list[Node]:
        """Topological sort of nodes by dependency order."""
        visited: set[str] = set()
        order: list[Node] = []
        lookup = {n.id: n for n in graph.nodes}

        def visit(node: Node) -> None:
            if node.id in visited:
                return
            visited.add(node.id)
            for dep_id in node.depends_on:
                visit(lookup[dep_id])
            order.append(node)

        for node in graph.nodes:
            visit(node)
        return order

    def _execute_node(self, node: Node, graph: ExecutionGraph) -> None:
        """Run a single node. Override to add real LLM dispatch."""
        node.status = NodeStatus.RUNNING
        self._tracer.on_node_start(node)

        try:
            agent = self._registry.get(node.agent_id) if node.agent_id else None
            dep_results = {
                dep_id: lookup.result
                for dep_id in node.depends_on
                if (lookup := self._node_by_id(dep_id, graph)) is not None
            }
            node.result = self._call(node, agent, dep_results)
            node.status = NodeStatus.COMPLETED
        except Exception as exc:
            node.status = NodeStatus.FAILED
            node.result = str(exc)
            self._tracer.on_node_error(node, exc)
            raise
        finally:
            self._tracer.on_node_end(node)

    def _call(self, node: Node, agent, dep_results: dict) -> str:
        """Placeholder for actual LLM invocation. Returns a stub response."""
        raise NotImplementedError(
            "LLM dispatch not yet implemented. "
            "Subclass Executor and override _call() to add provider integration."
        )

    @staticmethod
    def _node_by_id(node_id: str, graph: ExecutionGraph) -> Node | None:
        return next((n for n in graph.nodes if n.id == node_id), None)
