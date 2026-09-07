"""Optional graph limits bound to a durable text workflow's saved recipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from smythe.workflow_binding import WorkflowBindingError

if TYPE_CHECKING:
    from smythe.graph import ExecutionGraph


@dataclass(frozen=True, slots=True)
class WorkflowGraphPolicy:
    """Reject graphs that exceed explicitly frozen execution limits.

    ``node_model`` matches each node's effective model, including the Swarm
    model inherited when metadata omits it. Retry and regeneration limits cap
    the corresponding node fields; they do not alter or repair generated work.
    """

    max_nodes: int
    node_model: str | None = None
    max_retries: int | None = None
    max_regenerations: int | None = None

    def __post_init__(self) -> None:
        if type(self) is not WorkflowGraphPolicy:
            raise WorkflowBindingError("graph_policy must be the built-in WorkflowGraphPolicy")
        if type(self.max_nodes) is not int or self.max_nodes < 1:
            raise WorkflowBindingError("Graph policy max_nodes must be a positive integer")
        if self.node_model is not None and (
            type(self.node_model) is not str or not self.node_model.strip()
        ):
            raise WorkflowBindingError("Graph policy node_model must be a nonempty string or None")
        for name in ("max_retries", "max_regenerations"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise WorkflowBindingError(
                    f"Graph policy {name} must be a non-negative integer or None"
                )

    def to_dict(self) -> dict:
        self.__post_init__()
        return {"version": 1, "max_nodes": self.max_nodes, "node_model": self.node_model,
                "max_retries": self.max_retries, "max_regenerations": self.max_regenerations}

    @classmethod
    def from_dict(cls, value: dict) -> WorkflowGraphPolicy:
        keys = {"version", "max_nodes", "node_model", "max_retries", "max_regenerations"}
        if type(value) is not dict or value.keys() != keys:
            raise WorkflowBindingError("Graph policy has an invalid schema")
        if type(value["version"]) is not int or value["version"] != 1:
            raise WorkflowBindingError("Unsupported graph policy version")
        return cls(**{key: value[key] for key in keys - {"version"}})

    def validate(self, graph: ExecutionGraph, *, default_model: str) -> None:
        """Check without changing the graph, its nodes, or their metadata."""
        self.__post_init__()
        if len(graph.nodes) > self.max_nodes:
            raise WorkflowBindingError(
                f"Graph policy max_nodes={self.max_nodes} exceeded by {len(graph.nodes)} nodes"
            )
        for node in graph.nodes:
            model = node.metadata.get("model", default_model)
            if self.node_model is not None and (
                type(model) is not str or model != self.node_model
            ):
                raise WorkflowBindingError(
                    f"Graph policy requires node {node.id!r} model {self.node_model!r}; got {model!r}"
                )
            for name in ("max_retries", "max_regenerations"):
                ceiling = getattr(self, name)
                value = getattr(node, name)
                if ceiling is not None and (
                    type(value) is not int or value < 0 or value > ceiling
                ):
                    raise WorkflowBindingError(
                        f"Graph policy requires node {node.id!r} {name} <= {ceiling}; got {value!r}"
                    )


def snapshot_graph_policy(value: WorkflowGraphPolicy | None) -> WorkflowGraphPolicy | None:
    """Validate and detach a policy before binding it to a run."""
    if value is None:
        return None
    if type(value) is not WorkflowGraphPolicy:
        raise WorkflowBindingError("graph_policy must be the built-in WorkflowGraphPolicy")
    return WorkflowGraphPolicy.from_dict(value.to_dict())
