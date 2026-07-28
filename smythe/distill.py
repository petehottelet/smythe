"""Distillation — turn a run that worked into a template you can reuse.

`PlannerMemory` already records *that* a run succeeded. What it cannot
do is reuse the shape of that success: the next similar task is planned
from scratch, and pays the planning call, the planning variance, and
the risk of a worse plan all over again.

Distillation closes that loop. A completed graph becomes a
:class:`SubGraphTemplate` — the same object the `ConstrainedArchitect`
already selects from — so a proven topology can be offered as a menu
item instead of re-derived:

    template = distill_template(result.graph, name="diligence-review")
    architect = ConstrainedArchitect(provider=p, templates=[template])

What carries over is *structure*, not content: node labels, dependency
edges, agent personas and capabilities. Results, costs, and statuses
are deliberately left behind — a template is a shape to fill, not a
cached answer.
"""

from __future__ import annotations

import re

from smythe.agent import Agent, AgentProfile
from smythe.constrained_planner import SubGraphTemplate
from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.registry import Registry
from smythe.task import Task

_GOAL_TOKEN = re.compile(r"\{goal\}")


class DistillationError(ValueError):
    """Raised when a graph cannot be turned into a usable template."""


def distill_template(
    graph: ExecutionGraph,
    *,
    name: str,
    description: str | None = None,
    registry: Registry | None = None,
    require_success: bool = True,
) -> SubGraphTemplate:
    """Build a reusable template from a graph that ran.

    Args:
        graph: The executed graph to learn from.
        name: Template identifier shown in the architect's menu.
        description: What the template is for. Defaults to a summary of
            the shape it captures.
        registry: Registry holding the agents the graph used, so their
            personas travel with the template. Without it the template
            keeps the structure but not the personas.
        require_success: Refuse to distill a graph that did not fully
            succeed. Learning a shape from a failed run is how a bad
            plan becomes a permanent one.

    Raises:
        DistillationError: The graph is empty, or (under
            ``require_success``) some node did not complete.
    """
    if not graph.nodes:
        raise DistillationError("cannot distill an empty graph")
    if require_success:
        unfinished = [
            n.id for n in graph.nodes if n.status is not NodeStatus.COMPLETED
        ]
        if unfinished:
            raise DistillationError(
                "cannot distill a graph with unfinished nodes: "
                f"{', '.join(sorted(unfinished))}"
            )

    # Snapshot structure now; the source graph must not be able to
    # mutate a template that was already handed to an architect.
    blueprint = [
        {
            "id": node.id,
            "label": node.label,
            "depends_on": list(node.depends_on),
            "capabilities": list(node.required_capabilities),
            "profile": _profile_of(node, registry),
        }
        for node in graph.nodes
    ]

    def builder(task: Task, params: dict | None = None) -> tuple[list[Node], Registry]:
        template_registry = Registry()
        nodes: list[Node] = []
        for entry in blueprint:
            agent_id = None
            profile = entry["profile"]
            if profile is not None:
                agent = Agent(profile=AgentProfile(**profile))
                template_registry.register(agent)
                agent_id = agent.id
            nodes.append(
                Node(
                    id=entry["id"],
                    label=_specialize(entry["label"], task),
                    depends_on=list(entry["depends_on"]),
                    agent_id=agent_id,
                    required_capabilities=list(entry["capabilities"]),
                )
            )
        return nodes, template_registry

    return SubGraphTemplate(
        name=name,
        description=description or _describe(graph),
        builder=builder,
    )


def _profile_of(node: Node, registry: Registry | None) -> dict | None:
    if registry is None or node.agent_id is None:
        return None
    agent = registry.get(node.agent_id)
    if agent is None:
        return None
    return {
        "name": agent.profile.name,
        "persona": agent.profile.persona,
        "capabilities": list(agent.profile.capabilities),
    }


def _specialize(label: str, task: Task) -> str:
    """Point a learned label at the new task.

    A distilled label describes the *step*, not the subject; carrying
    the original subject forward would make every reuse re-answer the
    task it was learned from.
    """
    if _GOAL_TOKEN.search(label):
        return _GOAL_TOKEN.sub(task.goal, label)
    return f"{label}\n\nApply this step to: {task.goal}"


def _describe(graph: ExecutionGraph) -> str:
    topology = " -> ".join(t.value for t in graph.topology) or "custom"
    return (
        f"Learned {topology} shape with {len(graph.nodes)} steps "
        f"(depth {graph.depth}): "
        + "; ".join(node.label.split("\n")[0][:60] for node in graph.nodes[:4])
    )
