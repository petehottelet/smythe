"""Parallel workers that each contribute something distinct.

    python examples/17_contribution_templates.py

Parallel workers add value when each one receives distinct evidence, a
distinct capability, a distinct responsibility or an output that can be
checked on its own. Each SubGraphTemplate below builds its workers that way:

- source_extracts: one extractor per source, so each reads different evidence;
- specialist_reviews: one review by specialists, each with its own persona
  and capabilities;
- separate_documents: documents that can each be checked alone.

A WhiteRabbit routes the task to a ConstrainedArchitect, which composes the
templates, and the contribution lint finds nothing to flag in that plan.
repeated_research is the pattern to avoid: three identical workers given the
same inputs. The lint records them under graph.planning["coordination"] and
leaves the plan unchanged.

The run is offline and deterministic: OfflineProvider answers every call, from
a script for routing and planning, even when an API key is set.

Docs: docs/architecture.md#contribution-lint
"""

import json
import sys

from smythe import (
    ConstrainedArchitect, OfflineProvider, SubGraphTemplate, Swarm, Task, WhiteRabbit,
    lint_contributions,
)
from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.registry import Registry

TASK = Task("Assess the solar-charger vendor before we sign")
SOURCES = ("supplier schedule", "field-trial log", "support tickets")
SPECIALISTS = {
    "legal": "You are a contracts lawyer. Flag terms that shift liability to us.",
    "security": "You are a security reviewer. Flag data-handling and access risks.",
    "finance": "You are a financial analyst. Flag cost and payment-term risks.",
}


def worker(agent_id: str, name: str, persona: str, capabilities: list[str]) -> Agent:
    return Agent(AgentProfile(name, persona=persona, capabilities=capabilities), id=agent_id)


def registry(*agents: Agent) -> Registry:
    holder = Registry()
    for agent in agents:
        holder.register(agent)
    return holder


def source_extracts(task: Task) -> tuple[list[Node], Registry]:
    """Distinct evidence: one extractor per source."""
    extractor = worker("extractor", "Extractor", "You quote dated claims and figures exactly.",
                       ["extraction"])
    return [Node(id=f"extract-{source.replace(' ', '-')}",
                 label=f"Extract the dated claims and figures in the {source}",
                 agent_id=extractor.id) for source in SOURCES], registry(extractor)


def specialist_reviews(task: Task) -> tuple[list[Node], Registry]:
    """Distinct capabilities: the same review by three specialists."""
    agents = [worker(f"{area}-reviewer", f"{area.title()}Reviewer", persona, [area])
              for area, persona in SPECIALISTS.items()]
    return [Node(id=f"review-{area}", label="Review the extracted claims for risks in your field",
                 agent_id=agent.id, required_capabilities=[area])
            for area, agent in zip(SPECIALISTS, agents)], registry(*agents)


def separate_documents(task: Task) -> tuple[list[Node], Registry]:
    """Distinct, checkable outputs: each document can be checked alone."""
    writer = worker("writer", "Writer", "You write precise documents that cite their sources.",
                    ["writing"])
    return [
        Node(id="risk-register", agent_id=writer.id,
             label="Write the risk register: one row per risk, with its owner and source"),
        Node(id="decision-memo", agent_id=writer.id,
             label="Write the one-page sign or renegotiate memo with a Sources section"),
    ], registry(writer)


def repeated_research(task: Task) -> tuple[list[Node], Registry]:
    """The pattern to avoid: identical workers given identical inputs."""
    return [Node(id=f"research-{i}", label="Research the vendor") for i in range(3)], Registry()


TEMPLATES = [
    SubGraphTemplate("source_extracts", "Extract claims from each source", source_extracts),
    SubGraphTemplate("specialist_reviews", "Legal, security and finance reviews",
                     specialist_reviews),
    SubGraphTemplate("separate_documents", "A risk register and a decision memo",
                     separate_documents),
]
REPEATED = SubGraphTemplate("repeated_research", "Research the vendor three times",
                            repeated_research)


def routed_swarm(selections: list[str]) -> Swarm:
    """A Swarm whose scripted classifier and architect compose *selections*."""
    reply = json.dumps([{"template": name} for name in selections])
    architect = ConstrainedArchitect(OfflineProvider(responses=[reply]), [*TEMPLATES, REPEATED],
                                     model="demo-model")
    router = WhiteRabbit(constrained=architect, classifier_model="demo-model",
                         classifier_provider=OfflineProvider(responses=["constrained"]))
    return Swarm(provider=OfflineProvider(), model="demo-model", router=router, artifact_dir=None)


def executed(swarm: Swarm, graph: ExecutionGraph) -> str:
    result = swarm.execute(graph)
    done = sum(node.status.value == "completed" for node in result.graph.nodes)
    return f"{done} of {len(result.graph.nodes)} nodes offline"


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding and stream.encoding.lower() not in ("utf-8", "utf8"):
            stream.reconfigure(encoding="utf-8")
    print("Running offline with smythe's built-in OfflineProvider (no keys, no cost).\n")

    print("=== Each template, linted on its own ===")
    for template in [*TEMPLATES, REPEATED]:
        nodes, agents = template.builder(TASK)
        groups = lint_contributions(ExecutionGraph([Topology.FORK_JOIN], nodes), agents)
        print(f"  {template.name:<20} {len(nodes)} workers  flagged: {groups or 'nothing'}")

    print("\n=== A routed plan composed from the contribution templates ===")
    swarm = routed_swarm([template.name for template in TEMPLATES])
    graph = swarm.plan(TASK)
    routing = graph.planning["routing"]
    print(f"  routing:      {routing['tier']} ({routing['method']})")
    for node in graph.nodes:
        print(f"  {node.id:<44} {node.agent_id}")
    print(f"  coordination: {graph.planning.get('coordination', 'nothing recorded')}")
    print(f"  executed:     {executed(swarm, graph)}")

    print("\n=== A routed plan with a repeated fan-out ===")
    swarm = routed_swarm(["source_extracts", REPEATED.name])
    graph = swarm.plan(TASK)
    print(f"  coordination: {json.dumps(graph.planning['coordination'])}")
    # The lint is advisory: the plan keeps every node and runs as planned.
    print(f"  executed:     {executed(swarm, graph)}")


if __name__ == "__main__":
    main()
