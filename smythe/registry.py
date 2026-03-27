"""Agent Registry — stores agents and matches them to graph nodes."""

from __future__ import annotations

from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph


class Registry:
    """Maintains the pool of available agents and assigns them to work.

    Future versions will use capability matching and performance history
    to route nodes to the best-fit agent.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.id] = agent

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def assign(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Assign agents to unassigned nodes in the graph.

        Default strategy: create a generalist agent for each unassigned node.
        """
        for node in graph.nodes:
            if node.agent_id is None:
                agent = Agent(profile=AgentProfile(name=f"agent-{node.id}"))
                self.register(agent)
                node.agent_id = agent.id
        return graph
