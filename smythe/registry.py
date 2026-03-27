"""Agent Registry — stores agents and matches them to graph nodes."""

from __future__ import annotations

import logging
import time

from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph
from smythe.skills import (
    CapabilityHydrationMode,
    CapabilityMapper,
    DefaultCapabilityMapper,
    SkillProvider,
)

logger = logging.getLogger("smythe.registry")


class Registry:
    """Maintains the pool of available agents and assigns them to work.

    Supports capability-aware matching: when a node has
    ``required_capabilities``, the registry prefers agents whose
    capabilities are a superset of the required set.  Ties are broken
    alphabetically by agent name for determinism.

    When a ``skill_provider`` is configured, agent capabilities are
    hydrated from external skill systems (e.g. OpenClaw AgentSkills)
    before matching.  Hydrated values are cached per agent with a
    configurable TTL.
    """

    def __init__(
        self,
        *,
        skill_provider: SkillProvider | None = None,
        capability_mapper: CapabilityMapper | None = None,
        hydration_mode: CapabilityHydrationMode = CapabilityHydrationMode.MERGE,
        capability_cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._agents: dict[str, Agent] = {}
        self._skill_provider = skill_provider
        self._capability_mapper: CapabilityMapper = capability_mapper or DefaultCapabilityMapper()
        self._hydration_mode = hydration_mode
        self._cache_ttl = capability_cache_ttl_seconds
        self._capability_cache: dict[str, tuple[set[str], float]] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.id] = agent

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def refresh_agent_capabilities(self, agent_id: str) -> None:
        """Invalidate the cached capabilities for a single agent."""
        self._capability_cache.pop(agent_id, None)

    def refresh_all_capabilities(self) -> None:
        """Invalidate all cached capabilities, forcing re-hydration."""
        self._capability_cache.clear()

    def find_by_capabilities(self, required: list[str]) -> Agent | None:
        """Return the best-matching agent for the required capabilities.

        An agent matches if its (possibly hydrated) capabilities are a
        superset of *required*.  Among matches, the one with the fewest
        extra capabilities wins.  Ties are broken alphabetically by
        agent name.  Returns None if no match is found.
        """
        required_set = set(required)
        candidates: list[tuple[int, str, Agent]] = []
        for agent in self._agents.values():
            agent_caps = set(self._hydrate(agent))
            if required_set.issubset(agent_caps):
                extra = len(agent_caps) - len(required_set)
                candidates.append((extra, agent.profile.name, agent))

        if not candidates:
            logger.debug(
                "find_by_capabilities: no match for %s among %d agent(s)",
                required, len(self._agents),
            )
            return None

        candidates.sort(key=lambda t: (t[0], t[1]))
        winner = candidates[0][2]
        source = "hydrated" if self._skill_provider is not None else "static"
        logger.debug(
            "find_by_capabilities: selected %s (%s caps, %s) "
            "from %d candidate(s) for %s",
            winner.profile.name, source, "exact" if candidates[0][0] == 0 else "superset",
            len(candidates), required,
        )
        return winner

    def assign(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Assign agents to unassigned nodes in the graph.

        Uses capability matching when a node has ``required_capabilities``.
        Falls back to creating a generalist agent when no match is found.
        """
        for node in graph.nodes:
            if node.agent_id is not None:
                continue

            if node.required_capabilities:
                match = self.find_by_capabilities(node.required_capabilities)
                if match is not None:
                    node.agent_id = match.id
                    continue

            agent = Agent(profile=AgentProfile(name=f"agent-{node.id}"))
            self.register(agent)
            node.agent_id = agent.id
        return graph

    def _hydrate(self, agent: Agent) -> list[str]:
        """Return the effective capabilities for *agent*.

        When a skill provider is configured and the hydration mode is
        not ``STATIC_ONLY``, capabilities are derived from the
        provider and merged or replaced according to policy.  Results
        are cached per agent with a TTL.
        """
        if self._skill_provider is None or self._hydration_mode == CapabilityHydrationMode.STATIC_ONLY:
            return agent.profile.capabilities

        cached = self._capability_cache.get(agent.id)
        if cached is not None:
            caps, ts = cached
            if time.monotonic() - ts < self._cache_ttl:
                return sorted(caps)

        try:
            skills = self._skill_provider.list_agent_skills(agent.id)
        except Exception:
            logger.warning(
                "Skill provider failed for agent %s; falling back to static capabilities",
                agent.id,
                exc_info=True,
            )
            return agent.profile.capabilities

        derived = self._capability_mapper.map_skills(skills)

        if self._hydration_mode == CapabilityHydrationMode.MERGE:
            effective = set(agent.profile.capabilities) | derived
        else:
            effective = derived

        self._capability_cache[agent.id] = (effective, time.monotonic())
        return sorted(effective)
