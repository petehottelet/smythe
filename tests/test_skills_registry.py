"""Tests for Registry skill-hydration integration."""

import time

from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.registry import Registry
from smythe.skills import (
    CapabilityHydrationMode,
    DefaultCapabilityMapper,
    SkillRef,
)


class FakeSkillProvider:
    """Returns a fixed set of skills per agent, tracking call count."""

    def __init__(self, skills_by_agent: dict[str, list[SkillRef]] | None = None) -> None:
        self._skills = skills_by_agent or {}
        self.call_count = 0

    def list_agent_skills(self, agent_id: str) -> list[SkillRef]:
        self.call_count += 1
        return self._skills.get(agent_id, [])


class FailingSkillProvider:
    """Always raises when listing skills."""

    def list_agent_skills(self, agent_id: str) -> list[SkillRef]:
        raise RuntimeError("provider unavailable")


def test_registry_hydrates_capabilities_merge_mode():
    agent = Agent(profile=AgentProfile(name="a1", capabilities=["static-cap"]))
    provider = FakeSkillProvider({agent.id: [SkillRef(name="derived-cap")]})

    reg = Registry(
        skill_provider=provider,
        hydration_mode=CapabilityHydrationMode.MERGE,
    )
    reg.register(agent)

    match = reg.find_by_capabilities(["static-cap"])
    assert match is agent
    match2 = reg.find_by_capabilities(["derived-cap"])
    assert match2 is agent
    match3 = reg.find_by_capabilities(["static-cap", "derived-cap"])
    assert match3 is agent


def test_registry_hydrates_capabilities_replace_mode():
    agent = Agent(profile=AgentProfile(name="a1", capabilities=["static-cap"]))
    provider = FakeSkillProvider({agent.id: [SkillRef(name="derived-cap")]})

    reg = Registry(
        skill_provider=provider,
        hydration_mode=CapabilityHydrationMode.REPLACE,
    )
    reg.register(agent)

    match_static = reg.find_by_capabilities(["static-cap"])
    assert match_static is None
    match_derived = reg.find_by_capabilities(["derived-cap"])
    assert match_derived is agent


def test_registry_static_only_ignores_provider():
    agent = Agent(profile=AgentProfile(name="a1", capabilities=["static-cap"]))
    provider = FakeSkillProvider({agent.id: [SkillRef(name="derived-cap")]})

    reg = Registry(
        skill_provider=provider,
        hydration_mode=CapabilityHydrationMode.STATIC_ONLY,
    )
    reg.register(agent)

    match_static = reg.find_by_capabilities(["static-cap"])
    assert match_static is agent
    match_derived = reg.find_by_capabilities(["derived-cap"])
    assert match_derived is None
    assert provider.call_count == 0


def test_registry_uses_cache_until_ttl_expired():
    agent = Agent(profile=AgentProfile(name="a1", capabilities=[]))
    provider = FakeSkillProvider({agent.id: [SkillRef(name="cap")]})

    reg = Registry(
        skill_provider=provider,
        capability_cache_ttl_seconds=300.0,
    )
    reg.register(agent)

    reg.find_by_capabilities(["cap"])
    assert provider.call_count == 1
    reg.find_by_capabilities(["cap"])
    assert provider.call_count == 1


def test_registry_cache_expires_after_ttl():
    agent = Agent(profile=AgentProfile(name="a1", capabilities=[]))
    provider = FakeSkillProvider({agent.id: [SkillRef(name="cap")]})

    reg = Registry(
        skill_provider=provider,
        capability_cache_ttl_seconds=0.05,
    )
    reg.register(agent)

    reg.find_by_capabilities(["cap"])
    assert provider.call_count == 1

    time.sleep(0.06)
    reg.find_by_capabilities(["cap"])
    assert provider.call_count == 2


def test_registry_refresh_agent_forces_reload():
    agent = Agent(profile=AgentProfile(name="a1", capabilities=[]))
    provider = FakeSkillProvider({agent.id: [SkillRef(name="cap")]})

    reg = Registry(skill_provider=provider)
    reg.register(agent)

    reg.find_by_capabilities(["cap"])
    assert provider.call_count == 1

    reg.refresh_agent_capabilities(agent.id)
    reg.find_by_capabilities(["cap"])
    assert provider.call_count == 2


def test_registry_refresh_all_clears_cache():
    a1 = Agent(profile=AgentProfile(name="a1", capabilities=[]))
    a2 = Agent(profile=AgentProfile(name="a2", capabilities=[]))
    provider = FakeSkillProvider({
        a1.id: [SkillRef(name="cap")],
        a2.id: [SkillRef(name="cap")],
    })

    reg = Registry(skill_provider=provider)
    reg.register(a1)
    reg.register(a2)

    reg.find_by_capabilities(["cap"])
    initial_calls = provider.call_count

    reg.refresh_all_capabilities()
    reg.find_by_capabilities(["cap"])
    assert provider.call_count > initial_calls


def test_assignment_uses_hydrated_capabilities():
    agent = Agent(profile=AgentProfile(name="researcher", capabilities=[]))
    provider = FakeSkillProvider({agent.id: [SkillRef(name="research")]})

    reg = Registry(skill_provider=provider)
    reg.register(agent)

    node = Node(label="Research task", id="r1", required_capabilities=["research"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    reg.assign(graph)

    assert node.agent_id == agent.id


def test_skill_provider_failure_falls_back_to_static():
    agent = Agent(profile=AgentProfile(name="a1", capabilities=["static-cap"]))

    reg = Registry(skill_provider=FailingSkillProvider())
    reg.register(agent)

    match = reg.find_by_capabilities(["static-cap"])
    assert match is agent


def test_deterministic_tiebreak_with_hydrated_caps():
    alice = Agent(profile=AgentProfile(name="alice", capabilities=[]))
    bob = Agent(profile=AgentProfile(name="bob", capabilities=[]))
    provider = FakeSkillProvider({
        alice.id: [SkillRef(name="code")],
        bob.id: [SkillRef(name="code")],
    })

    reg = Registry(skill_provider=provider)
    reg.register(bob)
    reg.register(alice)

    match = reg.find_by_capabilities(["code"])
    assert match is alice


def test_no_provider_behaves_like_before():
    """Registry() with no skill_provider should behave identically to the original."""
    reg = Registry()
    agent = Agent(profile=AgentProfile(name="worker", capabilities=["writing"]))
    reg.register(agent)

    assert reg.find_by_capabilities(["writing"]) is agent
    assert reg.find_by_capabilities(["unknown"]) is None


def test_hydration_with_alias_mapper():
    agent = Agent(profile=AgentProfile(name="a1", capabilities=[]))
    provider = FakeSkillProvider({agent.id: [SkillRef(name="search")]})
    mapper = DefaultCapabilityMapper(aliases={"search": "research"})

    reg = Registry(
        skill_provider=provider,
        capability_mapper=mapper,
    )
    reg.register(agent)

    assert reg.find_by_capabilities(["research"]) is agent
    assert reg.find_by_capabilities(["search"]) is None
