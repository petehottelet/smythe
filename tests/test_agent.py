"""Dedicated tests for Agent and AgentProfile models."""

from smythe.agent import Agent, AgentProfile


def test_agent_profile_defaults():
    """AgentProfile with only a name should have empty persona and capabilities."""
    profile = AgentProfile(name="Alice")
    assert profile.name == "Alice"
    assert profile.persona == ""
    assert profile.capabilities == []


def test_agent_profile_with_capabilities():
    """AgentProfile should store custom persona and capabilities."""
    profile = AgentProfile(
        name="Researcher",
        persona="You are a meticulous researcher.",
        capabilities=["research", "analysis"],
    )
    assert profile.persona == "You are a meticulous researcher."
    assert "research" in profile.capabilities
    assert len(profile.capabilities) == 2


def test_agent_auto_id():
    """Agent should generate a unique 12-char hex id automatically."""
    profile = AgentProfile(name="Bot")
    agent = Agent(profile=profile)
    assert len(agent.id) == 12
    assert all(c in "0123456789abcdef" for c in agent.id)


def test_agent_unique_ids():
    """Two agents should have distinct auto-generated IDs."""
    profile = AgentProfile(name="Bot")
    a1 = Agent(profile=profile)
    a2 = Agent(profile=profile)
    assert a1.id != a2.id


def test_agent_custom_id():
    """Agent should accept a custom ID."""
    profile = AgentProfile(name="Custom")
    agent = Agent(profile=profile, id="my-custom-id")
    assert agent.id == "my-custom-id"


def test_agent_name_delegates_to_profile():
    """Agent.name should return profile.name."""
    profile = AgentProfile(name="Delegated")
    agent = Agent(profile=profile)
    assert agent.name == "Delegated"


def test_agent_history_starts_empty():
    """Agent history should be an empty list by default."""
    agent = Agent(profile=AgentProfile(name="New"))
    assert agent.history == []


def test_agent_history_append():
    """Appending to history should work and be mutable."""
    agent = Agent(profile=AgentProfile(name="Logger"))
    agent.history.append({"task": "test", "status": "ok"})
    assert len(agent.history) == 1
    assert agent.history[0]["status"] == "ok"


def test_agent_history_independent():
    """Each Agent should have its own independent history list."""
    a1 = Agent(profile=AgentProfile(name="A"))
    a2 = Agent(profile=AgentProfile(name="B"))
    a1.history.append({"x": 1})
    assert len(a2.history) == 0


def test_agent_profile_capabilities_independent():
    """Each AgentProfile should have its own independent capabilities list."""
    p1 = AgentProfile(name="A")
    p2 = AgentProfile(name="B")
    p1.capabilities.append("coding")
    assert len(p2.capabilities) == 0
