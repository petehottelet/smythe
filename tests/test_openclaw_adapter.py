"""Tests for the OpenClaw adapter (mocked SDK — no live dependency)."""

from unittest.mock import MagicMock, patch

import pytest

from smythe.skills import SkillRef


class FakeSDKSkill:
    """Simulates an OpenClaw SDK skill object."""

    def __init__(self, name: str, version: str | None = None, description: str = "") -> None:
        self.name = name
        self.version = version
        self.description = description


def _make_mock_client(skills: list[FakeSDKSkill]) -> MagicMock:
    client = MagicMock()
    client.skills.list.return_value = skills
    return client


def test_openclaw_adapter_translates_skills():
    from smythe.openclaw_adapter import OpenClawSkillProvider

    sdk_skills = [
        FakeSDKSkill(name="web-search", version="1.2.0", description="Search the web"),
        FakeSDKSkill(name="summarize", version="0.9.0"),
    ]
    client = _make_mock_client(sdk_skills)
    provider = OpenClawSkillProvider(client=client)

    refs = provider.list_agent_skills("agent-123")

    assert len(refs) == 2
    assert refs[0] == SkillRef(
        name="web-search",
        version="1.2.0",
        source="openclaw",
        metadata={"description": "Search the web"},
    )
    assert refs[1].name == "summarize"
    assert refs[1].source == "openclaw"
    client.skills.list.assert_called_once_with(agent_id="agent-123")


def test_openclaw_adapter_handles_skill_without_version():
    from smythe.openclaw_adapter import OpenClawSkillProvider

    client = _make_mock_client([FakeSDKSkill(name="code")])
    provider = OpenClawSkillProvider(client=client)
    refs = provider.list_agent_skills("a1")

    assert refs[0].version is None
    assert refs[0].name == "code"


def test_openclaw_adapter_handles_raw_metadata():
    from smythe.openclaw_adapter import OpenClawSkillProvider

    skill = FakeSDKSkill(name="tool")
    skill.metadata = {"openclaw": {"requires": {"bins": ["uv"]}}}
    client = _make_mock_client([skill])
    provider = OpenClawSkillProvider(client=client)
    refs = provider.list_agent_skills("a1")

    assert refs[0].metadata["raw"] == {"openclaw": {"requires": {"bins": ["uv"]}}}


def test_openclaw_adapter_missing_sdk_raises():
    with patch.dict("sys.modules", {"openclaw_sdk": None}):
        from smythe import openclaw_adapter
        import importlib
        importlib.reload(openclaw_adapter)

        with pytest.raises(ImportError, match="pip install smythe\\[openclaw\\]"):
            openclaw_adapter.OpenClawSkillProvider()


def test_openclaw_adapter_empty_skills():
    from smythe.openclaw_adapter import OpenClawSkillProvider

    client = _make_mock_client([])
    provider = OpenClawSkillProvider(client=client)
    refs = provider.list_agent_skills("a1")

    assert refs == []
