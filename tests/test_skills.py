"""Tests for skill types and DefaultCapabilityMapper."""

import pytest

from smythe.skills import (
    CapabilityHydrationMode,
    CapabilityMapper,
    DefaultCapabilityMapper,
    SkillProvider,
    SkillRef,
)


def test_skill_ref_frozen():
    ref = SkillRef(name="web-search", version="1.0", source="openclaw")
    with pytest.raises(AttributeError):
        ref.name = "changed"


def test_skill_ref_defaults():
    ref = SkillRef(name="summarize")
    assert ref.version is None
    assert ref.source is None
    assert ref.metadata == {}


def test_default_mapper_name_passthrough():
    mapper = DefaultCapabilityMapper()
    skills = [SkillRef(name="research"), SkillRef(name="summarize")]
    assert mapper.map_skills(skills) == {"research", "summarize"}


def test_default_mapper_normalises_case_and_whitespace():
    mapper = DefaultCapabilityMapper()
    skills = [SkillRef(name="  Web-Search  "), SkillRef(name="SUMMARIZE")]
    assert mapper.map_skills(skills) == {"web-search", "summarize"}


def test_default_mapper_alias_mapping():
    mapper = DefaultCapabilityMapper(aliases={"search": "research", "summarize-text": "summarize"})
    skills = [SkillRef(name="search"), SkillRef(name="summarize-text")]
    assert mapper.map_skills(skills) == {"research", "summarize"}


def test_default_mapper_alias_case_insensitive():
    mapper = DefaultCapabilityMapper(aliases={"Search": "research"})
    skills = [SkillRef(name="SEARCH")]
    assert mapper.map_skills(skills) == {"research"}


def test_default_mapper_deduplicates():
    mapper = DefaultCapabilityMapper()
    skills = [SkillRef(name="research"), SkillRef(name="research"), SkillRef(name="Research")]
    assert mapper.map_skills(skills) == {"research"}


def test_default_mapper_empty_skills():
    mapper = DefaultCapabilityMapper()
    assert mapper.map_skills([]) == set()


def test_hydration_mode_values():
    assert CapabilityHydrationMode.MERGE.value == "merge"
    assert CapabilityHydrationMode.REPLACE.value == "replace"
    assert CapabilityHydrationMode.STATIC_ONLY.value == "static_only"


def test_default_mapper_satisfies_protocol():
    mapper = DefaultCapabilityMapper()
    assert isinstance(mapper, CapabilityMapper)


def test_skill_provider_protocol_check():
    """A class implementing list_agent_skills satisfies the protocol."""

    class FakeProvider:
        def list_agent_skills(self, agent_id: str) -> list[SkillRef]:
            return []

    assert isinstance(FakeProvider(), SkillProvider)
