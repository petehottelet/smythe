"""Skills — adapter types for external skill systems like OpenClaw AgentSkills."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SkillRef:
    """An immutable reference to an installed agent skill.

    Attributes:
        name: Canonical skill identifier (e.g. ``"web-search"``).
        version: Optional version string.
        source: Origin system (e.g. ``"openclaw"``).
        metadata: Arbitrary provider-specific data.
    """

    name: str
    version: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SkillProvider(Protocol):
    """Protocol for systems that expose per-agent skill inventories."""

    def list_agent_skills(self, agent_id: str) -> list[SkillRef]: ...


@runtime_checkable
class CapabilityMapper(Protocol):
    """Translates a list of skills into a set of capability tags."""

    def map_skills(self, skills: list[SkillRef]) -> set[str]: ...


class CapabilityHydrationMode(Enum):
    """How skill-derived capabilities are combined with static ones.

    MERGE: union of static profile capabilities and derived capabilities.
    REPLACE: derived capabilities only; static profile is ignored.
    STATIC_ONLY: ignore the skill provider entirely.
    """

    MERGE = "merge"
    REPLACE = "replace"
    STATIC_ONLY = "static_only"


class DefaultCapabilityMapper:
    """Maps skills to capabilities by normalised name with optional aliases.

    Each skill's name is lowercased and stripped.  If an alias mapping
    is provided, matching names are replaced with the alias target.
    """

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self._aliases = {k.lower().strip(): v for k, v in (aliases or {}).items()}

    def map_skills(self, skills: list[SkillRef]) -> set[str]:
        caps: set[str] = set()
        for skill in skills:
            normalised = skill.name.lower().strip()
            mapped = self._aliases.get(normalised, normalised)
            caps.add(mapped)
        return caps
