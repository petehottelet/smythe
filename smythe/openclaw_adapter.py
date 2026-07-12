"""OpenClaw adapter — translates OpenClaw AgentSkills into Smythe SkillRefs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from smythe.skills import SkillRef


class OpenClawSkillProvider:
    """SkillProvider backed by the ``openclaw-sdk`` package.

    Wraps the OpenClaw SDK client and translates each installed agent
    skill into a :class:`SkillRef` that the Smythe registry can use
    for capability hydration.

    The ``openclaw-sdk`` package is imported lazily so Smythe can be
    used without it installed.
    """

    def __init__(self, *, client: Any = None, **client_kwargs: Any) -> None:
        if client is not None:
            self._client = client
            return

        try:
            from openclaw_sdk import OpenClaw  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "OpenClaw SDK is required for OpenClawSkillProvider. "
                "Install it with: pip install smythe[openclaw]"
            ) from exc

        self._client = OpenClaw(**client_kwargs)

    def list_agent_skills(self, agent_id: str) -> list[SkillRef]:
        """Fetch installed skills for *agent_id* from OpenClaw.

        The SDK client is expected to expose a ``skills.list()`` or
        equivalent method.  Each SDK skill object is translated to a
        :class:`SkillRef`.
        """
        raw_skills = self._client.skills.list(agent_id=agent_id)
        if raw_skills is None:
            return []
        return [self._to_skill_ref(s) for s in raw_skills]

    @staticmethod
    def _to_skill_ref(sdk_skill: Any) -> SkillRef:
        """Convert an OpenClaw SDK object or mapping to a SkillRef."""

        def field(name: str, default: Any = None) -> Any:
            if isinstance(sdk_skill, Mapping):
                return sdk_skill.get(name, default)
            return getattr(sdk_skill, name, default)

        name = field("name") or str(sdk_skill)
        version = field("version")
        metadata: dict[str, Any] = {}
        description = field("description")
        if description:
            metadata["description"] = str(description)
        raw_metadata = field("metadata")
        if raw_metadata is not None:
            metadata["raw"] = (
                dict(raw_metadata)
                if isinstance(raw_metadata, Mapping)
                else raw_metadata
            )
        return SkillRef(
            name=str(name),
            version=str(version) if version is not None else None,
            source="openclaw",
            metadata=metadata,
        )
