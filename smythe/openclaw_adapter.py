"""OpenClaw adapter — translates OpenClaw AgentSkills into Smythe SkillRefs."""

from __future__ import annotations

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
        return [self._to_skill_ref(s) for s in raw_skills]

    @staticmethod
    def _to_skill_ref(sdk_skill: Any) -> SkillRef:
        """Convert an OpenClaw SDK skill object to a SkillRef."""
        name = getattr(sdk_skill, "name", None) or str(sdk_skill)
        version = getattr(sdk_skill, "version", None)
        metadata: dict[str, Any] = {}
        if hasattr(sdk_skill, "description"):
            metadata["description"] = sdk_skill.description
        if hasattr(sdk_skill, "metadata"):
            metadata["raw"] = sdk_skill.metadata
        return SkillRef(
            name=name,
            version=version,
            source="openclaw",
            metadata=metadata,
        )
