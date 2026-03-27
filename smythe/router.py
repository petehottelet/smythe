"""PlannerRouter — routes tasks to the appropriate planner tier."""

from __future__ import annotations

import asyncio

from smythe.planner import DeterministicPlanner, Planner, PlanningError
from smythe.provider import Provider
from smythe.task import Task


CLASSIFIER_SYSTEM_PROMPT = """\
You are a task classifier.  Given a user's goal, decide which planning \
strategy to use.  Respond with ONLY one of the following tokens:

{options}

Do not add any other text.
"""


class PlannerRouter:
    """Routes tasks to the appropriate planner tier based on classification.

    Supports two modes:
    - **Explicit routing**: user passes ``planner=`` directly to Swarm.
    - **Classifier routing**: a lightweight LLM call classifies the task
      and selects the appropriate planner from the registered tiers.

    When no classifier provider is set, falls back to the autonomous planner.
    """

    def __init__(
        self,
        *,
        deterministic: dict[str, DeterministicPlanner] | None = None,
        constrained: Planner | None = None,
        autonomous: Planner | None = None,
        classifier_provider: Provider | None = None,
        classifier_model: str = "claude-mythos",
    ) -> None:
        self._deterministic = deterministic or {}
        self._constrained = constrained
        self._autonomous = autonomous
        self._classifier_provider = classifier_provider
        self._classifier_model = classifier_model

    def route(self, task: Task) -> Planner:
        """Classify task and return the appropriate planner (sync)."""
        if self._classifier_provider is None:
            return self._fallback()
        return asyncio.run(self.aroute(task))

    async def aroute(self, task: Task) -> Planner:
        """Async classification — awaits the classifier provider directly."""
        if self._classifier_provider is None:
            return self._fallback()

        options = self._build_options()
        system = CLASSIFIER_SYSTEM_PROMPT.format(options=options)
        prompt = f"Goal: {task.goal}"

        result = await self._classifier_provider.complete(
            system, prompt, model=self._classifier_model
        )

        return self._parse_classification(result.text.strip())

    def _build_options(self) -> str:
        lines: list[str] = []
        for key in self._deterministic:
            lines.append(f"deterministic:{key}")
        if self._constrained is not None:
            lines.append("constrained")
        lines.append("autonomous")
        return "\n".join(f"- {opt}" for opt in lines)

    def _parse_classification(self, text: str) -> Planner:
        """Map classifier output to a planner instance."""
        cleaned = text.strip().lower()

        if cleaned.startswith("deterministic:"):
            key = cleaned.split(":", 1)[1].strip()
            if key in self._deterministic:
                return self._deterministic[key]

        if cleaned == "constrained" and self._constrained is not None:
            return self._constrained

        return self._fallback()

    def _fallback(self) -> Planner:
        """Return the autonomous planner, or raise if none configured."""
        if self._autonomous is not None:
            return self._autonomous
        raise PlanningError(
            "No autonomous planner configured and classifier unavailable"
        )
