"""Count benchmark usage at the provider boundary, including planning calls."""

from __future__ import annotations

from smythe.provider import CompletionResult, Provider
from smythe.tools import ChatMessage, ToolSpec


BLENDED_USD_PER_TOKEN = 0.000003


class UsageRecordingProvider(Provider):
    """Delegate unchanged requests and retain returned usage for the whole run.

    Executor budget totals historically excluded architect calls. Recording
    the shared provider measures planning, execution, and synthesis without
    relying on any one framework phase's budget accounting.
    """

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self.calls: list[dict[str, int | str]] = []

    def _record(self, result: CompletionResult, model: str) -> CompletionResult:
        self.calls.append({
            "model": model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        })
        return result

    @property
    def total_tokens(self) -> int:
        return sum(int(call["total_tokens"]) for call in self.calls)

    def snapshot(self) -> dict:
        return {
            "scope": "planning_execution_synthesis",
            "source": "provider_response_usage",
            "call_count": len(self.calls),
            "prompt_tokens": sum(int(call["prompt_tokens"]) for call in self.calls),
            "completion_tokens": sum(int(call["completion_tokens"]) for call in self.calls),
            "total_tokens": self.total_tokens,
            "calls": list(self.calls),
        }

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return self._record(await self.provider.complete(system, prompt, model), model)

    async def chat(
        self,
        system: str,
        messages: list[ChatMessage],
        model: str,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        return self._record(await self.provider.chat(system, messages, model, tools), model)

    def budget_estimate_usd(self, model: str) -> float | None:
        return self.provider.budget_estimate_usd(model)

    def requires_explicit_budget_estimate(self, model: str) -> bool:
        return self.provider.requires_explicit_budget_estimate(model)
