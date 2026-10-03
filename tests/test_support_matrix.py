"""Support matrix cells: what each execution path accepts, and what it refuses.

A rejection cell proves that the refusal happens before any provider method
runs and before any SDK client exists, including the planning and routing
calls a run would make.
"""

from __future__ import annotations

import pytest

from smythe import (
    AnthropicMessagesProvider, AnthropicProvider, CallableVerifier, CompletionResult,
    FileCheckpointStore, GeminiProvider, OfflineProvider, OpenAIImageProvider, OpenAIProvider,
    OpenAIResponsesProvider, PlannerMemory, Provider, SQLiteWorkflowStore, Swarm, Task,
    TokenVerifier, ToolRuntime, WorkflowBindingError,
)

BUILT_IN_PROVIDERS = (
    OfflineProvider, AnthropicProvider, OpenAIProvider, OpenAIImageProvider, GeminiProvider,
    OpenAIResponsesProvider, AnthropicMessagesProvider,
)


class CustomProvider(Provider):
    """A caller-written provider: ordinary runs accept it, durable runs do not."""

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return CompletionResult("custom")


class AuditedStore(SQLiteWorkflowStore):
    """A store subclass could change journal semantics, so durable runs refuse it."""


class StrictTokenVerifier(TokenVerifier):
    pass


class UnusedTools(ToolRuntime):
    def open(self, agent):
        raise AssertionError("A tool session opened")


def forbid_provider_calls(monkeypatch) -> list[str]:
    """Record, from here on, every provider method call and SDK client creation."""
    calls: list[str] = []

    def recorder(owner, name):
        def record(self, *args, **kwargs):
            calls.append(f"{owner}.{name}")
            raise AssertionError(f"{owner}.{name} ran before the rejection")
        return record

    for cls in (*BUILT_IN_PROVIDERS, CustomProvider):
        for name in ("complete", "chat", "_get_client"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, recorder(cls.__name__, name))
    return calls


@pytest.fixture
def store(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as value:
        yield value


# ---------------------------------------------------------------------------
# Durable text workflows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argument,make_component,shown", [
    pytest.param("run_store", lambda path: AuditedStore(path / "audited.db"), "AuditedStore",
                 id="store-subclass"),
    pytest.param("memory", lambda path: PlannerMemory(path / "history.jsonl"), None, id="memory"),
    pytest.param("tool_runtime", lambda path: UnusedTools(), None, id="tool-runtime"),
    pytest.param("checkpoint_store", lambda path: FileCheckpointStore(path / "checkpoints"), None,
                 id="checkpoint-store"),
    pytest.param("verifier", lambda path: CallableVerifier(lambda *_: True), "CallableVerifier",
                 id="callable-verifier"),
    pytest.param("verifier", lambda path: StrictTokenVerifier(), "StrictTokenVerifier",
                 id="token-verifier-subclass"),
])
def test_durable_run_rejects_component_before_any_call(
    store, tmp_path, monkeypatch, argument, make_component, shown,
):
    component = make_component(tmp_path)
    calls = forbid_provider_calls(monkeypatch)
    try:
        # The default LLMArchitect would make a paid planning call first.
        swarm = Swarm(model="offline", provider=OfflineProvider(), max_budget_usd=1,
                      **{"run_store": store, argument: component})
        with pytest.raises(WorkflowBindingError) as caught:
            swarm.execute(Task("Report"))
    finally:
        if isinstance(component, SQLiteWorkflowStore):
            component.close()
    message = str(caught.value)
    assert isinstance(caught.value, ValueError)
    assert message.startswith(f"{argument} "), message
    assert shown is None or shown in message
    assert calls == []
    assert store.list_runs() == []
