"""Support matrix cells: what each execution path accepts, and what it refuses.

docs/support-matrix.md cites these tests by node id. A rejection cell proves
that the refusal happens before any provider method runs and before any SDK
client exists, including the planning and routing calls a run would make.
The ownership tests at the end pin who owns an execution responsibility
where no other test does, including a responsibility that nothing owns.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from fakes import RATE_LIMITED, Gate, ManualClock, MessagesTransport, NativeTransport
from smythe import (
    AcceptancePolicy, AnthropicMessagesProvider, AnthropicProvider, Artifact,
    BudgetEstimateRequired, CallableVerifier, ChatMessage, CompletionResult, FailurePolicy,
    FileCheckpointStore, GeminiProvider, LineagePolicy, LLMArchitect, LocalTool, LocalToolRuntime,
    OfflineProvider,
    OpenAIImageProvider, OpenAIProvider, OpenAIResponsesProvider, PlannerMemory, Provider,
    ProviderAccountingError, ProviderFeatures, ProviderResponseError, SimpleArchitect,
    SQLiteWorkflowStore, Swarm, SynthesisStrategy, Synthesizer, Task, TextValidator,
    TokenVerifier, ToolRuntime, ToolSession, ToolSpec, WhiteRabbit, WorkflowBindingError,
)
from smythe.checkpoint import build_state
from smythe.graph import SYNTHESIS_NODE_ID, ExecutionGraph, Node, NodeStatus, Topology
from smythe.registry import Registry
from smythe.jobs import JobManifestV1, make_approval, preflight_job
from smythe.jobs.models import ManifestValidationError
from smythe.jobs.providers import ProviderPreflightError, ProviderPreflightKind
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import SQLiteRunStore
from smythe.pricing_anthropic import MODEL as FABLE
from smythe.workflow_store import WorkflowConflictError, WorkflowLeaseError

BUILT_IN_PROVIDERS = (
    OfflineProvider, AnthropicProvider, OpenAIProvider, OpenAIImageProvider, GeminiProvider,
    OpenAIResponsesProvider, AnthropicMessagesProvider,
)


class CustomProvider(Provider):
    """A caller-written provider: ordinary runs accept it, durable runs do not."""

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return CompletionResult("custom")


class OfflineSubclass(OfflineProvider):
    pass


class ResponsesSubclass(OpenAIResponsesProvider):
    pass


class MessagesSubclass(AnthropicMessagesProvider):
    pass


class AuditedStore(SQLiteWorkflowStore):
    """A store subclass could change journal semantics, so durable runs refuse it."""


class StrictTokenVerifier(TokenVerifier):
    pass


class UnusedTools(ToolRuntime):
    def open(self, agent):
        raise AssertionError("A tool session opened")


def unused_handler(arguments):
    raise AssertionError("A local tool ran")


class NoTools(ToolSession):
    @property
    def tools(self):
        return []

    async def call(self, tool_call):
        raise AssertionError("no tool is offered")


class EmptyTools(ToolRuntime):
    """A tool runtime whose sessions offer no tools, so nodes send plain requests."""

    def open(self, agent):
        @asynccontextmanager
        async def session():
            yield NoTools()

        return session()


class CeilingProvider(CustomProvider):
    """A caller-written provider whose capped calls need a per-call ceiling."""

    def requires_explicit_budget_estimate(self, model: str) -> bool:
        return True


class OwnChatOffline(OfflineProvider):
    """Replacing chat() may serve more than OfflineProvider, so nothing is declared."""

    async def chat(self, system, messages, model, tools=None):
        return CompletionResult("own chat", prompt_tokens=1, completion_tokens=1)


class CountingProvider(Provider):
    """A caller-written provider with no feature declaration that counts its calls."""

    def __init__(self, text: str = "done") -> None:
        self.text = text
        self.calls: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.calls.append(model)
        return CompletionResult(self.text, prompt_tokens=1, completion_tokens=1)

    async def chat(self, system, messages, model, tools=None):
        self.calls.append(model)
        return CompletionResult(self.text, prompt_tokens=1, completion_tokens=1)


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


def durable_swarm(store, **options):
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **options)


# ---------------------------------------------------------------------------
# Durable text workflows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_provider,model", [
    pytest.param(lambda: AnthropicProvider(api_key="test-key"), "claude-opus-5-5", id="anthropic"),
    pytest.param(lambda: OpenAIProvider(api_key="test-key"), "gpt-5", id="openai-chat"),
    pytest.param(lambda: OpenAIImageProvider(api_key="test-key", max_cost_per_call_usd=0.25),
                 "gpt-image-1", id="openai-image"),
    pytest.param(lambda: GeminiProvider(api_key="test-key"), "gemini-3-pro", id="gemini"),
    pytest.param(lambda: GeminiProvider(api_key="test-key", max_cost_per_call_usd=0.25),
                 "gemini-3.1-flash-image", id="gemini-image"),
    pytest.param(lambda: OfflineProvider(artifacts_per_call=1), "offline", id="offline-artifacts"),
    pytest.param(lambda: OfflineProvider(responses=["scripted"]), "offline", id="offline-script"),
    pytest.param(CustomProvider, "custom", id="custom"),
    pytest.param(OfflineSubclass, "offline", id="offline-subclass"),
    pytest.param(lambda: ResponsesSubclass(api_key="test-key"), "gpt-6-astra", id="responses-subclass"),
    pytest.param(lambda: MessagesSubclass(api_key="test-key"), FABLE, id="messages-subclass"),
    pytest.param(lambda: OpenAIResponsesProvider(api_key="test-key"), "gpt-4o",
                 id="responses-unsupported-model"),
    pytest.param(lambda: AnthropicMessagesProvider(api_key="test-key"), "claude-opus-5-5",
                 id="messages-unsupported-model"),
])
def test_durable_run_rejects_provider_before_any_call(store, monkeypatch, make_provider, model):
    calls = forbid_provider_calls(monkeypatch)
    # The default LLMArchitect would make a paid planning call first.
    swarm = Swarm(model=model, provider=make_provider(), run_store=store, max_budget_usd=1)
    for entry in (swarm.plan, swarm.execute):
        with pytest.raises(ValueError, match="(?i)durable"):
            entry(Task("Report"))
    assert calls == []
    assert store.list_runs() == []


@pytest.mark.parametrize("argument,make_component,shown", [
    pytest.param("run_store", lambda path: AuditedStore(path / "audited.db"), "AuditedStore",
                 id="store-subclass"),
    pytest.param("memory", lambda path: PlannerMemory(path / "history.jsonl"), None, id="memory"),
    pytest.param("tool_runtime", lambda path: UnusedTools(), None, id="tool-runtime"),
    pytest.param("tool_runtime", lambda path: LocalToolRuntime(
        [LocalTool(ToolSpec("notes.search", "Search.", {"type": "object"}), unused_handler, "1")],
        agents={"writer": ["notes.search"]}), None, id="local-tool-runtime"),
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


@pytest.mark.parametrize("change", [
    pytest.param(lambda node: setattr(node, "attach_dep_artifacts", True), id="attach-dep-artifacts"),
    pytest.param(lambda node: node.metadata.update(attachments=["logo.png"]), id="attachments"),
    pytest.param(lambda node: node.metadata.update(artifacts=[{"path": "a.png"}]), id="artifacts"),
])
def test_durable_run_rejects_image_inputs_on_a_caller_graph_before_any_call(store, monkeypatch, change):
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"), Node(id="review", label="Review", depends_on=["draft"]),
    ])
    change(graph.nodes[1])
    calls = forbid_provider_calls(monkeypatch)
    with pytest.raises(WorkflowBindingError, match="plain text nodes"):
        durable_swarm(store).execute(graph)
    assert calls == []
    assert [store.inspect_run(run["run_id"])["call_count"] for run in store.list_runs()] == [0]


@pytest.mark.parametrize("entry", ["resume-completed", "execute-planned"])
def test_durable_run_rejects_a_changed_retry_backoff_before_any_call(store, monkeypatch, entry):
    if entry == "resume-completed":
        run_id = durable_swarm(store).execute(Task("Report")).execution_id
    else:
        planned = durable_swarm(store).plan(Task("Report"))
        run_id = planned.run_ref["run_id"]
    before = store.inspect_run(run_id)
    calls = forbid_provider_calls(monkeypatch)
    changed = durable_swarm(store, retry_backoff_s=0.5)
    with pytest.raises(WorkflowConflictError, match="saved recipe"):
        if entry == "resume-completed":
            changed.resume(run_id)
        else:
            changed.execute(planned)
    assert calls == []
    after = store.inspect_run(run_id)
    assert after["calls"] == before["calls"] and after["revision"] == before["revision"]


@pytest.mark.parametrize("durable,acceptance,needs", [
    pytest.param(False, "strict", "durable run_store", id="ordinary"),
    pytest.param(True, None, "strict acceptance", id="no-policy"),
    pytest.param(True, "advisory", "strict acceptance", id="advisory-policy"),
])
def test_lineage_is_refused_without_a_durable_store_or_strict_policy_before_any_call(
    store, tmp_path, monkeypatch, durable, acceptance, needs,
):
    policy = {"strict": AcceptancePolicy(mode="strict", final_validators=[TextValidator()]),
              "advisory": AcceptancePolicy(), None: None}[acceptance]
    path = tmp_path / "graph.yaml"
    path.write_text("nodes:\n  - id: first\n    label: First\n", encoding="utf-8")
    options = {"run_store": store} if durable else {}
    calls = forbid_provider_calls(monkeypatch)
    # The default LLMArchitect would make a paid planning call first.
    with pytest.raises(WorkflowBindingError, match=needs) as caught:
        Swarm(model="claude-opus-5-5", provider=AnthropicProvider(api_key="test-key"),
              max_budget_usd=1, acceptance=policy, lineage=LineagePolicy(), **options)
    assert str(caught.value).startswith("lineage ")
    with pytest.raises(WorkflowBindingError, match=needs):
        Swarm.from_yaml(str(path), model="claude-opus-5-5",
                        provider=AnthropicProvider(api_key="test-key"), max_budget_usd=1,
                        acceptance=policy, lineage=LineagePolicy(), **options)
    assert calls == []
    assert store.list_runs() == []


def test_durable_run_ignores_artifact_dir_and_tool_iterations_and_allows_no_budget(store, tmp_path):
    swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                  run_store=store, artifact_dir=tmp_path / "artifacts", max_tool_iterations=1)
    result = swarm.execute(Task("Report"))
    assert result.output == "offline: Report"
    assert result.cost_scope == "complete_text_workflow" and result.cost_is_complete
    assert store.load_run(result.execution_id)["budget_nanousd"] is None
    assert not (tmp_path / "artifacts").exists()


# ---------------------------------------------------------------------------
# Ordinary Swarm runs
# ---------------------------------------------------------------------------


def test_anthropic_messages_provider_runs_an_ordinary_swarm(monkeypatch):
    transport = MessagesTransport().install(monkeypatch)
    swarm = Swarm(model=FABLE, artifact_dir=None, max_budget_usd=5,
                  provider=AnthropicMessagesProvider(api_key="test-key", max_cost_per_call_usd=0.5))
    result = swarm.execute(Task("Write and check a short summary"))
    assert result.output == "Draft accepted output."
    assert [(record.phase, record.outcome) for record in transport.log] == [
        ("planning", "delivered"), ("execution", "delivered"), ("execution", "delivered"),
    ]
    # Ordinary runs dispatch directly: no input-token quote, no journal, and
    # each call closes the client it opened.
    assert transport.counts == [] and result.workflow_accounting is None
    assert [client.close.await_count for client in transport.clients] == [1, 1, 1]
    receipts = [entry["receipt"] for node in result.graph.nodes
                for entry in node.metadata["native_receipts"]]
    assert [receipt["cost_nanousd"] for receipt in receipts] == [2_535_000, 2_535_000]
    # The ordinary ledger covers execution and synthesis; planning is outside it.
    assert result.cost_scope == "execution_and_synthesis"
    assert result.total_cost_usd == pytest.approx(0.00507)
    assert result.cost_is_complete and not result.cost_contains_estimates


@pytest.mark.parametrize("transport_class,provider_class,model,fault,error", [
    pytest.param(MessagesTransport, AnthropicMessagesProvider, FABLE, RATE_LIMITED,
                 ProviderAccountingError, id="messages-rate-limited"),
    pytest.param(MessagesTransport, AnthropicMessagesProvider, FABLE, {"stop_reason": "refusal"},
                 ProviderResponseError, id="messages-refused"),
    pytest.param(NativeTransport, OpenAIResponsesProvider, "gpt-6-astra", RATE_LIMITED,
                 ProviderAccountingError, id="responses-rate-limited"),
    pytest.param(NativeTransport, OpenAIResponsesProvider, "gpt-6-astra", {"status": "incomplete"},
                 ProviderResponseError, id="responses-truncated"),
])
def test_native_provider_failure_on_an_ordinary_run_is_terminal_under_a_retry_policy(
    monkeypatch, transport_class, provider_class, model, fault, error,
):
    transport = transport_class().install(monkeypatch)
    if isinstance(fault, dict):
        transport.overrides["execution"] = fault
    else:
        transport.schedule("execution", 0, fault)
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft", failure_policy=FailurePolicy.RETRY, max_retries=2),
    ])
    swarm = Swarm(model=model, artifact_dir=None, max_budget_usd=5,
                  provider=provider_class(api_key="test-key", max_cost_per_call_usd=0.5))
    with pytest.raises(error) as caught:
        swarm.execute(graph)
    # A rate limit leaves the bill unresolved; a refused or truncated reply
    # has a known bill. Either way the node's retry policy never calls again.
    assert type(caught.value) is error
    outcome = "http_429" if fault is RATE_LIMITED else "delivered"
    assert [(record.phase, record.outcome) for record in transport.log] == [("execution", outcome)]
    assert graph.nodes[0].status is NodeStatus.FAILED


@pytest.mark.parametrize("make_provider,model,charged", [
    pytest.param(OfflineProvider, "offline", True, id="blended-estimate"),
    pytest.param(CustomProvider, "custom", False, id="missing-usage"),
])
def test_ordinary_ledger_does_not_mark_uncertain_costs_from_other_providers(
    make_provider, model, charged,
):
    graph = ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft")])
    result = Swarm(model=model, provider=make_provider(), artifact_dir=None,
                   max_budget_usd=5).execute(graph)
    assert result.cost_scope == "execution_and_synthesis"
    # A blended token estimate is recorded as an exact charge, and a reply
    # without usage as zero tokens, so the result still reads as complete.
    assert (result.total_cost_usd > 0) is charged
    assert not result.cost_contains_estimates and result.cost_is_complete


@pytest.mark.parametrize("message,tools", [
    pytest.param(ChatMessage("user", "Look this up"),
                 [ToolSpec("lookup", "Look up a fact", {"type": "object"})], id="tools"),
    pytest.param(ChatMessage("user", "Describe the image", attachments=[Artifact(b"png")]), None,
                 id="attachments"),
])
def test_anthropic_messages_provider_rejects_tools_and_attachments_before_any_request(
    monkeypatch, message, tools,
):
    transport = MessagesTransport().install(monkeypatch)
    provider = AnthropicMessagesProvider(api_key="test-key")
    with pytest.raises(ValueError):
        asyncio.run(provider.chat("System", [message], FABLE, tools))
    assert transport.log == [] and transport.clients == []


@pytest.mark.parametrize("make_provider,expected", [
    # No max_retries argument: the SDK's own default retries apply.
    pytest.param(lambda: AnthropicProvider(api_key="test-key"), {"api_key": "test-key"},
                 id="anthropic-sdk-default"),
    pytest.param(lambda: OpenAIProvider(api_key="test-key"), {"api_key": "test-key"},
                 id="openai-sdk-default"),
    pytest.param(lambda: OpenAIProvider(api_key="test-key", max_retries=0),
                 {"api_key": "test-key", "max_retries": 0}, id="openai-disabled"),
    # google-genai retries only when a retry option is configured.
    pytest.param(lambda: GeminiProvider(api_key="test-key"), {"api_key": "test-key"},
                 id="gemini-not-configured"),
])
def test_legacy_provider_clients_leave_retries_to_the_sdk_configuration(
    monkeypatch, make_provider, expected,
):
    created = []

    def client(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AsyncAnthropic=client))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=client))
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=SimpleNamespace(Client=client)))
    make_provider()._get_client()
    assert created == [expected]


@pytest.mark.parametrize("transport_class,make_provider,model", [
    pytest.param(NativeTransport, lambda: OpenAIResponsesProvider(api_key="test-key"), "gpt-6-astra",
                 id="responses"),
    pytest.param(MessagesTransport, lambda: AnthropicMessagesProvider(api_key="test-key"), FABLE,
                 id="messages"),
])
def test_native_providers_refuse_a_client_with_sdk_retries(monkeypatch, transport_class, make_provider,
                                                           model):
    transport = transport_class().install(monkeypatch)
    provider = make_provider()
    provider._client = transport._client()
    provider._client.max_retries = 2
    with pytest.raises(ValueError, match="retries"):
        asyncio.run(provider.complete("System", "Question", model))
    assert transport.log == []


@pytest.mark.parametrize("cls", [*BUILT_IN_PROVIDERS, Provider],
                         ids=lambda cls: cls.__name__)
def test_no_provider_offers_structured_output_or_streaming(cls):
    # Planners parse JSON from ordinary text; no provider takes a response
    # schema or returns a stream.
    assert list(inspect.signature(cls.chat).parameters) == [
        "self", "system", "messages", "model", "tools",
    ]
    assert list(inspect.signature(cls.complete).parameters) == ["self", "system", "prompt", "model"]
    options = set(inspect.signature(cls.__init__).parameters)
    assert not options & {"response_format", "json_schema", "output_schema", "stream", "streaming"}
    assert not inspect.isasyncgenfunction(cls.chat)


# ---------------------------------------------------------------------------
# Ordinary Swarm runs: provider features checked before any call
# ---------------------------------------------------------------------------

FEATURES = ("model_supported", "text", "tools", "attachments", "image_output", "cost_ceiling_required")
OFFLINE = (None, True, False, True, False, False)


@pytest.mark.parametrize("make_provider,model,declared", [
    pytest.param(OfflineProvider, "offline", OFFLINE, id="offline"),
    pytest.param(lambda: OfflineProvider(artifacts_per_call=1), "offline",
                 (None, True, False, True, True, False), id="offline-artifacts"),
    pytest.param(lambda: AnthropicProvider(api_key="test-key"), "claude-opus-5-5",
                 (None, True, True, True, False, False), id="anthropic"),
    pytest.param(lambda: OpenAIProvider(api_key="test-key"), "gpt-5",
                 (None, True, True, True, False, False), id="openai-chat"),
    pytest.param(lambda: OpenAIImageProvider(api_key="test-key"), "gpt-image-1",
                 (True, False, False, False, True, True), id="openai-image"),
    pytest.param(lambda: OpenAIImageProvider(api_key="test-key"), "gpt-5",
                 (False, False, False, False, True, True), id="openai-image-text-model"),
    pytest.param(lambda: GeminiProvider(api_key="test-key"), "gemini-3-pro",
                 (None, True, True, True, False, False), id="gemini"),
    pytest.param(lambda: GeminiProvider(api_key="test-key"), "gemini-3.1-flash-image",
                 (None, True, True, True, True, True), id="gemini-image"),
    pytest.param(lambda: OpenAIResponsesProvider(api_key="test-key"), "gpt-6-astra",
                 (True, True, True, False, False, True), id="responses"),
    pytest.param(lambda: OpenAIResponsesProvider(api_key="test-key"), "gpt-4o",
                 (False, True, True, False, False, True), id="responses-unsupported-model"),
    pytest.param(lambda: AnthropicMessagesProvider(api_key="test-key"), FABLE,
                 (True, True, False, False, False, True), id="messages"),
    pytest.param(lambda: AnthropicMessagesProvider(api_key="test-key"), "claude-opus-5-5",
                 (False, True, False, False, False, True), id="messages-unsupported-model"),
    pytest.param(OfflineSubclass, "offline", OFFLINE, id="offline-subclass"),
    pytest.param(CustomProvider, "custom", (None, None, None, None, None, False), id="custom"),
    pytest.param(CeilingProvider, "custom", (None, None, None, None, None, True), id="custom-ceiling"),
    pytest.param(OwnChatOffline, "offline", (None, None, None, None, None, False),
                 id="subclass-with-its-own-chat"),
])
def test_providers_declare_what_each_model_supports(make_provider, model, declared):
    features = make_provider().feature_support(model)
    assert isinstance(features, ProviderFeatures)
    assert tuple(getattr(features, name) for name in FEATURES) == declared


def enter(swarm, entry, graph):
    """Start a run the way *entry* names; every way must refuse before any call."""
    if entry == "plan":
        return swarm.plan(Task("Report"))
    if entry == "aplan":
        return asyncio.run(swarm.aplan(Task("Report")))
    if entry == "graph":
        return swarm.execute(graph)
    if entry == "execute-async":
        return asyncio.run(swarm.execute_async(Task("Report")))
    return swarm.execute(Task("Report"))


ENTRIES = ["plan", "aplan", "execute", "execute-parallel", "graph"]
# The entries that run what they plan; plan() and aplan() return the graph first.
RUN_ENTRIES = ["execute", "execute-async", "execute-parallel", "graph"]


def two_step_graph(**review):
    # The first step could be paid for before the second one fails.
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft", metadata={"estimated_cost_usd": 0.5}),
        Node(id="review", label="Review", depends_on=["draft"], **review),
    ])


@pytest.mark.parametrize("entry", RUN_ENTRIES)
@pytest.mark.parametrize("make_provider,model", [
    pytest.param(lambda: OpenAIImageProvider(api_key="test-key"), "gpt-image-1", id="openai-image"),
    pytest.param(lambda: GeminiProvider(api_key="test-key"), "gemini-3.1-flash-image", id="gemini-image"),
    pytest.param(lambda: OpenAIResponsesProvider(api_key="test-key"), "gpt-6-astra", id="responses"),
    pytest.param(lambda: AnthropicMessagesProvider(api_key="test-key"), FABLE, id="messages"),
    pytest.param(CeilingProvider, "custom", id="custom-ceiling"),
])
def test_ordinary_run_rejects_a_missing_cost_ceiling_before_any_call(monkeypatch, make_provider, model,
                                                                      entry):
    calls = forbid_provider_calls(monkeypatch)
    # The default LLMArchitect plans with the run's provider; a generated
    # plan cannot give a node its own model or estimate.
    swarm = Swarm(model=model, provider=make_provider(), max_budget_usd=1, artifact_dir=None,
                  parallel=entry == "execute-parallel")
    with pytest.raises(BudgetEstimateRequired) as caught:
        enter(swarm, entry, two_step_graph())
    assert calls == []
    if entry == "graph":
        assert caught.value.node_id == "review"
    else:
        assert caught.value.node_id is None and "generated plan" in str(caught.value)
        assert "estimated_cost_usd" in str(caught.value)


@pytest.mark.parametrize("entry", ["plan", "aplan"])
def test_plan_returns_a_graph_whose_node_estimates_stand_in_for_a_missing_ceiling(monkeypatch, entry):
    transport = MessagesTransport().install(monkeypatch)
    swarm = Swarm(model=FABLE, provider=AnthropicMessagesProvider(api_key="test-key"), max_budget_usd=1,
                  artifact_dir=None)
    # Planning and running in one call gives no chance to set estimates, so
    # it is refused before any request.
    with pytest.raises(BudgetEstimateRequired) as caught:
        swarm.execute(Task("Report"))
    assert caught.value.node_id is None and transport.log == [] and transport.clients == []

    graph = enter(swarm, entry, None)
    assert [record.phase for record in transport.log] == ["planning"]
    # Each node is checked before the first node call, and the refusal names
    # the remedy.
    with pytest.raises(BudgetEstimateRequired, match="estimated_cost_usd") as caught:
        swarm.execute(graph)
    assert caught.value.node_id == graph.nodes[0].id
    assert [record.phase for record in transport.log] == ["planning"]

    for node in graph.nodes:
        node.metadata["estimated_cost_usd"] = 0.5
    result = swarm.execute(graph) if entry == "plan" else asyncio.run(swarm.execute_async(graph))
    assert result.output == "Draft accepted output."
    assert [(record.phase, record.outcome) for record in transport.log] == [
        ("planning", "delivered"), ("execution", "delivered"), ("execution", "delivered"),
    ]
    assert result.total_cost_usd == pytest.approx(0.00507)
    assert result.cost_is_complete and not result.cost_contains_estimates


def test_ordinary_run_checks_a_routed_plan_before_the_classifier_call(monkeypatch):
    calls = forbid_provider_calls(monkeypatch)
    router = WhiteRabbit(autonomous=LLMArchitect(provider=CustomProvider(), planning_model=FABLE),
                         classifier_provider=CustomProvider())
    swarm = Swarm(model=FABLE, provider=AnthropicMessagesProvider(api_key="test-key"), router=router,
                  max_budget_usd=1, artifact_dir=None)
    with pytest.raises(BudgetEstimateRequired):
        swarm.execute(Task("Report"))
    assert calls == []


@pytest.mark.parametrize("entry", ENTRIES)
def test_ordinary_run_rejects_an_llm_merge_it_cannot_budget_before_any_call(monkeypatch, entry):
    calls = forbid_provider_calls(monkeypatch)
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft", metadata={"estimated_cost_usd": 0.5}),
    ])
    swarm = Swarm(model=FABLE, provider=AnthropicMessagesProvider(api_key="test-key"),
                  synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE), max_budget_usd=1,
                  artifact_dir=None, parallel=entry == "execute-parallel")
    # Without the check, every node would run before the merge failed.
    with pytest.raises(BudgetEstimateRequired) as caught:
        enter(swarm, entry, graph)
    assert caught.value.node_id == SYNTHESIS_NODE_ID
    assert calls == []


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("make_provider,model", [
    pytest.param(OfflineProvider, "offline", id="offline"),
    pytest.param(lambda: OpenAIImageProvider(api_key="test-key", max_cost_per_call_usd=0.25),
                 "gpt-image-1", id="openai-image"),
    pytest.param(lambda: AnthropicMessagesProvider(api_key="test-key", max_cost_per_call_usd=0.5),
                 FABLE, id="messages"),
])
def test_ordinary_run_rejects_a_tool_runtime_its_provider_cannot_serve_before_any_call(
    monkeypatch, make_provider, model, entry,
):
    calls = forbid_provider_calls(monkeypatch)
    swarm = Swarm(model=model, provider=make_provider(), tool_runtime=UnusedTools(), max_budget_usd=1,
                  artifact_dir=None, parallel=entry == "execute-parallel")
    with pytest.raises(ValueError, match="tool_runtime needs a provider that can call tools"):
        enter(swarm, entry, two_step_graph())
    assert calls == []


def test_tool_runtime_is_checked_before_an_overridden_plan_runs(monkeypatch):
    class ReviewedSwarm(Swarm):
        def plan(self, task):
            self.planned.append(task.goal)
            return two_step_graph()

    calls = forbid_provider_calls(monkeypatch)
    swarm = ReviewedSwarm(model="offline", provider=OfflineProvider(), tool_runtime=UnusedTools(),
                          artifact_dir=None)
    swarm.planned = []
    with pytest.raises(ValueError, match="tool_runtime"):
        swarm.execute(Task("Report"))
    assert swarm.planned == [] and calls == []


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("make_provider,model", [
    pytest.param(lambda: OpenAIImageProvider(api_key="test-key", max_cost_per_call_usd=0.25),
                 "gpt-image-1", id="openai-image"),
    pytest.param(lambda: OpenAIResponsesProvider(api_key="test-key", max_cost_per_call_usd=0.5),
                 "gpt-6-astra", id="responses"),
    pytest.param(lambda: AnthropicMessagesProvider(api_key="test-key", max_cost_per_call_usd=0.5),
                 FABLE, id="messages"),
])
def test_ordinary_run_rejects_image_inputs_its_provider_cannot_take_before_any_call(
    monkeypatch, make_provider, model, parallel,
):
    calls = forbid_provider_calls(monkeypatch)
    swarm = Swarm(model=model, provider=make_provider(), max_budget_usd=5, artifact_dir=None,
                  parallel=parallel)
    graph = two_step_graph(attach_dep_artifacts=True)
    with pytest.raises(ValueError, match="attach_dep_artifacts on node 'review'"):
        swarm.execute(graph)
    assert calls == []
    assert [node.status for node in graph.nodes] == [NodeStatus.PENDING, NodeStatus.PENDING]


class PlanOnly(Provider):
    """Plans for a run whose nodes use another provider, counting its calls."""

    def __init__(self, plan: dict) -> None:
        self.plan = plan
        self.calls = 0

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.calls += 1
        return CompletionResult(json.dumps(self.plan), prompt_tokens=1, completion_tokens=1)


def test_generated_plan_with_image_inputs_is_rejected_before_any_node_call(monkeypatch):
    # Whether a generated plan asks for image inputs is known only once it exists.
    planner = PlanOnly({"topology": ["serial"], "nodes": [
        {"id": "draft", "label": "Draft", "depends_on": []},
        {"id": "review", "label": "Review", "depends_on": ["draft"], "attach_dep_artifacts": True},
    ]})
    calls = forbid_provider_calls(monkeypatch)
    swarm = Swarm(model=FABLE, planning_provider=planner, max_budget_usd=5, artifact_dir=None,
                  provider=AnthropicMessagesProvider(api_key="test-key", max_cost_per_call_usd=0.5))
    with pytest.raises(ValueError, match="attach_dep_artifacts on node 'review'"):
        swarm.execute(Task("Report"))
    assert planner.calls == 1 and calls == []


def test_a_plan_refused_before_its_nodes_resumes_without_planning_again(tmp_path, monkeypatch):
    planner = PlanOnly({"topology": ["serial"], "nodes": [
        {"id": "draft", "label": "Draft", "depends_on": []},
        {"id": "review", "label": "Review", "depends_on": ["draft"], "attach_dep_artifacts": True},
    ]})
    calls = forbid_provider_calls(monkeypatch)
    store = FileCheckpointStore(tmp_path)
    refused = Swarm(model=FABLE, planning_provider=planner, checkpoint_store=store, artifact_dir=None,
                    provider=AnthropicMessagesProvider(api_key="test-key"))
    with pytest.raises(ValueError, match="attach_dep_artifacts"):
        refused.execute(Task("Report"))
    [execution_id] = store.list_ids()
    assert store.load(execution_id)["status"] == "failed"

    worker = CountingProvider()
    result = Swarm(model=FABLE, provider=worker, checkpoint_store=store, artifact_dir=None,
                   planning_provider=planner).resume(execution_id)
    assert result.output == "done"
    assert planner.calls == 1 and worker.calls == [FABLE, FABLE] and calls == []


@pytest.mark.parametrize("entry", ["planned", "plan", "graph"])
def test_ordinary_run_rejects_a_model_its_provider_does_not_support_before_any_call(monkeypatch, entry):
    planner = PlanOnly({"topology": ["serial"], "nodes": [{"id": "draft", "label": "Draft"}]})
    calls = forbid_provider_calls(monkeypatch)
    provider = OpenAIResponsesProvider(api_key="test-key", max_cost_per_call_usd=0.5)
    if entry in ("planned", "plan"):
        # plan() refuses it too: no estimate on the returned graph could fix it.
        swarm = Swarm(model="gpt-4o", provider=provider, planning_provider=planner, artifact_dir=None)
        with pytest.raises(ValueError, match="does not support model 'gpt-4o'"):
            swarm.execute(Task("Report")) if entry == "planned" else swarm.plan(Task("Report"))
    else:
        swarm = Swarm(model="gpt-6-astra", provider=provider, artifact_dir=None)
        graph = two_step_graph(metadata={"model": "gpt-4o"})
        with pytest.raises(ValueError, match="does not support model 'gpt-4o', which node 'review'"):
            swarm.execute(graph)
    assert planner.calls == 0 and calls == []


def test_resume_checks_pending_nodes_before_any_call(tmp_path, monkeypatch):
    store = FileCheckpointStore(tmp_path)
    graph = two_step_graph()
    for node in graph.nodes:
        node.metadata["model"] = "gpt-6-astra"
    store.save("halted", build_state(
        execution_id="halted", status="failed", model="gpt-6-astra", graph=graph,
        registry=Registry(), task=None, max_budget_usd=5, node_costs={},
    ))
    calls = forbid_provider_calls(monkeypatch)
    swarm = Swarm(model="gpt-6-astra", provider=OpenAIResponsesProvider(api_key="test-key"),
                  checkpoint_store=store, artifact_dir=None)
    with pytest.raises(BudgetEstimateRequired) as caught:
        swarm.resume("halted")
    assert caught.value.node_id == "review" and calls == []


@pytest.mark.parametrize("parallel", [False, True])
def test_providers_that_declare_nothing_still_run_with_tools_image_inputs_and_a_budget(parallel):
    # A custom provider, and a subclass that replaces a built-in's chat(),
    # declare nothing, so nothing about them is refused in advance.
    custom = CountingProvider()
    graph = two_step_graph(attach_dep_artifacts=True)
    result = Swarm(model="custom", provider=custom, tool_runtime=EmptyTools(), max_budget_usd=5,
                   synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE), artifact_dir=None,
                   parallel=parallel).execute(graph)
    assert result.output == "done"
    assert custom.calls == ["custom", "custom", "custom"]

    planned = Swarm(model="offline", provider=OwnChatOffline(), planning_provider=CountingProvider(
        json.dumps({"topology": ["serial"], "nodes": [{"id": "draft", "label": "Draft"}]})),
        tool_runtime=EmptyTools(), max_budget_usd=5, artifact_dir=None, parallel=parallel,
    ).execute(Task("Report"))
    assert planned.output == "own chat"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def jobs_manifest(provider, model, ceiling, artifact=None, **extra):
    operation = {"key": "tile", "prompt": "One tile", "profile": "default", **extra}
    if artifact is not None:
        operation["artifact"] = artifact
    return {
        "version": 1, "name": "support-matrix",
        "profiles": [{"name": "default", "provider": provider, "model": model,
                      "max_cost_per_call_usd": ceiling}],
        "operations": [operation],
        "execution": {"max_concurrency": 1, "max_attempts": 1, "max_budget_usd": ceiling},
    }


@pytest.mark.parametrize("provider,model,ceiling,artifact", [
    pytest.param("offline", "offline-image", "0", {"mime_type": "image/jpeg"}, id="offline-jpeg"),
    pytest.param("offline", "offline-image", "0",
                 {"mime_type": "image/png", "width": 1024, "height": 1024}, id="offline-1024"),
    pytest.param("gemini_image", "gemini-3.1-flash-image", "0.25", {"mime_type": "application/pdf"},
                 id="gemini-pdf"),
    pytest.param("openai_image", "gpt-image-1", "0.25", {"mime_type": "image/jpeg"},
                 id="openai-format"),
])
def test_jobs_rejects_an_artifact_its_profile_cannot_produce_before_any_call(
    tmp_path, monkeypatch, provider, model, ceiling, artifact,
):
    plan = preflight_job(JobManifestV1.from_dict(jobs_manifest(provider, model, ceiling, artifact)),
                         manifest_root=tmp_path)
    calls = forbid_provider_calls(monkeypatch)
    with SQLiteRunStore(tmp_path / "jobs.db") as jobs:
        with pytest.raises(ProviderPreflightError) as caught:
            asyncio.run(JobRunner(jobs).start(plan, make_approval(plan), manifest_root=tmp_path))
        assert jobs.list_runs() == []
    assert caught.value.kind is ProviderPreflightKind.LOCAL
    assert calls == []


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini", "openai_responses",
                                      "anthropic_messages"])
def test_jobs_manifest_accepts_only_image_provider_kinds(provider):
    with pytest.raises(ManifestValidationError, match="provider is unsupported"):
        JobManifestV1.from_dict(jobs_manifest(provider, "any-model", "0.25"))


@pytest.mark.parametrize("section,field", [
    ("profile", "validators"), ("operation", "verifier"), ("operation", "tools"),
    ("execution", "memory"),
])
def test_jobs_manifest_has_no_validator_tool_or_memory_fields(section, field):
    manifest = jobs_manifest("offline", "offline-image", "0")
    target = {"profile": manifest["profiles"][0], "operation": manifest["operations"][0],
              "execution": manifest["execution"]}[section]
    target[field] = []
    with pytest.raises(ManifestValidationError, match="unknown fields"):
        JobManifestV1.from_dict(manifest)


# ---------------------------------------------------------------------------
# Who owns each execution responsibility
# ---------------------------------------------------------------------------


class Unavailable(Provider):
    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        raise RuntimeError("provider unavailable")


class Resumer(Provider):
    """Answers with its own name once the test lets its call through."""

    def __init__(self, name: str) -> None:
        self.name, self.gate, self.calls = name, Gate(), []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.calls.append(prompt)
        await self.gate.wait()
        return CompletionResult(self.name, prompt_tokens=1, completion_tokens=1)


def test_ordinary_checkpoints_do_not_fence_a_second_resumer(tmp_path):
    graph = ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft")])
    with pytest.raises(RuntimeError, match="provider unavailable"):
        Swarm(model="custom", provider=Unavailable(), checkpoint_store=FileCheckpointStore(tmp_path),
              artifact_dir=None).execute(graph)
    [execution_id] = FileCheckpointStore(tmp_path).list_ids()
    first, second = Resumer("first"), Resumer("second")

    async def resume_twice():
        # Two processes, each with its own store on one directory, resume one run.
        runs = [asyncio.create_task(Swarm(model="custom", provider=provider, artifact_dir=None,
                                          checkpoint_store=FileCheckpointStore(tmp_path))
                                    .aresume(execution_id))
                for provider in (first, second)]
        assert await first.gate.reached() and await second.gate.reached()
        first.gate.release()
        earlier = await runs[0]
        second.gate.release()
        return earlier, await runs[1]

    earlier, later = asyncio.run(resume_twice())

    # Nothing fenced the second resumer: each sent the pending node's call.
    assert len(first.calls) == 1 and second.calls == first.calls
    assert (earlier.output, later.output) == ("first", "second")
    # The snapshot written last replaced the first resumer's.
    saved = FileCheckpointStore(tmp_path).load(execution_id)
    assert saved["status"] == "completed"
    assert [node["result"] for node in saved["graph"]["nodes"]] == ["second"]
    assert [path.name for path in tmp_path.iterdir()] == [f"{execution_id}.json"]


def offline_calls(monkeypatch) -> list[str]:
    """Record, from here on, every prompt a durable run sends to OfflineProvider."""
    calls, complete = [], OfflineProvider.complete

    async def record(self, system, prompt, model):
        calls.append(prompt)
        return await complete(self, system, prompt, model)

    monkeypatch.setattr(OfflineProvider, "complete", record)
    return calls


def test_a_durable_resume_waits_for_another_owners_live_lease_before_any_call(tmp_path, monkeypatch):
    clock = ManualClock()
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as store:
        run_id = durable_swarm(store).plan(Task("Report")).run_ref["run_id"]
        # Another process holds the run, as a killed one does until its lease lapses.
        held = store.acquire_lease(run_id, "another-process", ttl_s=60)
        before = store.inspect_run(run_id)
        calls = offline_calls(monkeypatch)
        clock.advance(59)
        with pytest.raises(WorkflowLeaseError, match="active lease"):
            durable_swarm(store).resume(run_id)
        assert calls == [] and store.inspect_run(run_id) == before

        clock.advance_past(held.expires_at_ns)
        result = durable_swarm(store).resume(run_id)
        # The lapsed owner can no longer write to the run.
        with pytest.raises(WorkflowLeaseError, match="expired or was superseded"):
            store.release_lease(held)
    assert result.output == "offline: Report" and calls == ["Report"]


def recorded(method, into: list):
    """Wrap *method* so that each value it returns is appended to *into*."""
    def call(*args, **kwargs):
        into.append(method(*args, **kwargs))
        return into[-1]
    return call


def test_a_durable_run_holds_a_60_second_lease_that_its_event_loop_renews_every_15_seconds(
    tmp_path, monkeypatch,
):
    clock, sleep, complete = ManualClock(), asyncio.sleep, OfflineProvider.complete
    intervals, beat, leases, renewals = [], {}, [], []

    async def heartbeat_sleep(delay, result=None):
        if delay != 15:
            return await sleep(delay, result)
        # Each heartbeat interval ends only when the test lets it.
        intervals.append(delay)
        await beat.setdefault("next", asyncio.Event()).wait()
        del beat["next"]
        return result

    async def slow(self, system, prompt, model):
        clock.advance(50)
        beat.setdefault("next", asyncio.Event()).set()
        for _ in range(1000):  # Yield until the heartbeat has renewed the lease.
            if renewals:
                break
            await sleep(0)
        # Past the first lease's 60 seconds, the renewed lease still holds the run.
        clock.advance(50)
        return await complete(self, system, prompt, model)

    monkeypatch.setattr(asyncio, "sleep", heartbeat_sleep)
    monkeypatch.setattr(OfflineProvider, "complete", slow)
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as store:
        monkeypatch.setattr(store, "acquire_lease", recorded(store.acquire_lease, leases))
        monkeypatch.setattr(store, "heartbeat", recorded(store.heartbeat, renewals))
        started = clock()
        result = durable_swarm(store).execute(Task("Report"))

    [lease], [renewed] = leases, renewals
    assert lease.expires_at_ns == started + 60 * 10**9
    assert renewed.epoch == lease.epoch and renewed.expires_at_ns == started + (50 + 60) * 10**9
    assert intervals[:2] == [15, 15]
    assert result.output == "offline: Report"


def test_a_durable_run_whose_lease_lapses_during_a_stall_fails_and_resumes_without_calling_again(
    tmp_path, monkeypatch,
):
    clock = ManualClock()
    complete, calls = OfflineProvider.complete, []

    async def stalled(self, system, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            # Blocking work that never yields lets no heartbeat run.
            clock.advance(61)
        return await complete(self, system, prompt, model)

    monkeypatch.setattr(OfflineProvider, "complete", stalled)
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as store:
        with pytest.raises(WorkflowLeaseError):
            durable_swarm(store).execute(Task("Report"))
        [run] = store.list_runs()
        [call] = store.inspect_run(run["run_id"])["calls"]
        result = durable_swarm(store).resume(run["run_id"])
        accounting = store.inspect_run(run["run_id"])
    assert (call["state"], call["billing_state"]) == ("response_saved", "reserved")
    # Resuming settles the response the stalled call saved instead of sending it again.
    assert result.output == "offline: Report" and calls == ["Report"]
    assert accounting["status"] == "completed" and accounting["unknown_nanousd"] == 0
