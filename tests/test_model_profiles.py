"""Developer-declared model profiles route the nodes that name them on ordinary runs.

A ``ModelProfile`` pairs a provider with a model, a per-call cost ceiling and
the features its provider must declare. A node that names one in
``metadata["profile"]`` calls that provider with that model, whether a graph,
an architect or a developer supervisor's revision added it; every other node
and synthesis call the run's provider. Only developer code can name a
profile: a generated plan cannot. A profile that cannot serve its nodes, a
node that names a profile the run does not allow, and a model-less
``LLMSupervisor`` that would review with another provider's model are refused
with ``NoAllowedProfileError`` before any provider call, planning included.
Checkpoints of a run with profiled nodes are version 5 and list
``model_profiles``, so a reader that would call the run's provider instead
refuses them.
"""

from __future__ import annotations

import asyncio
import json
import typing

import pytest

from fakes import Crash, release_price_version
from smythe import (
    FileCheckpointStore, ModelProfile, NoAllowedProfileError, OfflineProvider, RetryPolicy,
    SentinelAlert, SimpleArchitect, Swarm, Synthesizer, SynthesisStrategy, Task,
)
from smythe import checkpoint as checkpoints
from smythe import swarm as swarm_module
from smythe._failures import INPUT_ERROR, classify_failure, is_terminal
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel
from smythe.checkpoint import CheckpointStore, model_profiles_from_state
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Revision, Topology
from smythe.loader import _MODEL_METADATA_KEYS, build_graph_from_model_output
from smythe.planner import ArchitectError, DeterministicArchitect
from smythe.pricing_anthropic import MODEL as FABLE
from smythe.provider import (
    AnthropicProvider, CompletionResult, GeminiProvider, OpenAIImageProvider, OpenAIProvider,
    Provider, ProviderFeatures,
)
from smythe.provider_messages import AnthropicMessagesProvider
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.registry import Registry
from smythe.supervisor import LLMSupervisor, Supervisor
from smythe.tools import ToolRuntime, ToolSession
from smythe.tracer import Tracer
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import SQLiteWorkflowStore

MODES = ["serial", "parallel"]
SENDS_ONCE = ProviderFeatures(sdk_retries=0)
EVERY_FEATURE = ProviderFeatures(tools=True, attachments=True, image_output=True,
                                 exact_accounting=True, sdk_retries=0)


class Recorder(Provider):
    """Records each call's model and step and answers with its own name.

    It declares *features* for every model. An outcome in *outcomes* is raised
    or answered by the call it lines up with; later calls answer normally.
    """

    def __init__(self, name: str, features: ProviderFeatures = SENDS_ONCE, *outcomes) -> None:
        self.name = name
        self.features = features
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str]] = []

    def feature_support(self, model):
        return self.features

    async def complete(self, system, prompt, model):
        step = next((line.removeprefix("Your step: ") for line in prompt.splitlines()
                     if line.startswith("Your step: ")), prompt.split("\n")[0])
        self.calls.append((model, step))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        return CompletionResult(f"{self.name} wrote {step}", prompt_tokens=1, completion_tokens=1)


class Store(CheckpointStore):
    """Keeps each saved state as detached JSON data, as a file store would, without file writes."""

    def __init__(self) -> None:
        self.states: dict[str, dict] = {}

    def save(self, execution_id, state):
        self.states[execution_id] = json.loads(json.dumps(state))

    def load(self, execution_id):
        state = self.states.get(execution_id)
        return None if state is None else json.loads(json.dumps(state))

    def delete(self, execution_id):
        self.states.pop(execution_id, None)

    def list_ids(self):
        return sorted(self.states)


class NoTools(ToolSession):
    @property
    def tools(self):
        return []

    async def call(self, tool_call):
        raise AssertionError("no tool is offered")


class EmptyTools(ToolRuntime):
    """A tool runtime whose sessions offer no tools, so nodes send plain requests."""

    def open(self, agent):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def session():
            yield NoTools()

        return session()


class Profiled(DeterministicArchitect):
    """Plans a draft on the run's provider and a polish step that names *profile*."""

    def __init__(self, profile: str = "fast") -> None:
        self.profile = profile

    def plan(self, task):
        return graph(profile=self.profile), Registry()


def graph(*, profile: str = "fast", metadata: dict | None = None, **options) -> ExecutionGraph:
    """A draft on the run's provider, then a polish step that names *profile*."""
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="polish", label="Polish", depends_on=["draft"],
             metadata={"profile": profile, **(metadata or {})}, **options),
    ])


def fast_profile(provider: Provider, **options) -> ModelProfile:
    options.setdefault("max_cost_per_call_usd", 0.25)
    return ModelProfile("fast", provider, "fast-model", **options)


def build(default: Provider, *profiles: ModelProfile, **options) -> Swarm:
    options.setdefault("artifact_dir", None)
    return Swarm(model="default-model", provider=default, profiles=profiles, **options)


def enter(swarm: Swarm, entry: str, run_graph: ExecutionGraph | None = None):
    """Start a run the way *entry* names; ``graph`` runs *run_graph*."""
    if entry == "plan":
        return swarm.plan(Task("Report"))
    if entry == "aplan":
        return asyncio.run(swarm.aplan(Task("Report")))
    if entry == "graph":
        return swarm.execute(run_graph if run_graph is not None else graph())
    if entry == "execute-async":
        return asyncio.run(swarm.execute_async(Task("Report")))
    return swarm.execute(Task("Report"))


ENTRIES = ["plan", "aplan", "execute", "execute-async", "execute-parallel", "graph"]


def saved(store: Store) -> tuple[str, dict]:
    (execution_id,) = store.list_ids()
    return execution_id, store.load(execution_id)


# ---------------------------------------------------------------------------
# Profiles and their declarations
# ---------------------------------------------------------------------------


def test_a_profile_keeps_valid_fields():
    provider = Recorder("fast")
    profile = ModelProfile("fast.v1", provider, "fast-model", 1, requires=["tools", "tools"])
    assert (profile.name, profile.provider, profile.model) == ("fast.v1", provider, "fast-model")
    assert profile.max_cost_per_call_usd == 1.0 and type(profile.max_cost_per_call_usd) is float
    assert profile.requires == frozenset({"tools"})
    assert ModelProfile("fast", provider, "fast-model", 0).requires == frozenset()


def test_the_profile_type_resolves_its_type_hints():
    hints = typing.get_type_hints(ModelProfile)
    assert hints["provider"] is Provider and hints["requires"] == frozenset[str]


@pytest.mark.parametrize("options,error", [
    ({"name": "1fast"}, ValueError), ({"name": ""}, ValueError), ({"name": "a" * 65}, ValueError),
    ({"name": "has space"}, ValueError), ({"name": None}, ValueError),
    ({"provider": "fast"}, TypeError), ({"model": ""}, ValueError), ({"model": 3}, ValueError),
    ({"max_cost_per_call_usd": -1}, ValueError), ({"max_cost_per_call_usd": True}, ValueError),
    ({"max_cost_per_call_usd": float("inf")}, ValueError),
    ({"max_cost_per_call_usd": "0.5"}, ValueError),
    ({"requires": "tools"}, ValueError), ({"requires": {"capabilities"}}, ValueError),
    ({"requires": {"text"}}, ValueError),
])
def test_a_malformed_profile_is_refused_when_built(options, error):
    fields = {"name": "fast", "provider": Recorder("fast"), "model": "fast-model",
              "max_cost_per_call_usd": 0.25, **options}
    with pytest.raises(error):
        ModelProfile(**fields)


@pytest.mark.parametrize("profiles,error", [
    ("fast", TypeError), ({"fast": None}, TypeError), ([object()], TypeError),
    ([fast_profile(Recorder("a")), fast_profile(Recorder("b"))], ValueError),
])
def test_swarm_refuses_profiles_that_are_not_distinct_model_profiles(profiles, error):
    with pytest.raises(error):
        Swarm(model="default-model", provider=Recorder("default"), profiles=profiles)


@pytest.mark.parametrize("make_provider,model,declared", [
    pytest.param(OfflineProvider, "offline", False, id="offline"),
    pytest.param(lambda: AnthropicProvider(api_key="test-key"), "claude-opus-5-5", False,
                 id="anthropic"),
    pytest.param(lambda: OpenAIProvider(api_key="test-key"), "gpt-5", False, id="openai-chat"),
    pytest.param(lambda: OpenAIImageProvider(api_key="test-key"), "gpt-image-1", False,
                 id="openai-image"),
    pytest.param(lambda: GeminiProvider(api_key="test-key"), "gemini-3-pro", False, id="gemini"),
    pytest.param(lambda: OpenAIResponsesProvider(api_key="test-key"), "gpt-6-astra", True,
                 id="responses"),
    pytest.param(lambda: AnthropicMessagesProvider(api_key="test-key"), FABLE, True,
                 id="messages"),
    pytest.param(lambda: Recorder("custom", ProviderFeatures()), "custom", None, id="custom"),
])
def test_providers_declare_whether_they_price_each_call_exactly(make_provider, model, declared):
    assert make_provider().feature_support(model).exact_accounting is declared


def test_no_allowed_profile_is_an_input_error_that_does_not_stop_the_run_by_itself():
    error = NoAllowedProfileError("no profile fits")
    assert isinstance(error, ValueError)
    assert classify_failure(error) == INPUT_ERROR and not is_terminal(error)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_profiled_node_calls_its_profiles_provider_and_model(mode):
    default, fast = Recorder("default"), Recorder("fast")
    result = build(default, fast_profile(fast), parallel=mode == "parallel").execute(graph())

    assert default.calls == [("default-model", "Draft")]
    assert fast.calls == [("fast-model", "Polish")]
    polish = result.graph.nodes[1]
    assert polish.metadata["profile"] == "fast" and polish.metadata["model"] == "fast-model"
    assert result.graph.nodes[0].metadata["model"] == "default-model"
    assert result.output == "fast wrote Polish"


@pytest.mark.parametrize("entry", ["execute", "execute-async", "execute-parallel"])
def test_a_developer_plan_routes_its_profiled_nodes(entry):
    default, fast = Recorder("default"), Recorder("fast")
    swarm = build(default, fast_profile(fast), architect=Profiled(),
                  parallel=entry == "execute-parallel")
    result = enter(swarm, entry)

    assert default.calls == [("default-model", "Draft")]
    assert fast.calls == [("fast-model", "Polish")]
    assert result.output == "fast wrote Polish"


class AddAfterPolish(Supervisor):
    """Adds one step after the polish step, once."""

    async def review(self, graph, node, *, task, revisions_remaining):
        if node.id != "polish":
            return None
        return Revision(add_nodes=(Node(id="extra", label="Extra", depends_on=["polish"]),))


@pytest.mark.parametrize("mode", MODES)
def test_supervisor_additions_and_synthesis_call_the_runs_provider(mode):
    default, fast = Recorder("default"), Recorder("fast")
    only_profiled = ExecutionGraph([Topology.SERIAL], [
        Node(id="polish", label="Polish", metadata={"profile": "fast"}),
    ])
    swarm = build(default, fast_profile(fast), supervisor=AddAfterPolish(), max_revisions=1,
                  synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE),
                  parallel=mode == "parallel")
    result = swarm.execute(only_profiled)

    assert fast.calls == [("fast-model", "Polish")]
    # The added step has no profile, so it takes the run's model, not the
    # profiled node's; the merge calls the run's provider too.
    assert [model for model, _ in default.calls] == ["default-model", "default-model"]
    assert default.calls[0][1] == "Extra"
    extra = next(node for node in result.graph.nodes if node.id == "extra")
    assert "profile" not in extra.metadata and extra.metadata["model"] == "default-model"


class AddProfiled(Supervisor):
    """Adds one step after the draft whose metadata names *profile*, once."""

    def __init__(self, profile: str = "fast", **metadata) -> None:
        self.metadata = {"profile": profile, **metadata}

    async def review(self, graph, node, *, task, revisions_remaining):
        if node.id != "draft":
            return None
        return Revision(add_nodes=(Node(id="extra", label="Extra", depends_on=["draft"],
                                        metadata=dict(self.metadata)),))


def draft_only() -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft")])


@pytest.mark.parametrize("mode", MODES)
def test_a_supervisor_addition_that_names_a_profile_calls_it_and_resumes(mode):
    store = Store()
    default = Recorder("default")
    fast = Recorder("fast", SENDS_ONCE, Crash("process lost during the call"))
    with pytest.raises(Crash):
        build(default, fast_profile(fast), checkpoint_store=store, supervisor=AddProfiled(),
              max_revisions=1, parallel=mode == "parallel").execute(draft_only())
    assert default.calls == [("default-model", "Draft")]
    assert fast.calls == [("fast-model", "Extra")]
    execution_id, state = saved(store)
    extra = next(node for node in state["graph"]["nodes"] if node["id"] == "extra")
    # The saved node records the model its profile called, not the run's.
    assert extra["metadata"]["profile"] == "fast" and extra["metadata"]["model"] == "fast-model"
    assert state["version"] == 5 and state["requires"] == ["model_profiles"]

    default, fast = Recorder("default"), Recorder("fast")
    result = build(default, fast_profile(fast), checkpoint_store=store, supervisor=AddProfiled(),
                   max_revisions=1, parallel=mode == "parallel").resume(execution_id)
    assert default.calls == [] and fast.calls == [("fast-model", "Extra")]
    assert result.output == "fast wrote Extra"
    extra = next(node for node in result.graph.nodes if node.id == "extra")
    assert extra.metadata["model"] == "fast-model"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("supervisor,match", [
    pytest.param(AddProfiled("slow"), "'slow'", id="not-allowed"),
    pytest.param(AddProfiled(model="other-model"), "other-model", id="other-model"),
])
def test_a_supervisor_addition_whose_profile_cannot_serve_it_is_refused_before_it_applies(
    mode, supervisor, match,
):
    default, fast = Recorder("default"), Recorder("fast")
    result = build(default, fast_profile(fast), supervisor=supervisor, max_revisions=1,
                   parallel=mode == "parallel").execute(draft_only())

    (refused,) = [span for span in result.trace if span["status"] == "revision_rejected"]
    assert match in refused["error"]
    assert [node.id for node in result.graph.nodes] == ["draft"]
    assert default.calls == [("default-model", "Draft")] and fast.calls == []
    assert result.output == "default wrote Draft"


@pytest.mark.parametrize("entry", ["execute", "execute-async", "graph"])
def test_a_model_less_llm_supervisor_with_another_providers_profile_is_refused_before_any_call(
    entry,
):
    # It would review a profiled node with the profile's model on its own provider.
    default, fast = Recorder("default"), Recorder("fast")
    swarm = build(default, fast_profile(fast), supervisor=LLMSupervisor(default),
                  max_revisions=1)
    with pytest.raises(NoAllowedProfileError, match=r"LLMSupervisor\(provider, model="):
        enter(swarm, entry)
    # Planning included: the run's own provider plans the generated runs.
    assert default.calls == [] and fast.calls == []


@pytest.mark.parametrize("mode", MODES)
def test_an_executor_refuses_a_model_less_llm_supervisor_with_another_providers_profile(mode):
    default, fast = Recorder("default"), Recorder("fast")
    executor_class = AsyncExecutor if mode == "parallel" else Executor
    executor = executor_class(provider=default, registry=Registry(), tracer=Tracer(),
                              artifact_dir=None, supervisor=LLMSupervisor(default),
                              max_revisions=1, profiles=[fast_profile(fast)])
    run_graph = graph()
    run_graph.nodes[0].metadata["model"] = "default-model"
    with pytest.raises(NoAllowedProfileError, match="LLMSupervisor"):
        asyncio.run(executor.run(run_graph)) if mode == "parallel" else executor.run(run_graph)
    assert default.calls == [] and fast.calls == []


@pytest.mark.parametrize("case", ["own-model", "same-provider", "no-revisions"])
def test_an_llm_supervisor_reviews_profiled_nodes_with_a_model_its_provider_serves(case):
    default, fast = Recorder("default"), Recorder("fast")
    profile = (ModelProfile("fast", default, "fast-model", 0.25) if case == "same-provider"
               else fast_profile(fast))
    supervisor = LLMSupervisor(default, model="review-model" if case == "own-model" else None)
    build(default, profile, supervisor=supervisor,
          max_revisions=0 if case == "no-revisions" else 1).execute(graph())

    reviews = [model for model, step in default.calls if step == "Original task:"]
    assert reviews == {"own-model": ["review-model"], "same-provider": ["fast-model"],
                       "no-revisions": []}[case]
    assert ("fast-model", "Polish") in (default if case == "same-provider" else fast).calls


def test_a_generated_plan_cannot_name_a_profile():
    assert _MODEL_METADATA_KEYS == frozenset({"role"})
    plan = {"topology": "serial",
            "nodes": [{"id": "polish", "label": "Polish", "metadata": {"profile": "fast"}}]}
    with pytest.raises(ValueError, match="unsupported field"):
        build_graph_from_model_output(plan)

    fast = Recorder("fast")
    swarm = build(OfflineProvider(plan=plan), fast_profile(fast))
    with pytest.raises(ArchitectError, match="profile"):
        swarm.execute(Task("Report"))
    assert fast.calls == []


@pytest.mark.parametrize("mode", MODES)
def test_executors_route_profiled_nodes_without_a_swarm(mode):
    default, fast = Recorder("default"), Recorder("fast")
    run_graph = graph()
    run_graph.nodes[0].metadata["model"] = "default-model"
    executor_class = AsyncExecutor if mode == "parallel" else Executor
    executor = executor_class(provider=default, registry=Registry(), tracer=Tracer(),
                              artifact_dir=None, profiles=[fast_profile(fast)])
    if mode == "parallel":
        asyncio.run(executor.run(run_graph))
    else:
        executor.run(run_graph)

    assert default.calls == [("default-model", "Draft")]
    assert fast.calls == [("fast-model", "Polish")]


@pytest.mark.parametrize("mode", MODES)
def test_an_executor_without_the_profile_fails_the_node_without_a_call(mode):
    default = Recorder("default")
    run_graph = graph(failure_policy=FailurePolicy.SKIP)
    executor_class = AsyncExecutor if mode == "parallel" else Executor
    executor = executor_class(provider=default, registry=Registry(), tracer=Tracer(),
                              artifact_dir=None)
    if mode == "parallel":
        asyncio.run(executor.run(run_graph))
    else:
        executor.run(run_graph)

    polish = run_graph.nodes[1]
    assert polish.status is NodeStatus.SKIPPED and "does not allow" in polish.result
    assert [step for _, step in default.calls] == ["Draft"]


def test_executors_refuse_profiles_with_journal_managed_calls():
    with pytest.raises(ValueError, match="profiles"):
        AsyncExecutor(provider=Recorder("default"), registry=Registry(), tracer=Tracer(),
                      provider_call_factory=lambda *args: None, profiles=[fast_profile(Recorder("f"))])


# ---------------------------------------------------------------------------
# Refusals before any call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("requires,features,match", [
    pytest.param({"tools"}, ProviderFeatures(tools=False, sdk_retries=0), "tools",
                 id="declared-unsupported"),
    pytest.param({"exact_accounting"}, ProviderFeatures(), "exact_accounting",
                 id="not-declared"),
    pytest.param({"image_output", "attachments"}, ProviderFeatures(attachments=True),
                 "image_output", id="one-of-two"),
    pytest.param(set(), ProviderFeatures(model_supported=False), "does not support",
                 id="unsupported-model"),
])
def test_a_profile_that_does_not_fit_is_refused_before_any_call(entry, requires, features, match):
    default, fast = Recorder("default"), Recorder("fast", features)
    swarm = build(default, fast_profile(fast, requires=requires),
                  parallel=entry == "execute-parallel")
    with pytest.raises(NoAllowedProfileError, match=match):
        enter(swarm, entry)
    # Planning included: the run's own provider plans these runs.
    assert default.calls == [] and fast.calls == []


@pytest.mark.parametrize("entry", ["execute", "execute-parallel", "graph"])
def test_a_profile_whose_provider_declares_every_requirement_runs(entry):
    default, fast = Recorder("default"), Recorder("fast", EVERY_FEATURE)
    profile = fast_profile(fast, requires={"tools", "attachments", "image_output",
                                           "exact_accounting"})
    swarm = build(default, profile, architect=Profiled(), parallel=entry == "execute-parallel")
    enter(swarm, entry)
    assert fast.calls == [("fast-model", "Polish")]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_tool_run_refuses_a_profile_that_cannot_call_tools_before_any_call(entry):
    default = Recorder("default", ProviderFeatures(tools=True, sdk_retries=0))
    fast = Recorder("fast", ProviderFeatures(tools=False))
    swarm = build(default, fast_profile(fast), tool_runtime=EmptyTools(),
                  parallel=entry == "execute-parallel")
    with pytest.raises(NoAllowedProfileError, match="tool"):
        enter(swarm, entry)
    assert default.calls == [] and fast.calls == []


@pytest.mark.parametrize("swarm_profiles", [True, False], ids=["other-profiles", "no-profiles"])
@pytest.mark.parametrize("entry", ["graph", "plan", "execute"])
def test_an_unknown_profile_name_is_refused_before_any_call(entry, swarm_profiles):
    default, fast = Recorder("default"), Recorder("fast")
    profiles = [fast_profile(fast)] if swarm_profiles else []
    swarm = build(default, *profiles, architect=Profiled("slow"))
    with pytest.raises(NoAllowedProfileError, match="'slow'"):
        enter(swarm, entry, graph(profile="slow"))
    assert default.calls == [] and fast.calls == []


@pytest.mark.parametrize("value", [None, 7, ["fast"]])
def test_a_profile_name_that_is_not_text_is_refused(value):
    default, fast = Recorder("default"), Recorder("fast")
    run_graph = graph()
    run_graph.nodes[1].metadata["profile"] = value
    with pytest.raises(NoAllowedProfileError):
        build(default, fast_profile(fast)).execute(run_graph)
    assert default.calls == [] and fast.calls == []


def test_a_node_that_sets_another_model_than_its_profiles_is_refused():
    default, fast = Recorder("default"), Recorder("fast")
    run_graph = graph()
    run_graph.nodes[1].metadata["model"] = "other-model"
    with pytest.raises(NoAllowedProfileError, match="other-model"):
        build(default, fast_profile(fast)).execute(run_graph)
    assert default.calls == [] and fast.calls == []
    # Nothing is stamped on a refused graph.
    assert "model" not in run_graph.nodes[0].metadata


def test_a_profile_naming_a_retired_model_is_refused_at_admission(monkeypatch):
    clients: list[str] = []
    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client",
                        lambda self: clients.append("client") or pytest.fail("SDK created"))
    native = OpenAIResponsesProvider(api_key="test-key", max_cost_per_call_usd=0.5)
    profile = ModelProfile("astra", native, "gpt-6-astra", 0.5, requires={"exact_accounting"})
    default = Recorder("default")
    run_graph = graph(profile="astra")
    # Admitted while the model is priced and not retired.
    planned = build(default, profile, architect=Profiled("astra")).plan(Task("Report"))
    assert planned.nodes[1].metadata["model"] == "gpt-6-astra"

    # A release retires the model on a day that has passed.
    release_price_version(monkeypatch, version="openai-test-v2",
                          retired_on={"gpt-6-astra": "2000-01-01"})
    for entry in ("plan", "execute", "graph"):
        swarm = build(default, profile, architect=Profiled("astra"))
        with pytest.raises(NoAllowedProfileError, match="retired"):
            enter(swarm, entry, run_graph)
    assert default.calls == [] and clients == []


@pytest.mark.parametrize("entry", ["execute", "graph"])
def test_a_retry_policy_refuses_a_profile_whose_sdk_may_retry_before_any_call(entry):
    default, fast = Recorder("default"), Recorder("fast", ProviderFeatures())
    swarm = build(default, fast_profile(fast), retry_policy=RetryPolicy(), architect=Profiled())
    with pytest.raises(ValueError, match="sdk_retries"):
        enter(swarm, entry)
    assert default.calls == [] and fast.calls == []


@pytest.mark.parametrize("mode", MODES)
def test_an_executor_with_a_retry_policy_refuses_a_profile_whose_sdk_may_retry(mode):
    default, fast = Recorder("default"), Recorder("fast", ProviderFeatures())
    executor_class = AsyncExecutor if mode == "parallel" else Executor
    executor = executor_class(provider=default, registry=Registry(), tracer=Tracer(),
                              artifact_dir=None, retry_policy=RetryPolicy(),
                              profiles=[fast_profile(fast)])
    run_graph = graph()
    run_graph.nodes[0].metadata["model"] = "default-model"
    with pytest.raises(ValueError, match="sdk_retries"):
        asyncio.run(executor.run(run_graph)) if mode == "parallel" else executor.run(run_graph)
    assert default.calls == [] and fast.calls == []


@pytest.mark.parametrize("mode", MODES)
def test_an_executor_with_a_retry_policy_checks_profiles_a_supervisor_addition_may_name(mode):
    # No node names the profile yet, but a revision may add one that does.
    default, fast = Recorder("default"), Recorder("fast", ProviderFeatures())
    executor_class = AsyncExecutor if mode == "parallel" else Executor
    executor = executor_class(provider=default, registry=Registry(), tracer=Tracer(),
                              artifact_dir=None, retry_policy=RetryPolicy(),
                              supervisor=AddProfiled(), max_revisions=1,
                              profiles=[fast_profile(fast)])
    run_graph = draft_only()
    run_graph.nodes[0].metadata["model"] = "default-model"
    with pytest.raises(ValueError, match="sdk_retries"):
        asyncio.run(executor.run(run_graph)) if mode == "parallel" else executor.run(run_graph)
    assert default.calls == [] and fast.calls == []


def test_durable_runs_refuse_profiles(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(WorkflowBindingError, match="profiles"):
            Swarm(model="offline", provider=OfflineProvider(), run_store=store,
                  profiles=[fast_profile(Recorder("fast"))])

        provider = OfflineProvider()
        swarm = Swarm(model="offline", provider=provider, architect=SimpleArchitect(),
                      run_store=store, max_budget_usd=1)
        with pytest.raises(WorkflowBindingError, match="profile"):
            swarm.execute(graph())
        assert provider.calls == []


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def record_reservations(monkeypatch) -> list[tuple[str, float]]:
    reserved: list[tuple[str, float]] = []
    reserve = Sentinel.reserve

    def recording(self, node_id, estimated_cost, **options):
        reserved.append((node_id, estimated_cost))
        return reserve(self, node_id, estimated_cost, **options)

    monkeypatch.setattr(Sentinel, "reserve", recording)
    return reserved


@pytest.mark.parametrize("mode", MODES)
def test_a_profiled_node_reserves_its_profiles_ceiling(monkeypatch, mode):
    reserved = record_reservations(monkeypatch)
    default, fast = Recorder("default"), Recorder("fast")
    build(default, fast_profile(fast), max_budget_usd=5,
          parallel=mode == "parallel").execute(graph())
    assert ("polish", 0.25) in reserved

    # A node's own estimate still wins over the ceiling.
    reserved.clear()
    build(default, fast_profile(fast), max_budget_usd=5,
          parallel=mode == "parallel").execute(graph(metadata={"estimated_cost_usd": 0.1}))
    assert ("polish", 0.1) in reserved and ("polish", 0.25) not in reserved


@pytest.mark.parametrize("mode", MODES)
def test_a_ceiling_the_budget_cannot_cover_is_refused_before_the_call(mode):
    default, fast = Recorder("default"), Recorder("fast")
    only_profiled = ExecutionGraph([Topology.SERIAL], [
        Node(id="polish", label="Polish", metadata={"profile": "fast"}),
    ])
    swarm = build(default, fast_profile(fast, max_cost_per_call_usd=2), max_budget_usd=1,
                  parallel=mode == "parallel")
    with pytest.raises(SentinelAlert):
        swarm.execute(only_profiled)
    assert default.calls == [] and fast.calls == []


def test_a_held_check_uses_its_profiles_ceiling(monkeypatch):
    held: list[tuple[str, float]] = []
    hold = Sentinel.hold

    def recording(self, key, amount):
        held.append((key, amount))
        return hold(self, key, amount)

    monkeypatch.setattr(Sentinel, "hold", recording)
    default, judge = Recorder("default"), Recorder("judge")
    gated = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="judge", label="Judge", depends_on=["draft"], verifies="draft",
             metadata={"profile": "judge"}),
    ])
    swarm = build(default, ModelProfile("judge", judge, "judge-model", 0.3), max_budget_usd=5,
                  reserve_for_checks=True, parallel=True)
    swarm.execute(gated)
    assert held == [("judge", 0.3)]
    assert judge.calls == [("judge-model", "Judge")]


@pytest.mark.parametrize("mode", MODES)
def test_persisted_attempts_on_a_profiled_node_use_its_provider_and_ceiling(monkeypatch, mode):
    reserved = record_reservations(monkeypatch)
    store = Store()
    default = Recorder("default")
    fast = Recorder("fast", SENDS_ONCE, ConnectionError("reset"), ConnectionError("reset"))
    swarm = build(default, fast_profile(fast), max_budget_usd=5, checkpoint_store=store,
                  retry_policy=RetryPolicy(persist_attempts=True), parallel=mode == "parallel")
    result = swarm.execute(graph(failure_policy=FailurePolicy.RETRY, max_retries=2))

    assert result.output == "fast wrote Polish"
    assert fast.calls == [("fast-model", "Polish")] * 3
    assert {amount for node_id, amount in reserved if node_id == "polish"} == {0.25}
    _, state = saved(store)
    assert state["requires"] == ["attempt_allowance", "model_profiles"]
    assert state["control"]["attempts_used"]["polish"]["attempts"] == 3


# ---------------------------------------------------------------------------
# Checkpoints and resume
# ---------------------------------------------------------------------------


def test_a_run_with_profiled_nodes_writes_version_5_and_others_version_4():
    store = Store()
    build(Recorder("default"), fast_profile(Recorder("fast")), checkpoint_store=store).execute(graph())
    _, state = saved(store)
    assert state["version"] == 5 and state["requires"] == ["model_profiles"]
    assert model_profiles_from_state(state) == {"polish": "fast"}

    store = Store()
    plain = ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft")])
    build(Recorder("default"), fast_profile(Recorder("fast")), checkpoint_store=store).execute(plain)
    _, state = saved(store)
    assert state["version"] == 4 and "requires" not in state
    assert model_profiles_from_state(state) == {}


def interrupted(store: Store, *, status: str = "running") -> tuple[str, dict]:
    """A store holding one run whose profiled step was ``completed`` or crashed during its call."""
    if status == "completed":
        build(Recorder("default"), fast_profile(Recorder("fast")),
              checkpoint_store=store).execute(graph())
        return saved(store)
    fast = Recorder("fast", SENDS_ONCE, Crash("process lost during the call"))
    with pytest.raises(Crash):
        build(Recorder("default"), fast_profile(fast), checkpoint_store=store).execute(graph())
    return saved(store)


@pytest.mark.parametrize("mode", MODES)
def test_resume_keeps_the_profile(tmp_path, mode):
    store = Store()
    execution_id, state = interrupted(store)
    assert state["status"] == "failed" and state["requires"] == ["model_profiles"]

    default, fast = Recorder("default"), Recorder("fast")
    result = build(default, fast_profile(fast), checkpoint_store=store,
                   parallel=mode == "parallel").resume(execution_id)

    assert default.calls == [] and fast.calls == [("fast-model", "Polish")]
    assert result.output == "fast wrote Polish"
    assert store.load(execution_id)["requires"] == ["model_profiles"]


def test_resume_through_a_file_store_keeps_the_profile(tmp_path):
    store = FileCheckpointStore(tmp_path)
    fast = Recorder("fast", SENDS_ONCE, Crash("process lost during the call"))
    with pytest.raises(Crash):
        build(Recorder("default"), fast_profile(fast), checkpoint_store=store).execute(graph())
    (execution_id,) = store.list_ids()

    default, fast = Recorder("default"), Recorder("fast")
    build(default, fast_profile(fast), checkpoint_store=store).resume(execution_id)
    assert default.calls == [] and fast.calls == [("fast-model", "Polish")]


@pytest.mark.parametrize("profiles,match", [
    pytest.param(lambda fast: [], "'fast'", id="no-profiles"),
    pytest.param(lambda fast: [ModelProfile("slow", fast, "fast-model", 0.25)], "'fast'",
                 id="other-profile"),
    pytest.param(lambda fast: [ModelProfile("fast", fast, "new-model", 0.25)], "new-model",
                 id="changed-model"),
])
def test_resume_refuses_a_profile_the_resuming_swarm_does_not_allow(profiles, match):
    store = Store()
    execution_id, state = interrupted(store)
    default, fast = Recorder("default"), Recorder("fast")
    with pytest.raises(NoAllowedProfileError, match=match):
        build(default, *profiles(fast), checkpoint_store=store).resume(execution_id)
    assert default.calls == [] and fast.calls == [] and store.load(execution_id) == state


def test_a_completed_run_resumes_without_its_profiles_and_without_a_call():
    store = Store()
    execution_id, _ = interrupted(store, status="completed")
    default = Recorder("default")
    result = build(default, checkpoint_store=store).resume(execution_id)
    assert result.output == "fast wrote Polish" and default.calls == []


@pytest.mark.parametrize("status", ["completed", "running"])
def test_a_reader_that_knows_only_delivery_acceptance_refuses_profiles(monkeypatch, status):
    store = Store()
    execution_id, state = interrupted(store, status=status)
    assert state["requires"] == ["model_profiles"]
    monkeypatch.setattr(checkpoints, "KNOWN_CHECKPOINT_REQUIREMENTS",
                        frozenset({"delivery_acceptance"}))
    default, fast = Recorder("default"), Recorder("fast")

    with pytest.raises(ValueError, match="model_profiles"):
        build(default, fast_profile(fast), checkpoint_store=store).resume(execution_id)
    assert default.calls == [] and fast.calls == [] and store.load(execution_id) == state


@pytest.mark.parametrize("status", ["completed", "running"])
def test_a_reader_of_versions_1_to_4_refuses_profiles(monkeypatch, status):
    store = Store()
    execution_id, state = interrupted(store, status=status)
    monkeypatch.setattr(swarm_module, "SUPPORTED_CHECKPOINT_VERSIONS", (1, 2, 3, 4))
    default = Recorder("default")

    with pytest.raises(ValueError, match="version 5"):
        build(default, fast_profile(Recorder("fast")), checkpoint_store=store).resume(execution_id)
    assert default.calls == [] and store.load(execution_id) == state


@pytest.mark.parametrize("status", ["completed", "running"])
@pytest.mark.parametrize("edit,match", [
    pytest.param(lambda state: state.update(requires=[]), "omits", id="unlisted"),
    pytest.param(lambda state: state["graph"]["nodes"][1]["metadata"].pop("profile"),
                 "name a profile", id="listed-without-a-profiled-node"),
])
def test_a_state_whose_profiles_and_requirements_disagree_is_refused(edit, match, status):
    store = Store()
    execution_id, state = interrupted(store, status=status)
    edit(state)
    store.save(execution_id, state)
    default, fast = Recorder("default"), Recorder("fast")

    with pytest.raises(ValueError, match=match):
        build(default, fast_profile(fast), checkpoint_store=store).resume(execution_id)
    with pytest.raises(ValueError, match=match):
        model_profiles_from_state(state)
    assert default.calls == [] and fast.calls == []


def test_a_version_4_state_that_names_a_profile_resumes_only_on_an_allowed_one():
    # Builds that write version 4 never route a profile, but a graph could
    # carry the key; the resuming Swarm decides where its nodes go.
    store = Store()
    execution_id, state = interrupted(store)
    state["version"] = 4
    del state["requires"]
    store.save(execution_id, state)
    assert model_profiles_from_state(state) == {"polish": "fast"}
    default, fast = Recorder("default"), Recorder("fast")

    with pytest.raises(NoAllowedProfileError, match="'fast'"):
        build(default, checkpoint_store=store).resume(execution_id)
    assert default.calls == [] and fast.calls == []
    build(default, fast_profile(fast), checkpoint_store=store).resume(execution_id)
    assert default.calls == [] and fast.calls == [("fast-model", "Polish")]
    assert store.load(execution_id)["requires"] == ["model_profiles"]
