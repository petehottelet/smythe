"""Public preflight and per-run component bindings never rewrite shared objects."""

import asyncio
import json

import pytest

from smythe.agent import Agent, AgentProfile
from smythe.constrained_planner import ConstrainedArchitect, SubGraphTemplate
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.planner import DeterministicArchitect, LLMArchitect, SimpleArchitect
from smythe.provider import CompletionResult, OfflineProvider, Provider, ProviderResponseError
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.registry import Registry
from smythe.router import WhiteRabbit
from smythe.skills import CapabilityHydrationMode
from smythe.supervisor import LLMSupervisor
from smythe.synthesizer import Synthesizer, SynthesisStrategy
from smythe.task import Task
from smythe.workflow_binding import (
    CallScope, ComponentBinding, LocalOnly, WorkflowBindingError, bind_component,
    describe_component, json_snapshot,
)


PLAN = json.dumps({"topology": ["serial"], "nodes": [{"id": "a", "label": "Write it"}]})


class Factory:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, source, scope):
        parent = self

        class BoundCall(Provider):
            async def complete(self, system, prompt, model):
                parent.calls.append((source, scope, system, prompt, model))
                value = parent.outputs.pop(0)
                if isinstance(value, BaseException):
                    raise value
                return CompletionResult(value, cost_usd=0)

        return BoundCall()


def binding(factory=None, *, phase="planning", component_id="architect", provider=None, model="test"):
    return ComponentBinding("run-a", component_id, phase, factory or Factory(), provider, model)


def completed_graph():
    return ExecutionGraph([Topology.SERIAL], [Node("A", id="a", result="text", status=NodeStatus.COMPLETED)])


def test_call_scope_is_detached_and_strictly_identifies_repair_attempts():
    trigger = {"node": "a", "inputs": [1]}
    scope = CallScope("planner", "planning", trigger, attempt=2)
    trigger["inputs"].append(2)
    scope.trigger["inputs"].append(3)
    assert scope.trigger == {"node": "a", "inputs": [1]}
    assert scope.to_dict()["attempt"] == 2


@pytest.mark.parametrize("field,value", [("attempt", True), ("turn", -1), ("generation", 1.5)])
def test_call_scope_rejects_ambiguous_numeric_identity(field, value):
    with pytest.raises(WorkflowBindingError):
        CallScope("planner", "planning", "task", **{field: value})


@pytest.mark.parametrize("value", [{1: "x"}, float("nan"), object()])
def test_configuration_requires_finite_json(value):
    with pytest.raises(WorkflowBindingError):
        json_snapshot(value)


def test_component_tree_shares_one_snapshot_per_source_and_isolates_runs():
    source = OfflineProvider(echo_prefix="same:")
    first, second = binding(), binding()
    a = first.snapshot_provider(source)
    assert first.child("routing").snapshot_provider(source) is a
    assert second.snapshot_provider(source) is not a
    assert a is not source
    assert not source.calls


def test_planner_repairs_have_explicit_attempts_and_fresh_bound_provider():
    source, factory = OfflineProvider(), Factory("not json", PLAN)
    original = LLMArchitect(source, planning_model="test", max_retries=2)
    bound = bind_component(original, binding(factory))
    graph, _ = bound.plan(Task("goal"))
    assert graph.nodes[0].id == "a"
    assert [entry[1].attempt for entry in factory.calls] == [0, 1]
    assert all(entry[1].trigger == "task" and entry[1].phase == "planning" for entry in factory.calls)
    assert bound.workflow_providers()[0] is not source
    assert not source.calls


def test_bound_planner_does_not_repair_terminal_native_failure():
    error = ProviderResponseError("native output unavailable")
    factory = Factory(error)
    bound = bind_component(LLMArchitect(OfflineProvider(), planning_model="test"), binding(factory))
    with pytest.raises(ProviderResponseError) as caught:
        bound.plan(Task("goal"))
    assert caught.value is error and len(factory.calls) == 1


@pytest.mark.parametrize("component", [
    LLMArchitect(OfflineProvider(), memory=object()),
    ConstrainedArchitect(OfflineProvider(), [SubGraphTemplate("x", "X", lambda task: ([], Registry()))]),
    LLMArchitect(OfflineProvider(responses=["scripted"])),
    LLMArchitect(OfflineProvider(artifacts_per_call=1)),
])
def test_preflight_rejects_unjournaled_dependencies(component):
    with pytest.raises(ValueError):
        describe_component(component)


def test_custom_subclass_cannot_silently_inherit_builtin_binding():
    class Custom(LLMArchitect):
        pass

    component = Custom(OfflineProvider())
    with pytest.raises(WorkflowBindingError):
        describe_component(component)
    with pytest.raises(WorkflowBindingError):
        component.bind_run(binding())


def test_router_validates_unselected_tiers_before_any_classification():
    class Unbound(DeterministicArchitect):
        def plan(self, task):
            raise AssertionError("must not run")

    factory = Factory("autonomous", PLAN)
    router = WhiteRabbit(
        classifier_provider=OfflineProvider(), classifier_model="test",
        deterministic={"bad": Unbound()}, autonomous=LLMArchitect(OfflineProvider(), planning_model="test"),
    )
    with pytest.raises(WorkflowBindingError):
        bind_component(router, binding(factory, phase="routing", component_id="router"))
    assert not factory.calls


def test_router_binds_all_tiers_and_preserves_component_scope():
    source, factory = OfflineProvider(), Factory("autonomous", PLAN)
    router = WhiteRabbit(
        classifier_provider=source, classifier_model="test", deterministic={"simple": SimpleArchitect()},
        autonomous=LLMArchitect(source, planning_model="test"),
    )
    bound = bind_component(router, binding(factory, phase="routing", component_id="router"))
    selected = bound.route(Task("goal"))
    selected.plan(Task("goal"))
    assert [c[1].phase for c in factory.calls] == ["routing", "planning"]
    assert [c[1].component_id for c in factory.calls] == ["router", "router/autonomous"]
    providers = bound.workflow_providers()
    assert providers[0] is providers[1] and providers[0] is not source


def test_preflight_checks_models_in_unselected_native_tiers_without_client_creation(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("preflight created SDK client")

    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", fail)
    router = WhiteRabbit(
        classifier_provider=OfflineProvider(), autonomous=SimpleArchitect(),
        constrained=LLMArchitect(OpenAIResponsesProvider(api_key="unused"), planning_model="unsupported"),
    )
    with pytest.raises(ValueError):
        describe_component(router)


def test_registry_snapshot_detaches_profiles_and_nested_history():
    source = Registry()
    agent = Agent(AgentProfile("Analyst", capabilities=["research"]), id="agent-a", history=[{"x": [1]}])
    source.register(agent)
    bound = bind_component(source, binding())
    agent.profile.capabilities.append("new")
    agent.history[0]["x"].append(2)
    bound.get("agent-a").profile.name = "run-only"
    assert bound.get("agent-a").profile.capabilities == ["research"]
    assert bound.get("agent-a").history == [{"x": [1]}]
    assert agent.name == "Analyst"


def test_public_registry_restore_preserves_full_snapshot_without_aliasing():
    original = Registry()
    original.register(Agent(AgentProfile("A"), id="a", history=[{"nested": [1]}]))
    description = original.workflow_description()
    restored = Registry.from_workflow_description(description)
    description["agents"][0]["history"][0]["nested"].append(2)
    assert restored.get("a").history == [{"nested": [1]}]
    assert restored.workflow_description() == original.workflow_description()


@pytest.mark.parametrize("bad", [
    None, {}, {"type": "registry", "version": True, "agents": []},
    {"type": "registry", "version": 1.0, "agents": []},
    {"type": "registry", "version": 1, "agents": {}},
    {"type": "registry", "version": 1, "agents": [None]},
])
def test_public_registry_restore_rejects_malformed_schema(bad):
    with pytest.raises(WorkflowBindingError):
        Registry.from_workflow_description(bad)


def test_public_registry_restore_rejects_duplicate_identity():
    original = Registry()
    original.register(Agent(AgentProfile("A"), id="a"))
    description = original.workflow_description()
    description["agents"].append(description["agents"][0])
    with pytest.raises(WorkflowBindingError):
        Registry.from_workflow_description(description)


def test_wrong_component_roles_are_rejected_before_classifier_calls():
    router = WhiteRabbit(classifier_provider=OfflineProvider(), autonomous=Synthesizer())
    with pytest.raises(WorkflowBindingError, match="role"):
        bind_component(router, binding(phase="routing"))


@pytest.mark.parametrize("component", [
    LLMArchitect(OfflineProvider(), max_retries=True),
    LLMArchitect(OfflineProvider(), cost_per_token=float("nan")),
    ConstrainedArchitect(OfflineProvider(), [], max_retries=-1),
    LLMSupervisor(OfflineProvider(), only_terminal="yes"),
])
def test_invalid_inactive_component_policies_fail_preflight(component):
    with pytest.raises(ValueError):
        describe_component(component)


def test_registry_never_invokes_external_hydration_during_preflight():
    class External:
        def list_agent_skills(self, agent_id):
            raise AssertionError("external hydration called")

    source = Registry(skill_provider=External())
    with pytest.raises(WorkflowBindingError):
        bind_component(source, binding())
    static = Registry(skill_provider=External(), hydration_mode=CapabilityHydrationMode.STATIC_ONLY)
    assert not bind_component(static, binding()).list_agents()


def test_registry_rejects_mcp_tools_before_binding():
    from smythe.mcp import MCPServerSpec

    source = Registry()
    source.register(Agent(AgentProfile("Tool user", mcp_servers=[
        MCPServerSpec("x", "http", url="https://example.invalid"),
    ])))
    with pytest.raises(WorkflowBindingError):
        bind_component(source, binding())


def test_local_only_factory_creates_independent_local_architects():
    class Local(DeterministicArchitect):
        def __init__(self):
            self.calls = 0

        def plan(self, task):
            self.calls += 1
            return ExecutionGraph([Topology.SERIAL], [Node(str(self.calls), id="a")]), Registry()

    declared = LocalOnly(Local, "local-planner", "1")
    a, b = bind_component(declared, binding()), bind_component(declared, binding())
    assert a.plan(Task("goal"))[0].nodes[0].label == "1"
    assert b.plan(Task("goal"))[0].nodes[0].label == "1"
    assert describe_component(declared) == {
        "type": "local_only", "role": "architect", "identity": "local-planner", "version": "1",
    }
    assert a.workflow_providers() == ()


def test_provider_backed_builtin_cannot_be_hidden_in_local_only():
    declared = LocalOnly(lambda: LLMArchitect(OfflineProvider()), "bad", "1")
    with pytest.raises(WorkflowBindingError):
        bind_component(declared, binding())


@pytest.mark.asyncio
async def test_local_synthesizer_preserves_custom_signature_without_runtime_kwargs():
    class LocalSynthesis:
        def synthesize(self, graph):
            return graph.nodes[0].result + " local"

    declared = LocalOnly(LocalSynthesis, "local-synthesis", "1", role="synthesizer")
    bound = bind_component(declared, binding(phase="synthesis"), role="synthesizer")
    assert await bound.asynthesize(completed_graph()) == "text local"
    with pytest.raises(WorkflowBindingError):
        await bound.asynthesize(completed_graph(), provider=OfflineProvider())


def test_constrained_template_builders_are_bound_and_repair_scopes_are_explicit():
    def build(task, **params):
        return [Node(params.get("label", task.goal), id="step")], Registry()

    declaration = LocalOnly(lambda: build, "builder", "1", role="template_builder")
    template = SubGraphTemplate("part", "local part", declaration)
    factory = Factory("bad json", '[{"template":"part","params":{"label":"done"}}]')
    original = ConstrainedArchitect(OfflineProvider(), [template], model="test")
    bound = bind_component(original, binding(factory))
    template.name = "mutated"
    graph, _ = bound.plan(Task("goal"))
    assert graph.nodes[0].label == "done" and graph.nodes[0].id == "part-0-step"
    assert [c[1].attempt for c in factory.calls] == [0, 1]


def test_bad_local_template_registry_is_terminal_after_one_selection():
    class ExternalRegistry(Registry):
        pass

    declaration = LocalOnly(lambda: lambda task: ([Node("x")], ExternalRegistry()),
                            "builder", "1", role="template_builder")
    original = ConstrainedArchitect(OfflineProvider(), [SubGraphTemplate("part", "part", declaration)],
                                    model="test")
    factory = Factory('[{"template":"part"}]')
    with pytest.raises(WorkflowBindingError):
        bind_component(original, binding(factory)).plan(Task("goal"))
    assert len(factory.calls) == 1


@pytest.mark.asyncio
async def test_supervision_scope_uses_node_generation_and_copies_review_filter():
    filters, factory = {"a"}, Factory('{"change":false}')
    original = LLMSupervisor(OfflineProvider(), model="test", review_after=filters)
    bound = bind_component(original, binding(factory, phase="supervision", component_id="supervisor"))
    filters.clear()
    graph = completed_graph()
    graph.nodes[0].metadata["execution_generation"] = 2
    await bound.review(graph, graph.nodes[0], task=Task("goal"), revisions_remaining=1)
    [(_, scope, *rest)] = factory.calls
    assert scope.phase == "supervision" and scope.generation == 2
    assert scope.trigger == {"node_id": "a", "generation": 2}


@pytest.mark.parametrize("strategy", list(SynthesisStrategy))
def test_bound_synthesis_rejects_legacy_runtime_override_even_for_local_strategy(strategy):
    original = Synthesizer(strategy, provider=OfflineProvider(), model="test")
    bound = bind_component(original, binding(Factory("merged"), phase="synthesis"))
    with pytest.raises(WorkflowBindingError):
        bound.synthesize(completed_graph(), provider=OfflineProvider())


def test_bound_synthesis_pins_actual_swarm_resolution_and_uses_ledger_facade():
    configured, effective, factory = OfflineProvider(echo_prefix="unused"), OfflineProvider(), Factory("merged")
    original = Synthesizer(SynthesisStrategy.LLM_MERGE, provider=configured, model="configured")
    run_binding = binding(factory, phase="synthesis", provider=effective, model="actual")
    descriptor = describe_component(original, default_provider=effective, default_model="actual")
    bound = bind_component(original, run_binding)
    assert descriptor["completion"]["model"] == "actual"
    assert bound.synthesize(completed_graph()) == "merged"
    [source, scope, _, _, model] = factory.calls[0]
    assert source is run_binding.snapshot_provider(effective)
    assert source is not run_binding.snapshot_provider(configured)
    assert model == "actual" and scope.phase == "synthesis"
    assert not effective.calls and not configured.calls


@pytest.mark.asyncio
async def test_shared_components_bind_to_separate_concurrent_runs():
    original = LLMArchitect(OfflineProvider(), planning_model="test")
    a, b = Factory(PLAN), Factory(PLAN)
    left, right = bind_component(original, binding(a)), bind_component(original, binding(b))
    await asyncio.gather(left.aplan(Task("one")), right.aplan(Task("two")))
    assert len(a.calls) == len(b.calls) == 1
    assert a.calls[0][0] is not b.calls[0][0]
    assert "one" in a.calls[0][3] and "two" in b.calls[0][3]
