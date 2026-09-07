"""Explicit, isolated component bindings for journaled text workflows.

Descriptions and binding perform local validation only. The orchestrator owns
durable invocation allocation; components supply stable triggers and attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import inspect
import json
from typing import Any, Callable

from smythe.provider import Provider


class WorkflowBindingError(ValueError):
    """A component cannot satisfy the strict text-workflow binding contract."""


def json_snapshot(value: Any) -> Any:
    def check(item, active):
        if item is None or type(item) in (str, bool, int, float):
            return
        if type(item) not in (list, dict) or id(item) in active:
            raise WorkflowBindingError("Workflow configuration must be acyclic JSON data")
        active.add(id(item))
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise WorkflowBindingError("Workflow configuration keys must be strings")
            children = item.values()
        else:
            children = item
        for child in children:
            check(child, active)
        active.remove(id(item))

    check(value, set())
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise WorkflowBindingError("Workflow configuration must be finite JSON data") from exc


def require_exact(component: object, expected: type) -> None:
    if type(component) is not expected:
        raise WorkflowBindingError(
            f"Custom {type(component).__name__} requires an explicit LocalOnly declaration"
        )


def provider_description(provider: Provider, model: str | None = None) -> dict:
    from smythe.workflow_provider import describe_provider, validate_workflow_model

    description = describe_provider(provider)
    if model is not None:
        validate_workflow_model(provider, model)
    return {"provider": description, "model": model}


def provider_snapshot(provider: Provider) -> Provider:
    from smythe.workflow_provider import snapshot_provider

    return snapshot_provider(provider)


@dataclass(frozen=True, init=False)
class CallScope:
    component_id: str
    phase: str
    _trigger_json: str = field(repr=False)
    generation: int = 0
    attempt: int = 0
    turn: int = 0

    def __init__(self, component_id, phase, trigger, generation=0, attempt=0, turn=0):
        if not isinstance(component_id, str) or not component_id or phase not in {
            "routing", "planning", "execution", "verification", "supervision", "synthesis",
        }:
            raise WorkflowBindingError("Call scope requires a component ID and supported phase")
        for name, value in (("generation", generation), ("attempt", attempt), ("turn", turn)):
            if type(value) is not int or value < 0:
                raise WorkflowBindingError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "_trigger_json", json.dumps(
            json_snapshot(trigger), sort_keys=True, separators=(",", ":"),
        ))

    @property
    def trigger(self):
        return json.loads(self._trigger_json)

    def to_dict(self):
        return {"component_id": self.component_id, "phase": self.phase, "trigger": self.trigger,
                "generation": self.generation, "attempt": self.attempt, "turn": self.turn}


@dataclass(frozen=True)
class ComponentBinding:
    run_id: str
    component_id: str
    phase: str
    call_factory: Callable[[Provider, CallScope], Provider] = field(repr=False, compare=False)
    default_provider: Provider | None = field(default=None, repr=False, compare=False)
    default_model: str = ""
    _provider_snapshots: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not self.run_id:
            raise WorkflowBindingError("Binding requires a run ID")
        if not callable(self.call_factory):
            raise WorkflowBindingError("Binding requires a call factory")
        CallScope(self.component_id, self.phase, None)

    def child(self, name: str, *, phase: str | None = None) -> ComponentBinding:
        if not isinstance(name, str) or not name:
            raise WorkflowBindingError("Child binding requires a name")
        return replace(self, component_id=f"{self.component_id}/{name}", phase=phase or self.phase)

    def snapshot_provider(self, source: Provider) -> Provider:
        saved = self._provider_snapshots.get(id(source))
        if saved is None:
            saved = (source, provider_snapshot(source))
            self._provider_snapshots[id(source)] = saved
        return saved[1]

    def for_call(
        self, source_provider: Provider, *, trigger: Any,
        attempt: int = 0, turn: int = 0, generation: int = 0,
    ) -> Provider:
        return self.call_factory(source_provider, CallScope(
            self.component_id, self.phase, trigger, generation, attempt, turn,
        ))


@dataclass(frozen=True)
class LocalOnly:
    """Declare a fresh local component with no provider calls or side effects.

    This is a caller contract, not a sandbox. ``factory`` must return a fresh
    instance for each binding. Stable identity/version are persisted in recipes.
    """

    factory: Callable[[], object] = field(repr=False, compare=False)
    identity: str
    version: str
    role: str = "architect"

    def __post_init__(self):
        if not callable(self.factory) or any(
            not isinstance(value, str) or not value.strip() for value in (self.identity, self.version)
        ):
            raise WorkflowBindingError("LocalOnly requires a factory, identity, and version")
        if self.role not in {"architect", "supervisor", "synthesizer", "template_builder"}:
            raise WorkflowBindingError("Unsupported LocalOnly component role")

    def workflow_description(self, **defaults) -> dict:
        require_exact(self, LocalOnly)
        return {"type": "local_only", "role": self.role,
                "identity": self.identity, "version": self.version}

    def workflow_providers(self) -> tuple[Provider, ...]:
        return ()

    def bind_run(self, binding: ComponentBinding):
        self.workflow_description()
        component = self.factory()
        if component is self or isinstance(component, Provider):
            raise WorkflowBindingError("LocalOnly factory must produce a local component")
        if type(component) in _supported_types() and component.workflow_providers():
            raise WorkflowBindingError("Provider-backed built-ins cannot be declared LocalOnly")
        required = {"architect": "plan", "supervisor": "review",
                    "synthesizer": "synthesize", "template_builder": "__call__"}[self.role]
        if not callable(getattr(component, required, None)):
            raise WorkflowBindingError(f"LocalOnly {self.role} requires {required}()")
        return _BoundLocal(self, component, binding)


class _BoundLocal:
    def __init__(self, declaration, component, binding):
        self.declaration = declaration
        self.component = component
        self.binding = binding

    def workflow_description(self, **defaults):
        return self.declaration.workflow_description()

    def workflow_providers(self):
        return ()

    def plan(self, task):
        return self.component.plan(task)

    async def aplan(self, task):
        method = getattr(self.component, "aplan", None)
        return await method(task) if method is not None else self.component.plan(task)

    async def review(self, graph, node, *, task, revisions_remaining):
        result = self.component.review(
            graph, node, task=task, revisions_remaining=revisions_remaining,
        )
        return await result if inspect.isawaitable(result) else result

    def synthesize(self, graph, **kwargs):
        if any(value is not None for value in kwargs.values()):
            raise WorkflowBindingError("LocalOnly synthesis does not accept provider overrides")
        return self.component.synthesize(graph)

    async def asynthesize(self, graph, **kwargs):
        if any(value is not None for value in kwargs.values()):
            raise WorkflowBindingError("LocalOnly synthesis does not accept provider overrides")
        method = getattr(self.component, "asynthesize", None)
        return await method(graph) if method is not None else self.component.synthesize(graph)

    def __call__(self, *args, **kwargs):
        return self.component(*args, **kwargs)


def _supported_types():
    from smythe.constrained_planner import ConstrainedArchitect
    from smythe.planner import LLMArchitect, SimpleArchitect
    from smythe.registry import Registry
    from smythe.router import WhiteRabbit
    from smythe.supervisor import LLMSupervisor
    from smythe.synthesizer import Synthesizer

    return (LLMArchitect, SimpleArchitect, ConstrainedArchitect, Registry,
            WhiteRabbit, LLMSupervisor, Synthesizer, LocalOnly, _BoundLocal)


def describe_component(component, *, role=None, default_provider=None, default_model="") -> dict:
    if component is None:
        return {"type": "none"}
    if type(component) not in _supported_types():
        raise WorkflowBindingError(
            f"Unsupported workflow component {type(component).__name__}; use LocalOnly for local code"
        )
    description = json_snapshot(component.workflow_description(
        default_provider=default_provider, default_model=default_model,
    ))
    roles = {
        "architect": {"llm_architect", "simple_architect", "constrained_architect"},
        "router": {"white_rabbit"}, "registry": {"registry"},
        "supervisor": {"llm_supervisor"}, "synthesizer": {"synthesizer"},
        "template_builder": set(),
    }
    if role is not None and (
        role not in roles or not (
            description["type"] in roles[role]
            or description["type"] == "local_only" and description["role"] == role
        )
    ):
        raise WorkflowBindingError(f"Component cannot serve workflow role {role!r}")
    return description


def bind_component(component, binding: ComponentBinding, *, role=None):
    if component is None:
        return None
    describe_component(component, role=role, default_provider=binding.default_provider,
                       default_model=binding.default_model)
    return component.bind_run(binding)
