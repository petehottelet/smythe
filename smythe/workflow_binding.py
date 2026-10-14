"""Explicit, isolated component bindings for journaled text workflows.

Descriptions and binding perform local validation only. The orchestrator owns
durable invocation allocation; components supply stable triggers and attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import inspect
import json
from typing import TYPE_CHECKING, Any, Callable

from smythe.provider import Provider

if TYPE_CHECKING:
    from smythe.graph import ExecutionGraph
    from smythe.registry import Registry


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
        # The journal's call phases (workflow_store._PHASES).
        if not isinstance(component_id, str) or not component_id or phase not in {
            "routing", "planning", "execution", "verification", "supervision", "synthesis", "tool",
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
    # Set by the durable runtime. Planners call it on each candidate plan inside
    # their retry loop, so a plan the run would reject is repaired, not final.
    plan_check: Callable[[ExecutionGraph, Registry], None] | None = field(
        default=None, repr=False, compare=False,
    )
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

    With ``role="validator"`` the factory builds a validator: an object with a
    :class:`~smythe.validation.ValidatorIdentity` whose name and version are
    ``identity`` and ``version``, and a synchronous ``validate(subject)``. A
    ``Provider``, a coroutine function or an awaitable is refused as the
    factory and as what it builds, as is an ``async`` ``validate``. The
    recipe records the identity, including its ``config_sha256``.

    With ``role="renderer"`` the factory builds a deterministic document
    renderer for a durable run's published output: a synchronous
    ``render(output, task)`` that returns text, with an optional
    ``filename`` and ``media_type`` (see
    :func:`smythe.workflow_artifact.renderer_settings`).
    """

    factory: Callable[[], object] = field(repr=False, compare=False)
    identity: str
    version: str
    role: str = "architect"

    def __post_init__(self):
        if self.role == "validator":
            _check_validator_factory(self.factory)
        if not callable(self.factory) or any(
            not isinstance(value, str) or not value.strip() for value in (self.identity, self.version)
        ):
            raise WorkflowBindingError("LocalOnly requires a factory, identity, and version")
        if self.role not in {"architect", "supervisor", "synthesizer", "template_builder",
                             "validator", "renderer"}:
            raise WorkflowBindingError("Unsupported LocalOnly component role")

    def workflow_description(self, **defaults) -> dict:
        require_exact(self, LocalOnly)
        description = {"type": "local_only", "role": self.role,
                       "identity": self.identity, "version": self.version}
        if self.role == "validator":
            description["config_sha256"] = self.validator().identity.config_sha256
        elif self.role == "renderer":
            self.renderer()  # Refuse an unusable renderer when it is declared.
        return description

    def workflow_providers(self) -> tuple[Provider, ...]:
        return ()

    def validator(self):
        """Build a fresh validator and check it against this declaration."""
        from smythe.acceptance import check_validator

        if self.role != "validator":
            raise WorkflowBindingError(f"LocalOnly {self.role} is not a validator")
        require_exact(self, LocalOnly)
        component = self.factory()
        if inspect.isawaitable(component):
            if inspect.iscoroutine(component):
                component.close()
            raise WorkflowBindingError("LocalOnly validator factory returned an awaitable")
        if component is self or isinstance(component, Provider):
            raise WorkflowBindingError("LocalOnly factory must produce a local component")
        try:
            identity = check_validator(component)
        except TypeError as error:
            raise WorkflowBindingError(f"LocalOnly validator {self.identity!r}: {error}") from error
        if (identity.name, identity.version) != (self.identity, self.version):
            raise WorkflowBindingError(
                f"LocalOnly validator {self.identity!r} version {self.version!r} built "
                f"{identity.name!r} version {identity.version!r}"
            )
        return component

    def renderer(self):
        """Build a fresh renderer and check it against the renderer contract."""
        from smythe.workflow_artifact import renderer_settings

        if self.role != "renderer":
            raise WorkflowBindingError(f"LocalOnly {self.role} is not a renderer")
        require_exact(self, LocalOnly)
        component = self.factory()
        if inspect.isawaitable(component):
            if inspect.iscoroutine(component):
                component.close()
            raise WorkflowBindingError("LocalOnly renderer factory returned an awaitable")
        if component is self or isinstance(component, Provider):
            raise WorkflowBindingError("LocalOnly factory must produce a local component")
        render = getattr(component, "render", None)
        if not callable(render):
            raise WorkflowBindingError("LocalOnly renderer requires render()")
        if inspect.iscoroutinefunction(render) or inspect.isasyncgenfunction(render):
            raise WorkflowBindingError("LocalOnly renderer render() must be synchronous")
        renderer_settings(component)
        return component

    def bind_run(self, binding: ComponentBinding):
        if self.role == "validator":
            return self.validator()
        if self.role == "renderer":
            return _BoundLocal(self, self.renderer(), binding)
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


def _check_validator_factory(factory):
    if isinstance(factory, Provider):
        raise WorkflowBindingError("A Provider cannot be a LocalOnly validator factory")
    if inspect.isawaitable(factory):
        if inspect.iscoroutine(factory):
            factory.close()
        raise WorkflowBindingError("An awaitable cannot be a LocalOnly validator factory")
    if any(inspect.iscoroutinefunction(item) or inspect.isasyncgenfunction(item)
           for item in (factory, getattr(type(factory), "__call__", None))):
        raise WorkflowBindingError("A LocalOnly validator factory must not be a coroutine function")


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

    def render(self, output, task):
        return self.component.render(output, task)

    def __call__(self, *args, **kwargs):
        return self.component(*args, **kwargs)


def local_component(component: object) -> object:
    """The component a ``LocalOnly`` declaration built when a run bound it.

    Binding a declaration wraps the component its factory builds, so read
    the component's own attributes, such as a template builder's
    ``template_version``, from what this returns. Any other *component* is
    returned unchanged.
    """
    return component.component if isinstance(component, _BoundLocal) else component


def _supported_types():
    from smythe.constrained_planner import ConstrainedArchitect
    from smythe.planner import LLMArchitect, SimpleArchitect
    from smythe.registry import Registry
    from smythe.router import WhiteRabbit
    from smythe.supervisor import EvidenceTriggeredSupervisor, LLMSupervisor
    from smythe.synthesizer import Synthesizer
    from smythe.workflow_artifact import MarkdownRenderer

    return (LLMArchitect, SimpleArchitect, ConstrainedArchitect, Registry,
            WhiteRabbit, LLMSupervisor, EvidenceTriggeredSupervisor, Synthesizer,
            MarkdownRenderer, LocalOnly, _BoundLocal)


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
        "supervisor": {"llm_supervisor", "evidence_triggered_supervisor"},
        "synthesizer": {"synthesizer"},
        "template_builder": set(), "validator": set(), "renderer": {"markdown_renderer"},
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
