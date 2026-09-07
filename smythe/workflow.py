"""Run-scoped orchestration over an exact, durable text-call ledger."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from decimal import Decimal
import hashlib
import json
import math
from uuid import uuid4

from smythe.budget import BudgetValidationError, Sentinel, validate_completion_usage
from smythe.checkpoint import (
    graph_from_dict, graph_to_dict, node_from_dict, node_to_dict, reset_incomplete_nodes,
)
from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, NodeStatus, Revision, snapshot_run_ref
from smythe.registry import Registry
from smythe.provider import ProviderResponseError
from smythe.task import Task, snapshot_task, task_from_dict, task_to_dict
from smythe.tracer import Tracer
from smythe.verifier import (
    TokenVerifier, node_generation, validate_verification_checkpoint, verification_pending,
)
from smythe.workflow_binding import (
    CallScope, ComponentBinding, WorkflowBindingError, bind_component, describe_component,
    json_snapshot,
)
from smythe.workflow_provider import (
    WorkflowProviderContext, describe_provider, validate_workflow_model,
)
from smythe.workflow_policy import snapshot_graph_policy
from smythe.workflow_store import (
    CallKey, SQLiteWorkflowStore, WorkflowConflictError, WorkflowError, WorkflowLeaseError,
    WorkflowStateError,
)


def _canonical(value):
    return json.dumps(json_snapshot(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _nanousd(value):
    # Decimal multiplication must not inherit the caller's rounding context.
    if value is None:
        return None
    amount = Decimal(str(value)).as_tuple()
    coefficient = int("".join(map(str, amount.digits)))
    exponent = amount.exponent + 9
    return coefficient * 10 ** exponent if exponent >= 0 else coefficient // 10 ** -exponent


class WorkflowBudget:
    """Read-only compatibility projection; the journal alone admits and settles."""

    workflow_managed = True
    cost_per_token = 0.0

    def __init__(self, store, run_id, max_budget_usd):
        self.store, self.run_id, self.max_budget_usd = store, run_id, max_budget_usd
        self._node_calls = {}

    def snapshot(self):
        return self.store.inspect_run(self.run_id)

    @property
    def total_cost_usd(self):
        return self.snapshot()["confirmed_nanousd"] / 1_000_000_000

    @property
    def cost_is_complete(self):
        state = self.snapshot()
        return not state["unknown_calls"] and not state["reserved_nanousd"]

    @property
    def cost_contains_estimates(self):
        return False

    def check(self, node_id):
        state = self.store.load_run(self.run_id)
        if state["blocked_reason"]:
            raise WorkflowStateError("Workflow admission is closed: " + state["blocked_reason"])

    def reserve(self, *args, **kwargs):
        raise WorkflowStateError("Only the call journal may reserve workflow funds")

    def release(self, node_id):
        # A node failure cannot release a dispatched call's unknown exposure.
        pass

    def mark_unknown(self, node_id):
        pass  # The provider facade has already persisted the raw failure.

    def add_cost(self, node_id, result, **kwargs):
        validate_completion_usage(result)
        receipt = result.native_receipt or {}
        if (receipt.get("workflow_run_id") != self.run_id
                or receipt.get("workflow_charge_recorded") is not True):
            raise WorkflowStateError("Completion is not settled in this workflow")
        call_id = receipt.get("workflow_call_id")
        records = {call["call_id"]: call for call in self.snapshot()["calls"]}
        if call_id not in records or records[call_id]["billing_state"] != "known":
            raise WorkflowStateError("Completion has no confirmed journal charge")
        self._node_calls.setdefault(node_id, set()).add(call_id)
        return sum(records[key]["cost_nanousd"] for key in self._node_calls[node_id]) / 1e9

    record = add_cost

    def breakdown(self):
        records = {call["call_id"]: call for call in self.snapshot()["calls"]}
        return {node: sum(records[key]["cost_nanousd"] for key in calls) / 1e9
                for node, calls in self._node_calls.items()}


class WorkflowRuntime:
    """One isolated binding per plan/execute/resume entry, one persistent run."""

    def __init__(self, *, store, model, max_budget_usd, provider, architect, registry,
                 router, synthesizer, supervisor, max_revisions, verifier, memory,
                 tool_runtime, checkpoint_store, checkpoint_every_n_nodes, retry_backoff_s,
                 max_concurrency, graph_policy=None):
        if type(store) is not SQLiteWorkflowStore:
            raise WorkflowBindingError("run_store must be SQLiteWorkflowStore")
        Sentinel(max_budget_usd)
        if memory is not None or tool_runtime is not None or checkpoint_store is not None:
            raise WorkflowBindingError(
                "Durable text workflows own their checkpoints and require no live memory or tools"
            )
        if verifier is not None and type(verifier) is not TokenVerifier:
            raise WorkflowBindingError("Durable text workflows require the built-in TokenVerifier")
        if type(max_revisions) is not int or max_revisions < 0:
            raise WorkflowBindingError("max_revisions must be a non-negative integer")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise WorkflowBindingError("Durable workflows require a positive concurrency cap")
        if (type(retry_backoff_s) not in (int, float) or not math.isfinite(retry_backoff_s)
                or retry_backoff_s < 0):
            raise WorkflowBindingError("retry_backoff_s must be finite and non-negative")
        self.store, self.model, self.max_budget_usd = store, model, max_budget_usd
        self.provider = provider
        self.sources = {"architect": architect, "registry": registry, "router": router,
                        "synthesizer": synthesizer, "supervisor": supervisor}
        self.max_revisions, self.verifier = max_revisions, verifier
        self.max_concurrency = max_concurrency
        self.execution_concurrency = None
        self.retry_backoff_s = retry_backoff_s
        self.checkpoint_every_n_nodes = checkpoint_every_n_nodes
        self.graph_policy = snapshot_graph_policy(graph_policy)
        self.recipe = self._describe()
        self.graph = None
        self.executor = None
        self.revision = 0
        self._operation = ContextVar("workflow_operation", default=None)
        self._operation_keys = {}
        self._worker_keys = {}
        self._pending_operations = set()
        self._updates = 0
        self.trace = Tracer()

    def _describe(self):
        validate_workflow_model(self.provider, self.model)
        recipe = {
            "type": "text_workflow", "version": 1, "model": self.model,
            "provider": describe_provider(self.provider),
            "components": {name: describe_component(
                component, role=name, default_provider=self.provider, default_model=self.model,
            ) for name, component in self.sources.items()},
            "max_revisions": self.max_revisions, "retry_backoff_s": self.retry_backoff_s,
            "max_concurrency": self.max_concurrency,
            "verifier": "token-v1", "checkpoint_every_n_nodes": self.checkpoint_every_n_nodes,
        }
        # Omit the field for existing unbounded recipes: adding a null would
        # change their identity and prevent otherwise compatible recovery.
        if self.graph_policy is not None:
            recipe["graph_policy"] = self.graph_policy.to_dict()
        return json_snapshot(recipe)

    def _reference(self):
        return {"version": 1, "store_id": self.store.store_id, "run_id": self.run_id,
                "recipe_sha256": self.run["config_sha256"]}

    def _bind(self):
        binding = ComponentBinding(self.run_id, "workflow", "execution", self._component_call,
                                   default_provider=self.provider, default_model=self.model)
        self.execution_provider = binding.snapshot_provider(self.provider)
        phases = {"architect": "planning", "router": "routing", "registry": "execution",
                  "synthesizer": "synthesis", "supervisor": "supervision"}
        self.bound = {name: bind_component(component, binding.child(name, phase=phases[name]),
                                          role=name)
                      for name, component in self.sources.items()}
        sources = [self.execution_provider]
        for component in self.bound.values():
            if component is not None:
                sources.extend(component.workflow_providers())
        self.providers, self.provider_ids = {}, {}
        for source in sources:
            if id(source) not in self.provider_ids:
                key = f"provider-{len(self.providers)}"
                self.providers[key] = source
                self.provider_ids[id(source)] = key
        self.registry = self.bound["registry"]

    def _component_call(self, source, scope):
        operation_key = _hash(scope.trigger)
        invocation = self.store.allocate_invocation(
            self.context.lease, scope.phase, scope.component_id, scope.generation, operation_key,
        )
        key = CallKey(scope.phase, scope.component_id, scope.generation,
                      invocation, scope.attempt, scope.turn)
        current = self._operation.get()
        if current is not None:
            self._operation_keys.setdefault(current, set()).add(key)
        return self.context.for_call(self.provider_ids[id(source)], key)

    def _worker_call(self, node, phase, attempt, turn):
        scope = CallScope(f"node/{node.id}", phase, {"node_id": node.id},
                          generation=node_generation(node), attempt=attempt, turn=turn)
        result = self._component_call(self.execution_provider, scope)
        self._worker_keys.setdefault((node.id, node_generation(node)), set()).add(result.key)
        return result

    def _call_ids(self, keys):
        ids = []
        for key in keys:
            record = self.store.lookup_call(self.run_id, key)
            if record is not None and record["result_state"] in {"accepted", "rejected", "applied"}:
                ids.append(record["call_id"])
        return sorted(set(ids))

    @asynccontextmanager
    async def _session(self, run):
        self.run, self.run_id = run, run["run_id"]
        if (_canonical(run["config"]) != _canonical(self.recipe)
                or run["budget_nanousd"] != _nanousd(self.max_budget_usd)):
            raise WorkflowConflictError("Workflow components or budget differ from the saved recipe")
        self._bind()
        lease = self.store.acquire_lease(self.run_id, uuid4().hex, ttl_s=60)
        self.context = WorkflowProviderContext(self.store, lease, self.providers)
        self.budget = WorkflowBudget(self.store, self.run_id, self.max_budget_usd)
        self.revision = run["revision"]
        parent = asyncio.current_task()
        heartbeat_error = []

        async def heartbeat():
            try:
                while True:
                    await asyncio.sleep(15)
                    self.context.update_lease(self.store.heartbeat(self.context.lease, ttl_s=60))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                heartbeat_error.append(error)
                parent.cancel()

        pulse = asyncio.create_task(heartbeat())
        try:
            async with self.context:
                self.store.recover(self.context.lease)
                # Late raw evidence can resolve a previously unknown call.
                # Settle every saved envelope before any new admission; decoding
                # remains local to the original logical call's replay.
                for call in self.store.inspect_run(self.run_id)["calls"]:
                    if call["evidence_id"] and call["billing_state"] != "known":
                        self.store.settle_call(self.context.lease, call["call_id"], call["evidence_id"])
                yield
        finally:
            pulse.cancel()
            await asyncio.gather(pulse, return_exceptions=True)
            try:
                self.store.release_lease(self.context.lease)
            except WorkflowLeaseError:
                if not heartbeat_error:
                    raise
            if heartbeat_error:
                raise WorkflowLeaseError("Workflow lease renewal failed") from heartbeat_error[0]

    def _new_run(self, task):
        return self.store.create_run(task_to_dict(task), self.recipe, _nanousd(self.max_budget_usd))

    def _validate_graph_policy(self, graph):
        if self.graph_policy is not None:
            self.graph_policy.validate(graph, default_model=self.model)

    def _validate_graph(self, graph, *, fresh=False):
        graph.validate()
        json_snapshot(graph_to_dict(graph))
        self._validate_graph_policy(graph)
        for node in graph.nodes:
            if fresh and (node.status is not NodeStatus.PENDING or node.result is not None):
                raise WorkflowBindingError("A new workflow requires pending nodes without prior results")
            if fresh and any(key in node.metadata for key in (
                "execution_generation", "regenerations_used", "verification_receipt",
                "regeneration_intent", "workflow_supervision", "native_receipts", "response_error",
                "accounting_invalid", "accounting_error", "workflow_accounting_invalid", "cost_usd",
                "cost_usd_unknown", "cost_usd_is_estimate",
            )):
                raise WorkflowBindingError("A new workflow cannot inherit execution or accounting history")
            for name in ("max_retries", "max_regenerations"):
                value = getattr(node, name)
                if type(value) is not int or value < 0:
                    raise WorkflowBindingError(f"Node {name} must be a non-negative integer")
            if node.timeout_s is not None and (
                type(node.timeout_s) not in (int, float)
                or not math.isfinite(node.timeout_s) or node.timeout_s <= 0
            ):
                raise WorkflowBindingError("Node timeout must be finite and positive")
            if node.attach_dep_artifacts or node.metadata.get("attachments") or node.metadata.get("artifacts"):
                raise WorkflowBindingError("Durable workflows currently support plain text nodes")
            validate_workflow_model(self.execution_provider, node.metadata.get("model", self.model))
            if node.agent_id is not None and self.registry.get(node.agent_id) is None:
                raise WorkflowBindingError(f"Node {node.id!r} names an unknown agent")

    def _prepare_graph(self, graph, *, fresh=False):
        self._validate_graph(graph, fresh=fresh)
        for node in graph.nodes:
            node.metadata.setdefault("model", self.model)
        self.registry.assign(graph)
        self.registry.workflow_description()
        graph.run_ref = self._reference()
        if graph.task is not None:
            graph.task = snapshot_task(graph.task)
            ExecutorBase.stamp_task_context(graph.nodes, graph.task)
        self.graph = graph
        return graph

    def _save(self, status="running", *, output=None):
        if self.graph is None:
            return
        self._validate_graph_policy(self.graph)
        self.registry.workflow_description()
        calls = []
        nodes = {node.id: node for node in self.graph.nodes}
        for node in self.graph.nodes:
            if node.status in {NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED}:
                calls.extend(self._call_ids(self._worker_keys.get((node.id, node_generation(node)), ())))
        # Regeneration can discard an accepted descendant during cancellation
        # after the previous batched snapshot. Consume that settled call with
        # the checkpoint that durably advances its node generation.
        for call in self.store.inspect_run(self.run_id)["calls"]:
            key = call["key"]
            node = nodes.get(key["scope_id"].removeprefix("node/"))
            if (key["phase"] in {"execution", "verification"} and key["scope_id"].startswith("node/")
                    and node is not None and key["generation"] < node_generation(node)
                    and call["billing_state"] == "known"
                    and call["result_state"] in {"accepted", "rejected", "applied"}):
                calls.append(call["call_id"])
        checkpoint = {
            "version": 1, "status": status, "graph": graph_to_dict(self.graph),
            "registry": self.registry.workflow_description(), "output": output,
            "revisions_used": self.executor.revisions_used if self.executor else 0,
            "max_concurrency": self.execution_concurrency,
        }
        saved = self.store.save_checkpoint(
            self.context.lease, self.revision, checkpoint, consumed_call_ids=calls,
            consumed_operation_ids=sorted(self._pending_operations), transition_kind=status,
        )
        self.revision = saved["revision"]
        self._pending_operations.clear()

    def _node_update(self, node):
        if (node.status is NodeStatus.COMPLETED and "response_error" in node.metadata
                and "workflow_accounting_invalid" not in node.metadata):
            self._clear_resolved_native_error(node)
        self._updates += 1
        if self._updates >= self.checkpoint_every_n_nodes:
            self._updates = 0
            self._save()

    def _clear_resolved_native_error(self, node):
        records = {call["call_id"]: call for call in self.store.inspect_run(self.run_id)["calls"]}
        ids = set()
        for entry in node.metadata.get("native_receipts", []):
            receipt = entry.get("receipt", {})
            if receipt.get("workflow_run_id") == self.run_id:
                ids.add(receipt.get("workflow_call_id"))
        ids.update(call["call_id"] for call in records.values()
                   if call["key"]["scope_id"] == f"node/{node.id}"
                   and call["key"]["generation"] == node_generation(node))
        if ids and all(key in records and records[key]["billing_state"] == "known"
                       and records[key]["result_state"] in {"accepted", "applied"} for key in ids):
            for key in ("response_error", "accounting_invalid", "accounting_error", "cost_usd_unknown"):
                node.metadata.pop(key, None)

    async def _operation_result(self, key, kind, inputs, function):
        operation = self.store.load_operation(self.run_id, key)
        if operation is None:
            operation = self.store.begin_operation(self.context.lease, key, kind, inputs)
        if operation["state"] == "started":
            token = self._operation.set(key)
            try:
                result = await function(operation["inputs"])
            finally:
                self._operation.reset(token)
            operation = self.store.complete_operation(
                self.context.lease, key, result,
                consumed_call_ids=self._call_ids(self._operation_keys.get(key, ())),
            )
        self._pending_operations.add(operation["operation_id"])
        return operation["result"]

    async def _plan(self, task):
        async def build(inputs):
            frozen = task_from_dict(inputs["task"])
            architect = (await self.bound["router"].aroute(snapshot_task(frozen))
                         if self.bound["router"] else self.bound["architect"])
            graph, registry = await architect.aplan(snapshot_task(frozen))
            self._validate_graph_policy(graph)
            registry.workflow_description()
            for agent in registry.list_agents():
                self.registry.register(agent)
            graph.task = snapshot_task(frozen)
            self._prepare_graph(graph, fresh=True)
            return {"graph": graph_to_dict(graph), "registry": self.registry.workflow_description()}

        result = await self._operation_result("planning", "planning", {"task": task_to_dict(task)}, build)
        registry = Registry.from_workflow_description(result["registry"])
        graph = graph_from_dict(result["graph"])
        # Completed planning operations replay without invoking build(). Apply
        # the same graph policy before adopting or checkpointing that result.
        self._validate_graph_policy(graph)
        self.registry = registry
        self._prepare_graph(graph, fresh=True)
        self._save("planned")
        return self.graph

    async def plan(self, task):
        task = snapshot_task(task)
        async with self._session(self._new_run(task)):
            return await self._plan(task)

    async def execute(self, value, *, max_concurrency):
        if isinstance(value, Task):
            task = snapshot_task(value)
            async with self._session(self._new_run(task)):
                await self._plan(task)
                return await self._execute(max_concurrency)
        if not isinstance(value, ExecutionGraph):
            raise TypeError("Expected Task or ExecutionGraph")
        # Capture caller-owned input before binding or waiting for a lease.
        for node in value.nodes:
            json_snapshot(node.metadata)
        value = graph_from_dict(json_snapshot(graph_to_dict(value)))
        reference = snapshot_run_ref(value.run_ref)
        if reference is None:
            async with self._session(self._new_run(value.task)):
                self._prepare_graph(value, fresh=True)
                self._save("planned")
                return await self._execute(max_concurrency)
        if reference["store_id"] != self.store.store_id:
            raise WorkflowConflictError("Graph belongs to a different workflow store")
        run = self.store.load_run(reference["run_id"])
        if (reference["recipe_sha256"] != run["config_sha256"]
                or _canonical(task_to_dict(value.task)) != _canonical(run["task"])):
            raise WorkflowConflictError("Graph Task or recipe differs from its durable run")
        async with self._session(run):
            state = self._restore()
            if state["status"] == "planned":
                self._prepare_graph(value, fresh=True)
                self._save("planned")
            elif _canonical(graph_to_dict(value)) != _canonical(graph_to_dict(self.graph)):
                raise WorkflowConflictError("Graph has progressed; resume its execution_id")
            return await self._execute(max_concurrency, state=state)

    def _restore(self):
        saved = self.store.get_checkpoint(self.run_id)
        if saved is None:
            return None
        self.revision = saved["revision"]
        state = saved["checkpoint"]
        if type(state.get("version")) is not int or state["version"] != 1:
            raise WorkflowStateError("Unsupported workflow checkpoint version")
        registry = Registry.from_workflow_description(state["registry"])
        graph = graph_from_dict(state["graph"])
        if (graph.run_ref != self._reference()
                or _canonical(task_to_dict(graph.task)) != _canonical(self.run["task"])):
            raise WorkflowConflictError("Checkpoint graph does not match its durable run")
        self._validate_graph_policy(graph)
        self.registry = registry
        self._prepare_graph(graph)
        return state

    async def resume(self, run_id, *, max_concurrency):
        async with self._session(self.store.load_run(run_id)):
            state = self._restore()
            if state is None:
                task = task_from_dict(self.run["task"])
                if task is None:
                    raise WorkflowStateError("Caller-built workflow has no saved initial graph")
                await self._plan(task)
            return await self._execute(max_concurrency, state=state)

    async def _execute(self, max_concurrency, *, state=None):
        from smythe.async_executor import AsyncExecutor

        # A verdict's forced checkpoint can precede its ordinary node callback.
        # Resolve proven native replay markers before consuming that control.
        for node in self.graph.nodes:
            if (node.status is NodeStatus.COMPLETED and "response_error" in node.metadata
                    and "workflow_accounting_invalid" not in node.metadata):
                self._clear_resolved_native_error(node)
        if state and state["status"] == "completed":
            validate_verification_checkpoint(self.graph, version=3, completed=True)
            if verification_pending(self.graph) or any(
                any(key in node.metadata for key in ("workflow_accounting_invalid", "response_error"))
                or node.metadata.get("accounting_invalid") for node in self.graph.nodes
            ):
                raise WorkflowStateError("Completed workflow has an unresolved control or accounting marker")
            return self._result(state["output"])
        self.execution_concurrency = (state.get("max_concurrency") if state else None) or max_concurrency
        if (type(self.execution_concurrency) is not int
                or not 1 <= self.execution_concurrency <= self.max_concurrency):
            raise WorkflowBindingError("Execution concurrency exceeds the saved workflow policy")
        validate_verification_checkpoint(self.graph, version=3, completed=False)
        reset_incomplete_nodes(self.graph)
        self.executor = AsyncExecutor(
            provider=self.execution_provider, registry=self.registry, tracer=self.trace,
            budget=self.budget, max_concurrency=self.execution_concurrency, artifact_dir=None,
            retry_backoff_s=self.retry_backoff_s, task=self.graph.task,
            verifier=self.verifier, max_revisions=self.max_revisions,
            revisions_used=state["revisions_used"] if state else 0,
            supervisor=_WorkflowSupervisor(self) if self.bound["supervisor"] else None,
            provider_call_factory=self._worker_call, on_node_update=self._node_update,
            on_control_update=self._save, on_supervision_update=self._supervision_update,
        )
        self._save()
        try:
            # A crash can follow node acceptance but precede its review.
            self.executor.prepare_execution(self.graph)
            self.executor.recover_verification(self.graph)
            for node in tuple(self.graph.nodes):
                disposition = node.metadata.get("workflow_supervision", {})
                if (node.status is NodeStatus.COMPLETED
                        and (disposition.get("generation") != node_generation(node)
                             or disposition.get("state") != "applied")):
                    await self.executor.maybe_revise(node, self.graph)
            await self.executor.run(self.graph)

            async def synthesize(inputs):
                graph = graph_from_dict(inputs["graph"])
                output = await self.bound["synthesizer"].asynthesize(graph)
                if type(output) is not str:
                    raise WorkflowStateError("Synthesis must return text")
                return {"output": output}

            final = await self._operation_result(
                "synthesis", "synthesis", {"graph": graph_to_dict(self.graph)}, synthesize,
            )
            self._save("completed", output=final["output"])
            return self._result(final["output"])
        except BaseException:
            self._save("failed")
            raise

    def _supervision_update(self, node, applied):
        disposition = node.metadata.get("workflow_supervision")
        if disposition is None:
            return
        operation = self.store.load_operation(self.run_id, disposition["operation_key"])
        if operation is None or operation["state"] == "started":
            raise WorkflowStateError("Supervisor disposition has no durable result")
        disposition.update(state="applied", revision_applied=applied)
        self._pending_operations.add(operation["operation_id"])
        # Added agent assignments and model validation precede the checkpoint.
        self._prepare_graph(self.graph)
        self._save()

    def _result(self, output):
        from smythe.swarm import SwarmResult

        accounting = self.budget.snapshot()
        return SwarmResult(
            output=output, graph=self.graph, trace=self.trace.summary(), execution_id=self.run_id,
            total_cost_usd=accounting["confirmed_nanousd"] / 1e9,
            cost_is_complete=not accounting["unknown_calls"] and not accounting["reserved_nanousd"],
            cost_contains_estimates=False, cost_scope="complete_text_workflow",
            workflow_accounting={key: accounting[key] for key in (
                "confirmed_nanousd", "reserved_nanousd", "unknown_nanousd", "unknown_calls", "call_count",
            )},
        )


class _WorkflowSupervisor:
    def __init__(self, runtime):
        self.runtime = runtime

    async def review(self, graph, node, *, task, revisions_remaining):
        runtime = self.runtime
        key = f"supervision/{node.id}/{node_generation(node)}"
        node.metadata["workflow_supervision"] = {
            "operation_key": key, "generation": node_generation(node), "state": "pending",
        }
        operation = runtime.store.load_operation(runtime.run_id, key)
        if operation is None:
            runtime.store.begin_operation(runtime.context.lease, key, "supervision", {
                "graph": graph_to_dict(graph), "node_id": node.id, "task": task_to_dict(task),
                "revisions_remaining": revisions_remaining,
            })
        runtime._save()

        async def review(inputs):
            frozen = graph_from_dict(inputs["graph"])
            frozen_node = next(item for item in frozen.nodes if item.id == inputs["node_id"])
            try:
                revision = await runtime.bound["supervisor"].review(
                    frozen, frozen_node, task=task_from_dict(inputs["task"]),
                    revisions_remaining=inputs["revisions_remaining"],
                )
            except BaseExceptionGroup as error:
                pending = list(error.exceptions)
                while pending:
                    nested = pending.pop()
                    if isinstance(nested, BaseExceptionGroup):
                        pending.extend(nested.exceptions)
                    elif (not isinstance(nested, Exception)
                          or isinstance(nested, (WorkflowError, ProviderResponseError, BudgetValidationError))):
                        raise
                return {"revision": None, "local_error_type": type(error).__name__}
            except (WorkflowError, ProviderResponseError, BudgetValidationError):
                raise
            except Exception as error:
                # Match the legacy no-change disposition for local review
                # failures, but persist it so recovery never repeats the review.
                return {"revision": None, "local_error_type": type(error).__name__}
            if revision is None:
                return {"revision": None}
            if not isinstance(revision, Revision):
                raise WorkflowStateError("Supervisor must return Revision or None")
            return {"revision": {
                "add_nodes": [node_to_dict(item) for item in revision.add_nodes],
                "drop_node_ids": list(revision.drop_node_ids),
                "rewire": {key: list(value) for key, value in revision.rewire.items()},
                "reason": revision.reason,
            }}

        result = await runtime._operation_result(key, "supervision", {}, review)
        revision = result["revision"]
        if revision is None:
            return None
        proposal = Revision(
            add_nodes=tuple(node_from_dict(item) for item in revision["add_nodes"]),
            drop_node_ids=tuple(revision["drop_node_ids"]),
            rewire={key: tuple(value) for key, value in revision["rewire"].items()},
            reason=revision["reason"],
        )
        # A checkpoint can lag an already admitted worker. Resetting RUNNING
        # to PENDING on recovery must not make that work rewritable history.
        targets = set(proposal.drop_node_ids) | set(proposal.rewire)
        generations = {item.id: node_generation(item) for item in graph.nodes if item.id in targets}
        for call in runtime.store.inspect_run(runtime.run_id)["calls"]:
            call_key = call["key"]
            target = call_key["scope_id"].removeprefix("node/")
            if (call_key["phase"] in {"execution", "verification"}
                    and call_key["scope_id"].startswith("node/") and target in generations
                    and call_key["generation"] == generations[target]):
                runtime.trace.on_revision(
                    node, proposal, applied=False,
                    detail="Revision targets work already admitted in the durable journal",
                )
                return None
        if runtime.graph_policy is not None:
            try:
                candidate = graph_from_dict(json_snapshot(graph_to_dict(graph)))
                additions = tuple(node_from_dict(json_snapshot(node_to_dict(item)))
                                  for item in proposal.add_nodes)
                candidate.apply_revision(Revision(
                    add_nodes=additions, drop_node_ids=proposal.drop_node_ids,
                    rewire=proposal.rewire, reason=proposal.reason,
                ))
                # Validate the exact context the executor will give new nodes,
                # without changing the live graph or the returned proposal.
                runtime.executor._inherit_execution_context(additions, candidate)
                runtime._validate_graph(candidate)
            except (ValueError, TypeError, KeyError) as error:
                runtime.trace.on_revision(node, proposal, applied=False, detail=str(error))
                return None
        return proposal
