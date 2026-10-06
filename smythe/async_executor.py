"""AsyncExecutor — concurrent DAG execution via asyncio."""

from __future__ import annotations

import asyncio
import inspect
from bisect import bisect_right
from collections import deque
from itertools import islice, zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from smythe._failures import SECURITY_STOP, classify_failure, is_terminal
from smythe._redact import error_text
from smythe.budget import BudgetValidationError, Sentinel, validate_token_count
from smythe.executor_base import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    PROMPT_ASSEMBLY_VERSION,
    ExecutorBase,
    NodeFinalizationError,
    validate_max_concurrency,
)
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus
from smythe.provider import Provider
from smythe.registry import Registry
from smythe.scheduling import RetryDeadlineError, TerminationPolicy
from smythe.tools import ToolRuntime
from smythe.tracer import Tracer
from smythe.verifier import VerificationRecoveryError

if TYPE_CHECKING:
    from smythe.acceptance import AcceptancePolicy
    from smythe.context import ContextSnapshot
    from smythe.scheduling import CapacityPolicy, ModelProfile, RetryPolicy
    from smythe.supervisor import Supervisor
    from smythe.task import Task
    from smythe.verifier import Verifier


def _branch(node: Node, order: dict[str, int]) -> int:
    """The graph position of *node*'s first dependency, or -1 for a node without one."""
    return order.get(node.depends_on[0], -1) if node.depends_on else -1


def _in_turn(ready: deque[Node], order: dict[str, int], last: int | None) -> deque[Node]:
    """*ready* ordered by turns across branches, from the branch after *last*.

    A branch is the ready nodes that share a first dependency (see
    ``_branch``), in their own order; each turn takes one node per branch.
    """
    branches: dict[int, list[Node]] = {}
    for node in ready:
        branches.setdefault(_branch(node, order), []).append(node)
    keys = sorted(branches)
    first = 0 if last is None else bisect_right(keys, last)
    turns = zip_longest(*(branches[key] for key in keys[first:] + keys[:first]))
    return deque(node for turn in turns for node in turn if node is not None)


class AsyncExecutor(ExecutorBase):
    """Executes an assigned graph with maximum concurrency.

    Independent nodes (those whose dependencies are all completed)
    are launched as capacity becomes available.  Only the configured
    number of tasks is admitted to the event loop at once; a broadcast
    with thousands of nodes therefore does not create thousands of
    coroutines that merely wait behind a semaphore.

    Budget safety: estimated cost is reserved before each node is admitted,
    so concurrent admissions cannot collectively exceed the configured policy.
    Reservations are reconciled with reported cost or configured estimates on
    completion, or released on failure/cancellation; under a retry policy
    with ``retain_uncertain_dispatch``, a sent call that never answered keeps
    its reservation as unknown cost instead.

    Concurrency safety: ``max_concurrency`` caps in-flight provider
    calls, so a wide broadcast doesn't fire every call
    at once and trip provider rate limits.  None means unlimited.
    Under a ``capacity_policy``, at most ``max_in_flight`` nodes of each
    provider type and model run at once: a ready node whose model has no
    free slot is passed over, still ready, for work behind it, and never
    stops the run. With ``round_robin``, ready nodes take turns by branch.

    Termination: a failure that stops the run, or a node refused at
    admission, cancels the nodes in flight at once. Under ``termination``, a
    :class:`~smythe.scheduling.TerminationPolicy`, the executor first stops
    admission and lets their calls in flight finish for up to
    ``drain_timeout_s``, starting no retry, then cancels the rest and raises
    the original error; :attr:`drain_record` reports how the drain ended.
    ``sleep``, when given, also times the drain's deadline.
    """

    DEFAULT_ESTIMATED_TOKENS = 2000

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        tracer: Tracer,
        budget: Sentinel | None = None,
        estimated_tokens_per_node: int = DEFAULT_ESTIMATED_TOKENS,
        max_concurrency: int | None = None,
        on_node_update: Callable[[Node], None] | None = None,
        tool_runtime: ToolRuntime | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        artifact_dir: str | Path | None = "smythe_artifacts",
        retry_backoff_s: float = 0.0,
        supervisor: Supervisor | None = None,
        max_revisions: int = 0,
        task: Task | None = None,
        verifier: Verifier | None = None,
        revisions_used: int = 0,
        on_control_update: Callable[[], None] | None = None,
        provider_call_factory: Callable[[Node, str, int, int], Provider] | None = None,
        on_supervision_update: Callable[[Node, bool], None] | None = None,
        *,
        prompt_assembly: int = PROMPT_ASSEMBLY_VERSION,
        read_call_factory: Callable[[Node, int], Any] | None = None,
        context_assembler: Callable[
            [Node, ExecutionGraph, dict[str, Any]], Awaitable[ContextSnapshot]
        ] | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], Any] | None = None,
        rng: Any = None,
        attempts_used: dict[str, dict[str, int]] | None = None,
        attempts_spent: Callable[[Node], int] | None = None,
        profiles: tuple[ModelProfile, ...] | list[ModelProfile] = (),
        default_model: str | None = None,
        capacity_policy: CapacityPolicy | None = None,
        termination: TerminationPolicy | None = None,
        acceptance: AcceptancePolicy | None = None,
    ) -> None:
        super().__init__(
            provider=provider, registry=registry, tracer=tracer, budget=budget,
            on_node_update=on_node_update, tool_runtime=tool_runtime,
            max_tool_iterations=max_tool_iterations, artifact_dir=artifact_dir,
            retry_backoff_s=retry_backoff_s, supervisor=supervisor,
            max_revisions=max_revisions, task=task, verifier=verifier,
            revisions_used=revisions_used,
            on_control_update=on_control_update,
            provider_call_factory=provider_call_factory,
            on_supervision_update=on_supervision_update,
            prompt_assembly=prompt_assembly,
            read_call_factory=read_call_factory,
            acceptance=acceptance,
            context_assembler=context_assembler,
            retry_policy=retry_policy,
            sleep=sleep,
            rng=rng,
            attempts_used=attempts_used,
            attempts_spent=attempts_spent,
            profiles=profiles,
            default_model=default_model,
            capacity_policy=capacity_policy,
        )
        self._estimated_tokens_per_node = validate_token_count(
            estimated_tokens_per_node, "estimated_tokens_per_node",
        )
        self._max_concurrency = validate_max_concurrency(max_concurrency)
        if termination is not None and not isinstance(termination, TerminationPolicy):
            raise TypeError(
                f"termination must be a TerminationPolicy or None, not {type(termination).__name__}"
            )
        self._termination = termination
        # The last run's drain: settled and cancelled node ids, and whether
        # its deadline ended it.
        self._drained: tuple[tuple[str, ...], tuple[str, ...], bool] | None = None
        # Whether the current run is draining, so no node starts a retry, and
        # the nodes waiting out a retry backoff, which have no call in flight.
        self._draining = False
        self._backing_off: set[str] = set()

    @property
    def drain_record(self) -> dict[str, Any] | None:
        """How the last run drained after a failure, or None when it did not drain.

        ``settled`` names the nodes in flight whose work ended during the
        drain, ``cancelled`` those it stopped, still in flight when it ended
        or before a retry, and ``deadline_hit`` whether ``drain_timeout_s``
        ended it. A Swarm's checkpoints save it as ``control["drain"]``.
        """
        if self._drained is None:
            return None
        settled, cancelled, deadline_hit = self._drained
        return {"settled": list(settled), "cancelled": list(cancelled),
                "deadline_hit": deadline_hit}

    def _schedule_state(
        self, graph: ExecutionGraph, active: dict[asyncio.Task[None], Node],
    ) -> tuple[
        dict[str, Node], set[str], dict[str, list[Node]], dict[str, int],
        deque[Node], dict[str, int],
    ]:
        """Derive scheduling state from the graph's live node statuses.

        Also used to rebuild after a supervisor revision: node status is
        the authority, so recomputing from it is both simpler and safer
        than patching the derived structures node by node.  Nodes already
        in flight are excluded — they are running, not schedulable.
        """
        active_ids = {node.id for node in active.values()}
        pending = {
            node.id: node
            for node in graph.nodes
            if node.status == NodeStatus.PENDING and node.id not in active_ids
        }
        resolved = {
            node.id for node in graph.nodes
            if node.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
        }
        dependents: dict[str, list[Node]] = {node.id: [] for node in graph.nodes}
        unresolved: dict[str, int] = {}
        for node in graph.nodes:
            for dep_id in node.depends_on:
                dependents.setdefault(dep_id, []).append(node)
            if node.id in pending:
                unresolved[node.id] = sum(
                    dep_id not in resolved for dep_id in node.depends_on
                )
        ready = deque(
            node for node in graph.nodes
            if node.id in pending and unresolved[node.id] == 0
        )
        order = {node.id: i for i, node in enumerate(graph.nodes)}
        return pending, resolved, dependents, unresolved, ready, order

    async def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Execute every node, fanning out independent nodes concurrently."""
        self._drained, self._draining = None, False
        self._backing_off.clear()
        self.prepare_execution(graph)
        with self._tracer.phase("verification"):
            self.recover_verification(graph)
        active: dict[asyncio.Task[None], Node] = {}
        pending, resolved, dependents, unresolved, ready, order = self._schedule_state(
            graph, active,
        )
        if not pending:
            return graph

        # None intentionally retains the documented unlimited mode.  With a
        # concrete cap, the active task set can never grow beyond that cap.
        concurrency = self._max_concurrency or max(1, len(pending))
        # When each ready node became ready, and how long supervisor reviews
        # then held it, for its admission wait.
        ready_at = dict.fromkeys((node.id for node in ready), self._tracer.now())
        held_s: dict[str, float] = {}
        # Under a capacity policy: when a full limit first passed over each
        # ready node, and the branch last admitted, for round-robin turns.
        deferred_at: dict[str, float] = {}
        last_branch: int | None = None

        try:
            while ready or active:
                admission_error: Exception | None = None
                capacity = self._capacity
                if capacity is not None and capacity.policy.round_robin:
                    ready = _in_turn(ready, order, last_branch)
                passed: list[Node] = []
                while ready and len(active) < concurrency:
                    node = ready.popleft()
                    bucket = None
                    if capacity is not None:
                        provider, model, _ = self._node_route(node)
                        bucket = capacity.bucket(provider, str(model))
                    if bucket is not None and not capacity.try_acquire(bucket):
                        # Its provider and model have no free slot: it stays
                        # ready, and work behind it may start instead.
                        passed.append(node)
                        deferred_at.setdefault(node.id, self._tracer.now())
                        continue
                    try:
                        self._reserve_node(node)
                    except Exception as exc:
                        # No provider call has started for this node. Stop
                        # admitting work, settle/cancel what is already in
                        # flight, then surface the admission failure.
                        if bucket is not None:
                            capacity.release(bucket)
                        if isinstance(exc, BudgetValidationError):
                            self.mark_accounting_invalid(node, exc)
                            self.notify_update(node)
                        ready.appendleft(node)
                        admission_error = exc
                        break
                    # Review time is a supervision wait, not part of this one.
                    # since() keeps a rounding error from going below zero.
                    waited = self._tracer.since(
                        ready_at.pop(node.id, self._tracer.now()) + held_s.pop(node.id, 0.0),
                    )
                    # Time since a full limit passed it over waited for capacity.
                    blocked = (min(waited, self._tracer.since(deferred_at.pop(node.id)))
                               if node.id in deferred_at else 0.0)
                    self._tracer.on_wait("admission", waited - blocked, node)
                    if blocked:
                        self._tracer.on_wait("capacity", blocked, node)
                    task = asyncio.create_task(
                        self._execute_node(node, graph),
                        name=f"smythe-node-{node.id}",
                    )
                    if bucket is not None:
                        # However the task ends, even cancelled before it starts.
                        task.add_done_callback(lambda _, bucket=bucket: capacity.release(bucket))
                        last_branch = _branch(node, order)
                    active[task] = node
                ready.extendleft(reversed(passed))

                if admission_error is not None:
                    await self._stop_after(admission_error, active)
                    raise admission_error
                if not active:
                    break

                done, _ = await asyncio.wait(
                    active, return_when=asyncio.FIRST_COMPLETED,
                )
                first_error: Exception | None = None
                just_completed: list[Node] = []
                for task in sorted(done, key=lambda item: order[active[item].id]):
                    node = active.pop(task)
                    try:
                        task.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                        continue

                    pending.pop(node.id, None)
                    resolved.add(node.id)
                    just_completed.append(node)
                    for child in dependents.get(node.id, []):
                        if child.id not in pending:
                            continue
                        unresolved[child.id] -= 1
                        if unresolved[child.id] == 0:
                            ready.append(child)
                            ready_at.setdefault(child.id, self._tracer.now())

                if first_error is None and just_completed:
                    revised = False
                    for node in just_completed:
                        with self._tracer.phase("verification"):
                            revised |= await self._regenerate_and_settle(node, graph, active)
                        if self._supervisor is not None:
                            revised |= await self._supervise(
                                node, graph, ready, held_s, concurrency - len(active),
                            )
                    if revised:
                        (
                            pending, resolved, dependents, unresolved, ready, order,
                        ) = self._schedule_state(graph, active)
                        # A node the change made unready again was blocked on
                        # new work, not on a slot: its wait restarts when it
                        # is next ready. Nodes still ready keep theirs.
                        ready_ids = {waiting.id for waiting in ready}
                        for stale in (ready_at.keys() | held_s.keys()) - ready_ids:
                            ready_at.pop(stale, None)
                            held_s.pop(stale, None)
                            deferred_at.pop(stale, None)
                        now = self._tracer.now()
                        for waiting in ready:
                            ready_at.setdefault(waiting.id, now)

                if first_error is not None:
                    # HALT means no queued sibling may start after the error
                    # is observed. Cancel in-flight siblings, after a drain
                    # policy has let them finish, and always await them so no
                    # provider task escapes the failed run.
                    await self._stop_after(first_error, active)
                    raise first_error
        except BaseException:
            await self._cancel_and_settle(active)
            raise

        if pending:
            failed = [n for n in graph.nodes if n.status == NodeStatus.FAILED]
            if failed:
                raise RuntimeError(
                    f"Execution halted: {len(pending)} node(s) blocked by "
                    f"{len(failed)} failed upstream node(s)"
                )
            raise RuntimeError("Deadlock: pending nodes exist but none are ready")
        return graph

    async def _supervise(
        self, node: Node, graph: ExecutionGraph, ready: deque[Node],
        held_s: dict[str, float], free_slots: int,
    ) -> bool:
        """Review after *node*; ready nodes that had a free slot wait for it."""
        held = list(islice(ready, max(0, free_slots)))
        started = self._tracer.now()
        revised = await self.maybe_revise(node, graph)
        if held:
            stalled = self._tracer.since(started)
            self._tracer.on_wait("supervision", stalled * len(held))
            # The review, not a full queue, held these nodes: keep that time
            # out of their admission waits.
            for waiting in held:
                held_s[waiting.id] = held_s.get(waiting.id, 0.0) + stalled
        return revised

    def _reserve_node(self, node: Node) -> None:
        """Reserve the best available per-call estimate before admission."""
        self.reserve_node_budget(
            node, default_estimated_tokens=self._estimated_tokens_per_node,
        )

    async def _stop_after(self, error: Exception, active: dict[asyncio.Task[None], Node]) -> None:
        """Cancel and settle the nodes in flight; under a drain policy, let them finish first.

        A failure of class ``security_stop`` cancels them at once. A node
        admitted in the same pass as a refused one has not begun an attempt,
        so it has no call to finish: it is cancelled before it starts. Nor
        has a node waiting out a retry backoff: it is cancelled at once and
        reported as cancelled by the drain.
        """
        if self._termination is not None and active and classify_failure(error) != SECURITY_STOP:
            self._draining = True
            unstarted = {node.id for node in active.values() if node.status is NodeStatus.PENDING}
            waiting = {node.id for node in active.values() if node.id in self._backing_off}
            if (await self._cancel_and_settle(active, unstarted | waiting))[1]:
                raise asyncio.CancelledError  # A cancelled run never drains.
            await self._drain(active, waiting)
        await self._cancel_and_settle(active)

    async def _drain(self, active: dict[asyncio.Task[None], Node], stopped: set[str]) -> None:
        """Let the nodes in flight finish, for up to the policy's ``drain_timeout_s``.

        Nothing is admitted meanwhile, and no node starts a retry: one whose
        attempt fails returns to pending with its reservation released, as a
        cancelled node does, and joins *stopped*, the nodes the drain has
        already cancelled. A node that finishes has already saved its result,
        cost and status through its own completion path; it is neither
        supervised nor regenerated, and its own failure does not replace the
        run's. Nodes still in flight when the drain ends stay in *active* to
        be cancelled. A cancellation of the run, or a node that ends
        cancelled, ends the drain at once.
        """
        settled: list[str] = []
        started = self._tracer.now()
        deadline = asyncio.create_task(self._drain_wait(self._termination.drain_timeout_s))
        try:
            while active and not deadline.done():
                done, _ = await asyncio.wait(
                    {deadline, *active}, return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done - {deadline}:
                    node = active.pop(task)
                    try:
                        task.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass  # The node settled its own failure.
                    if node.status is NodeStatus.PENDING:
                        stopped.add(node.id)  # Only a refused retry ends a node pending.
                    else:
                        settled.append(node.id)
        finally:
            deadline_hit = deadline.done() and bool(active)
            if deadline.done() and not deadline.cancelled():
                deadline.exception()  # A wait that failed ends the drain like its deadline.
            deadline.cancel()
            self._tracer.on_wait("drain", self._tracer.since(started))
            cancelled = stopped | {node.id for node in active.values()}
            self._drained = (tuple(sorted(settled)), tuple(sorted(cancelled)), deadline_hit)

    async def _drain_wait(self, seconds: float) -> None:
        """Wait out a drain's timeout with the executor's ``sleep``, or ``asyncio.sleep``."""
        waited = (self._sleep or asyncio.sleep)(seconds)
        if inspect.isawaitable(waited):
            await waited

    async def _cancel_and_settle(
        self, active: dict[asyncio.Task[None], Node],
        affected_ids: set[str] | None = None,
    ) -> tuple[list[BaseException], bool]:
        """Cancel and await active node tasks, releasing unused reservations."""
        tasks = [task for task, node in active.items()
                 if affected_ids is None or node.id in affected_ids]
        if not tasks:
            return [], False
        for task in tasks:
            task.cancel()
        async def outcome(task):
            try:
                return await task
            except BaseException as exc:
                # gather() normalizes cancelled tasks to plain CancelledError;
                # retain native accounting evidence carried by its subclass.
                return exc

        settled = asyncio.gather(*(outcome(task) for task in tasks))
        interrupted = False
        while not settled.done():
            try:
                await asyncio.shield(settled)
            except asyncio.CancelledError:
                # Repeated cancellation cannot detach a billed artifact write.
                interrupted = True
        errors = [result for result in settled.result() if isinstance(result, BaseException)]
        for task in tasks:
            node = active.pop(task)
            if node.status in (NodeStatus.PENDING, NodeStatus.RUNNING):
                self.release_node_budget(node)
                node.status = NodeStatus.PENDING
                node.result = None
        return errors, interrupted

    async def _regenerate_and_settle(
        self, node: Node, graph: ExecutionGraph, active: dict[asyncio.Task[None], Node],
    ) -> bool:
        intent = self.prepare_regeneration(node, graph)
        if intent is None:
            return False
        errors, interrupted = await self._cancel_and_settle(
            active, set(intent["affected_generations"]),
        )
        for error in errors:
            if is_terminal(error):
                # Retain the intent and all accounting: recovery must not hide
                # an unusable bill or buy a replacement for a failed write.
                raise error
        if interrupted:
            raise asyncio.CancelledError
        self.apply_regeneration(node, graph, intent)
        return True

    async def _execute_node(self, node: Node, graph: ExecutionGraph) -> None:
        """Run a single node through the provider, respecting its failure policy."""
        last_exc: Exception | None = None
        attempts = 1 + max(node.max_retries, 0) if node.failure_policy == FailurePolicy.RETRY else 1

        # A resumed run continues the allowance its checkpoint recorded.
        for attempt in range(self._attempts_spent(node), attempts):
            if last_exc is not None and self._draining:
                # A drain lets calls in flight finish but sends no new one: the
                # node returns to pending, so a resume runs it again.
                self._requeue_cancelled(node)
                return
            self._call_attempts[node.id] = attempt
            try:
                delay = self._retry_wait_s(node, attempt, last_exc)
            except RetryDeadlineError as exc:
                last_exc = exc
                break
            if delay:
                started = self._tracer.now()
                # A drain that begins meanwhile cancels the wait.
                self._backing_off.add(node.id)
                try:
                    waited = (self._sleep or asyncio.sleep)(delay)
                    if inspect.isawaitable(waited):
                        await waited
                finally:
                    self._backing_off.discard(node.id)
                    self._tracer.on_wait("retry_backoff", self._tracer.since(started), node)
            self._spend_attempt(node, attempt)
            node.status = NodeStatus.RUNNING
            self.begin_verification(node, graph)
            self._tracer.on_node_start(node, attempt=attempt)

            try:
                # Cost recording happens inside acall_node (per provider call).
                result = await self.acall_node(node, graph)
                # Artifact writes are file I/O — keep them off the event
                # loop so a wide wave's completions don't serialize.
                finalize_task = asyncio.create_task(
                    asyncio.to_thread(self.finalize_node_result, node, result),
                )
                while not finalize_task.done():
                    try:
                        await asyncio.shield(finalize_task)
                    except asyncio.CancelledError:
                        # Once billed, finish persistence even after repeated
                        # cancels. The scheduler awaits us before invalidation.
                        continue
                    except Exception as exc:
                        raise NodeFinalizationError(node.id, exc) from exc
                try:
                    finalize_task.result()
                except Exception as exc:
                    raise NodeFinalizationError(node.id, exc) from exc
                node.status = NodeStatus.COMPLETED
                self._tracer.on_node_end(node)
                self.complete_verification(node)
                self.notify_update(node)
                return
            except VerificationRecoveryError:
                raise
            except Exception as exc:
                # A cancellation can also be an Exception. A terminal one, such
                # as ProviderAccountingCancelledError, fails the node here; any
                # other is a cancellation and is never retried or skipped.
                if is_terminal(exc):
                    # A reconciliation alert or post-billing persistence failure
                    # is non-retryable here: another provider call would compound
                    # the spend rather than repair the local failure. Invalid
                    # accounting also retains its reservation until reconciled.
                    if isinstance(exc, BudgetValidationError):
                        self.mark_accounting_invalid(node, exc)
                    node.status = NodeStatus.FAILED
                    node.result = error_text(exc)
                    self._tracer.on_node_error(node, exc)
                    self._tracer.on_node_end(node)
                    self.notify_update(node)
                    raise
                if isinstance(exc, asyncio.CancelledError):
                    self._requeue_cancelled(node)
                    raise
                last_exc = exc
                self._tracer.on_node_error(node, exc)
                self._end_failed_attempt(node)
                if not self._may_retry(exc):
                    break
            except asyncio.CancelledError:
                self._requeue_cancelled(node)
                raise
        if last_exc is None:  # No attempt was left to make.
            last_exc = self._allowance_spent(node, attempts)

        self.release_node_budget(node)

        node.result = error_text(last_exc)

        if node.failure_policy == FailurePolicy.SKIP:
            node.status = NodeStatus.SKIPPED
            self.notify_update(node)
            return

        node.status = NodeStatus.FAILED
        self.notify_update(node)
        raise last_exc  # type: ignore[misc]

    def _requeue_cancelled(self, node: Node) -> None:
        """Return a cancelled attempt's node to pending and release its reservation."""
        self.release_node_budget(node)
        node.status = NodeStatus.PENDING
        node.result = None
        self._tracer.on_node_end(node)
