"""ExecutorBase — shared logic for serial and async executors."""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from smythe.agent import Agent
from smythe.budget import (
    BudgetEstimateRequired,
    BudgetValidationError,
    Sentinel,
    validate_completion_usage,
    validate_token_count,
)
from smythe.graph import ExecutionGraph, Node, NodeStatus, RevisionError
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.task import render_task, snapshot_task
from smythe.tools import ChatMessage, ToolLoopLimitError, ToolResult, ToolRuntime
from smythe.tracer import Tracer
from smythe.verifier import (
    TokenVerifier, Verifier, VerificationRecoveryError, node_generation, verification_integer,
    validate_verification_receipt,
)

if TYPE_CHECKING:
    from smythe.supervisor import Supervisor
    from smythe.task import Task

DEFAULT_MAX_TOOL_ITERATIONS = 10


class NodeFinalizationError(RuntimeError):
    """Raised after a billed response cannot be persisted safely.

    This error is non-retryable at the executor layer: repeating the provider
    call would buy the same output again merely because local finalization
    failed.
    """

    def __init__(self, node_id: str, cause: Exception) -> None:
        self.node_id = node_id
        self.cause = cause
        super().__init__(f"Node {node_id!r} finalization failed: {cause}")


def _safe_filename_component(raw: str) -> str:
    """Make an arbitrary node id safe as a filename component.

    Node ids come from YAML and LLM plan JSON with no character
    validation, so they can carry path separators (traversal out of the
    artifact dir) or Windows-illegal characters (OSError after an
    already-billed provider call).  When sanitization changes anything,
    a short hash of the original keeps distinct ids distinct.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("._") or "node"
    if safe != raw:
        safe = f"{safe}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:6]}"
    return safe

# When False, root nodes ignore the task_context stamped by Swarm.plan()
# and see only their planned label - exists for benchmark ablations.
INCLUDE_TASK_CONTEXT = True

# Appended to a terminal node's prompt so the deliverable survives the
# chain: without it, final nodes tend to reference or summarize upstream
# findings instead of reproducing them, and the specifics are lost
# (benchmarks/README.md documents the failure mode this addresses).
#
# The second paragraph exists because the first was not enough. A node
# labelled "state the surviving claims" obeyed its label and emitted only
# the survivors, dropping the case and the critique the goal had also
# asked for - the label scopes the model's behaviour more strongly than a
# general instruction to be self-contained. The shape suite scored those
# runs 1-2 out of 10, because most of what the task asked for was never
# returned. The fix is to say plainly that the step name describes a
# contribution, not the extent of the output.
TERMINAL_DELIVERABLE_NOTE = (
    "You are the final step in this workflow: your output is the "
    "deliverable, and the context above will not be shown alongside it. "
    "Make your output self-contained - carry forward the concrete "
    "findings, evidence, and specifics from the context rather than "
    "referencing or summarizing them.\n\n"
    "Your step name describes what you contribute, not how much you "
    "output. Deliver everything the overall task asked for, including "
    "the parts produced in earlier steps: if the task asked for an "
    "argument and then a critique of it, return both, not just your "
    "own increment."
)


class ExecutorBase:
    """Shared infrastructure for all executor variants.

    Provides constructor, prompt building, dependency lookup, the
    tool-calling loop, and node-by-id helpers.  Subclasses implement
    ``run()`` and drive ``acall_node()`` (sync or async).
    """

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        tracer: Tracer,
        budget: Sentinel | None = None,
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
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._tracer = tracer
        self._budget = budget
        self._on_node_update = on_node_update
        self._on_control_update = on_control_update
        self._tool_runtime = tool_runtime
        self._max_tool_iterations = max_tool_iterations
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._prepared_graph: ExecutionGraph | None = None
        self._node_lookup: dict[str, Node] = {}
        self._dependent_ids: set[str] = set()
        if retry_backoff_s < 0:
            raise ValueError(f"retry_backoff_s must be >= 0, got {retry_backoff_s}")
        self._retry_backoff_s = retry_backoff_s
        if max_revisions < 0:
            raise ValueError(f"max_revisions must be >= 0, got {max_revisions}")
        self._supervisor = supervisor
        self._max_revisions = max_revisions
        self._default_task = snapshot_task(task) if task is not None else None
        self._task = snapshot_task(self._default_task) if self._default_task is not None else None
        # Restored from the checkpoint on resume: the revision cap is a
        # per-run guarantee, so a crash must not refill the allowance.
        self._revisions_used = revisions_used
        self._verifier = verifier or TokenVerifier()

    @property
    def revisions_used(self) -> int:
        """How many supervisor revisions this executor has applied."""
        return self._revisions_used

    async def maybe_revise(self, node: Node, graph: ExecutionGraph) -> bool:
        """Let the supervisor revise the plan after *node* completed.

        Returns True when the graph changed, so the caller can refresh
        any scheduling state it derived from it. Ordinary supervisor
        errors and invalid changes are contained and traced. Invalid
        provider accounting is terminal because continuing would spend
        more while the supervisor's charge remains unresolved.
        """
        if self._supervisor is None or node.status is not NodeStatus.COMPLETED:
            return False
        remaining = self._max_revisions - self._revisions_used
        if remaining <= 0:
            return False

        try:
            revision = await self._supervisor.review(
                graph, node, task=snapshot_task(self._task) if self._task is not None else None,
                revisions_remaining=remaining,
            )
        except BudgetValidationError as exc:
            self.mark_accounting_invalid(node, exc)
            self._tracer.on_revision(
                node, None, applied=False, detail=f"supervisor accounting invalid: {exc}",
            )
            self.notify_update(node)
            raise
        except Exception as exc:
            self._tracer.on_revision(
                node, None, applied=False, detail=f"supervisor raised: {exc}",
            )
            return False
        if revision is None or revision.is_empty:
            return False

        try:
            graph.apply_revision(revision)
        except RevisionError as exc:
            self._tracer.on_revision(node, revision, applied=False, detail=str(exc))
            return False

        self._revisions_used += 1
        self._inherit_execution_context(revision.add_nodes, graph)
        self.prepare_graph(graph)
        self._tracer.on_revision(node, revision, applied=True)
        return True

    def maybe_regenerate(self, node: Node, graph: ExecutionGraph) -> bool:
        """Apply a durable rejection when the caller has no active descendants."""
        intent = self.prepare_regeneration(node, graph)
        if intent is None:
            return False
        if any(n.status is NodeStatus.RUNNING for n in graph.nodes
               if n.id in intent["affected_generations"]):
            raise VerificationRecoveryError("Active descendants must settle before regeneration")
        self.apply_regeneration(node, graph, intent)
        return True

    def notify_control_update(self, node: Node) -> None:
        """Persist a control transition without batching or changing old hooks."""
        try:
            if self._on_control_update is not None:
                self._on_control_update()
            else:
                self.notify_update(node)
        except Exception as exc:
            raise VerificationRecoveryError("Could not persist verification control") from exc

    def begin_verification(self, node: Node, graph: ExecutionGraph) -> None:
        """Bind the judge's request to the version of its target it observes."""
        node_generation(node)
        if node.verifies and node.max_regenerations > 0:
            target = self._cached_node_by_id(node.verifies, graph)
            node.metadata["verification_target_generation"] = (
                node_generation(target) if target is not None else None
            )

    def complete_verification(self, node: Node) -> None:
        """Stamp pending before any COMPLETED checkpoint can become visible."""
        if node.status is not NodeStatus.COMPLETED or not node.verifies or node.max_regenerations <= 0:
            return
        node.metadata["verification_receipt"] = {
            "version": 1, "state": "pending", "judge_generation": node_generation(node),
            "target_id": node.verifies,
            "target_generation": node.metadata.get("verification_target_generation"),
        }
        self.notify_control_update(node)

    def _affected_nodes(self, target: Node, graph: ExecutionGraph) -> list[Node]:
        children: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
        for candidate in graph.nodes:
            for dep_id in candidate.depends_on:
                children.setdefault(dep_id, []).append(candidate.id)
            # A judge also observes its target when a hand-built graph omitted
            # the dependency edge. Its old verdict must never survive a reset.
            if candidate.verifies:
                children.setdefault(candidate.verifies, []).append(candidate.id)
        stack = [target.id]
        seen: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(children.get(node_id, ()))
        return [n for n in graph.nodes if n.id in seen]

    def prepare_regeneration(self, node: Node, graph: ExecutionGraph) -> dict | None:
        """Persist one rejection intent before cancellation or graph mutation."""
        existing = node.metadata.get("regeneration_intent")
        if existing is not None:
            self._validate_regeneration(node, graph, existing)
            return existing
        if node.status is not NodeStatus.COMPLETED or not node.verifies or node.max_regenerations <= 0:
            return None
        receipt = node.metadata.get("verification_receipt")
        # Completed legacy/manual nodes are not silently rejudged.
        if receipt is None:
            return None
        validate_verification_receipt(node, graph)
        if receipt.get("state") == "consumed":
            return None
        if receipt.get("state") != "pending" or receipt.get("judge_generation") != node_generation(node):
            raise VerificationRecoveryError(f"Stale verification receipt on {node.id!r}")
        target = self._cached_node_by_id(node.verifies, graph)
        used = verification_integer(node.metadata.get("regenerations_used", 0), "regenerations_used")
        valid_target = target is not None and target.status is NodeStatus.COMPLETED
        if valid_target and receipt.get("target_generation") != node_generation(target):
            raise VerificationRecoveryError(f"Verifier {node.id!r} observed an obsolete target")
        if receipt.get("target_id") != node.verifies:
            raise VerificationRecoveryError(f"Verifier {node.id!r} target identity changed")
        if not valid_target or used >= node.max_regenerations:
            node.metadata["verification_receipt"] = dict(receipt, state="consumed", reason="unavailable or exhausted")
            self.notify_control_update(node)
            return None
        verdict = self._verifier.verdict(node, target)
        if verdict.passed:
            node.metadata["verification_receipt"] = dict(receipt, state="consumed", passed=True, reason=verdict.reason)
            self.notify_control_update(node)
            return None
        intent = {
            "version": 1, "target_id": target.id, "target_generation": node_generation(target),
            "judge_generation": node_generation(node), "reason": verdict.reason,
            "regenerations_used": used + 1,
            "affected_generations": {n.id: node_generation(n) for n in self._affected_nodes(target, graph)},
        }
        node.metadata["regenerations_used"] = used + 1
        node.metadata["regeneration_intent"] = intent
        self.notify_control_update(node)
        return intent

    def _validate_regeneration(self, node: Node, graph: ExecutionGraph, intent: dict) -> None:
        if (not isinstance(intent, dict) or type(intent.get("version")) is not int
                or intent["version"] != 1 or not isinstance(intent.get("reason"), str)
                or not isinstance(intent.get("target_id"), str)):
            raise VerificationRecoveryError("Invalid regeneration intent")
        target = self._cached_node_by_id(intent.get("target_id"), graph)
        if target is None or node.verifies != target.id:
            raise VerificationRecoveryError("Regeneration target identity changed")
        affected = intent.get("affected_generations")
        if not isinstance(affected, dict) or set(affected) != {n.id for n in self._affected_nodes(target, graph)}:
            raise VerificationRecoveryError("Regeneration affected-node inventory changed")
        used = verification_integer(intent.get("regenerations_used"), "intent regenerations_used")
        saved_used = verification_integer(node.metadata.get("regenerations_used"), "regenerations_used")
        if used < 1 or used > node.max_regenerations or saved_used != used:
            raise VerificationRecoveryError("Regeneration allowance differs from the saved intent")
        for candidate in graph.nodes:
            if candidate.id in affected:
                old = verification_integer(affected[candidate.id], "source generation")
                if node_generation(candidate) not in (old, old + 1):
                    raise VerificationRecoveryError("Regeneration generation identity changed")
                if (node_generation(candidate) == old + 1
                        and (candidate.status is not NodeStatus.PENDING or candidate.result is not None)):
                    raise VerificationRecoveryError("Regeneration intent would overwrite a newer result")
                if candidate is not node and "regeneration_intent" in candidate.metadata:
                    raise VerificationRecoveryError("Overlapping regeneration intents require reconciliation")
        target_generation = verification_integer(intent.get("target_generation"), "intent target generation")
        judge_generation = verification_integer(intent.get("judge_generation"), "intent judge generation")
        if target_generation != affected[target.id] or judge_generation != affected[node.id]:
            raise VerificationRecoveryError("Regeneration source identity is inconsistent")

    def apply_regeneration(self, node: Node, graph: ExecutionGraph, intent: dict) -> None:
        """Commit an idempotent reset after every affected worker has settled."""
        self._validate_regeneration(node, graph, intent)
        target = self._cached_node_by_id(intent["target_id"], graph)
        reset = [n for n in graph.nodes if n.id in intent["affected_generations"]]
        if any(n.metadata.get("accounting_invalid") for n in reset):
            raise BudgetValidationError("Cannot regenerate unresolved provider accounting")
        for candidate in reset:
            candidate.status = NodeStatus.PENDING
            candidate.result = None
            candidate.metadata["execution_generation"] = intent["affected_generations"][candidate.id] + 1
            for key in ("artifacts", "artifacts_discarded", "verification_receipt", "verification_target_generation"):
                candidate.metadata.pop(key, None)
        node.metadata.pop("regeneration_intent", None)
        self.notify_control_update(node)
        self._tracer.on_regeneration(
            node, target, reason=intent["reason"], attempt=intent["regenerations_used"],
            limit=node.max_regenerations, reset_ids=[n.id for n in reset],
        )
        for candidate in reset:
            self.notify_update(candidate)

    def recover_verification(self, graph: ExecutionGraph) -> None:
        """Finish saved control transitions before initial scheduling."""
        # Validate every saved intent before changing any node. In particular,
        # malformed overlapping intents must not partially invalidate a graph.
        for node in graph.nodes:
            if "regeneration_intent" in node.metadata:
                self._validate_regeneration(node, graph, node.metadata["regeneration_intent"])
        for node in graph.nodes:
            intent = node.metadata.get("regeneration_intent")
            if intent is not None:
                self.apply_regeneration(node, graph, intent)
        for node in graph.nodes:
            self.maybe_regenerate(node, graph)

    def _inherit_execution_context(
        self, new_nodes: tuple[Node, ...], graph: ExecutionGraph,
    ) -> None:
        """Give supervisor-added nodes the run's model and task context.

        Nodes minted mid-run never passed through ``Swarm.plan``, so
        they carry none of the metadata the executor depends on; without
        this a revision would produce nodes that call the provider with
        an empty model name.
        """
        if not new_nodes:
            return
        model = next(
            (n.metadata["model"] for n in graph.nodes if n.metadata.get("model")), None,
        )
        context = next(
            (
                n.metadata["task_context"]
                for n in graph.nodes
                if n.metadata.get("task_context")
            ),
            None,
        )
        for node in new_nodes:
            if model is not None:
                node.metadata.setdefault("model", model)
            if self._task is None and context is not None:
                node.metadata.setdefault("task_context", context)
        if self._task is not None:
            self.stamp_task_context(new_nodes, self._task)

    @staticmethod
    def stamp_task_context(nodes: list[Node] | tuple[Node, ...], task: Task) -> None:
        """Keep constraints/data/criteria even when the label repeats the goal."""
        # Serialize source data once per variant, not once per fan-out node.
        full_context = render_task(task)
        same_goal_context = render_task(task, include_goal=False)
        for node in nodes:
            context = same_goal_context if node.label.strip() == task.goal.strip() else full_context
            if context:
                node.metadata["task_context"] = context
            else:
                # A goal-only SimpleArchitect retains its original bare prompt.
                node.metadata.pop("task_context", None)

    def retry_delay_s(self, attempt: int) -> float:
        """Full-jitter exponential backoff before retry `attempt` (1-based).

        Returns 0 when backoff is disabled (the default) so existing
        RETRY behavior and test timing are unchanged; opt in via
        ``retry_backoff_s`` for rate-limited workloads (429s at wide
        parallel image fan-out).
        """
        if attempt <= 0 or not self._retry_backoff_s:
            return 0.0
        ceiling = self._retry_backoff_s * (2 ** (attempt - 1))
        return random.uniform(0, ceiling)

    def reserve_node_budget(
        self,
        node: Node,
        *,
        default_estimated_tokens: int | None = None,
    ) -> None:
        """Reserve a defensible pre-call cost for one node.

        Node metadata wins, then the provider's model-aware estimate.  A
        generic token fallback is allowed only for providers that declare it
        safe.  Serial execution passes no fallback to preserve its historical
        pre-call behavior for ordinary text providers while still reserving
        image calls.
        """
        if self._budget is None:
            return

        estimate = node.metadata.get("estimated_cost_usd")
        model = str(node.metadata.get("model", ""))
        requires_explicit = self._provider.requires_explicit_budget_estimate(model)
        if estimate is None:
            estimate = self._provider.budget_estimate_usd(model)

        if estimate is None and requires_explicit:
            if self._budget.max_budget_usd is not None:
                raise BudgetEstimateRequired(
                    node.id, model, type(self._provider).__name__,
                )
            # Unlimited execution may proceed without a price, but must not
            # consume a fake text-token reservation and report it as complete.
            # The provider result will mark its USD cost unknown.
            return

        if estimate is not None:
            self._budget.reserve(
                node.id,
                estimate,
                hard_ceiling=requires_explicit,
            )
        elif default_estimated_tokens is not None:
            tokens = validate_token_count(default_estimated_tokens, "default_estimated_tokens")
            try:
                estimate = tokens * self._budget.cost_per_token
            except OverflowError as exc:
                raise BudgetValidationError("Token-derived estimate must remain finite") from exc
            self._budget.reserve(
                node.id,
                estimate,
            )
        else:
            self._budget.check(node.id)

    def notify_update(self, node: Node) -> None:
        """Invoke the node-update hook (used for checkpointing) if one is set.

        Called whenever a node reaches a terminal status: COMPLETED,
        SKIPPED, or FAILED.
        """
        if self._on_node_update is not None:
            try:
                self._on_node_update(node)
            except Exception as exc:
                if node.status is NodeStatus.COMPLETED and "verification_receipt" in node.metadata:
                    raise VerificationRecoveryError("Could not persist completed verifier") from exc
                raise

    @staticmethod
    def mark_accounting_invalid(node: Node, error: BudgetValidationError) -> None:
        """Persist the need for reconciliation before this run can resume."""
        node.metadata["accounting_invalid"] = True
        node.metadata["accounting_error"] = str(error)

    def prepare_graph(self, graph: ExecutionGraph) -> None:
        """Cache immutable graph structure used throughout an execution.

        Node statuses, results, and metadata change while a graph runs, but
        node IDs and dependency edges do not.  Building these indexes once
        keeps wide DAGs from repeatedly scanning every node while assembling
        dependency prompts or deciding whether a node is terminal.
        """
        if self._prepared_graph is not graph:
            # Reusing an executor must not carry a previous graph's source
            # material into a later taskless graph. Only the constructor Task
            # is a default; the current graph binding is scoped to that graph.
            task = graph.task if graph.task is not None else self._default_task
            self._task = snapshot_task(task) if task is not None else None
            if self._task is not None:
                graph.task = snapshot_task(self._task)
                self.stamp_task_context(graph.nodes, self._task)
        self._prepared_graph = graph
        self._node_lookup = {node.id: node for node in graph.nodes}
        self._dependent_ids = {
            dep_id for node in graph.nodes for dep_id in node.depends_on
        }

    def prepare_execution(self, graph: ExecutionGraph) -> None:
        """Bind a new run even when a caller reuses the same graph object."""
        self._prepared_graph = None
        self.prepare_graph(graph)

    def _ensure_graph_prepared(self, graph: ExecutionGraph) -> None:
        if self._prepared_graph is not graph:
            self.prepare_graph(graph)

    def _cached_node_by_id(self, node_id: str, graph: ExecutionGraph) -> Node | None:
        self._ensure_graph_prepared(graph)
        return self._node_lookup.get(node_id)

    def _is_terminal(self, node: Node, graph: ExecutionGraph) -> bool:
        self._ensure_graph_prepared(graph)
        return node.id not in self._dependent_ids

    @staticmethod
    def build_system_prompt(agent: Agent | None) -> str:
        if agent and agent.profile.persona:
            return agent.profile.persona
        return "You are a helpful assistant completing a step in a larger task."

    @staticmethod
    def build_user_prompt(
        node: Node, dep_results: dict[str, Any], *, is_terminal: bool = False,
    ) -> str:
        task_context = (
            node.metadata.get("task_context") if INCLUDE_TASK_CONTEXT else None
        )
        if task_context:
            parts = [f"Overall task:\n{task_context}", f"\nYour step: {node.label}"]
        else:
            parts = [node.label]
        if dep_results:
            parts.append("\n\nContext from prior steps:")
            for dep_id, result in dep_results.items():
                parts.append(f"\n[{dep_id}]: {result}")
            if is_terminal and TERMINAL_DELIVERABLE_NOTE:
                parts.append("\n" + TERMINAL_DELIVERABLE_NOTE)
        return "\n".join(parts)

    @staticmethod
    def node_by_id(node_id: str, graph: ExecutionGraph) -> Node | None:
        return next((n for n in graph.nodes if n.id == node_id), None)

    def deps_satisfied(self, node: Node, graph: ExecutionGraph) -> bool:
        """True if every dependency is COMPLETED or SKIPPED."""
        for dep_id in node.depends_on:
            dep = self._cached_node_by_id(dep_id, graph)
            if dep is None or dep.status not in (NodeStatus.COMPLETED, NodeStatus.SKIPPED):
                return False
        return True

    async def acall_node(self, node: Node, graph: ExecutionGraph) -> CompletionResult:
        """Run the node's conversation (including any tool loop) under timeout_s.

        Owns cost recording: every provider call inside is billed via
        Sentinel.add_cost, so executors must not record costs again.
        """
        coro = self._run_node_conversation(node, graph)
        if node.timeout_s is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=node.timeout_s)
        except TimeoutError:
            raise TimeoutError(
                f"Node {node.id!r} timed out after {node.timeout_s}s"
            ) from None

    async def _run_node_conversation(
        self, node: Node, graph: ExecutionGraph,
    ) -> CompletionResult:
        agent = self._registry.get(node.agent_id) if node.agent_id else None
        dep_results = self.gather_dep_results(node, graph)
        system = self.build_system_prompt(agent)
        prompt = self.build_user_prompt(
            node, dep_results, is_terminal=self._is_terminal(node, graph),
        )
        model = node.metadata.get("model", "")
        attachments = (
            self.load_dep_image_artifacts(node, graph)
            if node.attach_dep_artifacts
            else []
        )
        messages = [ChatMessage(role="user", content=prompt, attachments=attachments)]

        if self._tool_runtime is None:
            result = await self._provider.chat(system, messages, model)
            self._record_cost(node, result)
            return result

        collected_artifacts: list = []
        async with self._tool_runtime.open(agent) as session:
            tools = list(session.tools) or None
            limit = node.max_tool_iterations or self._max_tool_iterations
            for _ in range(limit):
                if self._budget:
                    self._budget.check(node.id)
                result = await self._provider.chat(system, messages, model, tools=tools)
                self._record_cost(node, result)
                # Artifacts on intermediate turns are already billed —
                # carry them to the final result so they get persisted.
                if result.artifacts:
                    collected_artifacts.extend(result.artifacts)

                if result.stop_reason == "pause_turn" and not result.tool_calls:
                    # Provider paused a server-side loop; re-send to continue.
                    messages.append(ChatMessage(role="assistant", content=result.text))
                    continue
                if not result.tool_calls:
                    if len(collected_artifacts) != len(result.artifacts):
                        result.artifacts = list(collected_artifacts)
                    return result

                messages.append(ChatMessage(
                    role="assistant",
                    content=result.text,
                    tool_calls=list(result.tool_calls),
                ))
                tool_results: list[ToolResult] = []
                for tc in result.tool_calls:
                    started = time.monotonic()
                    try:
                        outcome = await session.call(tc)
                    except BudgetValidationError:
                        # A tool may itself call a paid provider. Numeric
                        # accounting failures must not become retryable model
                        # feedback that dispatches another provider turn.
                        raise
                    except Exception as exc:
                        # Tool failures go back to the model, not up the stack —
                        # it can adapt or try another tool.
                        outcome = ToolResult(
                            tool_call_id=tc.id, content=str(exc), is_error=True,
                        )
                    duration_ms = (time.monotonic() - started) * 1000
                    self._tracer.on_tool_call(node, tc.name, duration_ms, outcome.is_error)
                    tool_results.append(outcome)
                messages.append(ChatMessage(role="user", tool_results=tool_results))

        raise ToolLoopLimitError(
            f"Node {node.id!r} hit max_tool_iterations={limit} without completing"
        )

    def _record_cost(self, node: Node, result: CompletionResult) -> None:
        # Custom providers can mutate CompletionResult after construction.
        # Reject malformed usage even when this executor has no Sentinel.
        validate_completion_usage(result)
        if self._budget:
            cost = self._budget.add_cost(node.id, result)
            if result.cost_usd_unknown:
                node.metadata["cost_usd_unknown"] = True
            if result.cost_usd_is_estimate:
                node.metadata["cost_usd_is_estimate"] = True
            node.metadata["cost_usd"] = cost

    def finalize_node_result(self, node: Node, result: CompletionResult) -> None:
        """Store the node's text result, persisting any artifacts to disk.

        Artifact bytes never land on the node itself — checkpoints are
        plain JSON and planner memory is JSONL, so binary payloads would
        break or bloat both.  Files go under ``artifact_dir`` with a
        filesystem-safe name; absolute paths are recorded in
        ``node.metadata["artifacts"]``, which ``gather_dep_results``
        surfaces to dependent nodes.  ``node.result`` stays the
        provider's text verbatim (downstream JSON parsers rely on it) —
        only when a node produced artifacts and no text does the result
        become a path listing.  With ``artifact_dir=None``, artifacts
        are dropped and counted in ``metadata["artifacts_discarded"]``.
        """
        node.result = result.text
        artifacts = result.artifacts
        if not artifacts:
            return
        if self._artifact_dir is None:
            node.metadata["artifacts_discarded"] = len(artifacts)
            if not result.text.strip():
                node.result = (
                    f"[{len(artifacts)} artifact(s) discarded: artifact_dir is None]"
                )
            return
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_filename_component(node.id)
        records = []
        for i, art in enumerate(artifacts):
            path = self._artifact_dir / f"{stem}_{i:02d}{art.suffix}"
            self._atomic_write_bytes(path, art.data)
            records.append({"path": str(path.resolve()), "mime_type": art.mime_type})
        node.metadata["artifacts"] = records
        if not result.text.strip():
            node.result = "Generated artifacts:\n" + "\n".join(
                r["path"] for r in records
            )

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        """Replace *path* atomically after fully writing a same-dir temp file.

        Artifact generation is commonly checkpointed immediately after this
        method returns.  Writing directly to the final path could therefore
        leave a checkpoint pointing at a truncated file after a crash.  A
        flushed temporary file plus ``os.replace`` gives readers either the
        prior complete artifact or the new complete artifact, never a partial
        write.  Same-directory placement preserves atomic rename semantics.
        """
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(data)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    MAX_ATTACHED_IMAGES = 12
    MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024  # per image

    def load_dep_image_artifacts(self, node: Node, graph: ExecutionGraph) -> list:
        """Load dependencies' image artifacts as multimodal attachments.

        Used when ``node.attach_dep_artifacts`` is set — the vision-judge
        pattern: an ArtDirector node that must *see* the images its
        dependencies generated, not just their paths.  Missing or
        oversized files are skipped (the path listing in the prompt
        still names them); count is capped so a wide fan-in can't blow
        the context window.
        """
        from smythe.provider import Artifact

        attachments: list[Artifact] = []
        for dep_id in node.depends_on:
            dep = self._cached_node_by_id(dep_id, graph)
            if dep is None:
                continue
            for record in dep.metadata.get("artifacts", []):
                if len(attachments) >= self.MAX_ATTACHED_IMAGES:
                    return attachments
                mime = record.get("mime_type", "")
                if not mime.startswith("image/"):
                    continue
                try:
                    data = Path(record["path"]).read_bytes()
                except OSError:
                    continue
                if len(data) > self.MAX_ATTACHMENT_BYTES:
                    continue
                attachments.append(Artifact(data=data, mime_type=mime))
        return attachments

    def gather_dep_results(self, node: Node, graph: ExecutionGraph) -> dict[str, Any]:
        """Collect dependency results for prompt building.

        Artifact paths live in dep metadata (not in the result text, so
        JSON results stay parseable); they are appended here so
        downstream nodes can reference the files.
        """
        dep_results: dict[str, Any] = {}
        for dep_id in node.depends_on:
            dep_node = self._cached_node_by_id(dep_id, graph)
            if dep_node is None:
                continue
            value = dep_node.result
            records = dep_node.metadata.get("artifacts") or []
            if records and isinstance(value, str) and records[0]["path"] not in value:
                listing = "\n".join(
                    f"- {r['path']} ({r['mime_type']})" for r in records
                )
                value = f"{value}\n\nArtifact files from this step:\n{listing}"
            dep_results[dep_id] = value
        return dep_results
