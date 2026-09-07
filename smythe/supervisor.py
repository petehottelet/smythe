"""Supervisor — the adaptive tier that revises a plan while it runs.

The Architect plans once, before any work happens.  Without a
supervisor the executor walks that plan to the end no matter what the
results show: a plan that turns out to be wrong is still executed in
full, and the benchmark record for generated topology carries the cost
of that (a single bad plan drags a whole task's score down).

A supervisor closes the loop.  After a node completes it reviews the
work so far against the goal and may return a :class:`Revision` that
changes the *unexecuted* remainder — appending a step that closes a
gap, dropping planned work the results made unnecessary, or inserting a
step ahead of pending work.  History is never touched.

Two guardrails keep this from becoming an unbounded agent loop:

- ``max_revisions`` caps how many times a run may be revised.
- Every revision is validated against the graph before it applies
  (:meth:`ExecutionGraph.apply_revision`), so a malformed proposal
  costs a trace entry rather than a corrupt run.

The design follows the adaptive-orchestration pattern: evaluate state,
find the gap between what exists and what was asked for, then decide
whether to add work, cut work, or stop.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from smythe.budget import validate_completion_usage
from smythe.graph import ExecutionGraph, Node, NodeStatus, Revision
from smythe.task import render_task
from smythe.verifier import node_generation
from smythe.workflow_binding import (
    ComponentBinding, WorkflowBindingError, provider_description, require_exact,
)

if TYPE_CHECKING:
    from smythe.provider import Provider
    from smythe.task import Task

SUPERVISOR_SYSTEM_PROMPT = (
    "You supervise a running multi-agent execution graph. Your job is "
    "to notice when the remaining plan no longer fits what the work has "
    "revealed, and to correct it — not to redesign a plan that is "
    "working. Revising costs money and time; most reviews should "
    "conclude that no change is needed."
)

REVIEW_PROMPT = """Original task:
{task_description}

The work is not done until its stated acceptance criteria hold.

Completed work so far:
{completed}

Remaining planned steps (not yet executed):
{remaining}

The step that just finished was {node_id!r}.

Identify the gap, if any, between what the completed work has produced \
and what the goal requires. Then decide:

- If the remaining steps will close that gap, make no change.
- If an acceptance criterion will not be met by any remaining step, add
  the step that meets it.
- If a needed step is missing, add it.
- If a remaining step is now pointless or redundant, drop it.

Revisions remaining: {budget}.

Respond with STRICT JSON only, no prose, no code fences:
{{"change": false, "reason": "<why no change is needed>"}}
or
{{"change": true, "reason": "<the gap being closed>",
  "add": [{{"id": "<new-id>", "label": "<what this step must produce>",
            "depends_on": ["<existing-or-new-id>"]}}],
  "drop": ["<pending-node-id>"],
  "rewire": {{"<pending-node-id>": ["<new-dependency-id>"]}}}}

Only pending steps may be dropped or rewired. Keep additions minimal."""


class Supervisor(ABC):
    """Reviews a running graph and may revise its unexecuted remainder."""

    @abstractmethod
    async def review(
        self,
        graph: ExecutionGraph,
        node: Node,
        *,
        task: Task | None,
        revisions_remaining: int,
    ) -> Revision | None:
        """Return a revision to apply, or None to let the plan stand.

        Args:
            graph: The running graph, with live node statuses and results.
            node: The node that just completed.
            task: The originating task, when the run started from one.
            revisions_remaining: How many revisions this run may still make.
        """


class LLMSupervisor(Supervisor):
    """Asks a model whether the remaining plan still fits the goal.

    Reviewing after every node is usually wasteful — the interesting
    moments are when a stage of work concludes.  ``review_after``
    restricts reviews to specific node ids; ``only_terminal`` (the
    default) reviews at a fan-in boundary or when the whole graph has
    finished.  Parallel terminal leaves therefore produce one review,
    not one review per leaf.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        model: str | None = None,
        review_after: set[str] | None = None,
        only_terminal: bool = True,
        run_binding: ComponentBinding | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._review_after = review_after
        self._only_terminal = only_terminal
        self._run_binding = run_binding

    def workflow_description(self, **defaults) -> dict:
        require_exact(self, LLMSupervisor)
        if type(self._only_terminal) is not bool or (
            self._review_after is not None and (
                type(self._review_after) is not set
                or any(type(value) is not str for value in self._review_after)
            )
        ):
            raise WorkflowBindingError("Supervisor review policy requires a bool and set of node IDs")
        model = self._model or defaults.get("default_model") or None
        return {"type": "llm_supervisor", "version": 1,
                **provider_description(self._provider, model),
                "review_after": sorted(self._review_after) if self._review_after is not None else None,
                "only_terminal": self._only_terminal}

    def workflow_providers(self) -> tuple[Provider, ...]:
        return (self._provider,)

    def bind_run(self, binding: ComponentBinding) -> LLMSupervisor:
        self.workflow_description(default_model=binding.default_model)
        return LLMSupervisor(
            binding.snapshot_provider(self._provider), model=self._model,
            review_after=set(self._review_after) if self._review_after is not None else None,
            only_terminal=self._only_terminal, run_binding=binding,
        )

    def _should_review(self, graph: ExecutionGraph, node: Node) -> bool:
        if self._review_after is not None:
            return node.id in self._review_after
        if not self._only_terminal:
            return True
        if not any(n.status is NodeStatus.PENDING for n in graph.nodes):
            return True
        return any(
            dependent.status is NodeStatus.PENDING
            and len(dependent.depends_on) > 1
            and graph.is_ready(dependent)
            for dependent in graph.dependents(node.id)
        )

    async def review(
        self,
        graph: ExecutionGraph,
        node: Node,
        *,
        task: Task | None,
        revisions_remaining: int,
    ) -> Revision | None:
        if revisions_remaining <= 0 or not self._should_review(graph, node):
            return None

        model = self._model or node.metadata.get("model", "")
        provider = (self._run_binding.for_call(
            self._provider, trigger={"node_id": node.id, "generation": node_generation(node)},
            generation=node_generation(node),
        ) if self._run_binding else self._provider)
        result = await provider.complete(
            SUPERVISOR_SYSTEM_PROMPT,
            self._build_prompt(graph, node, task, revisions_remaining),
            model,
        )
        validate_completion_usage(result)
        return self._parse(result.text)

    @staticmethod
    def _build_prompt(
        graph: ExecutionGraph,
        node: Node,
        task: Task | None,
        revisions_remaining: int,
    ) -> str:
        completed = [
            f"[{n.id}] {n.label}\n{str(n.result)[:1500]}"
            for n in graph.nodes
            if n.status is NodeStatus.COMPLETED
        ]
        remaining = [
            f"[{n.id}] {n.label} (depends on: {', '.join(n.depends_on) or 'nothing'})"
            for n in graph.nodes
            if n.status is NodeStatus.PENDING
        ]
        resolved_task = task if task is not None else graph.task
        task_description = (
            render_task(resolved_task) if resolved_task is not None
            else node.metadata.get("task_context") or "(not recorded)"
        )
        return REVIEW_PROMPT.format(
            task_description=task_description,
            completed="\n\n".join(completed) or "(nothing yet)",
            remaining="\n".join(remaining) or "(none — this is the last step)",
            node_id=node.id,
            budget=revisions_remaining,
        )

    @staticmethod
    def _parse(text: str) -> Revision | None:
        """Parse a proposal, treating anything malformed as 'no change'.

        A supervisor that cannot produce valid JSON must not be able to
        halt a run that is otherwise going fine.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) < 2:
                return None
            cleaned = parts[1].removeprefix("json").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or not data.get("change"):
            return None

        add_nodes: list[Node] = []
        for entry in data.get("add") or []:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label")
            if not isinstance(label, str) or not label.strip():
                continue
            node_id = entry.get("id")
            depends_on = [
                dep for dep in (entry.get("depends_on") or []) if isinstance(dep, str)
            ]
            add_nodes.append(
                Node(
                    label=label,
                    depends_on=depends_on,
                    **({"id": node_id} if isinstance(node_id, str) and node_id else {}),
                )
            )

        drop = tuple(
            item for item in (data.get("drop") or []) if isinstance(item, str)
        )
        rewire_raw = data.get("rewire") or {}
        rewire = {
            key: tuple(dep for dep in value if isinstance(dep, str))
            for key, value in rewire_raw.items()
            if isinstance(key, str) and isinstance(value, list)
        }

        revision = Revision(
            add_nodes=tuple(add_nodes),
            drop_node_ids=drop,
            rewire=rewire,
            reason=str(data.get("reason", ""))[:500],
        )
        return None if revision.is_empty else revision
