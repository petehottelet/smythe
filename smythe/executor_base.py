"""ExecutorBase — shared logic for serial and async executors."""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

from smythe.agent import Agent
from smythe.budget import Sentinel
from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tools import ChatMessage, ToolLoopLimitError, ToolResult, ToolRuntime
from smythe.tracer import Tracer

DEFAULT_MAX_TOOL_ITERATIONS = 10


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
TERMINAL_DELIVERABLE_NOTE = (
    "You are the final step in this workflow: your output is the "
    "deliverable, and the context above will not be shown alongside it. "
    "Make your output self-contained - carry forward the concrete "
    "findings, evidence, and specifics from the context rather than "
    "referencing or summarizing them."
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
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._tracer = tracer
        self._budget = budget
        self._on_node_update = on_node_update
        self._tool_runtime = tool_runtime
        self._max_tool_iterations = max_tool_iterations
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        if retry_backoff_s < 0:
            raise ValueError(f"retry_backoff_s must be >= 0, got {retry_backoff_s}")
        self._retry_backoff_s = retry_backoff_s

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

    def notify_update(self, node: Node) -> None:
        """Invoke the node-update hook (used for checkpointing) if one is set.

        Called whenever a node reaches a terminal status: COMPLETED,
        SKIPPED, or FAILED.
        """
        if self._on_node_update is not None:
            self._on_node_update(node)

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
            dep = self.node_by_id(dep_id, graph)
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
            node, dep_results, is_terminal=not graph.dependents(node.id),
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
        if self._budget:
            node.metadata["cost_usd"] = self._budget.add_cost(node.id, result)

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
            path.write_bytes(art.data)
            records.append({"path": str(path.resolve()), "mime_type": art.mime_type})
        node.metadata["artifacts"] = records
        if not result.text.strip():
            node.result = "Generated artifacts:\n" + "\n".join(
                r["path"] for r in records
            )

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
            dep = self.node_by_id(dep_id, graph)
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
            dep_node = self.node_by_id(dep_id, graph)
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
