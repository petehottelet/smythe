"""Synthesizer — merges outputs from parallel execution branches."""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from typing import Any

from smythe.budget import (
    BudgetEstimateRequired, BudgetValidationError, Sentinel, validate_completion_usage,
)
from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.provider import (
    Provider, ProviderResponseError, _native_receipt, _native_response_errors,
    _settle_response_error, _settle_response_group,
)
from smythe.task import Task, render_task
from smythe.tracer import Tracer
from smythe.workflow_binding import ComponentBinding, WorkflowBindingError, provider_description, require_exact

logger = logging.getLogger("smythe.synthesizer")


class SynthesisStrategy(Enum):
    DELIVERABLE = "deliverable"
    CONCATENATE = "concatenate"
    LLM_MERGE = "llm_merge"
    STRUCTURED = "structured"


MERGE_SYSTEM_PROMPT = """\
You are a synthesis agent.  You receive outputs from multiple parallel \
execution steps.  Your job is to merge them into a single coherent output.  \
Preserve key information from every input.  Be concise and well-organized.
"""

DEFAULT_SYNTHESIS_ESTIMATED_TOKENS = 2000


class Synthesizer:
    """Combines completed node results into a coherent final output.

    Strategies:
    - DELIVERABLE: return completed terminal outputs (default, zero cost).
    - CONCATENATE: join every completed result with newlines (zero cost).
    - LLM_MERGE: send all results to an LLM for intelligent merging.
    - STRUCTURED: parse each result as JSON and shallow-merge.
    """

    def __init__(
        self,
        strategy: SynthesisStrategy = SynthesisStrategy.DELIVERABLE,
        provider: Provider | None = None,
        model: str | None = None,
        budget: Sentinel | None = None,
        tracer: Tracer | None = None,
        *,
        run_binding: ComponentBinding | None = None,
    ) -> None:
        self._strategy = strategy
        self._provider = provider
        self._model = model or ""
        self._budget = budget
        self._tracer = tracer
        self._run_binding = run_binding

    def workflow_description(self, **defaults) -> dict:
        require_exact(self, Synthesizer)
        if type(self._strategy) is not SynthesisStrategy:
            raise WorkflowBindingError("Unsupported synthesis strategy")
        if self._budget is not None or self._tracer is not None:
            raise WorkflowBindingError("Bound synthesis uses the workflow ledger and run trace")
        provider = defaults.get("default_provider") or self._provider
        model = defaults.get("default_model") or self._model
        paid = self._strategy is SynthesisStrategy.LLM_MERGE
        if paid and provider is None:
            raise WorkflowBindingError("Journaled LLM synthesis requires an explicit provider")
        return {"type": "synthesizer", "version": 1, "strategy": self._strategy.value,
                "completion": provider_description(provider, model) if paid else None}

    def workflow_providers(self) -> tuple[Provider, ...]:
        return (self._provider,) if self._strategy is SynthesisStrategy.LLM_MERGE and self._provider else ()

    def bind_run(self, binding: ComponentBinding) -> Synthesizer:
        self.workflow_description(default_provider=binding.default_provider,
                                  default_model=binding.default_model)
        source = binding.default_provider or self._provider
        return Synthesizer(
            self._strategy,
            provider=binding.snapshot_provider(source)
            if source is not None and self._strategy is SynthesisStrategy.LLM_MERGE else None,
            model=binding.default_model or self._model, run_binding=binding,
        )

    def _check_bound_overrides(self, provider, model, budget, tracer):
        if self._run_binding is not None and any(
            value is not None for value in (provider, model, budget, tracer)
        ):
            raise WorkflowBindingError("Bound synthesis does not accept call-time runtime overrides")

    def synthesize(
        self,
        graph: ExecutionGraph,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        budget: Sentinel | None = None,
        tracer: Tracer | None = None,
    ) -> str:
        """Produce a single output from the completed graph."""
        self._check_bound_overrides(provider, model, budget, tracer)
        completed = [
            n for n in graph.nodes
            if n.status == NodeStatus.COMPLETED and n.result is not None
        ]
        if not completed:
            return ""

        if self._strategy == SynthesisStrategy.DELIVERABLE:
            return self._deliverable(graph, completed)
        if self._strategy == SynthesisStrategy.CONCATENATE:
            return self._concatenate(completed)
        if self._strategy == SynthesisStrategy.STRUCTURED:
            return self._structured_merge(completed)
        if self._strategy == SynthesisStrategy.LLM_MERGE:
            return asyncio.run(
                self._llm_merge(
                    completed,
                    task=graph.task,
                    provider=provider,
                    model=model,
                    budget=budget,
                    tracer=tracer,
                )
            )
        return self._concatenate(completed)

    async def asynthesize(
        self,
        graph: ExecutionGraph,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        budget: Sentinel | None = None,
        tracer: Tracer | None = None,
    ) -> str:
        """Async variant for use inside an existing event loop."""
        self._check_bound_overrides(provider, model, budget, tracer)
        completed = [
            n for n in graph.nodes
            if n.status == NodeStatus.COMPLETED and n.result is not None
        ]
        if not completed:
            return ""

        if self._strategy == SynthesisStrategy.DELIVERABLE:
            return self._deliverable(graph, completed)
        if self._strategy == SynthesisStrategy.CONCATENATE:
            return self._concatenate(completed)
        if self._strategy == SynthesisStrategy.STRUCTURED:
            return self._structured_merge(completed)
        if self._strategy == SynthesisStrategy.LLM_MERGE:
            return await self._llm_merge(
                completed,
                task=graph.task,
                provider=provider,
                model=model,
                budget=budget,
                tracer=tracer,
            )
        return self._concatenate(completed)

    @staticmethod
    def _deliverable(graph: ExecutionGraph, nodes: list[Node]) -> str:
        """Return what the graph produced, not a transcript of producing it.

        The executor already tells terminal nodes their output *is* the
        deliverable (TERMINAL_DELIVERABLE_NOTE); this makes the returned
        result agree with that instruction. Joining every node instead
        pads a memo with the research and analysis that fed it, which is
        rarely what a caller wants and measurably scores worse.

        Verifier nodes are excluded twice over: they judge the
        deliverable rather than being it, and the edge from a verifier to
        the node it judges does not make that node non-terminal.  Without
        both exclusions a gated run returns "PASS" and throws away the
        artefact it approved.

        Falls back to every completed node when a graph has no terminal
        result to hand back, so nothing is ever silently lost.
        """
        verifier_ids = {node.id for node in graph.nodes if node.verifies}
        depended_on = {
            dep
            for node in graph.nodes if node.id not in verifier_ids
            for dep in node.depends_on
        }
        terminal = [
            n for n in nodes
            if n.id not in depended_on and n.id not in verifier_ids
        ]
        return Synthesizer._concatenate(terminal or nodes)

    @staticmethod
    def _concatenate(nodes: list[Node]) -> str:
        return "\n\n".join(str(n.result) for n in nodes)

    @staticmethod
    def _structured_merge(nodes: list[Node]) -> str:
        """Parse each result as JSON and shallow-merge into a single dict."""
        merged: dict[str, Any] = {}
        for node in nodes:
            try:
                data = json.loads(str(node.result))
                if isinstance(data, dict):
                    merged.update(data)
                else:
                    merged[node.id] = data
            except (json.JSONDecodeError, TypeError):
                merged[node.id] = str(node.result)
        return json.dumps(merged, indent=2)

    async def _llm_merge(
        self,
        nodes: list[Node],
        *,
        task: Task | None = None,
        provider: Provider | None = None,
        model: str | None = None,
        budget: Sentinel | None = None,
        tracer: Tracer | None = None,
    ) -> str:
        """Send all results to an LLM for intelligent synthesis."""
        resolved_provider = provider or self._provider
        resolved_model = model if model is not None else self._model
        resolved_budget = budget or self._budget
        resolved_tracer = tracer or self._tracer

        if self._run_binding is not None:
            resolved_provider = self._run_binding.for_call(self._provider, trigger="graph_completed")

        if resolved_provider is None:
            logger.warning("LLM_MERGE requested but no provider set; falling back to concatenation")
            return self._concatenate(nodes)

        parts = []
        if task is not None:
            parts.append(
                "## Original task\n\n" + render_task(task)
                + "\n\nProduce the complete deliverable, satisfying the stated "
                "constraints and acceptance criteria."
            )
        for node in nodes:
            parts.append(f"## {node.label} (id: {node.id})\n\n{node.result}")
        prompt = "\n\n---\n\n".join(parts)

        if resolved_budget:
            requires_explicit = resolved_provider.requires_explicit_budget_estimate(
                resolved_model or ""
            )
            estimate = resolved_provider.budget_estimate_usd(resolved_model or "")
            if estimate is None and requires_explicit:
                if resolved_budget.max_budget_usd is not None:
                    raise BudgetEstimateRequired(
                        "__synthesis__",
                        resolved_model or "",
                        type(resolved_provider).__name__,
                    )
            elif estimate is not None:
                resolved_budget.reserve(
                    "__synthesis__", estimate, hard_ceiling=requires_explicit,
                )
            else:
                resolved_budget.reserve(
                    "__synthesis__",
                    DEFAULT_SYNTHESIS_ESTIMATED_TOKENS
                    * resolved_budget.cost_per_token,
                )

        synth_node = Node(label="Synthesis merge", id="__synthesis__")
        if resolved_tracer:
            resolved_tracer.on_node_start(synth_node)

        try:
            result = await resolved_provider.complete(
                MERGE_SYSTEM_PROMPT, prompt, model=resolved_model
            )
            validate_completion_usage(result)
            _native_receipt(synth_node.metadata, result.native_receipt, phase="synthesis")

            if resolved_budget:
                cost = resolved_budget.add_cost("__synthesis__", result)
                synth_node.metadata["cost_usd"] = cost

            synth_node.status = NodeStatus.COMPLETED
            return result.text
        except BaseExceptionGroup as exc:
            synth_node.status = NodeStatus.FAILED
            if resolved_tracer:
                resolved_tracer.on_node_error(synth_node, exc)
            if not _native_response_errors(exc) and resolved_budget:
                resolved_budget.release("__synthesis__")
            _settle_response_group(exc, lambda error: _settle_response_error(
                error, budget=resolved_budget, node_id="__synthesis__",
                metadata=synth_node.metadata, phase="synthesis",
            ))
        except ProviderResponseError as exc:
            synth_node.status = NodeStatus.FAILED
            if resolved_tracer:
                resolved_tracer.on_node_error(synth_node, exc)
            _settle_response_error(
                exc, budget=resolved_budget, node_id="__synthesis__",
                metadata=synth_node.metadata, phase="synthesis",
            )
            raise
        except Exception as exc:
            if resolved_budget and not isinstance(exc, BudgetValidationError):
                resolved_budget.release("__synthesis__")
            synth_node.status = NodeStatus.FAILED
            if resolved_tracer:
                resolved_tracer.on_node_error(synth_node, exc)
            raise
        finally:
            if resolved_tracer:
                resolved_tracer.on_node_end(synth_node)
