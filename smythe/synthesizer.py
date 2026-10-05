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
from smythe.evidence import ClaimRecord, read_source_evidence, render_contradictions
from smythe.executor_base import check_provider_support
from smythe.graph import SYNTHESIS_NODE_ID, ExecutionGraph, Node, NodeStatus, _deliverable_ids
from smythe.provider import (
    Provider, ProviderResponseError, _native_receipt, _native_response_errors,
    _raise_if_incomplete, _settle_response_error, _settle_response_group,
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

# Appended to the merge's user prompt only by Synthesizer(retain_contradictions=True);
# MERGE_SYSTEM_PROMPT is unchanged.
CONTRADICTION_NOTE = (
    "Some inputs may carry a source-evidence block of claim records. Keep every "
    "contradiction those records list: quote each contradicting passage exactly, beside "
    "the passages it contradicts, and say which cited passage resolves the conflict if "
    "one does. Never settle a contradiction by silently choosing one side. Carry one "
    "source-evidence block that lists every claim the inputs record."
)

DEFAULT_SYNTHESIS_ESTIMATED_TOKENS = 2000


def _completed(graph: ExecutionGraph) -> list[Node]:
    """The nodes whose results synthesis reads, in graph order."""
    return [n for n in graph.nodes if n.status == NodeStatus.COMPLETED and n.result is not None]


def _select_deliverable(graph: ExecutionGraph, nodes: list[Node]) -> tuple[list[Node], bool]:
    deliverable = _deliverable_ids(graph.nodes)
    terminal = [n for n in nodes if n.id in deliverable]
    if terminal:
        return terminal, False
    return nodes, bool(nodes)


def deliverable_sources(graph: ExecutionGraph) -> tuple[list[Node], bool]:
    """The nodes ``DELIVERABLE`` synthesis returns, and whether it fell back.

    These are the completed terminal nodes that are not verifiers, in graph
    order. When there are none, synthesis falls back to every completed
    node, verifiers included, and the flag is True. A graph with no
    completed node gives ``([], False)``.
    """
    return _select_deliverable(graph, _completed(graph))


def _with_contradictions(nodes: list[Node], output: str) -> str:
    """*output* and a section for each contradiction its nodes' evidence lists and it leaves out.

    Each node's claim records come from the ``source-evidence`` block of its
    own result; a node without a readable block adds none. A contradiction is
    left out when its passage is not in the results without their blocks.
    """
    claims: list[ClaimRecord] = []
    bodies: list[str] = []
    for node in nodes:
        text = str(node.result)
        try:
            evidence, body = read_source_evidence(text)
        except ValueError:
            evidence, body = None, text
        bodies.append(body)
        if evidence is not None:
            claims.extend(evidence.claims)
    section = render_contradictions(claims, "\n\n".join(bodies))
    if not section:
        return output
    return output + ("\n" if output.endswith("\n") else "\n\n") + section


class Synthesizer:
    """Combines completed node results into a coherent final output.

    Strategies:
    - DELIVERABLE: return completed terminal outputs (default, zero cost).
    - CONCATENATE: join every completed result with newlines (zero cost).
    - LLM_MERGE: send all results to an LLM for intelligent merging.
    - STRUCTURED: parse each result as JSON and shallow-merge.

    ``retain_contradictions=True`` keeps the contradictions that source
    evidence lists. ``DELIVERABLE`` appends a ``Contradictions`` section
    for each one the delivered text leaves out
    (:func:`~smythe.evidence.render_contradictions`); ``LLM_MERGE`` appends
    ``CONTRADICTION_NOTE`` to the merge prompt.
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
        retain_contradictions: bool = False,
    ) -> None:
        if type(retain_contradictions) is not bool:
            raise TypeError("retain_contradictions must be a bool")
        if retain_contradictions and strategy not in (SynthesisStrategy.DELIVERABLE,
                                                      SynthesisStrategy.LLM_MERGE):
            raise ValueError("retain_contradictions applies to DELIVERABLE and LLM_MERGE synthesis")
        self._strategy = strategy
        self._provider = provider
        self._model = model or ""
        self._budget = budget
        self._tracer = tracer
        self._run_binding = run_binding
        self._retain_contradictions = retain_contradictions

    @property
    def strategy(self) -> SynthesisStrategy:
        """The strategy this synthesizer applies."""
        return self._strategy

    @property
    def retain_contradictions(self) -> bool:
        """Whether synthesis keeps the contradictions its inputs' source evidence lists."""
        return self._retain_contradictions

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
        description = {"type": "synthesizer", "version": 1, "strategy": self._strategy.value,
                       "completion": provider_description(provider, model) if paid else None}
        # Named only when set, so that other recipes keep their identity.
        if self._retain_contradictions:
            description["retain_contradictions"] = True
        return description

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
            retain_contradictions=self._retain_contradictions,
        )

    def _check_bound_overrides(self, provider, model, budget, tracer):
        if self._run_binding is not None and any(
            value is not None for value in (provider, model, budget, tracer)
        ):
            raise WorkflowBindingError("Bound synthesis does not accept call-time runtime overrides")

    def check_merge_support(
        self,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        max_budget_usd: float | None = None,
    ) -> None:
        """Raise now what an ``LLM_MERGE`` call would raise for this configuration.

        Resolves the provider and model as the merge does. With
        *max_budget_usd* set, a provider that requires an inclusive per-call
        ceiling and has none raises ``BudgetEstimateRequired``; a model the
        provider declares unsupported raises ``ValueError``. Other strategies
        make no call and never raise. An ordinary ``Swarm`` calls this before
        planning and before its first node call.
        """
        resolved_provider = provider or self._provider
        if (self._strategy is not SynthesisStrategy.LLM_MERGE or self._run_binding is not None
                or resolved_provider is None):
            return
        resolved_model = model if model is not None else self._model
        check_provider_support(resolved_provider, resolved_model or "", node_id=SYNTHESIS_NODE_ID,
                               max_budget_usd=max_budget_usd)

    def merge_estimate_usd(
        self, budget: Sentinel, *, provider: Provider | None = None, model: str | None = None,
    ) -> float | None:
        """What an ``LLM_MERGE`` synthesis would reserve under *budget*, or None.

        The provider's estimate for the model, else
        ``DEFAULT_SYNTHESIS_ESTIMATED_TOKENS`` at the budget's
        ``cost_per_token``. None for another strategy, without a provider, or
        when the provider requires an explicit ceiling and has none.
        """
        resolved_provider = provider or self._provider
        if self._strategy is not SynthesisStrategy.LLM_MERGE or resolved_provider is None:
            return None
        return self._merge_estimate(resolved_provider, model if model is not None else self._model,
                                    budget)[0]

    @staticmethod
    def _merge_estimate(
        provider: Provider, model: str | None, budget: Sentinel,
    ) -> tuple[float | None, bool]:
        """The merge's estimate and whether its provider requires an explicit one."""
        requires_explicit = provider.requires_explicit_budget_estimate(model or "")
        estimate = provider.budget_estimate_usd(model or "")
        if estimate is None and not requires_explicit:
            estimate = DEFAULT_SYNTHESIS_ESTIMATED_TOKENS * budget.cost_per_token
        return estimate, requires_explicit

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
        completed = _completed(graph)
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
        completed = _completed(graph)
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

    def _deliverable(self, graph: ExecutionGraph, nodes: list[Node]) -> str:
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
        :func:`deliverable_sources` names the same nodes. With
        ``retain_contradictions``, the contradictions those nodes' source
        evidence lists and their text leaves out follow in a section.
        """
        selected = _select_deliverable(graph, nodes)[0]
        output = Synthesizer._concatenate(selected)
        return _with_contradictions(selected, output) if self._retain_contradictions else output

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
        if self._retain_contradictions:
            parts.append(CONTRADICTION_NOTE)
        prompt = "\n\n---\n\n".join(parts)

        if resolved_budget:
            estimate, requires_explicit = self._merge_estimate(
                resolved_provider, resolved_model, resolved_budget,
            )
            if estimate is None:
                if resolved_budget.max_budget_usd is not None:
                    raise BudgetEstimateRequired(
                        SYNTHESIS_NODE_ID,
                        resolved_model or "",
                        type(resolved_provider).__name__,
                    )
            else:
                # Allowance held for the merge (Swarm reserve_for_checks)
                # becomes its reservation.
                resolved_budget.reserve(
                    SYNTHESIS_NODE_ID, estimate, hard_ceiling=requires_explicit,
                    from_hold=SYNTHESIS_NODE_ID in resolved_budget.holds,
                )

        synth_node = Node(label="Synthesis merge", id=SYNTHESIS_NODE_ID)
        if resolved_tracer:
            resolved_tracer.on_node_start(synth_node, phase="synthesis")

        try:
            result = await resolved_provider.complete(
                MERGE_SYSTEM_PROMPT, prompt, model=resolved_model
            )
            validate_completion_usage(result)
            _native_receipt(synth_node.metadata, result.native_receipt, phase="synthesis")

            if resolved_budget:
                cost = resolved_budget.add_cost(SYNTHESIS_NODE_ID, result)
                synth_node.metadata["cost_usd"] = cost
            # Checked after billing so an incomplete merge keeps its charge.
            _raise_if_incomplete(result, where="Synthesis")

            synth_node.status = NodeStatus.COMPLETED
            return result.text
        except BaseExceptionGroup as exc:
            synth_node.status = NodeStatus.FAILED
            if resolved_tracer:
                resolved_tracer.on_node_error(synth_node, exc)
            if not _native_response_errors(exc) and resolved_budget:
                resolved_budget.release(SYNTHESIS_NODE_ID)
            _settle_response_group(exc, lambda error: _settle_response_error(
                error, budget=resolved_budget, node_id=SYNTHESIS_NODE_ID,
                metadata=synth_node.metadata, phase="synthesis",
            ))
        except ProviderResponseError as exc:
            synth_node.status = NodeStatus.FAILED
            if resolved_tracer:
                resolved_tracer.on_node_error(synth_node, exc)
            _settle_response_error(
                exc, budget=resolved_budget, node_id=SYNTHESIS_NODE_ID,
                metadata=synth_node.metadata, phase="synthesis",
            )
            raise
        except Exception as exc:
            if resolved_budget and not isinstance(exc, BudgetValidationError):
                resolved_budget.release(SYNTHESIS_NODE_ID)
            synth_node.status = NodeStatus.FAILED
            if resolved_tracer:
                resolved_tracer.on_node_error(synth_node, exc)
            raise
        finally:
            if resolved_tracer:
                resolved_tracer.on_node_end(synth_node)
