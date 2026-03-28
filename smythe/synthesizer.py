"""Synthesizer — merges outputs from parallel execution branches."""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from typing import Any

from smythe.budget import Sentinel
from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.provider import Provider
from smythe.tracer import Tracer

logger = logging.getLogger("smythe.synthesizer")


class SynthesisStrategy(Enum):
    CONCATENATE = "concatenate"
    LLM_MERGE = "llm_merge"
    STRUCTURED = "structured"


MERGE_SYSTEM_PROMPT = """\
You are a synthesis agent.  You receive outputs from multiple parallel \
execution steps.  Your job is to merge them into a single coherent output.  \
Preserve key information from every input.  Be concise and well-organized.
"""


class Synthesizer:
    """Combines completed node results into a coherent final output.

    Strategies:
    - CONCATENATE: join results with newlines (default, zero cost).
    - LLM_MERGE: send all results to an LLM for intelligent merging.
    - STRUCTURED: parse each result as JSON and shallow-merge.
    """

    def __init__(
        self,
        strategy: SynthesisStrategy = SynthesisStrategy.CONCATENATE,
        provider: Provider | None = None,
        model: str | None = None,
        budget: Sentinel | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._strategy = strategy
        self._provider = provider
        self._model = model or ""
        self._budget = budget
        self._tracer = tracer

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
        completed = [
            n for n in graph.nodes
            if n.status == NodeStatus.COMPLETED and n.result is not None
        ]
        if not completed:
            return ""

        if self._strategy == SynthesisStrategy.CONCATENATE:
            return self._concatenate(completed)
        if self._strategy == SynthesisStrategy.STRUCTURED:
            return self._structured_merge(completed)
        if self._strategy == SynthesisStrategy.LLM_MERGE:
            return asyncio.run(
                self._llm_merge(
                    completed,
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
        completed = [
            n for n in graph.nodes
            if n.status == NodeStatus.COMPLETED and n.result is not None
        ]
        if not completed:
            return ""

        if self._strategy == SynthesisStrategy.CONCATENATE:
            return self._concatenate(completed)
        if self._strategy == SynthesisStrategy.STRUCTURED:
            return self._structured_merge(completed)
        if self._strategy == SynthesisStrategy.LLM_MERGE:
            return await self._llm_merge(
                completed,
                provider=provider,
                model=model,
                budget=budget,
                tracer=tracer,
            )
        return self._concatenate(completed)

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

        if resolved_provider is None:
            logger.warning("LLM_MERGE requested but no provider set; falling back to concatenation")
            return self._concatenate(nodes)

        parts = []
        for node in nodes:
            parts.append(f"## {node.label} (id: {node.id})\n\n{node.result}")
        prompt = "\n\n---\n\n".join(parts)

        if resolved_budget:
            resolved_budget.check("__synthesis__")

        synth_node = Node(label="Synthesis merge", id="__synthesis__")
        if resolved_tracer:
            resolved_tracer.on_node_start(synth_node)

        try:
            result = await resolved_provider.complete(
                MERGE_SYSTEM_PROMPT, prompt, model=resolved_model
            )

            if resolved_budget:
                cost = resolved_budget.record("__synthesis__", result)
                synth_node.metadata["cost_usd"] = cost

            synth_node.status = NodeStatus.COMPLETED
            return result.text
        except Exception as exc:
            synth_node.status = NodeStatus.FAILED
            if resolved_tracer:
                resolved_tracer.on_node_error(synth_node, exc)
            raise
        finally:
            if resolved_tracer:
                resolved_tracer.on_node_end(synth_node)
