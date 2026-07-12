"""Sentinel — deterministic cost guardrails for execution.

The Sentinels patrol the boundaries.  They enforce the budget.
"""

from __future__ import annotations

from smythe.provider import CompletionResult


class SentinelAlert(Exception):
    """Raised when cumulative execution cost would exceed the budget limit."""

    def __init__(self, spent: float, limit: float, node_id: str) -> None:
        self.spent = spent
        self.limit = limit
        self.node_id = node_id
        super().__init__(
            f"Budget exhausted before node {node_id!r}: "
            f"${spent:.4f} spent of ${limit:.4f} limit"
        )


class Sentinel:
    """Accumulates token costs per node and enforces a USD spending cap.

    Supports a reservation protocol for parallel execution: ``reserve()``
    pre-commits estimated cost so concurrent nodes cannot collectively
    exceed the budget.  ``record()`` reconciles the reservation with the
    actual cost once the node completes.  ``release()`` frees a reservation
    if the node fails before ``record()`` is called.

    Attributes:
        max_budget_usd: Hard cap in USD.  None means unlimited.
        cost_per_token: Blended $/token rate (default ~$3/1M tokens).
    """

    def __init__(
        self,
        max_budget_usd: float | None = None,
        cost_per_token: float = 0.000003,
    ) -> None:
        self.max_budget_usd = max_budget_usd
        self.cost_per_token = cost_per_token
        self._node_costs: dict[str, float] = {}
        self._reservations: dict[str, float] = {}
        self._spent: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self._spent

    def check(self, node_id: str) -> None:
        """Raise SentinelAlert if the budget is already exhausted.

        Used by the serial executor where reservation is unnecessary.
        """
        if self.max_budget_usd is not None and self._spent >= self.max_budget_usd:
            raise SentinelAlert(self._spent, self.max_budget_usd, node_id)

    def reserve(self, node_id: str, estimated_cost: float) -> None:
        """Pre-commit estimated cost before a node starts executing.

        Raises SentinelAlert if the reservation would exceed the
        budget.  The reservation is held until ``record()`` replaces it
        with actual cost, or ``release()`` cancels it on failure.
        """
        if self.max_budget_usd is not None:
            if self._spent + estimated_cost > self.max_budget_usd:
                raise SentinelAlert(self._spent, self.max_budget_usd, node_id)
        self._reservations[node_id] = estimated_cost
        self._spent += estimated_cost

    def release(self, node_id: str) -> None:
        """Cancel a reservation (e.g. on node failure before record)."""
        reserved = self._reservations.pop(node_id, 0.0)
        self._spent -= reserved

    def _cost_of(self, result: CompletionResult) -> float:
        """Price a provider call: explicit provider cost wins over token math.

        Providers set ``cost_usd`` when they can price the call better
        than a blended token rate — per-image billing, for instance.
        """
        explicit = getattr(result, "cost_usd", None)
        if explicit is not None:
            # Clamp: a buggy provider must never "refund" the budget.
            return max(0.0, explicit)
        return result.total_tokens * self.cost_per_token

    def record(self, node_id: str, result: CompletionResult) -> float:
        """Record the actual cost, replacing any outstanding reservation.

        Overwrites the node's cost — use add_cost() for multi-call
        nodes (tool loops), where costs must accumulate.
        """
        reserved = self._reservations.pop(node_id, 0.0)
        self._spent -= reserved
        cost = self._cost_of(result)
        self._node_costs[node_id] = cost
        self._spent += cost
        return cost

    def add_cost(self, node_id: str, result: CompletionResult) -> float:
        """Accumulate cost for a node across multiple provider calls.

        The first call for a node also releases any outstanding
        reservation (the estimate is superseded by actuals).  Returns
        the node's cumulative cost.
        """
        reserved = self._reservations.pop(node_id, 0.0)
        self._spent -= reserved
        cost = self._cost_of(result)
        self._node_costs[node_id] = self._node_costs.get(node_id, 0.0) + cost
        self._spent += cost
        return self._node_costs[node_id]

    def breakdown(self) -> dict[str, float]:
        """Per-node cost map in USD."""
        return dict(self._node_costs)

    def restore(self, node_costs: dict[str, float]) -> None:
        """Seed per-node costs from a checkpoint so a resumed execution
        keeps counting against the same budget."""
        for node_id, cost in node_costs.items():
            self._node_costs[node_id] = cost
        self._spent = sum(self._node_costs.values()) + sum(self._reservations.values())
