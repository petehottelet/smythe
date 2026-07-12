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


class BudgetEstimateRequired(ValueError):
    """Raised before a priced call whose cost cannot be bounded safely.

    Image APIs charge on dimensions, quality, inputs, and outputs.  A generic
    token estimate is not a defensible reservation for them, so hard-budget
    execution fails closed until the caller supplies an inclusive per-call
    ceiling on the provider or node.
    """

    def __init__(self, node_id: str, model: str, provider: str) -> None:
        self.node_id = node_id
        self.model = model
        self.provider = provider
        super().__init__(
            f"Cannot enforce max_budget_usd for node {node_id!r}: "
            f"{provider} has no inclusive cost ceiling for model {model!r}. "
            "Configure max_cost_per_call_usd on the image provider or set "
            "node.metadata['estimated_cost_usd'] to a conservative upper bound."
        )


class BudgetReconciliationError(SentinelAlert):
    """Raised when a completed call reports more cost than was reserved.

    The cost has already been incurred and remains in the Sentinel breakdown;
    this exception stops the workflow immediately so it cannot compound an
    inaccurate ceiling with additional calls.
    """

    def __init__(
        self,
        *,
        spent: float,
        limit: float,
        node_id: str,
        actual_cost: float,
        reserved_cost: float | None,
    ) -> None:
        self.spent = spent
        self.limit = limit
        self.node_id = node_id
        self.actual_cost = actual_cost
        self.reserved_cost = reserved_cost
        if reserved_cost is None:
            detail = "the remaining budget"
        else:
            detail = f"its ${reserved_cost:.4f} reservation"
        Exception.__init__(
            self,
            f"Budget reconciliation failed for node {node_id!r}: "
            f"actual cost ${actual_cost:.4f} exceeded {detail}; "
            f"${spent:.4f} is now recorded against the ${limit:.4f} limit",
        )


class Sentinel:
    """Accumulates per-node costs and enforces a USD admission policy.

    Supports a reservation protocol for parallel execution: ``reserve()``
    pre-commits estimated cost so concurrent admissions cannot collectively
    exceed the budget. ``record()`` reconciles the reservation with the
    provider-reported cost or configured estimate once the node completes.
    ``release()`` frees a reservation if the node fails before ``record()``.

    Attributes:
        max_budget_usd: Hard cap in USD.  None means unlimited.
        cost_per_token: Blended $/token rate (default ~$3/1M tokens).
    """

    def __init__(
        self,
        max_budget_usd: float | None = None,
        cost_per_token: float = 0.000003,
    ) -> None:
        if max_budget_usd is not None and max_budget_usd < 0:
            raise ValueError(
                f"max_budget_usd must be non-negative, got {max_budget_usd}"
            )
        if cost_per_token < 0:
            raise ValueError(
                f"cost_per_token must be non-negative, got {cost_per_token}"
            )
        self.max_budget_usd = max_budget_usd
        self.cost_per_token = cost_per_token
        self._node_costs: dict[str, float] = {}
        self._reservations: dict[str, float] = {}
        self._hard_reservations: set[str] = set()
        self._spent: float = 0.0
        self._unknown_cost_nodes: set[str] = set()
        self._estimated_cost_nodes: set[str] = set()

    @property
    def total_cost_usd(self) -> float:
        return self._spent

    @property
    def cost_is_complete(self) -> bool:
        """Whether every recorded call had a USD cost or safe ceiling."""
        return not self._unknown_cost_nodes

    @property
    def cost_contains_estimates(self) -> bool:
        """Whether the total contains provider/configured cost estimates."""
        return bool(self._estimated_cost_nodes)

    def check(self, node_id: str) -> None:
        """Raise SentinelAlert if the budget is already exhausted.

        Used by the serial executor where reservation is unnecessary.
        """
        if self.max_budget_usd is not None and self._spent >= self.max_budget_usd:
            raise SentinelAlert(self._spent, self.max_budget_usd, node_id)

    def reserve(
        self,
        node_id: str,
        estimated_cost: float,
        *,
        hard_ceiling: bool = False,
    ) -> None:
        """Pre-commit estimated cost before a node starts executing.

        Raises SentinelAlert if the reservation would exceed the
        budget.  The reservation is held until ``record()`` replaces it
        with actual cost, or ``release()`` cancels it on failure.
        """
        if estimated_cost < 0:
            raise ValueError(
                f"estimated_cost must be non-negative, got {estimated_cost}"
            )
        if node_id in self._reservations:
            raise ValueError(f"Node {node_id!r} already has a budget reservation")
        if self.max_budget_usd is not None:
            if self._spent + estimated_cost > self.max_budget_usd:
                raise SentinelAlert(self._spent, self.max_budget_usd, node_id)
        self._reservations[node_id] = estimated_cost
        if hard_ceiling:
            self._hard_reservations.add(node_id)
        self._spent += estimated_cost

    def release(self, node_id: str) -> None:
        """Cancel a reservation (e.g. on node failure before record)."""
        reserved = self._reservations.pop(node_id, 0.0)
        self._hard_reservations.discard(node_id)
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

    def _reconcile(
        self,
        node_id: str,
        result: CompletionResult,
        *,
        accumulate: bool,
    ) -> float:
        """Replace a reservation with reconciled cost and validate the ceiling.

        Reconciliation is deliberately truthful: if a provider reports an
        overrun, the actual charge is retained in the breakdown before the
        error is raised.  Hiding or clamping an already-incurred charge would
        make checkpoints and user-visible totals unsafe.
        """
        had_reservation = node_id in self._reservations
        reserved = self._reservations.pop(node_id, 0.0)
        was_hard_ceiling = node_id in self._hard_reservations
        self._hard_reservations.discard(node_id)
        self._spent -= reserved

        if getattr(result, "cost_usd_unknown", False):
            if had_reservation:
                # An explicit reservation is the only defensible number when
                # a provider cannot return complete billable USD. Retain at
                # least that ceiling, but never hide a larger reported partial
                # charge behind a smaller reservation.
                explicit = getattr(result, "cost_usd", None)
                reported_partial = (
                    max(0.0, explicit) if explicit is not None else 0.0
                )
                call_cost = max(reserved, reported_partial)
                self._estimated_cost_nodes.add(node_id)
            else:
                # A provider may expose a useful partial/output-only estimate
                # while still marking the full invoice cost incomplete.
                explicit = getattr(result, "cost_usd", None)
                call_cost = max(0.0, explicit) if explicit is not None else 0.0
                self._unknown_cost_nodes.add(node_id)
                if getattr(result, "cost_usd_is_estimate", False):
                    self._estimated_cost_nodes.add(node_id)
        else:
            call_cost = self._cost_of(result)
            if getattr(result, "cost_usd_is_estimate", False):
                self._estimated_cost_nodes.add(node_id)
        node_cost = call_cost
        if accumulate:
            node_cost += self._node_costs.get(node_id, 0.0)
        else:
            self._spent -= self._node_costs.get(node_id, 0.0)

        self._node_costs[node_id] = node_cost
        self._spent += call_cost

        reservation_exceeded = was_hard_ceiling and call_cost > reserved + 1e-12
        limit_exceeded = (
            self.max_budget_usd is not None
            and self._spent > self.max_budget_usd + 1e-12
        )
        if (reservation_exceeded or limit_exceeded) and self.max_budget_usd is not None:
            raise BudgetReconciliationError(
                spent=self._spent,
                limit=self.max_budget_usd,
                node_id=node_id,
                actual_cost=call_cost,
                reserved_cost=reserved if had_reservation else None,
            )
        return node_cost

    def record(self, node_id: str, result: CompletionResult) -> float:
        """Record reconciled cost, replacing any outstanding reservation.

        Overwrites the node's cost — use add_cost() for multi-call
        nodes (tool loops), where costs must accumulate.
        """
        return self._reconcile(node_id, result, accumulate=False)

    def add_cost(self, node_id: str, result: CompletionResult) -> float:
        """Accumulate cost for a node across multiple provider calls.

        The first call for a node also releases any outstanding
        reservation (the estimate is superseded by actuals).  Returns
        the node's cumulative cost.
        """
        return self._reconcile(node_id, result, accumulate=True)

    def breakdown(self) -> dict[str, float]:
        """Per-node cost map in USD."""
        return dict(self._node_costs)

    def restore(
        self,
        node_costs: dict[str, float],
        *,
        unknown_cost_nodes: set[str] | None = None,
        estimated_cost_nodes: set[str] | None = None,
    ) -> None:
        """Seed per-node costs from a checkpoint so a resumed execution
        keeps counting against the same budget."""
        for node_id, cost in node_costs.items():
            if cost < 0:
                raise ValueError(
                    f"Checkpoint cost for node {node_id!r} must be non-negative, "
                    f"got {cost}"
                )
            self._node_costs[node_id] = cost
        self._spent = sum(self._node_costs.values()) + sum(self._reservations.values())
        self._unknown_cost_nodes.update(unknown_cost_nodes or set())
        self._estimated_cost_nodes.update(estimated_cost_nodes or set())
