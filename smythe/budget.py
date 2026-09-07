"""Sentinel — deterministic cost guardrails for execution.

The Sentinels patrol the boundaries.  They enforce the budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import chain
import math
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from smythe.provider import CompletionResult


class BudgetValidationError(ValueError):
    """Invalid monetary or usage data was rejected without changing the ledger."""


def _validate_usd(value: object, name: str) -> float:
    """Accept numeric USD only; never coerce booleans or numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetValidationError(f"{name} must be a finite non-negative number")
    try:
        amount = float(value)
    except OverflowError as exc:
        raise BudgetValidationError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(amount) or amount < 0:
        raise BudgetValidationError(f"{name} must be a finite non-negative number")
    return amount


def validate_token_count(value: object, name: str) -> int:
    """Validate usage or a pre-call token estimate without coercing its type."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BudgetValidationError(f"{name} must be a non-negative integer")
    return value


def validate_completion_usage(result: CompletionResult) -> tuple[int, float | None]:
    """Return validated total tokens and explicit USD, if supplied.

    Token usage remains subject to validation when explicit or unknown USD
    takes precedence over token pricing. Rechecking here also catches a
    CompletionResult mutated after construction by a custom provider.
    """
    prompt = validate_token_count(getattr(result, "prompt_tokens", None), "prompt_tokens")
    completion = validate_token_count(
        getattr(result, "completion_tokens", None), "completion_tokens"
    )
    explicit = getattr(result, "cost_usd", None)
    if explicit is not None:
        explicit = _validate_usd(explicit, "cost_usd")
    return prompt + completion, explicit


def _sum_costs(values: Iterable[float]) -> float:
    """Reject unrepresentable aggregates before they poison a finite ledger."""
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise BudgetValidationError("Total cost must remain finite") from exc
    return _validate_usd(total, "Total cost")


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
        self.max_budget_usd = max_budget_usd
        self.cost_per_token = cost_per_token
        self._node_costs: dict[str, float] = {}
        self._reservations: dict[str, float] = {}
        self._hard_reservations: set[str] = set()
        self._spent: float = 0.0
        self._unknown_cost_nodes: set[str] = set()
        self._estimated_cost_nodes: set[str] = set()

    @property
    def max_budget_usd(self) -> float | None:
        return self._max_budget_usd

    @max_budget_usd.setter
    def max_budget_usd(self, value: float | None) -> None:
        self._max_budget_usd = (
            None if value is None else _validate_usd(value, "max_budget_usd")
        )

    @property
    def cost_per_token(self) -> float:
        return self._cost_per_token

    @cost_per_token.setter
    def cost_per_token(self, value: float) -> None:
        self._cost_per_token = _validate_usd(value, "cost_per_token")

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

    def mark_unknown(self, node_id: str) -> None:
        """Mark unresolved billing without inventing usage or releasing exposure."""
        self._unknown_cost_nodes.add(node_id)

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
        estimated_cost = _validate_usd(estimated_cost, "estimated_cost")
        if node_id in self._reservations:
            raise ValueError(f"Node {node_id!r} already has a budget reservation")
        total = _sum_costs(chain(
            self._node_costs.values(), self._reservations.values(), (estimated_cost,)
        ))
        if self.max_budget_usd is not None:
            # A nano-dollar tolerance so an exact-fit budget (for example
            # 192 reservations of $0.06 against an $11.52 limit) is not
            # rejected by accumulated floating-point error. Observed live:
            # the 192nd reservation of a full-budget image run failed with
            # $0.06 nominally remaining.
            if total > self.max_budget_usd + 1e-9:
                raise SentinelAlert(self._spent, self.max_budget_usd, node_id)
        self._reservations[node_id] = estimated_cost
        if hard_ceiling:
            self._hard_reservations.add(node_id)
        self._spent = total

    def release(self, node_id: str) -> None:
        """Cancel a reservation (e.g. on node failure before record)."""
        reservations = dict(self._reservations)
        reservations.pop(node_id, None)
        total = _sum_costs(chain(self._node_costs.values(), reservations.values()))
        self._reservations = reservations
        self._hard_reservations.discard(node_id)
        self._spent = total

    def _cost_of(self, result: CompletionResult) -> float:
        """Price a provider call: explicit provider cost wins over token math.

        Providers set ``cost_usd`` when they can price the call better
        than a blended token rate — per-image billing, for instance.
        """
        total_tokens, explicit = validate_completion_usage(result)
        if explicit is not None:
            return explicit
        return self._token_cost(total_tokens)

    def _token_cost(self, total_tokens: int) -> float:
        if self.cost_per_token == 0:
            return 0.0
        try:
            cost = total_tokens * self.cost_per_token
        except OverflowError as exc:
            raise BudgetValidationError("Token-derived cost must remain finite") from exc
        return _validate_usd(cost, "Token-derived cost")

    def _reconcile(
        self,
        node_id: str,
        result: CompletionResult,
        *,
        accumulate: bool,
        preserve_reservation: bool = False,
    ) -> float:
        """Replace a reservation with reconciled cost and validate the ceiling.

        Reconciliation is deliberately truthful: if a provider reports an
        overrun, the actual charge is retained in the breakdown before the
        error is raised.  Hiding or clamping an already-incurred charge would
        make checkpoints and user-visible totals unsafe.
        """
        # Validate and calculate the complete replacement before touching any
        # reservation, cost, or completeness flag. Invalid reporting must not
        # release a held ceiling or erase an earlier valid charge.
        total_tokens, explicit = validate_completion_usage(result)
        had_reservation = node_id in self._reservations and not preserve_reservation
        reserved = self._reservations.get(node_id, 0.0) if had_reservation else 0.0
        was_hard_ceiling = node_id in self._hard_reservations and not preserve_reservation
        unknown_cost_nodes = set(self._unknown_cost_nodes)
        estimated_cost_nodes = set(self._estimated_cost_nodes)

        if getattr(result, "cost_usd_unknown", False):
            if had_reservation:
                # An explicit reservation is the only defensible number when
                # a provider cannot return complete billable USD. Retain at
                # least that ceiling, but never hide a larger reported partial
                # charge behind a smaller reservation.
                reported_partial = explicit if explicit is not None else 0.0
                call_cost = max(reserved, reported_partial)
                estimated_cost_nodes.add(node_id)
            else:
                # A provider may expose a useful partial/output-only estimate
                # while still marking the full invoice cost incomplete.
                call_cost = explicit if explicit is not None else 0.0
                unknown_cost_nodes.add(node_id)
                if getattr(result, "cost_usd_is_estimate", False):
                    estimated_cost_nodes.add(node_id)
        else:
            call_cost = explicit if explicit is not None else self._token_cost(total_tokens)
            if getattr(result, "cost_usd_is_estimate", False):
                estimated_cost_nodes.add(node_id)
        node_cost = call_cost
        if accumulate:
            node_cost = _sum_costs((self._node_costs.get(node_id, 0.0), call_cost))

        node_costs = dict(self._node_costs)
        node_costs[node_id] = node_cost
        reservations = dict(self._reservations)
        if not preserve_reservation:
            reservations.pop(node_id, None)
        total = _sum_costs(chain(node_costs.values(), reservations.values()))

        self._node_costs = node_costs
        self._reservations = reservations
        if not preserve_reservation:
            self._hard_reservations.discard(node_id)
        self._unknown_cost_nodes = unknown_cost_nodes
        self._estimated_cost_nodes = estimated_cost_nodes
        self._spent = total

        reservation_exceeded = was_hard_ceiling and call_cost > reserved + 1e-12
        # Matches the reserve() admission tolerance: an exact-fit budget may
        # carry up to a nano-dollar of accumulated float error without that
        # noise being reported as a genuine overrun.
        limit_exceeded = (
            self.max_budget_usd is not None
            and self._spent > self.max_budget_usd + 1e-9
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

    def add_cost(
        self, node_id: str, result: CompletionResult, *, preserve_reservation: bool = False,
    ) -> float:
        """Accumulate cost for a node across multiple provider calls.

        The first call for a node also releases any outstanding
        reservation (the estimate is superseded by actuals).  Returns
        the node's cumulative cost. ``preserve_reservation`` records a separate
        known call without consuming a ceiling held for unresolved billing.
        """
        return self._reconcile(
            node_id, result, accumulate=True, preserve_reservation=preserve_reservation,
        )

    def breakdown(self) -> dict[str, float]:
        """Per-node cost map in USD."""
        return dict(self._node_costs)

    def restore(
        self,
        node_costs: Mapping[str, float],
        *,
        unknown_cost_nodes: set[str] | None = None,
        estimated_cost_nodes: set[str] | None = None,
    ) -> None:
        """Seed per-node costs from a checkpoint so a resumed execution
        keeps counting against the same budget."""
        if not isinstance(node_costs, Mapping):
            raise BudgetValidationError("Checkpoint costs must be a mapping of node IDs to USD")
        restored = dict(self._node_costs)
        for node_id, cost in node_costs.items():
            restored[node_id] = _validate_usd(cost, f"Checkpoint cost for node {node_id!r}")
        total = _sum_costs(chain(restored.values(), self._reservations.values()))
        unknown = self._unknown_cost_nodes.union(unknown_cost_nodes or set())
        estimated = self._estimated_cost_nodes.union(estimated_cost_nodes or set())
        self._node_costs = restored
        self._spent = total
        self._unknown_cost_nodes = unknown
        self._estimated_cost_nodes = estimated
