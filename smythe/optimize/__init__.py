"""Public API for bounded, evidence-gated Autotune campaigns."""

from smythe.optimize.concurrency import ConcurrencyScenario, simulate_concurrency

from smythe.optimize.contracts import (
    CONTRACT_VERSION,
    Candidate,
    ContractValidationError,
    ExperimentContract,
    MetricObjective,
    MutableFieldRule,
    MutableValueType,
    ObjectiveDirection,
    canonical_json_bytes,
    normalize_gate_name,
    sha256_prefixed,
)
from smythe.optimize.engine import (
    OptimizationError,
    OptimizationLimitError,
    OptimizationNeedsAttention,
    OptimizationResult,
    OptimizationRunner,
    TrialContext,
    TrialOutcome,
)
from smythe.optimize.ledger import (
    ExperimentLedger,
    PromotionDecision,
    TrialRecord,
    TrialStatus,
)
from smythe.optimize.statistics import PromotionAssessment, assess_promotion

__all__ = [
    "CONTRACT_VERSION",
    "Candidate",
    "ConcurrencyScenario",
    "ContractValidationError",
    "ExperimentLedger",
    "ExperimentContract",
    "MetricObjective",
    "MutableFieldRule",
    "MutableValueType",
    "ObjectiveDirection",
    "OptimizationError",
    "OptimizationLimitError",
    "OptimizationNeedsAttention",
    "OptimizationResult",
    "OptimizationRunner",
    "PromotionAssessment",
    "PromotionDecision",
    "TrialContext",
    "TrialOutcome",
    "TrialRecord",
    "TrialStatus",
    "assess_promotion",
    "canonical_json_bytes",
    "normalize_gate_name",
    "sha256_prefixed",
    "simulate_concurrency",
]
