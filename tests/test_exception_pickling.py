"""Every smythe exception survives pickle and copy with its message and fields.

``BaseException`` rebuilds an exception by calling its class with ``args``. A
constructor whose parameters differ from ``args`` then raised TypeError on
unpickle, or wrapped its message twice. Exceptions are pickled whenever they
cross a process boundary, for example from a ``ProcessPoolExecutor``.
"""

import copy
import importlib
import inspect
import pickle
import pkgutil

import pytest

import smythe
from smythe.assets.models import AssetPreflightError
from smythe.budget import BudgetEstimateRequired, BudgetReconciliationError, SentinelAlert
from smythe.executor_base import NodeFinalizationError
from smythe.jobs.operator import WorkerStartupError, WorkerStartupInterrupted
from smythe.jobs.providers import ProviderPreflightError, ProviderPreflightKind
from smythe.provider import (
    CompletionResult, OutputRefusedError, OutputTruncatedError, ProviderAccountingCancelledError,
    ProviderAccountingError, ProviderResponseError,
)
from smythe.provider_responses import PreparedRequest, RawResponseEnvelope, ResponseQuoteError
from smythe.workflow_provider import ProviderRequestRejectedError, WorkflowQuoteError


def _exception_classes():
    """Every exception class defined at module level anywhere in the package."""
    found = set()
    for info in pkgutil.walk_packages(smythe.__path__, "smythe."):
        module = importlib.import_module(info.name)
        found.update(
            value for value in vars(module).values()
            if inspect.isclass(value) and issubclass(value, BaseException)
            and value.__module__ == module.__name__
        )
    return sorted(found, key=_name)


def _name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


EXCEPTION_CLASSES = _exception_classes()

_ENVELOPE = RawResponseEnvelope(
    PreparedRequest('{"model":"gpt-6-astra"}'), b'{"error":{"type":"rate_limit_error"}}',
    status_code=429, request_id="req_1",
)
_LAUNCH = {"run_id": "run-1", "log_path": "/runs/run-1/worker.log", "startup_authorized": True}

# One instance of each class whose constructor differs from BaseException's.
# A new exception with its own constructor fails the inventory test below
# until it has an entry here.
CUSTOM_CONSTRUCTED = {
    AssetPreflightError: lambda: AssetPreflightError(["logo is missing", "size is too large"]),
    SentinelAlert: lambda: SentinelAlert(1.5, 1.0, "draft"),
    BudgetEstimateRequired: lambda: BudgetEstimateRequired(
        "draft", "gpt-image-2", "OpenAIImageProvider",
    ),
    BudgetReconciliationError: lambda: BudgetReconciliationError(
        spent=1.5, limit=1.0, node_id="draft", actual_cost=0.7, reserved_cost=0.5,
    ),
    NodeFinalizationError: lambda: NodeFinalizationError("draft", OSError(28, "disk full")),
    WorkerStartupError: lambda: WorkerStartupError("Worker did not start", _LAUNCH),
    WorkerStartupInterrupted: lambda: WorkerStartupInterrupted(_LAUNCH),
    ProviderPreflightError: lambda: ProviderPreflightError(
        ProviderPreflightKind.KEY, "OPENAI_API_KEY is not set",
    ),
    OutputTruncatedError: lambda: OutputTruncatedError("max_tokens", where="Node 'report'"),
    OutputRefusedError: lambda: OutputRefusedError("content_filter", where="Synthesis"),
    ProviderResponseError: lambda: ProviderResponseError(
        "unusable output", envelope=_ENVELOPE, receipt={"cost_nanousd": "5"},
        billing_result=CompletionResult("", prompt_tokens=3, completion_tokens=4, cost_usd=0.5),
    ),
    ProviderAccountingError: lambda: ProviderAccountingError("unpriced", envelope=_ENVELOPE),
    ProviderAccountingCancelledError: lambda: ProviderAccountingCancelledError(
        "cancelled after dispatch", envelope=_ENVELOPE,
    ),
    ResponseQuoteError: lambda: ResponseQuoteError("Input counting failed", _ENVELOPE),
    WorkflowQuoteError: lambda: WorkflowQuoteError("Managed quote failed", _ENVELOPE),
    ProviderRequestRejectedError: lambda: ProviderRequestRejectedError(
        "HTTP 429", status_code=429, envelope=_ENVELOPE, receipt={"cost_nanousd": "0"},
    ),
}

ROUND_TRIPS = {
    **{f"pickle-{protocol}": (lambda error, protocol=protocol:
                              pickle.loads(pickle.dumps(error, protocol=protocol)))
       for protocol in range(pickle.HIGHEST_PROTOCOL + 1)},
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
}


def _has_own_constructor(cls):
    return any("__init__" in vars(base) for base in cls.__mro__
               if base.__module__.startswith("smythe"))


def test_every_exception_with_its_own_constructor_has_a_sample():
    assert OutputTruncatedError in EXCEPTION_CLASSES  # discovery reached the package
    assert {cls for cls in EXCEPTION_CLASSES if _has_own_constructor(cls)} == set(CUSTOM_CONSTRUCTED)


@pytest.mark.parametrize("how", ROUND_TRIPS)
@pytest.mark.parametrize("cls", EXCEPTION_CLASSES, ids=_name)
def test_exception_survives_pickle_and_copy(cls, how):
    make = CUSTOM_CONSTRUCTED.get(cls, lambda: cls("something went wrong"))
    original = make()
    original.add_note("added after construction")

    restored = ROUND_TRIPS[how](original)

    assert type(restored) is cls
    assert restored.args == original.args
    assert str(restored) == str(original)
    assert vars(restored).keys() == vars(original).keys()
    for name, value in vars(original).items():
        other = vars(restored)[name]
        if isinstance(value, BaseException):
            assert (type(other), str(other)) == (type(value), str(value))
        else:
            assert other == value
