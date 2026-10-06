"""Failure classes: how a node attempt failed, named in one place.

``classify_failure`` gives every exception a node attempt can raise one of the
classes below, and ``is_terminal`` says whether the executors stop the run at
once instead of applying the node's failure policy. Both executors take that
decision here. A class describes a failure and does not by itself decide a
retry: ``RETRY`` retries every exception that is not terminal, whatever its
class, unless a ``RetryPolicy`` lists the classes it retries.

The exception types are imported on first use, so any module can import this
one without an import cycle.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from smythe.workflow_provider import ProviderRequestRejectedError

# The request is wrong: a 4xx status other than those below, invalid read
# input, a missing cost ceiling, a model profile the run does not allow or a
# dependency file that changed.
INPUT_ERROR = "input_error"
# The credentials or account may not make the request: HTTP 401 or 403.
AUTHORITY_DENIED = "authority_denied"
# The model's work fell short: it did not finish within max_tool_iterations,
# or, under a retry policy that lists this class, a native reply with a known
# charge was refused or truncated.
QUALITY_DEFECT = "quality_defect"
# The provider answered and the charge is known, but the response cannot be
# used: a refusal, a truncation or an undecodable reply.
PROVIDER_REJECTED = "provider_rejected"
# Another attempt may succeed: rate limits, server errors, timeouts and any
# exception no other class names.
TRANSIENT = "transient"
# A dispatched call's charge is unresolved.
UNKNOWN_OUTCOME = "unknown_outcome"
# Usage or cost data is invalid, or the budget is exhausted or overrun.
TERMINAL_ACCOUNTING = "terminal_accounting"
# A billed result or verification control could not be saved or recovered.
PERSISTENCE = "persistence"
# A durable workflow journal refused or could not complete an operation.
WORKFLOW = "workflow"
# A validator could not decide (its outcome is "error"); no exception has it.
VALIDATOR_ERROR = "validator_error"
# Reserved for an explicit security stop; no exception has it yet.
SECURITY_STOP = "security_stop"
# A retry policy's deadline ran out before an attempt succeeded.
DEADLINE = "deadline"

FAILURE_CLASSES = frozenset({
    INPUT_ERROR, AUTHORITY_DENIED, QUALITY_DEFECT, PROVIDER_REJECTED, TRANSIENT,
    UNKNOWN_OUTCOME, TERMINAL_ACCOUNTING, PERSISTENCE, WORKFLOW, VALIDATOR_ERROR, SECURITY_STOP,
    DEADLINE,
})

# 4xx statuses that are not input errors: authentication and permission, and
# timeouts, conflicts and rate limits, which another attempt may get past.
_STATUS_CLASSES = {
    401: AUTHORITY_DENIED, 403: AUTHORITY_DENIED,
    408: TRANSIENT, 409: TRANSIENT, 429: TRANSIENT, 499: TRANSIENT,
}


def status_failure_class(status: object) -> str | None:
    """The class of an HTTP error status, or None for anything else.

    401 and 403 are ``authority_denied``; 408, 409, 429, 499 and every 5xx
    are ``transient``; any other 4xx is ``input_error``.
    """
    if type(status) is not int or not 400 <= status <= 599:
        return None
    if status >= 500:
        return TRANSIENT
    return _STATUS_CLASSES.get(status, INPUT_ERROR)


@cache
def _rules() -> tuple[tuple[type[BaseException], str | None, bool], ...]:
    """Each exception type with its class and whether it is terminal.

    The first matching type wins. Terminal types come first, so an exception
    is terminal exactly when it is an instance of one of them, whatever else
    it inherits; a subclass precedes its bases.
    """
    from smythe.budget import BudgetEstimateRequired, BudgetValidationError, SentinelAlert
    from smythe.executor_base import ArtifactIntegrityError, NodeFinalizationError
    from smythe.provider import (
        IncompleteOutputError, NativeOutputDefectError, ProviderAccountingError,
        ProviderResponseError,
    )
    from smythe.scheduling import NoAllowedProfileError, RetryDeadlineError, UncertainDispatchError
    from smythe.tools import ToolLoopLimitError
    from smythe.verifier import VerificationRecoveryError
    from smythe.workflow_binding import WorkflowBindingError
    from smythe.workflow_provider import ProviderRequestRejectedError
    from smythe.workflow_store import WorkflowError
    from smythe.workflow_tools import ReadToolError

    return (
        # Another provider call would compound a spend or abandon an unsettled
        # call rather than repair any of these.
        (VerificationRecoveryError, PERSISTENCE, True),
        (NodeFinalizationError, PERSISTENCE, True),
        (ProviderAccountingError, UNKNOWN_OUTCOME, True),
        (ProviderResponseError, PROVIDER_REJECTED, True),
        (SentinelAlert, TERMINAL_ACCOUNTING, True),
        (BudgetValidationError, TERMINAL_ACCOUNTING, True),
        (WorkflowError, WORKFLOW, True),
        # The node's failure policy decides these. A free provider
        # rejection's class follows its HTTP status.
        (ProviderRequestRejectedError, None, False),
        (NativeOutputDefectError, QUALITY_DEFECT, False),
        (IncompleteOutputError, PROVIDER_REJECTED, False),
        (ReadToolError, PROVIDER_REJECTED, False),
        (ToolLoopLimitError, QUALITY_DEFECT, False),
        (WorkflowBindingError, INPUT_ERROR, False),
        (BudgetEstimateRequired, INPUT_ERROR, False),
        (NoAllowedProfileError, INPUT_ERROR, False),
        (ArtifactIntegrityError, INPUT_ERROR, False),
        (RetryDeadlineError, DEADLINE, False),
        # A call sent and never answered; its reservation is kept as unknown cost.
        (UncertainDispatchError, UNKNOWN_OUTCOME, False),
    )


def _http_status(exc: BaseException) -> object:
    """The status an SDK-style error carries as ``status_code`` or ``response.status_code``."""
    try:
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
    except Exception:  # A broken attribute never replaces the failure being classified.
        return None
    return status


def classify_failure(exc: BaseException) -> str:
    """Name the failure class of an exception that a node attempt raised.

    Smythe's exceptions map by type. Any other exception that carries an HTTP
    status, as the provider SDKs' status errors do, maps by
    ``status_failure_class``. Everything else is ``transient``.
    """
    for kind, failure_class, _ in _rules():
        if isinstance(exc, kind):
            if failure_class is None:
                return cast("ProviderRequestRejectedError", exc).failure_class
            return failure_class
    return status_failure_class(_http_status(exc)) or TRANSIENT


def is_terminal(exc: BaseException) -> bool:
    """Whether the executors stop the run at once, whatever the node's failure policy.

    Invalid accounting and budget alerts, unusable or unresolved native
    responses, persistence failures and journal errors are terminal. Any
    other exception follows the node's failure policy.
    """
    for kind, _, terminal in _rules():
        if isinstance(exc, kind):
            return terminal
    return False
