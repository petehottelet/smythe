"""Opt-in scheduling policies for ordinary runs.

``RetryPolicy`` makes the executor the only retry owner of a node's calls:
which failure classes it calls again for, how long it waits first, and the
deadline that bounds a node's attempts and waits together. The validators
here check every policy's fields the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from smythe._failures import (
    AUTHORITY_DENIED, INPUT_ERROR, PROVIDER_REJECTED, QUALITY_DEFECT, TRANSIENT, UNKNOWN_OUTCOME,
    _http_status, status_failure_class,
)

# The classes of a failed call a policy may list. A failure whose exception is
# terminal stops the run whatever the list says.
RETRYABLE_CLASSES = frozenset({
    INPUT_ERROR, AUTHORITY_DENIED, QUALITY_DEFECT, PROVIDER_REJECTED, TRANSIENT, UNKNOWN_OUTCOME,
})


class RetryDeadlineError(TimeoutError):
    """A node used up its retry policy's ``deadline_s`` before an attempt succeeded.

    Raised when an attempt times out at the deadline, or when the wait before
    the next retry would end at or after it. Its class is ``deadline``, which
    is never retried; the node's failure policy settles it like any other
    failure.
    """


class UncertainDispatchError(TimeoutError):
    """A node attempt timed out after its provider call was sent.

    Raised in place of the node's ``TimeoutError`` under a retry policy with
    ``retain_uncertain_dispatch``, when the call had not answered: it may
    still be charged, so the node keeps its reservation as unknown cost. Its
    class is ``unknown_outcome``, retried only when the policy lists it.
    """


def validate_seconds(value: object, name: str, *, positive: bool = False) -> float:
    """Accept a finite number of seconds, non-negative or, with *positive*, above zero.

    A bool is not a number here, and an integer too large for a float is not
    finite.
    """
    message = f"{name} must be a finite {'positive' if positive else 'non-negative'} number"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{message}, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{message}, got {value!r}")
    return number


@dataclass(frozen=True)
class RetryPolicy:
    """Which failed node attempts the executor calls again, and when it stops.

    A node with ``FailurePolicy.RETRY`` gets up to ``max_retries`` more
    attempts, but only after a failure whose class is in ``retry_classes``;
    any other failure fails the node at once. Before a retry the executor
    waits as long as a transient status error's ``Retry-After`` header asks
    (with ``honor_retry_after``), otherwise a jittered backoff that doubles
    from ``retry_backoff_s``. No wait is longer than ``backoff_max_s``.

    ``deadline_s`` bounds each node's attempts and waits together, from the
    start of its first attempt, on the tracer's monotonic clock. Every
    attempt's timeout is the node's ``timeout_s`` or the time left, whichever
    is shorter, and a retry whose wait would reach the deadline is not made:
    the node fails with ``RetryDeadlineError``. Without a policy, ``RETRY``
    retries every failure that is not terminal, as before.

    With ``persist_attempts``, each attempt is saved as spent before its call
    and checkpoints record the policy, so a resumed run continues each node's
    allowance in its ``execution_generation`` instead of refilling it, and
    resumes only under an equal policy. Its deadline starts again in each
    process.

    With ``retain_uncertain_dispatch``, a node call that the executor's
    timeout, a cancellation or an interruption ends after it was sent, before
    the provider answered, keeps its reservation as the node's recorded cost,
    marked unknown and, with an amount, estimated, instead of releasing it,
    so checkpoints save the exposure and resume restores it. Such a timeout
    raises ``UncertainDispatchError``, class ``unknown_outcome``.
    """

    retry_classes: frozenset[str] = frozenset({TRANSIENT})
    honor_retry_after: bool = True
    backoff_max_s: float = 60.0
    deadline_s: float | None = None
    persist_attempts: bool = False
    retain_uncertain_dispatch: bool = False

    def to_dict(self) -> dict:
        """The policy's fields as JSON-ready values, as a checkpoint records them.

        ``retain_uncertain_dispatch`` is recorded only when set.
        """
        data = {"retry_classes": sorted(self.retry_classes),
                "honor_retry_after": self.honor_retry_after, "backoff_max_s": self.backoff_max_s,
                "deadline_s": self.deadline_s, "persist_attempts": self.persist_attempts}
        if self.retain_uncertain_dispatch:
            data["retain_uncertain_dispatch"] = True
        return data

    @classmethod
    def from_dict(cls, data: object) -> RetryPolicy:
        """The policy that :meth:`to_dict` *data* records; ValueError for any other value."""
        if not isinstance(data, dict):
            raise ValueError(f"A saved retry policy must be an object, got {data!r}")
        unknown = sorted(set(data) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"A saved retry policy has fields this build does not know: {unknown}")
        try:
            return cls(**data)
        except TypeError as exc:  # Such as an unhashable retry class.
            raise ValueError(f"A saved retry policy is not readable: {exc}") from exc

    def __post_init__(self) -> None:
        classes = self.retry_classes
        if isinstance(classes, (str, bytes)) or not hasattr(classes, "__iter__"):
            raise ValueError(f"retry_classes must be a set of failure classes, got {classes!r}")
        classes = frozenset(classes)
        unknown = ", ".join(sorted(map(repr, classes - RETRYABLE_CLASSES)))
        if unknown:
            raise ValueError(
                f"retry_classes may list only {sorted(RETRYABLE_CLASSES)}, not {unknown}"
            )
        for name in ("honor_retry_after", "persist_attempts", "retain_uncertain_dispatch"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        deadline = self.deadline_s
        if deadline is not None:
            deadline = validate_seconds(deadline, "deadline_s", positive=True)
        object.__setattr__(self, "retry_classes", classes)
        object.__setattr__(self, "backoff_max_s",
                           validate_seconds(self.backoff_max_s, "backoff_max_s"))
        object.__setattr__(self, "deadline_s", deadline)


def _retry_after_s(error: BaseException) -> float | None:
    """The wait a transient status error's response asks for, in seconds, or None.

    Reads ``retry-after-ms``, then ``retry-after``, as a number of seconds;
    an HTTP date, a negative or a non-finite value is not read.
    """
    if status_failure_class(_http_status(error)) != TRANSIENT:
        return None
    try:
        headers = {str(key).lower(): value for key, value in error.response.headers.items()}
    except Exception:  # A missing or broken response has no header to read.
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            wait = float(headers[name]) * scale
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(wait) and wait >= 0:
            return wait
    return None


def _attempt_records(data: object) -> dict[str, tuple[int, int]]:
    """Read attempts used, ``{node_id: {"execution_generation": g, "attempts": n}}``.

    Returns ``{node_id: (g, n)}``; raises ValueError unless every generation
    and count is a non-negative integer.
    """
    if not isinstance(data, dict):
        raise ValueError(f"attempts_used must map node ids to records, got {data!r}")
    records = {}
    for node_id, record in data.items():
        if (not isinstance(node_id, str) or not isinstance(record, dict)
                or set(record) != {"execution_generation", "attempts"}):
            raise ValueError(
                f"attempts_used of {node_id!r} must hold exactly execution_generation and attempts"
            )
        values = record["execution_generation"], record["attempts"]
        for name, value in zip(("execution_generation", "attempts"), values):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} in attempts_used of {node_id!r} must be an integer >= 0")
        records[node_id] = values
    return records
