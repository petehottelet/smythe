"""Opt-in scheduling policies for ordinary runs.

``RetryPolicy`` makes the executor the only retry owner of a node's calls:
which failure classes it calls again for, how long it waits first, and the
deadline that bounds a node's attempts and waits together. ``ModelProfile``
routes the nodes that name it to a developer-approved provider and model.
``CapacityPolicy`` limits how many calls each provider type and model takes
at once and how often one starts, and can admit ready work in turn across
branches. ``TerminationPolicy`` lets the async executor drain the nodes in
flight when a failure stops a run, instead of cancelling them at once. The
validators here check every policy's fields the same way.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import re
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from smythe._failures import (
    AUTHORITY_DENIED, INPUT_ERROR, PROVIDER_REJECTED, QUALITY_DEFECT, TRANSIENT, UNKNOWN_OUTCOME,
    _http_status, status_failure_class,
)
from smythe.budget import _validate_usd
from smythe.provider import Provider

if TYPE_CHECKING:
    from smythe.graph import Node

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


@dataclass(frozen=True)
class CapacityLimit:
    """How much of one provider type and model a run's calls may take.

    ``max_in_flight`` caps the nodes using the model that the async executor
    runs at once, and so their calls. ``min_interval_s`` is the least time
    between the starts of two calls. None leaves a field to the provider's
    ``capacity_hint()``, and without a hint the field sets no limit.
    """

    max_in_flight: int | None = None
    min_interval_s: float | None = None

    def __post_init__(self) -> None:
        cap = self.max_in_flight
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError(f"max_in_flight must be an integer >= 1 or None, got {cap!r}")
        if self.min_interval_s is not None:
            object.__setattr__(self, "min_interval_s",
                               validate_seconds(self.min_interval_s, "min_interval_s"))


@dataclass(frozen=True)
class CapacityPolicy:
    """Opt-in limits on the calls each provider type and model takes, and fair admission.

    ``limits`` maps a model name, or a ``(provider type name, model)`` pair
    such as ``("OpenAIResponsesProvider", "gpt-6-astra")``, to a
    :class:`CapacityLimit`. For each field a pair's value wins, then the
    model's, then the provider's ``capacity_hint(model)``. A ready node whose
    model has no free slot is passed over, still ready, and other ready work
    is admitted instead; its wait is a ``capacity`` wait, never a failure.

    A transient status error whose ``Retry-After`` asks for a wait cools its
    provider type and model down: no call to them starts for that long, at
    most ``cooldown_max_s`` (0 turns cooldowns off). With ``round_robin``,
    the async executor admits ready nodes in turn across branches, a branch
    being the ready nodes that share a first dependency (nodes without one
    share a branch), so one wide fan-out cannot hold back another's work.
    """

    limits: Mapping[str | tuple[str, str], CapacityLimit] = field(default_factory=dict)
    round_robin: bool = False
    cooldown_max_s: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.limits, Mapping):
            raise ValueError(f"limits must map models to CapacityLimit, got {self.limits!r}")
        for key, limit in self.limits.items():
            pair = (isinstance(key, tuple) and len(key) == 2
                    and all(isinstance(part, str) for part in key))
            if not isinstance(key, str) and not pair:
                raise ValueError(
                    f"A limit's key must be a model or a (provider type name, model) pair, "
                    f"got {key!r}"
                )
            if not isinstance(limit, CapacityLimit):
                raise ValueError(f"The limit for {key!r} must be a CapacityLimit, got {limit!r}")
        if type(self.round_robin) is not bool:
            raise ValueError(f"round_robin must be a bool, got {self.round_robin!r}")
        object.__setattr__(self, "limits", MappingProxyType(dict(self.limits)))
        object.__setattr__(self, "cooldown_max_s",
                           validate_seconds(self.cooldown_max_s, "cooldown_max_s"))


def _capacity_limit(policy: CapacityPolicy, provider: Any, model: str) -> CapacityLimit:
    """The limit *policy* sets for calls to *model* on *provider*, its hint filling gaps.

    Raises ValueError for a hint that is not None or a dict of
    ``CapacityLimit`` fields with valid values.
    """
    name = type(provider).__name__
    declare = getattr(provider, "capacity_hint", None)
    hint = None if declare is None else declare(model)
    if hint is not None:
        unknown = sorted(map(repr, set(hint) - {"max_in_flight", "min_interval_s"})
                         if isinstance(hint, dict) else ())
        if not isinstance(hint, dict) or unknown:
            raise ValueError(
                f"{name}.capacity_hint({model!r}) must return None or a dict of "
                f"max_in_flight and min_interval_s, got {hint!r}"
            )
        try:
            hint = CapacityLimit(**hint)
        except ValueError as exc:
            raise ValueError(f"{name}.capacity_hint({model!r}) is not valid: {exc}") from None
    layers = [policy.limits.get((name, model)), policy.limits.get(model), hint]
    return CapacityLimit(**{
        item.name: next((getattr(layer, item.name) for layer in layers
                         if layer is not None and getattr(layer, item.name) is not None), None)
        for item in fields(CapacityLimit)
    })


@dataclass(frozen=True)
class TerminationPolicy:
    """How the async executor stops a run after a failure: drain, then cancel.

    Without a policy, a failure cancels every node in flight at once. Under
    one, a failure that stops the run, or a node the budget refuses to admit,
    stops admission: no queued node starts. Each node in flight runs its
    current attempt on, including any further tool turns it takes, for up to
    ``drain_timeout_s`` seconds, and each node that finishes keeps and saves
    its result and cost as usual, without supervision or regeneration. No
    retry starts: a node whose attempt fails, or that is waiting to retry,
    returns to pending for a resume. Those still in flight
    at the deadline are cancelled, as without a policy, and the run raises
    its original error. A cancellation, an interrupt such as
    ``KeyboardInterrupt``, and a failure of class ``security_stop`` never
    drain. The serial executor has nothing in flight to drain.
    """

    drain_timeout_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "drain_timeout_s",
                           validate_seconds(self.drain_timeout_s, "drain_timeout_s", positive=True))


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


class _Bucket:
    """One provider type and model: its limit, the slots in use, and its next call start."""

    __slots__ = ("limit", "in_flight", "not_before", "turn")

    def __init__(self, limit: CapacityLimit) -> None:
        self.limit = limit
        self.in_flight = 0
        # No call starts before this reading of the tracer's clock.
        self.not_before = -math.inf
        # Held by the one call waiting for its start; the others queue in order.
        # Only calls in one async run's event loop ever queue for it.
        self.turn = asyncio.Lock()


class _CapacityLimiter:
    """The capacity that one run's calls take under a :class:`CapacityPolicy`.

    Slots and call starts are kept per provider type and model, on *tracer*'s
    monotonic clock, and each wait is recorded as its ``capacity`` wait.
    ``sleep(seconds)`` waits, and is awaited when it returns an awaitable.
    """

    def __init__(self, policy: CapacityPolicy, tracer: Any, sleep: Any) -> None:
        self.policy = policy
        self._tracer = tracer
        self._sleep = sleep
        self._buckets: dict[tuple[type, str], _Bucket] = {}

    def bucket(self, provider: Any, model: str) -> _Bucket:
        """The state of calls to *model* on *provider*'s type, its limit read on first use."""
        key = (type(provider), model)
        if key not in self._buckets:
            self._buckets[key] = _Bucket(_capacity_limit(self.policy, provider, model))
        return self._buckets[key]

    @staticmethod
    def try_acquire(bucket: _Bucket) -> bool:
        """Take one of *bucket*'s slots; False, taking none, when they are all in use."""
        cap = bucket.limit.max_in_flight
        if cap is not None and bucket.in_flight >= cap:
            return False
        bucket.in_flight += 1
        return True

    @staticmethod
    def release(bucket: _Bucket) -> None:
        bucket.in_flight -= 1

    @asynccontextmanager
    async def call(self, provider: Any, model: str, node: Any = None):
        """Make one call to *model* on *provider* once its turn comes.

        The call starts no sooner than ``min_interval_s`` after the previous
        call's start and after any cooldown, and calls take their turns in
        the order they ask. A call claims its start only once it begins, so a
        call cancelled while it waits claims none and the next call's turn is
        spaced from the calls that started. A transient status error it
        raises whose ``Retry-After`` asks for a wait starts a cooldown of that
        length, at most the policy's ``cooldown_max_s``.
        """
        bucket = self.bucket(provider, model)
        started, waited = self._tracer.now(), bucket.turn.locked()
        try:
            async with bucket.turn:
                while True:
                    now = self._tracer.now()
                    start = max(now, bucket.not_before)
                    if start <= now:
                        break
                    waited = True
                    pause = self._sleep(start - now)
                    if inspect.isawaitable(pause):
                        await pause
                    # Only a cooldown that began during the wait moves the start.
                    if bucket.not_before <= start:
                        break
                bucket.not_before = (max(start, self._tracer.now())
                                     + (bucket.limit.min_interval_s or 0.0))
        finally:
            if waited:
                self._tracer.on_wait("capacity", self._tracer.since(started), node)
        try:
            yield
        except Exception as error:
            asked = _retry_after_s(error)
            if asked is not None and self.policy.cooldown_max_s:
                until = self._tracer.now() + min(asked, self.policy.cooldown_max_s)
                bucket.not_before = max(bucket.not_before, until)
            raise


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


# The node metadata key that names a node's model profile. Only developer code
# sets it: a generated plan may set no metadata but ``role`` (smythe.loader).
PROFILE_KEY = "profile"
# What a profile may require: each is a ``ProviderFeatures`` field that its
# provider must declare True for the profile's model.
PROFILE_REQUIREMENTS = frozenset({"tools", "attachments", "image_output", "exact_accounting"})
_PROFILE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")


class NoAllowedProfileError(ValueError):
    """No allowed model profile can serve a node; raised before any provider call.

    A node names a profile the run does not allow or sets a model other than
    its profile's, or a profile's provider does not support its model, does
    not declare a feature the profile requires, or prices the model with a
    table that retires it. It is raised too for an ``LLMSupervisor`` without
    a model of its own that could review a node whose profile calls another
    provider than the supervisor's.
    """


@dataclass(frozen=True)
class ModelProfile:
    """A developer-approved provider and model that a node can name to be routed to them.

    A node whose ``metadata["profile"]`` is ``name`` calls ``provider`` with
    ``model``, and reserves ``max_cost_per_call_usd`` for each call unless it
    sets ``estimated_cost_usd``. ``requires`` lists the features the provider
    must declare ``True`` for the model through ``Provider.feature_support()``:
    ``tools``, ``attachments``, ``image_output`` and ``exact_accounting``.
    """

    name: str
    provider: Provider
    model: str
    max_cost_per_call_usd: float
    requires: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _PROFILE_NAME.fullmatch(self.name):
            raise ValueError(
                f"A model profile name must match {_PROFILE_NAME.pattern!r}, got {self.name!r}"
            )
        if not isinstance(self.provider, Provider):
            raise TypeError(
                f"Model profile {self.name!r} needs a Provider, not {type(self.provider).__name__}"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(f"Model profile {self.name!r} needs a model name, got {self.model!r}")
        requires = self.requires
        if isinstance(requires, (str, bytes)) or not hasattr(requires, "__iter__"):
            raise ValueError(f"requires must be a set of features, got {requires!r}")
        requires = frozenset(requires)
        unknown = ", ".join(sorted(map(repr, requires - PROFILE_REQUIREMENTS)))
        if unknown:
            raise ValueError(
                f"requires may list only {sorted(PROFILE_REQUIREMENTS)}, not {unknown}"
            )
        object.__setattr__(self, "requires", requires)
        object.__setattr__(self, "max_cost_per_call_usd",
                           _validate_usd(self.max_cost_per_call_usd, "max_cost_per_call_usd"))


def profile_map(profiles: object) -> dict[str, ModelProfile]:
    """*profiles*, a sequence of ``ModelProfile`` with distinct names, by name."""
    if profiles is None:
        return {}
    if isinstance(profiles, (str, bytes, Mapping)) or not hasattr(profiles, "__iter__"):
        raise TypeError("profiles must be a sequence of ModelProfile")
    named: dict[str, ModelProfile] = {}
    for profile in profiles:
        if not isinstance(profile, ModelProfile):
            raise TypeError(f"profiles must hold ModelProfile, not {type(profile).__name__}")
        if profile.name in named:
            raise ValueError(f"Model profile names must be distinct; {profile.name!r} repeats")
        named[profile.name] = profile
    return named


def node_profile(node: Node, profiles: Mapping[str, ModelProfile]) -> ModelProfile | None:
    """The profile *node* names in ``metadata["profile"]``, or None when it names none.

    Raises ``NoAllowedProfileError`` when the name is not one of *profiles*,
    or when the node sets a ``model`` other than that profile's: the model a
    node records is the one it calls.
    """
    if PROFILE_KEY not in node.metadata:
        return None
    name = node.metadata[PROFILE_KEY]
    profile = profiles.get(name) if isinstance(name, str) else None
    if profile is None:
        allowed = (f"its profiles are {sorted(profiles)}" if profiles
                   else "it has none; pass them as Swarm(profiles=...)")
        raise NoAllowedProfileError(
            f"Node {node.id!r} names model profile {name!r}, which this run does not allow: "
            + allowed
        )
    if node.metadata.get("model", profile.model) != profile.model:
        raise NoAllowedProfileError(
            f"Node {node.id!r} sets model {node.metadata['model']!r}, but its profile "
            f"{profile.name!r} calls {profile.model!r}"
        )
    return profile
