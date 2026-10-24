"""An in-process fake external service with independently inspectable state.

The service stands in for an external system that a workflow reads from and
writes to. A caller submits a request under a request key, which is both the
request's identity in the service log and its idempotency key. Each key keeps
separate counts of deliveries received, effects committed, deliveries answered
from an earlier commit, refusals, and acknowledgements that did or did not
reach the caller. A fault schedule can refuse a delivery before it commits,
commit it and then lose the acknowledgement, deliver it twice, or hold it at a
``Gate``; ``send`` and ``deliver`` reorder requests in flight. Sources are
versioned records a workflow reads, changed with ``change_source``; a version
whose access is revoked (``revoke_source``), or a source deleted with its
history (``delete_source``), can no longer be read.

A request may name a ``Target``, a tenant's collection. Its idempotency key
and object then belong to that tenant: another tenant, or a caller naming
none, can neither look them up nor update the object, and the same key from
another tenant is a separate request. The request log and fault schedule stay
keyed by request key alone. With ``send_notices`` the service puts a notice in
its outbox for each acknowledged commit, as a webhook would; ``forge_notice``
adds one the service never sent, and only ``snapshot()`` tells them apart.

``snapshot()`` returns the service's own state as detached JSON data, so a test
can compare what happened in the service with what the caller observed. Every
method is thread safe, and a held delivery blocks only its own thread.
``serve_http`` exposes the same service on an ephemeral loopback port for tests
that need a real socket or a separate process.

Import it by module name (``from fixture_service import FixtureService``).
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, unquote, urlsplit

from fakes import Gate

NS_PER_SECOND = 1_000_000_000
COUNTERS = ("received", "committed", "deduplicated", "refused", "acknowledged", "ack_dropped")


class ServiceError(Exception):
    """A failure the caller observes."""


class ServiceUnavailable(ServiceError):
    """The service refused the delivery before committing anything."""


class AcknowledgementLost(ServiceError):
    """The caller lost the answer; the service may have committed the effect."""


class NotFound(ServiceError, LookupError):
    """The named object or source version does not exist."""


class LookupUnavailable(ServiceError):
    """This service does not offer that kind of lookup."""


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    version: int
    content: str
    sha256: str
    changed_at_ns: int


@dataclass(frozen=True)
class Target:
    """Where a request writes: a tenant's collection."""

    tenant: str
    collection: str

    def __post_init__(self) -> None:
        for name in ("tenant", "collection"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"A target {name} is a nonempty string")


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    version: int
    value: Any
    request_key: str
    committed_at_ns: int
    target: Target | None = None


@dataclass(frozen=True)
class Receipt:
    """The service's answer for a committed request."""

    request_key: str
    object_id: str
    object_version: int
    committed_at_ns: int
    deduplicated: bool = False  # Answered from an earlier commit of the same key.
    target: Target | None = None


@dataclass(frozen=True)
class Fault:
    """What happens to one delivery of a request key."""

    fail_before_commit: bool = False
    drop_ack: bool = False  # Commit, then lose the acknowledgement.
    duplicate: bool = False  # The network delivers another copy afterwards.
    hold_before_commit: Gate | None = None
    hold_after_commit: Gate | None = None


NO_FAULT = Fault()
FAIL_BEFORE_COMMIT = Fault(fail_before_commit=True)
COMMIT_THEN_DROP_ACK = Fault(drop_ack=True)
DUPLICATE_DELIVERY = Fault(duplicate=True)


def _detached(value: Any) -> Any:
    """Copy JSON data; non-JSON values raise TypeError."""
    return json.loads(json.dumps(value, allow_nan=False))


def _nanoseconds(seconds: float) -> int:
    return round(seconds * NS_PER_SECOND)


def _tenant(target: Target | None) -> str | None:
    return None if target is None else target.tenant


def _target_of(item: Any) -> dict:
    """``{"target": ...}`` for an item that names a target, else nothing."""
    target = getattr(item, "target", None)
    return {} if target is None else {"target": asdict(target)}


def _public(record: Any) -> dict:
    """A record as JSON data; a record without a target has no ``target`` key."""
    return {**{name: value for name, value in asdict(record).items() if name != "target"},
            **_target_of(record)}


def _received(notice: dict) -> dict:
    """A notice as its receiver sees it, without the service's ``forged`` mark."""
    return _detached({name: value for name, value in notice.items() if name != "forged"})


class Message:
    """A request on its way to the service; ``result()`` is the caller's answer."""

    def __init__(self, request_key: str, value: Any, object_id: str | None,
                 target: Target | None = None) -> None:
        if type(request_key) is not str or not request_key:
            raise ValueError("A request key is a nonempty string")
        if object_id is not None and (type(object_id) is not str or not object_id):
            raise ValueError("An object ID is a nonempty string")
        if target is not None and not isinstance(target, Target):
            raise ValueError("A target is a Target")
        self.request_key = request_key
        self.value = _detached(value)
        self.object_id = object_id
        self.target = target
        self.delivered = False
        self._outcome: Receipt | ServiceError | None = None

    def result(self) -> Receipt:
        if not self.delivered:
            raise RuntimeError("Request is still in flight")
        if isinstance(self._outcome, ServiceError):
            raise self._outcome
        assert self._outcome is not None
        return self._outcome


class FixtureService:
    """A fake external service; see the module docstring.

    ``dedupe`` answers a repeated live request key from its earlier commit.
    ``key_ttl_s`` bounds how long a key is remembered, for deduplication and
    lookup. ``lookup_by_key`` and ``lookup_by_object`` choose which lookups
    exist. ``visibility_delay_s`` makes commits visible to lookups only after
    that long on ``clock_ns``; ``snapshot()`` always shows every commit.
    ``send_notices`` sends a notice for each acknowledged commit.
    """

    def __init__(self, *, clock_ns: Callable[[], int] = time.time_ns, dedupe: bool = True,
                 key_ttl_s: float | None = None, lookup_by_key: bool = True,
                 lookup_by_object: bool = True, visibility_delay_s: float = 0,
                 send_notices: bool = False) -> None:
        if key_ttl_s is not None and not key_ttl_s > 0:
            raise ValueError("Key lifetime must be positive")
        if not visibility_delay_s >= 0:
            raise ValueError("Visibility delay cannot be negative")
        self.dedupe = dedupe
        self.key_ttl_s = key_ttl_s
        self.lookup_by_key = lookup_by_key
        self.lookup_by_object = lookup_by_object
        self.visibility_delay_s = visibility_delay_s
        self.send_notices = send_notices
        self._clock_ns = clock_ns
        self._key_ttl_ns = None if key_ttl_s is None else _nanoseconds(key_ttl_s)
        self._delay_ns = _nanoseconds(visibility_delay_s)
        self._condition = threading.Condition()
        self._sources: dict[str, list[SourceRecord]] = {}
        self._source_reads: Counter[str] = Counter()
        self._revoked: set[tuple[str, int]] = set()  # (source id, version)
        self._objects: dict[str, list[ObjectRecord]] = {}
        # (tenant, request key) -> (receipt, expiry)
        self._keys: dict[tuple[str | None, str], tuple[Receipt, int | None]] = {}
        self._counters: dict[str, Counter[str]] = {}
        self._deliveries: list[dict] = []
        self._faults: dict[tuple[str, int], Fault] = {}
        self._in_flight: list[Message] = []
        self._notices: list[dict] = []
        self._active = 0

    # Sources

    def add_source(self, source_id: str, content: str) -> SourceRecord:
        with self._condition:
            if source_id in self._sources:
                raise ValueError(f"Source {source_id!r} already exists")
            record = self._source(source_id, 1, content)
            self._sources[source_id] = [record]
            return record

    def change_source(self, source_id: str, content: str) -> SourceRecord:
        """Replace a source's content as its next version, keeping the history."""
        with self._condition:
            history = self._sources.get(source_id)
            if history is None:
                raise NotFound(f"Unknown source {source_id!r}")
            record = self._source(source_id, history[-1].version + 1, content)
            history.append(record)
            return record

    def read_source(self, source_id: str, version: int | None = None) -> SourceRecord:
        """Read the latest or a named version; each successful read is counted.

        A revoked version cannot be read, even as the latest.
        """
        with self._condition:
            matches = [record for record in self._sources.get(source_id, ())
                       if version is None or record.version == version]
            if not matches or (source_id, matches[-1].version) in self._revoked:
                raise NotFound(f"Unknown source {source_id!r} version {version}")
            self._source_reads[source_id] += 1
            return matches[-1]

    def revoke_source(self, source_id: str, version: int | None = None) -> SourceRecord:
        """Revoke access to the latest or a named version, which stays in the history."""
        with self._condition:
            matches = [record for record in self._sources.get(source_id, ())
                       if version is None or record.version == version]
            if not matches:
                raise NotFound(f"Unknown source {source_id!r} version {version}")
            self._revoked.add((source_id, matches[-1].version))
            return matches[-1]

    def delete_source(self, source_id: str) -> None:
        """Delete a source and its whole history, as a retention rule would."""
        with self._condition:
            if self._sources.pop(source_id, None) is None:
                raise NotFound(f"Unknown source {source_id!r}")
            self._revoked = {item for item in self._revoked if item[0] != source_id}

    def _source(self, source_id, version, content):
        if type(content) is not str:
            raise TypeError("Source content is text")
        return SourceRecord(source_id, version, content,
                            hashlib.sha256(content.encode()).hexdigest(), self._clock_ns())

    # Fault schedule

    def schedule(self, request_key: str, fault: Fault, *, delivery: int = 0) -> None:
        """Apply ``fault`` to the ``delivery``-th arrival (from 0) of ``request_key``."""
        with self._condition:
            self._faults[request_key, delivery] = fault

    def fail_before_commit(self, request_key: str, *, delivery: int = 0) -> None:
        self.schedule(request_key, FAIL_BEFORE_COMMIT, delivery=delivery)

    def commit_then_drop_ack(self, request_key: str, *, delivery: int = 0) -> None:
        self.schedule(request_key, COMMIT_THEN_DROP_ACK, delivery=delivery)

    def deliver_duplicate(self, request_key: str, *, delivery: int = 0) -> None:
        self.schedule(request_key, DUPLICATE_DELIVERY, delivery=delivery)

    def hold(self, request_key: str, gate: Gate, *, delivery: int = 0,
             before_commit: bool = False) -> None:
        """Block that delivery's thread at ``gate``, before or after it commits."""
        fault = Fault(hold_before_commit=gate) if before_commit else Fault(hold_after_commit=gate)
        self.schedule(request_key, fault, delivery=delivery)

    # Requests

    def submit(self, request_key: str, value: Any, *, object_id: str | None = None,
               target: Target | None = None) -> Receipt:
        """Deliver one request now and return the caller's answer.

        Without ``object_id`` a commit creates a new object in ``target``; with
        it, a commit stores ``value`` as that object's next version. An update
        must name the object's own target: any other is refused with
        ``NotFound``, as for an unknown object. Raises the ``ServiceError`` the
        caller would see.
        """
        message = Message(request_key, value, object_id, target)
        self._deliver(message)
        return message.result()

    def send(self, request_key: str, value: Any, *, object_id: str | None = None,
             target: Target | None = None) -> Message:
        """Put a request in flight without delivering it; see ``deliver``."""
        message = Message(request_key, value, object_id, target)
        with self._condition:
            self._in_flight.append(message)
        return message

    def deliver(self, order: list[int] | None = None) -> list[Message]:
        """Deliver every request in flight, in ``order`` (queue indexes) or as sent."""
        with self._condition:
            queued = list(self._in_flight)
            order = list(range(len(queued))) if order is None else list(order)
            if sorted(order) != list(range(len(queued))):
                raise ValueError("Order must be a permutation of the requests in flight")
            self._in_flight = []
        messages = [queued[index] for index in order]
        for message in messages:
            self._deliver(message)
        return messages

    def _deliver(self, message: Message, respond: Callable[[Receipt], None] | None = None) -> None:
        fault, outcome = self._arrive(message, respond=respond)
        message.delivered, message._outcome = True, outcome
        while fault.duplicate:
            # Another copy arrives; nobody receives its answer.
            fault, _ = self._arrive(message, duplicate=True)

    def _arrive(self, message, *, respond=None, duplicate=False):
        key = message.request_key
        with self._condition:
            counts = self._counters.setdefault(key, Counter({name: 0 for name in COUNTERS}))
            fault = self._faults.get((key, counts["received"]), NO_FAULT)
            entry = {"delivery": f"dlv_{len(self._deliveries) + 1}", "request_key": key,
                     "sequence": counts["received"], "duplicate": duplicate,
                     "received_at_ns": self._clock_ns(), "outcome": "pending",
                     "object_id": None, "object_version": None, "acknowledged": None}
            entry.update(_target_of(message))
            counts["received"] += 1
            self._deliveries.append(entry)
            self._active += 1
        try:
            return fault, self._process(message, fault, entry, counts, respond, duplicate)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _process(self, message, fault, entry, counts, respond, duplicate):
        if fault.hold_before_commit is not None:
            fault.hold_before_commit.block()
        with self._condition:
            if fault.fail_before_commit:
                return self._refuse(entry, counts, ServiceUnavailable(
                    "The service refused the request before committing"))
            try:
                receipt = self._commit(message, entry, counts)
            except NotFound as error:
                return self._refuse(entry, counts, error)
        if fault.hold_after_commit is not None:
            fault.hold_after_commit.block()
        if fault.drop_ack:
            return self._lose_acknowledgement(entry, counts)
        if duplicate:
            with self._condition:
                entry["acknowledged"] = False
            return receipt
        if respond is not None:
            try:
                respond(receipt)
            except OSError:
                return self._lose_acknowledgement(entry, counts)
        with self._condition:
            counts["acknowledged"] += 1
            entry["acknowledged"] = True
            if self.send_notices and entry["outcome"] == "committed":
                self._notify(_public(receipt), forged=False)
        return receipt

    def _commit(self, message, entry, counts):
        now, key = self._clock_ns(), message.request_key
        scoped_key = _tenant(message.target), key
        known = self._keys.get(scoped_key)
        if self.dedupe and known is not None and (known[1] is None or now < known[1]):
            receipt = replace(known[0], deduplicated=True)
            counts["deduplicated"] += 1
            entry.update(outcome="deduplicated", object_id=receipt.object_id,
                         object_version=receipt.object_version)
            return receipt
        if message.object_id is None:
            object_id = f"obj_{len(self._objects) + 1}"
            history = self._objects.setdefault(object_id, [])
        elif (message.object_id in self._objects
              and self._objects[message.object_id][0].target == message.target):
            object_id = message.object_id
            history = self._objects[object_id]
        else:
            # Another tenant's object, or one in another collection, is unknown here.
            raise NotFound(f"Unknown object {message.object_id!r}")
        version = len(history) + 1
        history.append(ObjectRecord(object_id, version, message.value, key, now, message.target))
        receipt = Receipt(key, object_id, version, now, target=message.target)
        self._keys[scoped_key] = (
            receipt, None if self._key_ttl_ns is None else now + self._key_ttl_ns)
        counts["committed"] += 1
        entry.update(outcome="committed", object_id=object_id, object_version=version)
        return receipt

    def _refuse(self, entry, counts, error):
        with self._condition:
            counts["refused"] += 1
            entry.update(outcome="refused", acknowledged=False)
        return error

    def _lose_acknowledgement(self, entry, counts):
        with self._condition:
            counts["ack_dropped"] += 1
            entry["acknowledged"] = False
        return AcknowledgementLost("The acknowledgement was lost after the service committed")

    # Notices

    def _notify(self, payload: dict, *, forged: bool) -> dict:
        notice = {"notice_id": f"ntc_{len(self._notices) + 1}", "sent_at_ns": self._clock_ns(),
                  "payload": payload, "forged": forged}
        self._notices.append(notice)
        return _received(notice)

    def notices(self) -> list[dict]:
        """The outbox, oldest first, as its receiver sees it: a forged notice looks genuine.

        Each notice is ``{"notice_id", "sent_at_ns", "payload"}``; a genuine
        payload is the receipt of the commit it reports.
        """
        with self._condition:
            return [_received(notice) for notice in self._notices]

    def forge_notice(self, payload: dict) -> dict:
        """Put a notice the service never sent in the outbox and return it; nothing commits."""
        if type(payload) is not dict:
            raise TypeError("A notice payload is a JSON object")
        payload = _detached(payload)
        with self._condition:
            return self._notify(payload, forged=True)

    # Lookups and inspection

    def lookup_request(self, request_key: str, tenant: str | None = None) -> Receipt | None:
        """The receipt a live key of ``tenant`` committed, once visible; None otherwise."""
        if not self.lookup_by_key:
            raise LookupUnavailable("This service has no lookup by request key")
        with self._condition:
            now, known = self._clock_ns(), self._keys.get((tenant, request_key))
            if known is None or (known[1] is not None and now >= known[1]):
                return None
            return known[0] if now >= known[0].committed_at_ns + self._delay_ns else None

    def get_object(self, object_id: str, tenant: str | None = None) -> ObjectRecord | None:
        """The latest visible version of ``tenant``'s object; None if none is visible yet."""
        if not self.lookup_by_object:
            raise LookupUnavailable("This service has no lookup by object ID")
        with self._condition:
            now, history = self._clock_ns(), self._objects.get(object_id, [])
            if history and _tenant(history[0].target) != tenant:
                history = []
            visible = [record for record in history
                       if now >= record.committed_at_ns + self._delay_ns]
            return replace(visible[-1], value=_detached(visible[-1].value)) if visible else None

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Wait until no delivery is in progress, such as one held in a server thread."""
        with self._condition:
            return self._condition.wait_for(lambda: self._active == 0, timeout)

    def _revocations(self, source_id: str) -> dict:
        """``{"revoked": [versions]}`` for a source with a revoked version, else nothing."""
        versions = sorted(version for name, version in self._revoked if name == source_id)
        return {"revoked": versions} if versions else {}

    def snapshot(self) -> dict:
        """The service's complete state as detached JSON data, ignoring visibility delay.

        Targets, tenants' keys, notices and a source's revoked versions appear
        only once a request names a target, the outbox is in use or a version
        is revoked, so a snapshot without them keeps its original shape.
        """
        with self._condition:
            requests = {}
            for key, counts in self._counters.items():
                committed = [entry["object_id"] for entry in self._deliveries
                             if entry["request_key"] == key and entry["outcome"] == "committed"]
                requests[key] = {**{name: counts[name] for name in COUNTERS},
                                 "object_ids": list(dict.fromkeys(committed))}
            keys: dict[str | None, dict] = {None: {}}  # tenant -> request key -> entry
            for (tenant, key), (receipt, expires) in self._keys.items():
                keys.setdefault(tenant, {})[key] = {
                    "object_id": receipt.object_id, "object_version": receipt.object_version,
                    "expires_at_ns": expires, **_target_of(receipt)}
            state = {
                "now_ns": self._clock_ns(),
                "config": {"dedupe": self.dedupe, "key_ttl_s": self.key_ttl_s,
                           "lookup_by_key": self.lookup_by_key,
                           "lookup_by_object": self.lookup_by_object,
                           "visibility_delay_s": self.visibility_delay_s,
                           **({"send_notices": True} if self.send_notices else {})},
                "totals": {name: sum(counts[name] for counts in self._counters.values())
                           for name in COUNTERS},
                "requests": requests,
                "idempotency_keys": keys.pop(None),
                **({"tenant_idempotency_keys": keys} if keys else {}),
                "objects": {
                    object_id: {"version": history[-1].version, "value": history[-1].value,
                                "history": [_public(record) for record in history],
                                **_target_of(history[0])}
                    for object_id, history in self._objects.items()},
                "sources": {
                    source_id: {"version": history[-1].version, "content": history[-1].content,
                                "sha256": history[-1].sha256,
                                "reads": self._source_reads[source_id],
                                "history": [asdict(record) for record in history],
                                **self._revocations(source_id)}
                    for source_id, history in self._sources.items()},
                "deliveries": [dict(entry) for entry in self._deliveries],
                "in_flight": [{"request_key": message.request_key, "object_id": message.object_id,
                               "value": message.value, **_target_of(message)}
                              for message in self._in_flight],
            }
            if self.send_notices or self._notices:
                state["notices"] = self._notices
            return _detached(state)


# Loopback HTTP adapter


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False  # A handler held at a gate must not block shutdown.


class _Handler(BaseHTTPRequestHandler):
    service: FixtureService

    def do_POST(self):
        if urlsplit(self.path).path != "/requests":
            return self._json(404, {"error": "not_found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"null")
            target = body.get("target")
            message = Message(body["request_key"], body.get("value"), body.get("object_id"),
                              None if target is None else Target(**target))
        except (AttributeError, KeyError, TypeError, ValueError):
            return self._json(400, {"error": "bad_request"})
        self.service._deliver(message, respond=lambda receipt: self._json(200, _public(receipt)))
        try:
            message.result()
        except AcknowledgementLost:
            self.close_connection = True  # Nothing was written: the answer is lost.
        except ServiceUnavailable:
            self._json(503, {"error": "unavailable"})
        except NotFound:
            self._json(404, {"error": "not_found"})

    def do_GET(self):
        parts = urlsplit(self.path)
        segments = [unquote(segment) for segment in parts.path.strip("/").split("/")]
        if len(segments) != 2 or not segments[1]:
            return self._json(404, {"error": "not_found"})
        kind, name = segments
        query = parse_qs(parts.query)
        tenant = query["tenant"][0] if "tenant" in query else None
        try:
            if kind == "requests":
                found = self.service.lookup_request(name, tenant)
            elif kind == "objects":
                found = self.service.get_object(name, tenant)
            elif kind == "sources":
                version = query.get("version")
                found = self.service.read_source(name, None if version is None else int(version[0]))
            else:
                found = None
        except LookupUnavailable:
            return self._json(501, {"error": "lookup_unavailable"})
        except NotFound:
            found = None
        except ValueError:
            return self._json(400, {"error": "bad_request"})
        if found is None:
            return self._json(404, {"error": "not_found"})
        self._json(200, _public(found))

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


@contextmanager
def serve_http(service: FixtureService) -> Iterator[Endpoint]:
    """Serve ``service`` over HTTP on an ephemeral 127.0.0.1 port.

    ``POST /requests`` takes ``{"request_key", "value", "object_id"?, "target"?}``,
    where a target is ``{"tenant", "collection"}``, and answers 200 with the
    receipt, 503 for a refusal, 404 for an unknown object or 400 for a
    malformed body. A lost acknowledgement closes the connection without any
    response. ``GET /requests/<key>`` and ``/objects/<id>``, each scoped by an
    optional ``?tenant=``, and ``/sources/<id>[?version=N]`` answer 200, 404,
    or 501 for an unavailable lookup.
    """
    handler = type("FixtureServiceHandler", (_Handler,), {"service": service})
    server = _Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Endpoint("127.0.0.1", server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)
