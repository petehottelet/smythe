"""Journaled read steps for durable text workflows.

A :class:`ReadProfile` declares one versioned read operation: an HTTP GET of a
URL built from validated input, with byte, time, redirect and destination
limits, and an explicit statement of whether a read can incur a charge
(``metered``) and whether sending it again is harmless (``repeat_safe``).
Private and local network addresses are refused unless the profile allows
them. A developer graph or template declares a read step as a node whose
``metadata["read_step"]`` names a profile and its input; the durable runtime
runs that node without a model call, and a generated plan cannot declare one.

:class:`JournaledReadTool` records each read in the workflow journal like a
model call, in the ``tool`` phase with call kind ``read_tool``: the request
is prepared, quoted at the profile's declared ceiling, reserved, and claimed
for dispatch once. The raw response is saved before it is settled and
decoded, so a saved response is reused on resume without reading again. A
read never retries on its own; the node's failure policy decides whether a
failed read is read again, as a new journaled call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import inspect
import ipaddress
import math
import socket
import string
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, NamedTuple
from urllib.parse import quote, urljoin, urlsplit

from smythe._redact import redact_account_identifiers
from smythe.provider import CompletionResult
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import READ_PRICE_VERSION, WorkflowStateError, _canonical

if TYPE_CHECKING:
    from smythe.graph import Node
    from smythe.workflow_provider import WorkflowProviderContext
    from smythe.workflow_store import CallKey

# The node metadata key that declares a read step.
READ_STEP_KEY = "read_step"
# How a read is sent: one HTTP GET, its redirects, and the limits below.
READ_ADAPTER_VERSION = "http-get-v1"
# The built-in decoder: the body as strict UTF-8 text.
TEXT_DECODER_VERSION = "utf8-text-v1"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_TIMEOUT_S = 600.0
MAX_REDIRECTS = 10
_INPUT_TYPES = {"string": str, "integer": int}
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DESCRIPTOR_KEYS = frozenset({
    "kind", "adapter_version", "decoder_version", "name", "version", "url", "input_schema",
    "metered", "repeat_safe", "ceiling_nanousd", "max_response_bytes", "timeout_s",
    "max_redirects", "allowed_hosts", "allow_private_networks",
})


class ReadToolError(RuntimeError):
    """A read step ended without usable text, and the journal settled its charge.

    The response was refused by a limit, was not a 200, was lost before it
    was saved, or could not be decoded. The run stays open: the node's
    failure policy decides what happens next, and ``RETRY`` makes a new
    journaled read. The message names the status or failure, never the
    response body; a decoder's own error message is shortened and has
    account identifiers redacted.
    """


def _label(value: object, name: str, *, limit: int = 128) -> str:
    if (type(value) is not str or not value.strip() or len(value) > limit
            or any(ord(char) < 32 for char in value)):
        raise WorkflowBindingError(
            f"Read profile {name} must be nonempty text of at most {limit} characters "
            "without control characters"
        )
    return value


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise WorkflowBindingError(f"Read profile {name} must be an integer from {low} to {high}")
    return value


def _host(value: object) -> str:
    """A hostname or IP address as URLs compare it: lower case, without brackets or port."""
    valid = (type(value) is str and value and len(value) <= 253 and value == value.strip().lower()
             and not any(char in value for char in "/?#@[] \t")
             and not any(ord(char) < 32 for char in value))
    if valid and ":" in value:
        try:
            ipaddress.IPv6Address(value)
        except ValueError:
            valid = False
    if not valid:
        raise WorkflowBindingError(
            f"Read profile allowed host {value!r} must be a lower-case host name without a port")
    return value


def _template(url: object, schema: Mapping[str, str]) -> str:
    """Check a URL template and return its host.

    Placeholders such as ``{source_id}`` may appear only after the host, so
    input can never choose where a read goes. Each placeholder must be a
    declared input field, and each field must appear.
    """
    if (type(url) is not str or not url or len(url) > 2048 or not url.isascii()
            or any(ord(char) <= 32 or ord(char) == 127 for char in url)):
        raise WorkflowBindingError(
            "Read profile url must be an ASCII URL template of at most 2048 characters, "
            "without spaces or control characters"
        )
    if "#" in url:
        raise WorkflowBindingError("Read profile url cannot have a fragment")
    prefix, text, fields = None, "", set()
    try:
        for literal, name, spec, conversion in string.Formatter().parse(url):
            text += literal
            if name is None:
                continue
            if prefix is None:
                prefix = text
            if not name.isidentifier() or spec or conversion:
                raise WorkflowBindingError(
                    f"Read profile url placeholder {name!r} must be a plain input field name")
            fields.add(name)
    except ValueError as error:
        raise WorkflowBindingError(f"Read profile url is not a valid template: {error}") from error
    parts = urlsplit(url if prefix is None else prefix)
    try:
        parts.port
    except ValueError as error:
        raise WorkflowBindingError("Read profile url has an invalid port") from error
    # The text before the first placeholder must end the host and port.
    authority_ends = prefix is None or any(sep in prefix.partition("://")[2] for sep in "/?")
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username is not None or parts.password is not None or not authority_ends):
        raise WorkflowBindingError(
            "Read profile url must be an http or https URL with a fixed host and no credentials; "
            "placeholders may appear only in its path or query"
        )
    if fields != set(schema):
        raise WorkflowBindingError(
            "Read profile url placeholders must be exactly its input fields: "
            f"{sorted(fields)} differ from {sorted(schema)}"
        )
    return parts.hostname


def _check_descriptor(descriptor: object) -> dict:
    """Validate a read profile's recorded description; return it unchanged."""
    if type(descriptor) is not dict or descriptor.keys() != _DESCRIPTOR_KEYS:
        raise WorkflowBindingError("Read profile description has an invalid schema")
    if descriptor["kind"] != "read_tool" or descriptor["adapter_version"] != READ_ADAPTER_VERSION:
        raise WorkflowBindingError("Read profile description has an unsupported kind or adapter")
    for name in ("name", "version", "decoder_version"):
        _label(descriptor[name], name)
    schema = descriptor["input_schema"]
    if (type(schema) is not dict or len(schema) > 32
            or any(type(key) is not str or not key.isidentifier() or len(key) > 64
                   or type(value) is not str or value not in _INPUT_TYPES
                   for key, value in schema.items())):
        raise WorkflowBindingError(
            "Read profile input_schema must map at most 32 field names to 'string' or 'integer'")
    for name in ("metered", "repeat_safe", "allow_private_networks"):
        if type(descriptor[name]) is not bool:
            raise WorkflowBindingError(f"Read profile {name} must be True or False")
    ceiling = _exact_int(descriptor["ceiling_nanousd"], "ceiling_nanousd", 0, 10**18)
    if descriptor["metered"] != (ceiling > 0):
        raise WorkflowBindingError(
            "A metered read profile declares a positive ceiling_nanousd, its per-call price; "
            "a free one declares 0"
        )
    _exact_int(descriptor["max_response_bytes"], "max_response_bytes", 1, MAX_RESPONSE_BYTES)
    _exact_int(descriptor["max_redirects"], "max_redirects", 0, MAX_REDIRECTS)
    timeout = descriptor["timeout_s"]
    if type(timeout) is not float or not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT_S:
        raise WorkflowBindingError(f"Read profile timeout_s must be above 0 and at most {MAX_TIMEOUT_S}")
    host = _template(descriptor["url"], schema)
    hosts = descriptor["allowed_hosts"]
    if (type(hosts) is not list or [_host(item) for item in hosts] != sorted(set(hosts))
            or host not in hosts):
        raise WorkflowBindingError("Read profile allowed_hosts must be sorted, unique and include its url host")
    return descriptor


def _check_input(schema: Mapping[str, str], value: object) -> dict:
    if not isinstance(value, Mapping) or set(value) != set(schema):
        raise WorkflowBindingError(
            f"Read input must have exactly the fields {sorted(schema)}")
    checked = {}
    for name, kind in schema.items():
        item = value[name]
        if type(item) is not _INPUT_TYPES[kind]:
            raise WorkflowBindingError(f"Read input {name!r} must be a {kind}")
        if kind == "string" and (not item or len(item) > 512 or item in (".", "..")
                                 or any(ord(char) < 32 for char in item)):
            raise WorkflowBindingError(
                f"Read input {name!r} must be nonempty text of at most 512 characters, "
                "without control characters, and not '.' or '..'"
            )
        if kind == "integer" and abs(item) >= 10**18:
            raise WorkflowBindingError(f"Read input {name!r} is out of range")
        checked[name] = item
    return checked


def _request(descriptor: dict, value: object) -> dict:
    """The exact request a profile description builds from *value*."""
    checked = _check_input(descriptor["input_schema"], value)
    url = "".join(
        literal + ("" if name is None else quote(str(checked[name]), safe=""))
        for literal, name, _, _ in string.Formatter().parse(descriptor["url"])
    )
    return {"profile": descriptor["name"], "version": descriptor["version"], "method": "GET",
            "url": url, "input": checked}


def check_read_request(request: object, descriptor: object) -> None:
    """Raise ValueError unless *request* is what *descriptor* builds from its input.

    The journal applies this pure check before it admits a ``read_tool`` call.
    """
    _check_descriptor(descriptor)
    if type(request) is not dict or request != _request(descriptor, request.get("input")):
        raise WorkflowBindingError("Read request does not match its profile")


def reinvocable(record: Mapping | None) -> bool:
    """Whether a journaled call is a free, repeat-safe read settled as rejected.

    Such a read cost nothing and is safe to send again, so entering its node
    again makes a new call rather than replaying the rejection.
    """
    provider = record.get("provider") if record is not None else None
    return (type(provider) is dict and provider.get("kind") == "read_tool"
            and provider.get("metered") is False and provider.get("repeat_safe") is True
            and record["billing_state"] == "known" and record["result_state"] == "rejected")


def _utf8(body: bytes) -> str:
    return body.decode("utf-8")


@dataclass(frozen=True, kw_only=True)
class ReadProfile:
    """One versioned read operation that a durable run may journal.

    ``url`` is an http or https URL whose ``{field}`` placeholders, in its
    path or query only, take the percent-encoded values of a step's input.
    ``input_schema`` maps each required input field to ``"string"`` or
    ``"integer"``. ``metered`` states whether a read can incur a charge: a
    metered profile declares its per-call price as ``ceiling_nanousd``,
    which each read reserves and a successful one is charged, and a free one
    declares 0. ``repeat_safe`` states whether sending the read again is
    harmless. Neither has a default.

    ``decode`` turns a 200 response body into the step's text, as local code
    with no side effects; ``decoder_version`` names it, and a custom decoder
    needs its own. Without one the body is read as strict UTF-8.

    Each read follows at most ``max_redirects`` redirects, to http or https
    URLs on ``allowed_hosts`` (the url's host is always allowed) without a
    downgrade from https, reads at most ``max_response_bytes`` and takes at
    most ``timeout_s`` seconds. Unless ``allow_private_networks`` is set,
    every address a host resolves to must be a public one: loopback,
    private, link-local and other special addresses are refused, before the
    first request and at each redirect. Requests go to the address that was
    checked, never through a proxy, and ask for an uncompressed body.

    The run's recipe records the profile description, without the decoder
    code, so resuming with a changed profile is refused.
    """

    name: str
    version: str
    url: str
    input_schema: Mapping[str, str]
    metered: bool
    repeat_safe: bool
    ceiling_nanousd: int = 0
    decode: Callable[[bytes], str] | None = field(default=None, repr=False, compare=False)
    decoder_version: str = TEXT_DECODER_VERSION
    max_response_bytes: int = 1_000_000
    timeout_s: float = 30.0
    max_redirects: int = 0
    allowed_hosts: tuple[str, ...] = ()
    allow_private_networks: bool = False

    def __post_init__(self) -> None:
        if type(self) is not ReadProfile:
            raise WorkflowBindingError("Read profiles must be the built-in ReadProfile")
        if not isinstance(self.input_schema, Mapping):
            raise WorkflowBindingError("Read profile input_schema must be a mapping")
        object.__setattr__(self, "input_schema", MappingProxyType(dict(self.input_schema)))
        if isinstance(self.allowed_hosts, (str, bytes)):
            raise WorkflowBindingError("Read profile allowed_hosts must be a sequence of host names")
        object.__setattr__(self, "allowed_hosts", tuple(_host(host) for host in self.allowed_hosts))
        timeout = self.timeout_s
        if type(timeout) in (int, float):
            object.__setattr__(self, "timeout_s", float(timeout))
        if self.decode is not None:
            if (not callable(self.decode) or inspect.iscoroutinefunction(self.decode)
                    or inspect.iscoroutinefunction(getattr(type(self.decode), "__call__", None))):
                raise WorkflowBindingError("Read profile decode must be a synchronous callable")
            if self.decoder_version == TEXT_DECODER_VERSION:
                raise WorkflowBindingError("A custom read decoder needs its own decoder_version")
        self.workflow_descriptor()

    def workflow_descriptor(self) -> dict:
        """The profile as JSON data, without decoder code; the journal binds every call to it."""
        try:
            host = urlsplit(self.url).hostname if type(self.url) is str else None
        except ValueError:
            host = None  # The url check below reports it.
        descriptor = {
            "kind": "read_tool", "adapter_version": READ_ADAPTER_VERSION,
            "decoder_version": self.decoder_version, "name": self.name, "version": self.version,
            "url": self.url, "input_schema": dict(self.input_schema), "metered": self.metered,
            "repeat_safe": self.repeat_safe, "ceiling_nanousd": self.ceiling_nanousd,
            "max_response_bytes": self.max_response_bytes, "timeout_s": self.timeout_s,
            "max_redirects": self.max_redirects,
            "allowed_hosts": sorted({*self.allowed_hosts, *([host] if host else [])}),
            "allow_private_networks": self.allow_private_networks,
        }
        return _check_descriptor(descriptor)

    def request(self, input: Mapping[str, Any]) -> dict:
        """The exact request a step with *input* records; raises for invalid input."""
        return _request(self.workflow_descriptor(), input)

    def step(self, node_id: str, input: Mapping[str, Any] | None = None, *,
             label: str | None = None, **node_fields: Any) -> Node:
        """A graph node that reads with this profile, after checking *input*.

        ``node_fields`` are other ``Node`` fields, such as ``depends_on``,
        ``failure_policy`` or ``max_retries``.
        """
        from smythe.graph import Node

        value = dict(input or {})
        self.request(value)
        metadata = dict(node_fields.pop("metadata", None) or {})
        if READ_STEP_KEY in metadata:
            raise WorkflowBindingError(f"metadata already declares {READ_STEP_KEY!r}")
        metadata[READ_STEP_KEY] = {"profile": self.name, "input": value}
        return Node(id=node_id, label=label or f"Read {self.name}", metadata=metadata, **node_fields)

    def _decode(self, body: bytes) -> object:
        return (self.decode or _utf8)(body)


# ---------------------------------------------------------------------------
# HTTP reads
# ---------------------------------------------------------------------------


class _Refused(Exception):
    """A limit refused a destination or response; ``args[0]`` names the limit."""


class _Response(NamedTuple):
    status: int | None  # None when no HTTP response arrived
    body: bytes
    error: str | None  # an exception class or limit name, never response text


def _public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(address, "ipv4_mapped", None)
    address = mapped or address
    return address.is_global and not address.is_multicast


def _addresses(host: str, port: int, allow_private: bool) -> list[str]:
    """Resolve *host*; refuse unless every address is public or private ones are allowed.

    An IP address is used as written, without a lookup.
    """
    try:
        found = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as error:
            raise _Refused("DestinationUnresolved") from error
        found = [ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]) for info in infos]
    if not found or not allow_private and not all(_public(address) for address in found):
        raise _Refused("DestinationRefused")
    return list(dict.fromkeys(str(address) for address in found))


def _pinned(addresses: list[str]):
    """A connection factory for http.client that connects only to checked addresses."""

    def connect(address, timeout=None, source_address=None):
        error: OSError | None = None
        for candidate in addresses:
            try:
                return socket.create_connection((candidate, address[1]), timeout, source_address)
            except OSError as failure:
                error = failure
        raise error or OSError("No address to connect to")

    return connect


def _fetch(descriptor: dict, url: str, addresses: list[str], deadline: float) -> _Response:
    """Send one GET and follow its allowed redirects; never retry.

    Runs in a worker thread. A failure becomes the response's ``error``.
    """
    import http.client
    import ssl

    limit, hops, status = descriptor["max_response_bytes"], 0, None
    while True:
        parts = urlsplit(url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _Response(status, b"", "TimeoutError")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if parts.scheme == "https":
            connection = http.client.HTTPSConnection(
                parts.hostname, port, timeout=remaining, context=ssl.create_default_context())
        else:
            connection = http.client.HTTPConnection(parts.hostname, port, timeout=remaining)
        # Connect to the address that was checked, so a second lookup cannot
        # move the read; TLS still verifies the certificate for the host name.
        connection._create_connection = _pinned(addresses)
        status = None
        try:
            target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
            connection.request("GET", target, headers={"Accept": "*/*", "User-Agent": "smythe-read/1"})
            response = connection.getresponse()
            if not 100 <= response.status <= 599:
                return _Response(None, b"", "InvalidHTTPStatus")
            status = response.status
            location = response.getheader("Location")
            if status in _REDIRECT_STATUSES and location:
                if hops >= descriptor["max_redirects"]:
                    return _Response(status, b"", "RedirectLimitExceeded")
                url, scheme = urljoin(url, location), parts.scheme
                parts = urlsplit(url)
                if (parts.scheme not in ("http", "https") or scheme == "https" and parts.scheme == "http"
                        or parts.hostname not in descriptor["allowed_hosts"]
                        or parts.username is not None or parts.password is not None):
                    return _Response(status, b"", "DestinationRefused")
                try:
                    addresses = _addresses(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80),
                                           descriptor["allow_private_networks"])
                except _Refused as refused:
                    return _Response(status, b"", refused.args[0])
                hops += 1
                continue
            declared = response.getheader("Content-Length")
            if declared is not None and declared.strip().isdigit() and int(declared) > limit:
                return _Response(status, b"", "ResponseTooLarge")
            body = response.read(limit + 1)
            if len(body) > limit:
                return _Response(status, b"", "ResponseTooLarge")
            return _Response(status, body, None)
        except (OSError, http.client.HTTPException, ValueError) as error:
            return _Response(status, b"", type(error).__name__)
        finally:
            connection.close()


# ---------------------------------------------------------------------------
# Journaled reads
# ---------------------------------------------------------------------------


def _reason_text(value: str, limit: int) -> str:
    """Bounded, single-line text for a journal reason, with identifiers redacted."""
    text = " ".join("".join(" " if ord(char) < 32 else char
                            for char in redact_account_identifiers(value)).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


class JournaledReadTool:
    """One read step bound to one immutable journal call key.

    ``read()`` follows the call's recorded state. An accepted result replays
    its saved text, and a saved response is settled and decoded locally;
    neither reads again. A call never dispatched is quoted, reserved and sent
    once. A rejection replays as ``ReadToolError``. A call that was
    dispatched but holds no response or no known charge raises
    ``WorkflowStateError``: it is never sent again.
    """

    def __init__(self, context: WorkflowProviderContext, profile: ReadProfile, key: CallKey,
                 input: Mapping[str, Any]) -> None:
        if type(profile) is not ReadProfile:
            raise WorkflowBindingError("A journaled read needs a ReadProfile")
        self._context, self.profile, self.key = context, profile, key
        self._descriptor = profile.workflow_descriptor()
        self.request = _request(self._descriptor, input)
        self._request_json = _canonical(self.request)
        self._request_sha = hashlib.sha256(self._request_json.encode()).hexdigest()

    async def read(self) -> CompletionResult:
        context = self._context
        context._check_active()
        lock = context._call_locks.setdefault(self.key, asyncio.Lock())
        async with lock:
            context._check_active()
            return await self._execute()

    def _fail(self, reason: str) -> ReadToolError:
        return ReadToolError(f"Read step {self.profile.name!r} failed: {reason}")

    def _result(self, record: dict, text: str, *, replayed: bool) -> CompletionResult:
        receipt = dict(record["receipt"])
        receipt.update(workflow_run_id=record["run_id"], workflow_call_id=record["call_id"],
                       workflow_charge_recorded=True, workflow_replayed=replayed)
        return CompletionResult(text, cost_usd=record["cost_nanousd"] / 1_000_000_000,
                                native_receipt=receipt)

    async def _execute(self) -> CompletionResult:
        context, store = self._context, self._context.store
        record = store.prepare_call(context.lease, self.key, request_json=self._request_json,
                                    provider=self._descriptor, price_version=READ_PRICE_VERSION)
        call_id = record["call_id"]
        if record["result_state"] in ("accepted", "applied"):
            text = (record["decoded_result"] or {}).get("text")
            if type(text) is not str:
                raise WorkflowStateError("Saved read result has no text")
            return self._result(record, text, replayed=True)
        if record["result_state"] == "rejected":
            raise self._fail(record["error"] or "rejected")
        if record["evidence_id"] is not None:
            return self._finish(record, replayed=True)
        if record["state"] in ("dispatched", "unknown"):
            raise WorkflowStateError(
                f"Read step {self.profile.name!r} was dispatched without a saved response, "
                "or its charge is unknown; it cannot be sent again"
            )
        if record["quote_id"] is None:
            # The quote is the profile's declared per-call ceiling, local evidence.
            evidence_id = store.append_quote_evidence(context.lease, call_id, {
                "body": _canonical({"provider_kind": "read_tool", "version": 1,
                                    "ceiling_nanousd": self.profile.ceiling_nanousd}).encode(),
                "operation": "input_tokens", "request_sha256": self._request_sha,
                "status_code": 200, "request_id": None, "transport_error": None,
            })
            record["quote_id"] = store.accept_quote(context.lease, call_id, evidence_id)["quote_id"]
        store.reserve_call(context.lease, call_id, record["quote_id"])
        parts = urlsplit(self.request["url"])
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            addresses = await asyncio.to_thread(
                _addresses, parts.hostname, port, self.profile.allow_private_networks)
        except _Refused as refused:
            # Nothing was sent: the call returns its reservation and may be sent later.
            store.release_before_dispatch(context.lease, call_id, refused.args[0])
            raise self._fail(f"destination refused before dispatch ({refused.args[0]})") from None
        permit = store.claim_dispatch(context.lease, call_id, self._request_sha)
        interrupted: BaseException | None = None
        try:
            response = await asyncio.wait_for(asyncio.to_thread(
                _fetch, self._descriptor, self.request["url"], addresses,
                time.monotonic() + self.profile.timeout_s), self.profile.timeout_s)
        except BaseException as error:  # A timeout, or cancellation of the node.
            interrupted, response = error, _Response(None, b"", type(error).__name__)
        try:
            record["evidence_id"] = store.append_response(permit, {
                "body": response.body, "operation": "response", "request_sha256": self._request_sha,
                "status_code": response.status, "request_id": None,
                "transport_error": response.error,
            })
            if interrupted is not None and not isinstance(interrupted, Exception):
                store.settle_call(context.lease, call_id, record["evidence_id"])
        except BaseException as error:
            context._failed = True
            if not isinstance(error, Exception):
                raise
            if interrupted is not None and not isinstance(interrupted, Exception):
                raise interrupted from error
            raise WorkflowStateError("Read response could not complete its journal transition") from error
        if interrupted is not None and not isinstance(interrupted, Exception):
            raise interrupted  # Cancellation propagates once the outcome is saved.
        try:
            return self._finish(record, replayed=False)
        except (ReadToolError, WorkflowStateError) as error:
            if interrupted is not None and error.__cause__ is None:
                error.__cause__ = interrupted  # Keep a timeout's or failure's traceback.
            raise

    def _finish(self, record: dict, *, replayed: bool) -> CompletionResult:
        context, store = self._context, self._context.store
        call_id = record["call_id"]
        record = store.settle_call(context.lease, call_id, record["evidence_id"])
        if record["billing_state"] != "known":
            raise WorkflowStateError(
                f"Read step {self.profile.name!r} has no usable response and its charge is "
                f"unknown; workflow admission is closed: {record['blocked_reason']}"
            )
        if record["result_state"] == "rejected":
            raise self._fail(record["error"] or "rejected")
        if record["admission_closed"]:
            # A latch elsewhere is not this read's failure: leave its text undecoded.
            raise WorkflowStateError(
                "Workflow admission closed after settling this read: " + str(record["blocked_reason"]))
        body = store.load_evidence(call_id, record["evidence_id"])["body"]
        version = self.profile.decoder_version
        try:
            text = self.profile._decode(body)
        except Exception as error:
            detail = _reason_text(str(error), 200)
            reason = f"decoder {version} raised {type(error).__name__}" + (f": {detail}" if detail else "")
        else:
            if type(text) is str:
                store.accept_result(context.lease, call_id, {"text": text}, decoder_version=version)
                return self._result(record, text, replayed=replayed)
            if inspect.iscoroutine(text):
                text.close()
            reason = f"decoder {version} returned {type(text).__name__}, not text"
        store.reject_result(context.lease, call_id, _reason_text(reason, 500))
        raise self._fail(_reason_text(reason, 500))
