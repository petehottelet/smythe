"""A reference write connector and a test approval host for the fake external service.

``FixtureConnector`` implements ``smythe.actions.WriteConnector`` over an
in-process ``FixtureService``, and ``HttpFixtureConnector`` does the same over
``serve_http``'s loopback port. Each call makes exactly one request and never
retries, so every service fault maps to one outcome: a refusal before commit or
an unknown object is ``ConnectorRefused``, and a lost acknowledgement, closed
connection or timeout is ``ConnectorAmbiguous``. A connector's profile declares
only what the service's configuration guarantees.

The service stores each write as JSON holding the exact bytes, and a read
returns those bytes, so ``ObservedObject.content_ref`` hashes what the service
returned. An intent without content writes its canonical parameters as
``application/json``.

``HmacApprovalHost`` implements ``smythe.actions.ApprovalVerifier``. It signs
assertions with a random key drawn from ``secrets`` for each instance; pass
``key=`` only to share one instance's key with a child process.

Import it by module name (``from fixture_connector import FixtureConnector``).
It imports only the standard library, smythe and the other test fakes.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict
import hashlib
import hmac
import http.client
import ipaddress
import json
import secrets
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
import urllib.request

from fixture_service import (
    AcknowledgementLost, Endpoint, FixtureService, LookupUnavailable, NotFound, ServiceUnavailable,
    Target,
)
from smythe._canonical import canonical_bytes
from smythe.actions import (
    ActionIntent, ActionTarget, ApprovalRejected, ConnectorAmbiguous, ConnectorProfile,
    ConnectorReceipt, ConnectorRefused, ObservedObject, Principal,
)

NAME = "fixture-service"
VERSION = "1"
OPERATIONS = ("publish", "update")  # publish creates an object; update writes its next version
PARAMETERS_MEDIA_TYPE = "application/json"
_STORED_KEYS = frozenset({"media_type", "body_base64", "parameters"})
_DERIVED = frozenset({"deduplicates", "dedupe_window_s", "lookup_by_key", "lookup_by_object",
                      "visibility_bound_s"})


def fixture_profile(service: FixtureService, **settings: Any) -> ConnectorProfile:
    """The profile a connector to ``service`` declares.

    Deduplication, its window, lookups and the visibility bound come from the
    service's configuration; ``settings`` adds the rest, such as ceilings,
    attempts and readback. A key that expires without deduplication has no
    profile field for its lifetime, so lookup by key is then left undeclared:
    an empty lookup after expiry would prove nothing.
    """
    derived = sorted(_DERIVED & settings.keys())
    if derived:
        raise ValueError(f"{derived} come from the service configuration")
    return ConnectorProfile(
        NAME, VERSION, OPERATIONS, deduplicates=service.dedupe,
        dedupe_window_s=service.key_ttl_s if service.dedupe else None,
        lookup_by_key=service.lookup_by_key and (service.dedupe or service.key_ttl_s is None),
        lookup_by_object=service.lookup_by_object,
        visibility_bound_s=service.visibility_delay_s, **settings,
    )


def stored_value(intent: ActionIntent, payload: bytes | None) -> dict:
    """The JSON value the service stores for ``intent``; raises before anything is sent.

    ``publish`` needs a target without an object ID and ``update`` one with it.
    ``payload`` is required exactly when the intent names content.
    """
    if intent.operation not in OPERATIONS:
        raise ValueError(f"The fixture service has no {intent.operation!r} operation")
    if (intent.operation == "update") != (intent.target.object_id is not None):
        raise ValueError("publish creates an object; update names the object it writes")
    if (payload is None) != (intent.content is None):
        raise ValueError("A payload is given exactly when the intent names content")
    if payload is None:
        body, media_type = canonical_bytes(intent.parameters), PARAMETERS_MEDIA_TYPE
    elif isinstance(payload, bytes):
        body, media_type = payload, intent.content.media_type
    else:
        raise TypeError("A payload is bytes")
    return {"media_type": media_type, "body_base64": base64.b64encode(body).decode("ascii"),
            "parameters": intent.parameters}


def stored_bytes(value: Any) -> tuple[bytes, str]:
    """The bytes and media type an object's value holds.

    A value this connector wrote decodes to the bytes it sent; any other value,
    such as one from another writer, reads as its canonical JSON.
    """
    if (type(value) is dict and value.keys() == _STORED_KEYS
            and type(value["media_type"]) is str and "/" in value["media_type"]
            and type(value["body_base64"]) is str):
        try:
            return base64.b64decode(value["body_base64"], validate=True), value["media_type"]
        except (binascii.Error, ValueError):
            pass
    return canonical_bytes(value), PARAMETERS_MEDIA_TYPE


def _service_target(target: ActionTarget) -> Target:
    return Target(target.tenant, target.collection)


def _receipt(data: dict) -> ConnectorReceipt:
    """A connector receipt from the service's receipt JSON, naming the service's own target."""
    target = data["target"]
    return ConnectorReceipt(
        data["request_key"], ActionTarget(target["tenant"], target["collection"], data["object_id"]),
        object_version=data["object_version"], deduplicated=data["deduplicated"])


def _observed(data: dict) -> ObservedObject:
    """What a read returned, with the object's own target and its exact bytes."""
    target = data["target"]
    content, media_type = stored_bytes(data["value"])
    return ObservedObject(ActionTarget(target["tenant"], target["collection"], data["object_id"]),
                          content, media_type, version=data["version"], key=data["request_key"])


def _object_id(target: ActionTarget) -> str:
    if target.object_id is None:
        raise ValueError("A read names the target's object_id")
    return target.object_id


class FixtureConnector:
    """A ``WriteConnector`` for an in-process ``FixtureService``.

    ``settings`` go to :func:`fixture_profile`. ``timeout_s`` has no effect in
    process: a delivery held at a ``Gate`` waits for its release.
    """

    def __init__(self, service: FixtureService, **settings: Any) -> None:
        self.service = service
        self.profile = fixture_profile(service, **settings)

    def submit(self, key: str, intent: ActionIntent, payload: bytes | None,
               timeout_s: float) -> ConnectorReceipt:
        value = stored_value(intent, payload)
        try:
            receipt = self.service.submit(key, value, object_id=intent.target.object_id,
                                          target=_service_target(intent.target))
        except (ServiceUnavailable, NotFound) as error:
            raise ConnectorRefused(str(error)) from error
        except AcknowledgementLost as error:
            raise ConnectorAmbiguous(str(error)) from error
        return _receipt(asdict(receipt))

    def lookup(self, key: str, target: ActionTarget, timeout_s: float) -> ConnectorReceipt | None:
        """The receipt ``key`` committed in the target's tenant; raises ``LookupUnavailable``."""
        receipt = self.service.lookup_request(key, tenant=target.tenant)
        return None if receipt is None else _receipt(asdict(receipt))

    def read(self, target: ActionTarget, timeout_s: float) -> ObservedObject | None:
        """The object's latest visible version in the target's tenant."""
        record = self.service.get_object(_object_id(target), tenant=target.tenant)
        return None if record is None else _observed(asdict(record))


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HttpFixtureConnector:
    """A ``WriteConnector`` for ``serve_http``, over loopback ``urllib`` with no retries.

    A 200 answer is a receipt and 503 or 404 a refusal. A refused connection
    sent nothing, so it is a refusal too. A closed connection, a timeout, an
    unreadable answer or any other status may follow a commit, so it is
    ambiguous. Lookups and reads answer 404 with None and 501 with
    ``LookupUnavailable``, and raise ``OSError`` when the service cannot answer.
    """

    def __init__(self, endpoint: Endpoint, profile: ConnectorProfile) -> None:
        if not isinstance(profile, ConnectorProfile):
            raise TypeError("profile must be a ConnectorProfile")
        if not _loopback(endpoint.host):
            raise ValueError("The fixture connector talks only to a loopback address")
        self.endpoint = endpoint
        self.profile = profile
        # No proxy: a request never leaves the loopback interface.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _call(self, method: str, path: str, body: dict | None, timeout_s: float):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.endpoint.url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=timeout_s) as response:
                return response.status, response.read()
        except HTTPError as error:
            with error:
                return error.code, error.read()

    def submit(self, key: str, intent: ActionIntent, payload: bytes | None,
               timeout_s: float) -> ConnectorReceipt:
        body = {"request_key": key, "value": stored_value(intent, payload),
                "object_id": intent.target.object_id,
                "target": asdict(_service_target(intent.target))}
        try:
            status, answer = self._call("POST", "/requests", body, timeout_s)
        except URLError as error:
            if isinstance(error.reason, ConnectionRefusedError):
                raise ConnectorRefused("The service refused the connection") from error
            raise ConnectorAmbiguous(f"The request failed: {error.reason}") from error
        except (http.client.HTTPException, OSError) as error:
            raise ConnectorAmbiguous(f"The answer was lost: {error!r}") from error
        if status in (404, 503):
            raise ConnectorRefused(f"The service refused the request with {status}")
        if status != 200:
            raise ConnectorAmbiguous(f"The service answered {status}")
        try:
            return _receipt(json.loads(answer))
        except (KeyError, TypeError, ValueError) as error:
            raise ConnectorAmbiguous("The service's receipt was unreadable") from error

    def _get(self, kind: str, name: str, tenant: str, timeout_s: float) -> dict | None:
        path = f"/{kind}/{quote(name, safe='')}?{urlencode({'tenant': tenant})}"
        status, answer = self._call("GET", path, None, timeout_s)
        if status == 404:
            return None
        if status == 501:
            raise LookupUnavailable(f"The service has no lookup of {kind}")
        if status != 200:
            raise ConnectionError(f"The service answered {status}")
        return json.loads(answer)

    def lookup(self, key: str, target: ActionTarget, timeout_s: float) -> ConnectorReceipt | None:
        data = self._get("requests", key, target.tenant, timeout_s)
        return None if data is None else _receipt(data)

    def read(self, target: ActionTarget, timeout_s: float) -> ObservedObject | None:
        data = self._get("objects", _object_id(target), target.tenant, timeout_s)
        return None if data is None else _observed(data)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


_CLAIMS = frozenset({"iss", "sub", "tenant", "subject_sha256", "exp", "jti"})


class HmacApprovalHost:
    """A test ``ApprovalVerifier``: HMAC-signed assertions, revocation and expiry.

    ``issue`` signs an assertion that ``subject`` of ``tenant`` approves
    exactly one digest until ``expires_at_ns``: an action's approval subject,
    ``PreparedAction.approval_sha256``, or another digest such as an operator
    decision's. ``verify`` accepts only this host's own signature for that
    digest, before expiry, while neither the assertion nor its subject is
    revoked. ``still_authorized`` holds while
    the subject is not revoked and the intent is in the principal's tenant.
    """

    identity = "fixture-hmac/1"

    def __init__(self, *, issuer: str = "fixture-identity-host", key: bytes | None = None) -> None:
        if key is None:
            key = secrets.token_bytes(32)
        if type(key) is not bytes or len(key) < 32:
            raise ValueError("An HMAC key is at least 32 bytes")
        self.issuer = issuer
        self.key = key
        self._revoked_assertions: set[str] = set()
        self._revoked_subjects: set[str] = set()

    def _sign(self, claims: str) -> str:
        return _b64(hmac.new(self.key, claims.encode("ascii"), hashlib.sha256).digest())

    def issue(self, subject_sha256: str, *, subject: str, tenant: str, expires_at_ns: int) -> str:
        if type(expires_at_ns) is not int:
            raise ValueError("An expiry is integer nanoseconds")
        claims = _b64(canonical_bytes({
            "iss": self.issuer, "sub": subject, "tenant": tenant, "subject_sha256": subject_sha256,
            "exp": expires_at_ns, "jti": secrets.token_hex(16)}))
        return f"{claims}.{self._sign(claims)}"

    def _claims(self, assertion: Any) -> dict:
        """The claims of an assertion this host signed; anything else is rejected."""
        if type(assertion) is not str or not assertion.isascii() or assertion.count(".") != 1:
            raise ApprovalRejected("The assertion is malformed")
        claims, signature = assertion.split(".")
        if not claims or not hmac.compare_digest(signature, self._sign(claims)):
            raise ApprovalRejected("The assertion's signature is not this host's")
        try:
            data = json.loads(_unb64(claims))
        except (binascii.Error, ValueError) as error:
            raise ApprovalRejected("The assertion is malformed") from error
        if type(data) is not dict or data.keys() != _CLAIMS:
            raise ApprovalRejected("The assertion is malformed")
        return data

    def verify(self, assertion: str, subject_sha256: str, now_ns: int) -> Principal:
        claims = self._claims(assertion)
        if type(now_ns) is not int:
            raise ApprovalRejected("The verification time must be integer nanoseconds")
        if claims["iss"] != self.issuer:
            raise ApprovalRejected("The assertion names another issuer")
        if claims["subject_sha256"] != subject_sha256:
            raise ApprovalRejected("The assertion approves another digest")
        if now_ns >= claims["exp"]:
            raise ApprovalRejected("The assertion has expired")
        if claims["jti"] in self._revoked_assertions or claims["sub"] in self._revoked_subjects:
            raise ApprovalRejected("The approval was revoked")
        return Principal(self.issuer, claims["sub"], claims["tenant"])

    def still_authorized(self, principal: Principal, intent: ActionIntent, now_ns: int) -> bool:
        return (principal.issuer == self.issuer
                and principal.subject not in self._revoked_subjects
                and principal.tenant == intent.target.tenant)

    def revoke(self, assertion: str) -> None:
        """Revoke one assertion this host issued."""
        self._revoked_assertions.add(self._claims(assertion)["jti"])

    def revoke_subject(self, subject: str) -> None:
        """Revoke every approval by ``subject``, past and future."""
        self._revoked_subjects.add(subject)
