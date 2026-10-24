"""The reference connector and test approval host follow the external-action contract."""

from __future__ import annotations

import base64
from contextlib import ExitStack
from dataclasses import replace
import http.client
import json
import subprocess

import pytest

from fakes import Gate, ManualClock, child_executable, offline_env
from fixture_connector import (
    FixtureConnector, HmacApprovalHost, HttpFixtureConnector, fixture_profile, stored_bytes,
)
from fixture_service import Endpoint, FixtureService, LookupUnavailable, Target, serve_http
from smythe._canonical import canonical_bytes, sha256_hex
from smythe.actions import (
    ActionIntent, ActionTarget, ApprovalRejected, ConnectorAmbiguous, ConnectorReceipt,
    ConnectorRefused, ContentRef, Principal, action_key, check_connector, check_verifier,
)

STORE = "0" * 32
KEY = action_key(STORE, "run-1", "publish", 1)
BODY = "# Launch notes\n\nCafé opens at nine.\n".encode()
REVISED = b"# Launch notes\n\nOpens at ten.\n"
ACME = ActionTarget("acme", "posts")
CREATED = ActionTarget("acme", "posts", "obj_1")
NOW = 1_000_000_000_000
EXPIRES = NOW + 60 * 10**9


def intent(operation="publish", target=ACME, body=BODY, parameters=None):
    return ActionIntent(
        operation, target,
        parameters={"title": "Launch notes"} if parameters is None else parameters,
        content=None if body is None else ContentRef.for_bytes(body, "text/markdown"),
        connector="fixture-service", connector_version="1",
    )


def counters(service, key):
    state = service.snapshot()["requests"][key]
    return {name: state[name] for name in (
        "received", "committed", "deduplicated", "refused", "acknowledged", "ack_dropped")}


def expected(**values):
    return {"received": 0, "committed": 0, "deduplicated": 0, "refused": 0, "acknowledged": 0,
            "ack_dropped": 0, **values}


@pytest.fixture(params=["memory", "http"])
def connect(request):
    """Build a service and a connector to it, in process or over loopback HTTP."""
    with ExitStack() as stack:
        def build(**config):
            service = FixtureService(**config)
            if request.param == "memory":
                return service, FixtureConnector(service)
            endpoint = stack.enter_context(serve_http(service))
            return service, HttpFixtureConnector(endpoint, fixture_profile(service))

        yield build


# Profiles


@pytest.mark.parametrize("config,declared", [
    ({}, {"deduplicates": True, "lookup_by_key": True, "lookup_by_object": True,
          "visibility_bound_s": 0.0}),
    ({"key_ttl_s": 60, "visibility_delay_s": 2},
     {"deduplicates": True, "dedupe_window_s": 60.0, "lookup_by_key": True,
      "lookup_by_object": True, "visibility_bound_s": 2.0}),
    ({"dedupe": False, "lookup_by_object": False},
     {"lookup_by_key": True, "visibility_bound_s": 0.0}),
    # A key that expires without deduplication has no declared window, so an
    # empty lookup after it would prove nothing: the lookup is not declared.
    ({"dedupe": False, "key_ttl_s": 30},
     {"lookup_by_object": True, "visibility_bound_s": 0.0}),
], ids=["defaults", "window-and-delay", "no-dedupe", "expiring-key-without-dedupe"])
def test_profile_is_derived_from_the_service_configuration(config, declared):
    profile = FixtureConnector(FixtureService(**config)).profile
    assert profile.to_dict() == {
        "name": "fixture-service", "version": "1", "operations": ["publish", "update"],
        **declared}


def test_profile_settings_add_only_what_the_service_does_not_declare():
    service = FixtureService()
    connector = FixtureConnector(service, max_attempts=3, readback_reads=2,
                                 write_ceiling_nanousd=5_000)
    assert (connector.profile.max_attempts, connector.profile.readback_reads,
            connector.profile.write_ceiling_nanousd) == (3, 2, 5_000)
    assert fixture_profile(service, max_attempts=3, readback_reads=2,
                           write_ceiling_nanousd=5_000) == connector.profile
    for name, value in (("deduplicates", False), ("visibility_bound_s", 9),
                        ("lookup_by_key", False)):
        with pytest.raises(ValueError, match="service configuration"):
            FixtureConnector(service, **{name: value})


def test_connectors_pass_the_structural_check(connect):
    service, connector = connect()
    assert check_connector(connector) is connector.profile == fixture_profile(service)


# Writes, lookups and reads


def test_publish_lookup_and_read_round_trip(connect):
    service, connector = connect(send_notices=True)
    receipt = connector.submit(KEY, intent(), BODY, 5)
    assert receipt == ConnectorReceipt(KEY, CREATED, object_version=1)
    assert connector.lookup(KEY, ACME, 5) == receipt
    observed = connector.read(CREATED, 5)
    assert (observed.target, observed.version, observed.key, observed.media_type) == (
        CREATED, 1, KEY, "text/markdown")
    # The digest comes from the bytes the service returned.
    assert observed.content == BODY and observed.content_ref == intent().content
    update_key = action_key(STORE, "run-1", "publish", 2)
    revised = connector.submit(update_key, intent("update", CREATED, REVISED), REVISED, 5)
    assert revised == ConnectorReceipt(update_key, CREATED, object_version=2)
    assert connector.read(CREATED, 5).content_ref == ContentRef.for_bytes(REVISED, "text/markdown")
    assert connector.submit(KEY, intent(), BODY, 5) == replace(receipt, deduplicated=True)
    assert service.wait_idle(10)
    stored = service.snapshot()["objects"]["obj_1"]["history"][0]["value"]
    assert stored_bytes(stored) == (BODY, "text/markdown")
    assert stored["parameters"] == {"title": "Launch notes"}
    assert [notice["payload"]["request_key"] for notice in service.notices()] == [KEY, update_key]


@pytest.mark.parametrize("schedule,error,outcome", [
    ("fail_before_commit", ConnectorRefused, {"refused": 1}),
    ("commit_then_drop_ack", ConnectorAmbiguous, {"committed": 1, "ack_dropped": 1}),
], ids=["refused", "lost-acknowledgement"])
def test_each_fault_maps_to_one_outcome_from_one_delivery(connect, schedule, error, outcome):
    service, connector = connect()
    getattr(service, schedule)(KEY)
    with pytest.raises(error):
        connector.submit(KEY, intent(), BODY, 5)
    assert service.wait_idle(10)
    assert counters(service, KEY) == expected(received=1, **outcome)
    assert len(service.snapshot()["deliveries"]) == 1


def test_an_update_of_an_unknown_object_is_refused(connect):
    service, connector = connect()
    with pytest.raises(ConnectorRefused):
        connector.submit(KEY, intent("update", ActionTarget("acme", "posts", "obj_404")), BODY, 5)
    assert service.wait_idle(10)
    assert counters(service, KEY) == expected(received=1, refused=1)


def test_a_duplicate_delivery_has_one_effect(connect):
    service, connector = connect()
    service.deliver_duplicate(KEY)
    receipt = connector.submit(KEY, intent(), BODY, 5)
    assert not receipt.deduplicated
    assert service.wait_idle(10)
    state = service.snapshot()
    assert counters(service, KEY) == expected(received=2, committed=1, deduplicated=1,
                                              acknowledged=1)
    # The connector sent one copy; the network delivered the other.
    assert [delivery["duplicate"] for delivery in state["deliveries"]] == [False, True]
    assert list(state["objects"]) == ["obj_1"]


def test_other_tenants_keys_and_objects_are_hidden(connect):
    service, connector = connect()
    receipt = connector.submit(KEY, intent(), BODY, 5)
    globex = ActionTarget("globex", "posts", "obj_1")
    assert connector.lookup(KEY, ActionTarget("globex", "posts"), 5) is None
    assert connector.read(globex, 5) is None
    with pytest.raises(ConnectorRefused):
        connector.submit("hijack", intent("update", globex, REVISED), REVISED, 5)
    # A read names the collection the object is really in.
    assert connector.read(ActionTarget("acme", "pages", "obj_1"), 5).target == receipt.target
    assert service.wait_idle(10)
    assert counters(service, "hijack") == expected(received=1, refused=1)
    assert service.snapshot()["objects"]["obj_1"]["version"] == 1


def test_a_visibility_delay_hides_lookups_until_the_clock_passes_it(connect):
    clock = ManualClock(NOW)
    service, connector = connect(clock_ns=clock, visibility_delay_s=2)
    assert connector.profile.visibility_bound_s == 2
    receipt = connector.submit(KEY, intent(), BODY, 5)
    clock.advance(1.5)
    assert connector.lookup(KEY, ACME, 5) is None
    assert connector.read(CREATED, 5) is None
    clock.advance(0.5)
    assert connector.lookup(KEY, ACME, 5) == receipt
    assert connector.read(CREATED, 5).content == BODY


def test_key_expiry_hides_the_key_but_not_the_effect(connect):
    clock = ManualClock(NOW)
    service, connector = connect(clock_ns=clock, key_ttl_s=60)
    assert connector.profile.dedupe_window_s == 60
    receipt = connector.submit(KEY, intent(), BODY, 5)
    clock.advance(59)
    assert connector.lookup(KEY, ACME, 5) == receipt
    clock.advance(1)
    assert connector.lookup(KEY, ACME, 5) is None
    # The write still exists, so the empty lookup did not prove its absence.
    assert connector.read(CREATED, 5).content == BODY
    resent = connector.submit(KEY, intent(), BODY, 5)
    assert not resent.deduplicated and resent.target.object_id == "obj_2"


def test_undeclared_lookups_are_unavailable(connect):
    service, connector = connect(lookup_by_key=False, lookup_by_object=False)
    assert not (connector.profile.lookup_by_key or connector.profile.lookup_by_object)
    with pytest.raises(LookupUnavailable):
        connector.lookup(KEY, ACME, 5)
    with pytest.raises(LookupUnavailable):
        connector.read(CREATED, 5)
    with pytest.raises(ValueError, match="object_id"):
        connector.read(ACME, 5)


def test_a_parameter_only_intent_writes_its_canonical_parameters(connect):
    service, connector = connect()
    parameters = {"title": "Launch", "tags": ["news"]}
    receipt = connector.submit(KEY, intent(body=None, parameters=parameters), None, 5)
    observed = connector.read(receipt.target, 5)
    assert observed.content == canonical_bytes(parameters)
    assert observed.media_type == "application/json"
    assert observed.content_ref.sha256 == sha256_hex(parameters)


def test_a_value_from_another_writer_reads_as_its_canonical_json(connect):
    service, connector = connect()
    connector.submit(KEY, intent(), BODY, 5)
    service.submit("other-writer", {"title": "Replaced"}, object_id="obj_1",
                   target=Target("acme", "posts"))
    observed = connector.read(CREATED, 5)
    assert (observed.version, observed.key) == (2, "other-writer")
    assert observed.content == canonical_bytes({"title": "Replaced"})
    assert observed.content_ref != intent().content


@pytest.mark.parametrize("bad,payload", [
    (intent("publish", CREATED), BODY),
    (intent("update"), BODY),
    (intent("archive"), BODY),
    (intent(), None),
    (intent(body=None), BODY),
    (intent(), BODY.decode()),
], ids=["publish-names-an-object", "update-names-none", "undeclared-operation",
        "content-without-payload", "payload-without-content", "text-payload"])
def test_an_intent_the_service_cannot_take_is_refused_before_delivery(connect, bad, payload):
    service, connector = connect()
    with pytest.raises((TypeError, ValueError)):
        connector.submit(KEY, bad, payload, 5)
    assert service.snapshot()["deliveries"] == []


# Loopback HTTP


def test_http_timeout_is_ambiguous_although_the_service_commits_later():
    service = FixtureService()
    gate = Gate()
    service.hold(KEY, gate, before_commit=True)
    with serve_http(service) as endpoint:
        connector = HttpFixtureConnector(endpoint, fixture_profile(service))
        try:
            with pytest.raises(ConnectorAmbiguous):
                connector.submit(KEY, intent(), BODY, 0.2)
            assert gate.entered.wait(10)
        finally:
            gate.release()
        assert service.wait_idle(10)
    state = counters(service, KEY)
    assert (state["received"], state["committed"]) == (1, 1)


def test_http_connection_refused_is_a_refusal():
    service = FixtureService()
    with serve_http(service) as endpoint:
        profile = fixture_profile(service)
    # The server has closed, so nothing can have been delivered.
    connector = HttpFixtureConnector(endpoint, profile)
    with pytest.raises(ConnectorRefused):
        connector.submit(KEY, intent(), BODY, 10)
    assert service.snapshot()["deliveries"] == []


@pytest.mark.parametrize("host", ["192.0.2.1", "example.com", "0.0.0.0", ""])
def test_http_connector_talks_only_to_loopback(host):
    with pytest.raises(ValueError, match="loopback"):
        HttpFixtureConnector(Endpoint(host, 80), fixture_profile(FixtureService()))
    with pytest.raises(TypeError):
        HttpFixtureConnector(Endpoint("127.0.0.1", 80), fixture_profile(FixtureService()).to_dict())


def post(endpoint, body):
    connection = http.client.HTTPConnection(endpoint.host, endpoint.port, timeout=10)
    try:
        connection.request("POST", "/requests", json.dumps(body),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"null")
    finally:
        connection.close()


def test_http_adapter_passes_targets_and_refuses_malformed_ones():
    service = FixtureService()
    with serve_http(service) as endpoint:
        for target in ({"tenant": "acme"}, {"tenant": "acme", "collection": "posts", "region": "eu"},
                       ["acme", "posts"], {"tenant": "", "collection": "posts"}, "acme"):
            assert post(endpoint, {"request_key": "bad", "value": 1, "target": target})[0] == 400
        status, receipt = post(endpoint, {"request_key": "plain", "value": 1, "target": None})
        assert status == 200 and "target" not in receipt
        status, receipt = post(endpoint, {"request_key": "scoped", "value": 1,
                                          "target": {"tenant": "acme", "collection": "posts"}})
        assert status == 200 and receipt["target"] == {"tenant": "acme", "collection": "posts"}
    assert "bad" not in service.snapshot()["requests"]
    assert service.get_object(receipt["object_id"], tenant="acme").target == Target("acme", "posts")


# Test approval host


def _decode(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def test_hmac_host_verifies_only_its_own_assertion_for_the_exact_intent():
    host = HmacApprovalHost()
    assert check_verifier(host) == host.identity
    digest = intent().digest()
    assertion = host.issue(digest, subject="editor-7", tenant="acme", expires_at_ns=EXPIRES)
    principal = host.verify(assertion, digest, NOW)
    assert principal == Principal(host.issuer, "editor-7", "acme")
    assert host.still_authorized(principal, intent(), NOW)
    with pytest.raises(ApprovalRejected, match="approves another digest"):
        host.verify(assertion, intent(parameters={"title": "Changed"}).digest(), NOW)
    # Each host signs with its own random key, so the same claims from another
    # host are a forgery.
    impostor = HmacApprovalHost()
    assert len(host.key) == 32 and impostor.key != host.key
    forged = impostor.issue(digest, subject="editor-7", tenant="acme", expires_at_ns=EXPIRES)
    with pytest.raises(ApprovalRejected, match="signature"):
        host.verify(forged, digest, NOW)
    # Claims edited after signing no longer match the signature.
    claims, signature = assertion.split(".")
    edited = {**json.loads(_decode(claims)), "tenant": "globex"}
    with pytest.raises(ApprovalRejected, match="signature"):
        host.verify(_encode(canonical_bytes(edited)) + "." + signature, digest, NOW)
    for garbage in ("", "x", "a.b.c", assertion + "x", "." + signature, claims + ".",
                    claims + ".é", "é." + signature, None, 7, assertion.encode()):
        with pytest.raises(ApprovalRejected):
            host.verify(garbage, digest, NOW)
    with pytest.raises(ValueError, match="expiry"):
        host.issue(digest, subject="editor-7", tenant="acme", expires_at_ns=float(EXPIRES))


def test_hmac_host_honors_expiry_and_revocation():
    host = HmacApprovalHost()
    digest = intent().digest()
    first = host.issue(digest, subject="editor-7", tenant="acme", expires_at_ns=EXPIRES)
    second = host.issue(digest, subject="editor-7", tenant="acme", expires_at_ns=EXPIRES)
    assert first != second
    assert host.verify(first, digest, EXPIRES - 1).subject == "editor-7"
    with pytest.raises(ApprovalRejected, match="expired"):
        host.verify(first, digest, EXPIRES)
    with pytest.raises(ApprovalRejected, match="time"):
        host.verify(first, digest, float(NOW))
    host.revoke(first)
    with pytest.raises(ApprovalRejected, match="revoked"):
        host.verify(first, digest, NOW)
    principal = host.verify(second, digest, NOW)
    assert host.still_authorized(principal, intent(), NOW)
    host.revoke_subject("editor-7")
    with pytest.raises(ApprovalRejected, match="revoked"):
        host.verify(second, digest, NOW)
    assert not host.still_authorized(principal, intent(), NOW)
    with pytest.raises(ApprovalRejected, match="signature"):
        host.revoke(HmacApprovalHost().issue(digest, subject="editor-8", tenant="acme",
                                             expires_at_ns=EXPIRES))


def test_hmac_host_authorizes_a_principal_only_for_its_own_tenant_and_issuer():
    host = HmacApprovalHost()
    principal = Principal(host.issuer, "editor-7", "acme")
    assert host.still_authorized(principal, intent(), NOW)
    assert not host.still_authorized(principal, intent(target=ActionTarget("globex", "posts")), NOW)
    assert not host.still_authorized(Principal("another-issuer", "editor-7", "acme"), intent(), NOW)
    for key in (b"short", "0" * 64, bytearray(32)):
        with pytest.raises((TypeError, ValueError)):
            HmacApprovalHost(key=key)


CHILD_VERIFIER = r"""
import json, os, sys

from fixture_connector import HmacApprovalHost

host = HmacApprovalHost(key=bytes.fromhex(os.environ["FIXTURE_HMAC_KEY"]))
principal = host.verify(sys.argv[1], sys.argv[2], int(sys.argv[3]))
print(json.dumps(principal.to_dict()))
"""


def test_a_child_process_verifies_with_the_parents_key():
    host = HmacApprovalHost()
    digest = intent().digest()
    assertion = host.issue(digest, subject="editor-7", tenant="acme", expires_at_ns=EXPIRES)
    completed = subprocess.run(
        [child_executable(), "-c", CHILD_VERIFIER, assertion, digest, str(NOW)],
        env=offline_env({"FIXTURE_HMAC_KEY": host.key.hex()}), capture_output=True, text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"issuer": host.issuer, "subject": "editor-7",
                                            "tenant": "acme"}
