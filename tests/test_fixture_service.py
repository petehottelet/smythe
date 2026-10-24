"""Shared test fakes and the fake external service behave deterministically."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import hashlib
import http.client
import json
import os
import sqlite3
import threading

import pytest

from fakes import (
    DUPLICATE, INVALID_COUNT, LOST_RESPONSE, MALFORMED_USAGE, RATE_LIMITED, ROOT, SERVER_ERROR,
    TRANSPORT_ERROR, Crash, Gate, ManualClock, MessagesTransport, NativeTransport, copy_database,
    crash_when, kill_and_reap, launch_child, offline_env, wait_for_barrier,
)
from fixture_service import (
    AcknowledgementLost, Fault, FixtureService, LookupUnavailable, NotFound, ServiceUnavailable,
    Target, serve_http,
)
from smythe import Swarm, Task
import smythe.workflow_store as workflow_store
from smythe.graph import ExecutionGraph, FailurePolicy, Node, Topology
from smythe.pricing_anthropic import MODEL as MESSAGES_MODEL
from smythe.provider import ProviderAccountingError, ProviderResponseError
from smythe.provider_messages import AnthropicMessagesProvider
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.synthesizer import Synthesizer, SynthesisStrategy
from smythe.workflow_provider import ProviderRequestRejectedError, WorkflowProviderContext
from smythe.workflow_store import CallKey, SQLiteWorkflowStore, WorkflowError, WorkflowLeaseError

MODEL = "gpt-6-astra"
KEY = CallKey("execution", "node")


def counters(service, key):
    state = service.snapshot()["requests"][key]
    return {name: state[name] for name in (
        "received", "committed", "deduplicated", "refused", "acknowledged", "ack_dropped")}


def expected(**values):
    return {"received": 0, "committed": 0, "deduplicated": 0, "refused": 0, "acknowledged": 0,
            "ack_dropped": 0, **values}


# Fake external service


@pytest.mark.parametrize("dedupe", [True, False])
def test_committed_effect_with_a_lost_acknowledgement_is_visible_in_the_snapshot(dedupe):
    service = FixtureService(dedupe=dedupe)
    service.commit_then_drop_ack("order")
    with pytest.raises(AcknowledgementLost):
        service.submit("order", {"qty": 1})
    # The caller saw a failure; the service committed and kept the effect.
    state = service.snapshot()
    assert counters(service, "order") == expected(received=1, committed=1, ack_dropped=1)
    assert state["objects"]["obj_1"]["value"] == {"qty": 1}
    assert state["deliveries"][0]["outcome"] == "committed"
    assert state["deliveries"][0]["acknowledged"] is False
    retry = service.submit("order", {"qty": 1})
    if dedupe:
        assert retry.deduplicated and retry.object_id == "obj_1"
        assert counters(service, "order") == expected(
            received=2, committed=1, deduplicated=1, acknowledged=1, ack_dropped=1)
    else:
        # Without key deduplication the retry repeats the external effect.
        assert not retry.deduplicated and retry.object_id == "obj_2"
        assert counters(service, "order") == expected(
            received=2, committed=2, acknowledged=1, ack_dropped=1)
    assert len(service.snapshot()["objects"]) == (1 if dedupe else 2)


def test_refusal_before_commit_leaves_no_effect_and_the_retry_commits_once():
    service = FixtureService()
    service.fail_before_commit("order")
    service.fail_before_commit("other", delivery=1)
    with pytest.raises(ServiceUnavailable):
        service.submit("order", {"qty": 1})
    assert service.snapshot()["objects"] == {}
    receipt = service.submit("order", {"qty": 1})
    assert not receipt.deduplicated and receipt.object_version == 1
    assert counters(service, "order") == expected(
        received=2, refused=1, committed=1, acknowledged=1)
    # A schedule names the delivery ordinal of one request key.
    service.submit("other", 1)
    with pytest.raises(ServiceUnavailable):
        service.submit("other", 1)


@pytest.mark.parametrize("dedupe", [True, False])
def test_duplicate_delivery_commits_once_only_with_deduplication(dedupe):
    service = FixtureService(dedupe=dedupe)
    service.deliver_duplicate("order")
    receipt = service.submit("order", {"qty": 1})
    assert not receipt.deduplicated  # The caller receives the first copy's answer.
    state = service.snapshot()
    assert counters(service, "order") == expected(
        received=2, committed=1 if dedupe else 2, deduplicated=1 if dedupe else 0, acknowledged=1)
    assert len(state["objects"]) == (1 if dedupe else 2)
    assert [delivery["duplicate"] for delivery in state["deliveries"]] == [False, True]
    assert [delivery["acknowledged"] for delivery in state["deliveries"]] == [True, False]


def test_reordered_delivery_applies_updates_in_arrival_order():
    service = FixtureService()
    created = service.submit("create", {"status": "new"})
    packed = service.send("pack", {"status": "packed"}, object_id=created.object_id)
    shipped = service.send("ship", {"status": "shipped"}, object_id=created.object_id)
    with pytest.raises(RuntimeError):
        packed.result()
    in_flight = service.snapshot()["in_flight"]
    assert [message["request_key"] for message in in_flight] == ["pack", "ship"]
    with pytest.raises(ValueError):
        service.deliver(order=[0, 0])
    service.deliver(order=[1, 0])
    assert (shipped.result().object_version, packed.result().object_version) == (2, 3)
    state = service.snapshot()
    # The later request arrived first, so the earlier one wins.
    assert state["objects"][created.object_id]["value"] == {"status": "packed"}
    arrivals = [delivery["request_key"] for delivery in state["deliveries"]]
    assert arrivals == ["create", "ship", "pack"]
    assert state["in_flight"] == []
    with pytest.raises(NotFound):
        service.submit("missing", {}, object_id="obj_404")
    assert counters(service, "missing") == expected(received=1, refused=1)


def test_idempotency_keys_expire_after_their_lifetime():
    clock = ManualClock()
    service = FixtureService(clock_ns=clock, key_ttl_s=60)
    first = service.submit("order", {"qty": 1})
    assert first.committed_at_ns == clock.now_ns
    clock.advance(59)
    assert service.submit("order", {"qty": 1}).deduplicated
    clock.advance(1)
    late = service.submit("order", {"qty": 1})
    assert not late.deduplicated and late.object_id != first.object_id
    assert counters(service, "order") == expected(
        received=3, committed=2, deduplicated=1, acknowledged=3)


def test_lookups_respect_availability_and_eventual_consistency():
    clock = ManualClock()
    service = FixtureService(clock_ns=clock, visibility_delay_s=2)
    receipt = service.submit("order", {"qty": 1})
    assert service.lookup_request("order") is None
    assert service.get_object(receipt.object_id) is None
    assert service.snapshot()["objects"][receipt.object_id]["version"] == 1
    clock.advance(2)
    assert service.lookup_request("order") == receipt
    assert service.get_object(receipt.object_id).value == {"qty": 1}
    service.submit("resize", {"qty": 2}, object_id=receipt.object_id)
    assert service.get_object(receipt.object_id).version == 1
    clock.advance(2)
    assert service.get_object(receipt.object_id).version == 2
    assert service.lookup_request("unknown") is None
    blind = FixtureService(lookup_by_key=False, lookup_by_object=False)
    with pytest.raises(LookupUnavailable):
        blind.lookup_request("order")
    with pytest.raises(LookupUnavailable):
        blind.get_object("obj_1")


def test_sources_keep_every_version_and_count_reads():
    clock = ManualClock()
    service = FixtureService(clock_ns=clock)
    first = service.add_source("brief", "Version one")
    clock.advance(5)
    second = service.change_source("brief", "Version two")
    assert (first.version, second.version) == (1, 2)
    assert second.sha256 == hashlib.sha256(b"Version two").hexdigest()
    assert second.changed_at_ns == first.changed_at_ns + 5 * 10**9
    assert service.read_source("brief") == second
    assert service.read_source("brief", version=1) == first
    with pytest.raises(NotFound):
        service.read_source("brief", version=3)
    with pytest.raises(NotFound):
        service.change_source("missing", "Text")
    with pytest.raises(ValueError):
        service.add_source("brief", "Again")
    state = service.snapshot()["sources"]["brief"]
    assert state["version"] == 2 and state["content"] == "Version two" and state["reads"] == 2
    assert [entry["content"] for entry in state["history"]] == ["Version one", "Version two"]


@pytest.mark.parametrize("before_commit", [True, False])
def test_held_delivery_waits_until_released(before_commit):
    service = FixtureService()
    gate = Gate()
    service.hold("order", gate, before_commit=before_commit)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.submit, "order", {"qty": 1})
        try:
            assert gate.entered.wait(10)
            held = counters(service, "order")
            assert (held["received"], held["committed"], held["acknowledged"]) == (
                1, 0 if before_commit else 1, 0)
            assert not future.done()
        finally:
            gate.release()
        assert future.result(10).object_id == "obj_1"
    assert service.wait_idle(10)
    assert counters(service, "order") == expected(received=1, committed=1, acknowledged=1)


def test_snapshot_is_a_detached_json_copy():
    service = FixtureService()
    value = {"items": ["a"]}
    service.submit("order", value)
    value["items"].append("changed by caller")
    state = service.snapshot()
    state["objects"]["obj_1"]["value"]["items"].append("changed by test")
    assert service.snapshot()["objects"]["obj_1"]["value"] == {"items": ["a"]}
    assert json.loads(json.dumps(state)) == state
    with pytest.raises(TypeError):
        service.submit("invalid", {"value": object()})
    assert "invalid" not in service.snapshot()["requests"]


# Tenants and notices


ACME = Target("acme", "posts")
GLOBEX = Target("globex", "posts")


def test_snapshot_without_targets_or_notices_is_unchanged():
    clock = ManualClock()
    service = FixtureService(clock_ns=clock, key_ttl_s=60, visibility_delay_s=1)
    service.add_source("brief", "Version one")
    service.fail_before_commit("order")
    service.commit_then_drop_ack("order", delivery=1)
    service.deliver_duplicate("resize")
    with pytest.raises(ServiceUnavailable):
        service.submit("order", {"qty": 1})
    clock.advance(1)
    with pytest.raises(AcknowledgementLost):
        service.submit("order", {"qty": 1})
    service.submit("order", {"qty": 1})
    service.submit("resize", {"qty": 2}, object_id="obj_1")
    service.send("ship", {"status": "shipped"}, object_id="obj_1")
    service.read_source("brief")
    state = service.snapshot()
    # Pinned: the exact bytes, key order included, of a snapshot taken before
    # targets and notices existed.
    assert hashlib.sha256(json.dumps(state).encode()).hexdigest() == (
        "6e9510760e1ce542a4cf51c83b0d4e159fbb07f61c8ac6e1d71a18ccd7aa1ed0")
    assert service.notices() == []


def test_targets_scope_keys_and_objects_to_their_tenant():
    clock = ManualClock()
    service = FixtureService(clock_ns=clock)
    created = service.submit("order", {"qty": 1}, target=ACME)
    assert (created.object_id, created.target) == ("obj_1", ACME)
    # Another tenant, or a caller naming none, finds neither the key nor the object.
    for tenant in ("globex", None):
        assert service.lookup_request("order", tenant=tenant) is None
        assert service.get_object("obj_1", tenant=tenant) is None
    assert service.lookup_request("order", tenant="acme") == created
    assert service.get_object("obj_1", tenant="acme").target == ACME
    # An update outside the object's own target is refused as an unknown object.
    for target in (GLOBEX, None, Target("acme", "pages")):
        with pytest.raises(NotFound):
            service.submit("hijack", {"qty": 9}, object_id="obj_1", target=target)
    assert counters(service, "hijack") == expected(received=3, refused=3)
    updated = service.submit("resize", {"qty": 2}, object_id="obj_1", target=ACME)
    assert (updated.object_version, updated.target) == (2, ACME)
    # Keys belong to a tenant: the same key elsewhere is a new request.
    other = service.submit("order", {"qty": 5}, target=GLOBEX)
    assert not other.deduplicated and other.object_id == "obj_2"
    again = service.submit("order", {"qty": 1}, target=ACME)
    assert again.deduplicated and again == replace(created, deduplicated=True)
    assert service.lookup_request("order", tenant="globex") == other
    state = service.snapshot()
    acme, globex = {"tenant": "acme", "collection": "posts"}, {"tenant": "globex", "collection": "posts"}
    assert state["objects"]["obj_1"]["target"] == acme
    assert [record["target"] for record in state["objects"]["obj_1"]["history"]] == [acme, acme]
    assert state["idempotency_keys"] == {}
    assert state["tenant_idempotency_keys"] == {
        "acme": {
            "order": {"object_id": "obj_1", "object_version": 1, "expires_at_ns": None,
                      "target": acme},
            "resize": {"object_id": "obj_1", "object_version": 2, "expires_at_ns": None,
                       "target": acme},
        },
        "globex": {"order": {"object_id": "obj_2", "object_version": 1, "expires_at_ns": None,
                             "target": globex}},
    }
    pages = {"tenant": "acme", "collection": "pages"}
    assert [entry.get("target") for entry in state["deliveries"]] == [
        acme, globex, None, pages, acme, globex, acme]
    assert state["requests"]["order"]["object_ids"] == ["obj_1", "obj_2"]


def test_targets_are_validated_and_shown_in_flight():
    service = FixtureService()
    for target in (("acme", "posts"), {"tenant": "acme", "collection": "posts"}):
        with pytest.raises(ValueError):
            service.submit("bad", 1, target=target)
    for args in (("", "posts"), ("acme", ""), (None, "posts"), ("acme", 7)):
        with pytest.raises(ValueError):
            Target(*args)
    assert "bad" not in service.snapshot()["requests"]
    service.send("late", {"qty": 1}, target=ACME)
    service.send("plain", {"qty": 2})
    assert service.snapshot()["in_flight"] == [
        {"request_key": "late", "object_id": None, "value": {"qty": 1},
         "target": {"tenant": "acme", "collection": "posts"}},
        {"request_key": "plain", "object_id": None, "value": {"qty": 2}},
    ]


def test_notices_report_each_acknowledged_commit_only():
    clock = ManualClock()
    service = FixtureService(clock_ns=clock, send_notices=True)
    service.fail_before_commit("refused")
    service.commit_then_drop_ack("dropped")
    service.deliver_duplicate("twice")
    first = service.submit("order", {"qty": 1}, target=ACME)
    clock.advance(1)
    with pytest.raises(ServiceUnavailable):
        service.submit("refused", 1)
    with pytest.raises(AcknowledgementLost):
        service.submit("dropped", 1)
    second = service.submit("twice", 2)  # Its network duplicate is deduplicated.
    assert service.submit("order", {"qty": 1}, target=ACME).deduplicated
    assert service.notices() == [
        {"notice_id": "ntc_1", "sent_at_ns": first.committed_at_ns, "payload": {
            "request_key": "order", "object_id": "obj_1", "object_version": 1,
            "committed_at_ns": first.committed_at_ns, "deduplicated": False,
            "target": {"tenant": "acme", "collection": "posts"}}},
        {"notice_id": "ntc_2", "sent_at_ns": second.committed_at_ns, "payload": {
            "request_key": "twice", "object_id": "obj_3", "object_version": 1,
            "committed_at_ns": second.committed_at_ns, "deduplicated": False}},
    ]
    state = service.snapshot()
    assert state["config"]["send_notices"] is True
    assert [notice["forged"] for notice in state["notices"]] == [False, False]
    # The outbox is detached from the caller's copies.
    service.notices()[0]["payload"]["object_id"] = "changed"
    assert service.notices()[0]["payload"]["object_id"] == "obj_1"


def test_a_forged_notice_is_recorded_without_any_commit():
    clock = ManualClock()
    service = FixtureService(clock_ns=clock)
    service.submit("order", {"qty": 1}, target=ACME)
    assert service.notices() == [] and "notices" not in service.snapshot()
    before = service.snapshot()
    payload = {"request_key": "order", "object_id": "obj_9", "object_version": 1,
               "target": {"tenant": "acme", "collection": "posts"}}
    forged = service.forge_notice(payload)
    payload["object_id"] = "changed by caller"
    assert forged == {"notice_id": "ntc_1", "sent_at_ns": clock.now_ns, "payload": {
        "request_key": "order", "object_id": "obj_9", "object_version": 1,
        "target": {"tenant": "acme", "collection": "posts"}}}
    # A reader of the outbox cannot tell the forgery apart; the service can.
    assert service.notices() == [forged]
    after = service.snapshot()
    assert after["notices"] == [{**forged, "forged": True}]
    assert {name: value for name, value in after.items() if name != "notices"} == before
    assert service.get_object("obj_9", tenant="acme") is None
    for invalid in (["not", "an", "object"], {"value": object()}):
        with pytest.raises(TypeError):
            service.forge_notice(invalid)
    assert len(service.notices()) == 1


def call_http(port, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = None if body is None else json.dumps(body)
        connection.request(method, path, payload, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"null")
    finally:
        connection.close()


def test_loopback_http_adapter_serves_the_same_service():
    service = FixtureService(lookup_by_object=False)
    service.add_source("brief", "Version one")
    service.change_source("brief", "Version two")
    service.fail_before_commit("refused")
    service.commit_then_drop_ack("dropped")
    with serve_http(service) as endpoint:
        assert endpoint.url == f"http://127.0.0.1:{endpoint.port}"
        status, receipt = call_http(endpoint.port, "POST", "/requests",
                                    {"request_key": "order", "value": {"qty": 2}})
        assert status == 200 and receipt["object_id"] == "obj_1" and not receipt["deduplicated"]
        assert call_http(endpoint.port, "GET", "/requests/order") == (200, receipt)
        assert call_http(endpoint.port, "GET", "/requests/missing")[0] == 404
        assert call_http(endpoint.port, "GET", "/objects/obj_1")[0] == 501
        assert call_http(endpoint.port, "GET", "/sources/brief")[1]["content"] == "Version two"
        assert call_http(endpoint.port, "GET", "/sources/brief?version=1")[1]["version"] == 1
        assert call_http(endpoint.port, "POST", "/requests",
                         {"request_key": "refused", "value": 1})[0] == 503
        assert call_http(endpoint.port, "POST", "/requests", {"value": 1})[0] == 400
        # A dropped acknowledgement closes the connection without a response.
        with pytest.raises(http.client.RemoteDisconnected):
            call_http(endpoint.port, "POST", "/requests", {"request_key": "dropped", "value": 1})
    assert service.wait_idle(10)
    assert counters(service, "dropped") == expected(received=1, committed=1, ack_dropped=1)
    assert counters(service, "refused") == expected(received=1, refused=1)


CALLER = r"""
import http.client, json, os, sys
from pathlib import Path

from fakes import write_barrier

folder, port = Path(sys.argv[1]), int(sys.argv[2])
write_barrier(folder / "ready.json", {"pid": os.getpid()})
connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
connection.request("POST", "/requests", json.dumps({"request_key": "order", "value": {"qty": 1}}),
                   {"Content-Type": "application/json"})
write_barrier(folder / "acknowledged.json", {"status": connection.getresponse().status})
"""


def test_caller_killed_after_commit_leaves_one_effect_and_its_retry_is_deduplicated(tmp_path):
    service = FixtureService()
    gate = Gate()
    service.schedule("order", Fault(hold_after_commit=gate, drop_ack=True))
    with serve_http(service) as endpoint:
        process = launch_child(CALLER, tmp_path, endpoint.port)
        try:
            assert wait_for_barrier(process, tmp_path / "ready.json")["pid"] == process.pid
            assert gate.entered.wait(30), "the service never received the child's request"
            assert counters(service, "order") == expected(received=1, committed=1)
            reaped = kill_and_reap(process)
        finally:
            kill_and_reap(process)
            gate.release()
        assert reaped.killed and reaped.returncode != 0
        assert not (tmp_path / "acknowledged.json").exists()
        assert service.wait_idle(10)
    assert counters(service, "order") == expected(received=1, committed=1, ack_dropped=1)
    # The caller's successor retries with the same key and gets the original effect.
    retry = service.submit("order", {"qty": 1})
    assert retry.deduplicated and retry.object_id == "obj_1"
    assert list(service.snapshot()["objects"]) == ["obj_1"]


def test_barrier_wait_reports_a_child_that_exits_first(tmp_path):
    process = launch_child("import sys; sys.stderr.write('stopped early'); sys.exit(3)")
    try:
        with pytest.raises(AssertionError, match="stopped early"):
            wait_for_barrier(process, tmp_path / "never.json", timeout=30)
    finally:
        reaped = kill_and_reap(process)
    assert reaped.returncode == 3 and not reaped.killed


def test_child_environment_drops_credentials_and_keeps_imports(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "placeholder")
    monkeypatch.setenv("SMYTHE_MCP_URL", "http://127.0.0.1:9")
    environment = offline_env({"EXTRA": "1"})
    removed = {"OPENAI_API_KEY", "GITHUB_PERSONAL_ACCESS_TOKEN", "SMYTHE_MCP_URL"}
    assert not removed & environment.keys()
    assert environment["EXTRA"] == "1"
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT)


# Manual clock and crash images


def test_manual_clock_drives_journal_leases_without_waiting(tmp_path):
    clock = ManualClock()
    with SQLiteWorkflowStore(tmp_path / "leases.db", clock_ns=clock) as store:
        run = store.create_run({"goal": "test"}, {}, 10**9)
        first = store.acquire_lease(run["run_id"], "first", 30)
        assert first.expires_at_ns == clock.now_ns + 30 * 10**9
        clock.advance(29.5)
        with pytest.raises(WorkflowLeaseError):
            store.acquire_lease(run["run_id"], "second", 30)
        clock.advance_past(first.expires_at_ns)
        assert store.acquire_lease(run["run_id"], "second", 30).epoch == first.epoch + 1
        clock.expire_lease(30)
        assert store.acquire_lease(run["run_id"], "third", 30).epoch == first.epoch + 2
    started = clock.monotonic_ns()
    clock.jump_wall(-3600)
    assert clock.monotonic_ns() == started and clock() == clock.time_ns() == clock.now_ns
    clock.advance(1)
    assert clock.monotonic() == pytest.approx(started / 10**9 + 1)
    with pytest.raises(ValueError):
        clock.advance(-1)
    assert ManualClock.after(120).now_ns > ManualClock().now_ns


def responses_provider():
    return OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100)


def messages_provider():
    return AnthropicMessagesProvider(api_key="test-secret")


@pytest.fixture
def responses(monkeypatch):
    return NativeTransport().install(monkeypatch)


@pytest.fixture
def messages(monkeypatch):
    return MessagesTransport().install(monkeypatch)


@pytest.fixture
def journal(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "calls.db") as store:
        run = store.create_run({"goal": "test"}, {}, 10**9)
        yield store, store.acquire_lease(run["run_id"], "tests", 120)


async def invoke(journal, provider=None, model=MODEL, key=KEY, prompt="Question"):
    store, lease = journal
    providers = {"main": provider or responses_provider()}
    async with WorkflowProviderContext(store, lease, providers) as context:
        return await context.for_call("main", key).complete("System", prompt, model)


def test_crash_point_copies_the_journal_and_stops_later_writes(tmp_path, responses, monkeypatch):
    clock = ManualClock()
    image = tmp_path / "crashed.db"
    with SQLiteWorkflowStore(tmp_path / "live.db", clock_ns=clock) as store:
        run = store.create_run({"goal": "test"}, {}, 10**9)
        lease = store.acquire_lease(run["run_id"], "owner", 60)
        original = store.accept_result
        point = crash_when(monkeypatch, store, "accept_result", image=image, connection=store._db)
        # The provider context reports the interruption as unresolved accounting;
        # the image already holds the journal as the killed process left it.
        with pytest.raises(ProviderAccountingError) as caught:
            asyncio.run(invoke((store, lease)))
        assert isinstance(caught.value.__cause__, Crash)
        assert point.fired and image.exists()
        # A killed process makes no later writes either.
        with pytest.raises(Crash):
            store.accept_result(lease, "call", {}, "test-v1")
        assert [args[1] for args, _ in point.calls][1:] == ["call"]
        point.restore()
        assert store.accept_result == original
    clock.expire_lease(60)
    with SQLiteWorkflowStore(image, clock_ns=clock) as reopened:
        record = reopened.lookup_call(run["run_id"], KEY)
        assert record["state"] == "settled" and record["billing_state"] == "known"
        assert record["result_state"] != "accepted"
        assert reopened.acquire_lease(run["run_id"], "successor", 60).epoch == lease.epoch + 1
    assert len(responses.requests) == 1


def finishes(function, connection, timeout=20.0):
    """Whether ``function`` returns or raises within ``timeout`` in another thread.

    A copy still waiting on ``connection``'s write transaction is freed by
    rolling that transaction back, so a failing check cannot hang the test.
    """
    thread = threading.Thread(target=function, daemon=True)
    thread.start()
    thread.join(timeout)
    if not thread.is_alive():
        return True
    connection.rollback()
    thread.join(timeout)
    return False


@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_crash_image_inside_an_open_write_transaction_holds_only_committed_rows(
    tmp_path, journal_mode,
):
    image = tmp_path / "image.db"
    with closing(sqlite3.connect(tmp_path / "live.db", isolation_level=None,
                                 check_same_thread=False)) as writer:
        writer.execute(f"PRAGMA journal_mode={journal_mode}")
        writer.execute("CREATE TABLE rows (id INTEGER PRIMARY KEY)")
        writer.execute("INSERT INTO rows VALUES (1)")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO rows VALUES (2)")
        copied = []
        assert finishes(lambda: copied.append(copy_database(writer, image)), writer)
        assert copied == [image]
        # The copy neither waited for nor ended the caller's transaction.
        assert writer.in_transaction
        writer.execute("COMMIT")
        assert writer.execute("SELECT id FROM rows").fetchall() == [(1,), (2,)]
    with closing(sqlite3.connect(image)) as copy:
        assert copy.execute("SELECT id FROM rows").fetchall() == [(1,)]


def test_crash_image_of_an_in_memory_database_is_refused_inside_a_transaction(tmp_path):
    with closing(sqlite3.connect(":memory:", isolation_level=None,
                                 check_same_thread=False)) as memory:
        memory.execute("CREATE TABLE rows (id INTEGER PRIMARY KEY)")
        memory.execute("INSERT INTO rows VALUES (1)")
        with closing(sqlite3.connect(copy_database(memory, tmp_path / "copy.db"))) as copy:
            assert copy.execute("SELECT id FROM rows").fetchall() == [(1,)]
        # No other connection can read its committed state mid-transaction.
        memory.execute("BEGIN IMMEDIATE")
        memory.execute("INSERT INTO rows VALUES (2)")
        raised = []

        def copy_mid_transaction():
            try:
                copy_database(memory, tmp_path / "partial.db")
            except ValueError as error:
                raised.append(error)

        assert finishes(copy_mid_transaction, memory)
        assert len(raised) == 1 and memory.in_transaction


def test_crash_image_inside_a_journal_transaction_is_the_state_before_it(
    tmp_path, responses, monkeypatch,
):
    responses.schedule("execution", 0, RATE_LIMITED)
    clock = ManualClock()
    image = tmp_path / "crashed.db"
    with SQLiteWorkflowStore(tmp_path / "live.db", clock_ns=clock) as store:
        run = store.create_run({"goal": "test"}, {}, 10**9)
        lease = store.acquire_lease(run["run_id"], "owner", 60)
        # Settlement prices the rejection inside its write transaction.
        point = crash_when(monkeypatch, workflow_store, "_http_rejection_receipt",
                           image=image, connection=store._db)
        raised = []

        def settle():
            try:
                asyncio.run(invoke((store, lease)))
            except ProviderAccountingError as error:
                raised.append(error)

        assert finishes(settle, store._db)
        assert len(raised) == 1 and isinstance(raised[0].__cause__, Crash)
        assert point.fired and not store._db.in_transaction
    clock.expire_lease(60)
    with SQLiteWorkflowStore(image, clock_ns=clock) as reopened:
        record = reopened.lookup_call(run["run_id"], KEY)
        # The response was saved; its interrupted settlement left no trace.
        assert record["state"] == "response_saved" and record["billing_state"] != "known"
        assert reopened.acquire_lease(run["run_id"], "successor", 60).epoch == lease.epoch + 1


# SDK transport fakes


def test_native_transport_runs_a_durable_workflow_with_request_identities(tmp_path, responses):
    with SQLiteWorkflowStore(tmp_path / "run.db") as store:
        swarm = Swarm(model=MODEL, provider=responses_provider(), max_budget_usd=1, run_store=store,
                      synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE))
        result = swarm.execute(Task("Write and check"))
        accounting = store.inspect_run(result.execution_id)
    assert result.output == "Final accepted output." and result.total_cost_usd == .006
    generations = [record for record in responses.log if record.operation == "response"]
    assert [(record.phase, record.ordinal, record.outcome) for record in generations] == [
        ("planning", 0, "delivered"), ("execution", 0, "delivered"),
        ("execution", 1, "delivered"), ("synthesis", 0, "delivered")]
    assert len({record.request_id for record in responses.log}) == len(responses.log) == 8
    assert accounting["call_count"] == len(responses.requests) == len(responses.counts) == 4
    assert all(client.close.await_count == 1 for client in responses.clients)


@pytest.mark.parametrize("fault,outcome,received,rejected", [
    (RATE_LIMITED, "http_429", 1, True),
    (SERVER_ERROR, "http_500", 1, False),
    (MALFORMED_USAGE, "delivered", 1, False),
    (LOST_RESPONSE, "lost_response", 1, False),
    (TRANSPORT_ERROR, "not_delivered", 0, False),
], ids=["rate_limited", "server_error", "malformed_usage", "lost_response", "transport_error"])
def test_scheduled_generation_faults_reach_the_journal(
    journal, responses, fault, outcome, received, rejected,
):
    responses.schedule("execution", 0, fault)
    with pytest.raises((ProviderRequestRejectedError, ProviderAccountingError)):
        asyncio.run(invoke(journal))
    record = responses.log[-1]
    assert (record.operation, record.phase, record.ordinal, record.outcome) == (
        "response", "execution", 0, outcome)
    assert len(responses.requests) == received
    # A lost response was produced and billed by the provider, never received.
    assert (record.response_id is not None) == (fault in (MALFORMED_USAGE, LOST_RESPONSE))
    store, lease = journal
    audit = store.inspect_run(lease.run_id)
    if rejected:
        call = store.lookup_call(lease.run_id, KEY)
        assert (call["result_state"], call["cost_nanousd"]) == ("rejected", 0)
        assert audit["unknown_calls"] == 0
    else:
        assert audit["unknown_calls"] == 1 and audit["blocked_reason"] == "unknown_exposure"


def test_invalid_count_fails_the_quote_before_generation(journal, responses):
    responses.schedule("execution", None, INVALID_COUNT, operation="input_tokens")
    with pytest.raises(WorkflowError) as caught:
        asyncio.run(invoke(journal))
    assert caught.value.envelope.json() == {"input_tokens": False}
    assert responses.requests == [] and len(responses.counts) == 1
    assert responses.log[-1].outcome == "invalid_count"
    responses.clear("execution", None, operation="input_tokens")
    assert asyncio.run(invoke(journal)).text == "Draft accepted output."


def test_gated_response_is_held_until_released(journal, responses):
    gate = Gate()
    responses.schedule("execution", 0, gate)
    store, lease = journal

    async def exercise():
        call = asyncio.create_task(invoke(journal))
        try:
            assert await gate.reached(10)
            assert len(responses.requests) == 1 and not call.done()
            assert store.lookup_call(lease.run_id, KEY)["state"] == "dispatched"
        finally:
            gate.release()
        return await asyncio.wait_for(call, 10)

    assert asyncio.run(exercise()).text == "Draft accepted output."
    assert responses.log[-1].outcome == "delivered"


def test_duplicate_response_replays_the_previous_bytes(journal, responses):
    responses.schedule("execution", 1, DUPLICATE)
    first = asyncio.run(invoke(journal, key=CallKey("execution", "first"), prompt="First"))
    second = asyncio.run(invoke(journal, key=CallKey("execution", "second"), prompt="Second"))
    assert [record.outcome for record in responses.log if record.operation == "response"] == [
        "delivered", "duplicate"]
    for field in ("response_sha256", "request_id", "response_id"):
        assert first.native_receipt[field] == second.native_receipt[field]
    assert first.native_receipt["request_sha256"] != second.native_receipt["request_sha256"]


def test_scripted_replies_and_overrides_shape_responses(journal, responses):
    responses.script("execution", "Scripted draft.")
    assert asyncio.run(invoke(journal, key=CallKey("execution", "first"))).text == "Scripted draft."
    responses.overrides["execution"] = {"status": "incomplete"}
    with pytest.raises(ProviderResponseError) as caught:
        asyncio.run(invoke(journal, key=CallKey("execution", "second")))
    assert caught.value.billing_result.cost_usd == .0015


def test_messages_transport_retries_a_rate_limited_durable_node(tmp_path, messages):
    messages.schedule("execution", 0, RATE_LIMITED)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="Draft", failure_policy=FailurePolicy.RETRY, max_retries=1)])
    with SQLiteWorkflowStore(tmp_path / "messages.db") as store:
        swarm = Swarm(model=MESSAGES_MODEL, provider=messages_provider(), run_store=store,
                      max_budget_usd=1)
        result = swarm.execute(graph)
        calls = store.inspect_run(result.execution_id)["calls"]
    assert "Draft accepted output." in result.output and result.total_cost_usd == .002535
    assert sorted((call["key"]["attempt"], call["cost_nanousd"], call["result_state"])
                  for call in calls) == [(0, 0, "rejected"), (1, 2_535_000, "applied")]
    generations = [record for record in messages.log if record.operation == "response"]
    assert [record.outcome for record in generations] == ["http_429", "delivered"]
    assert generations[0].request_id != generations[1].request_id
    assert len(messages.requests) == len(messages.counts) == 2


def test_messages_transport_lost_response_keeps_unknown_exposure(journal, messages):
    messages.schedule("execution", 0, LOST_RESPONSE)
    with pytest.raises(ProviderAccountingError):
        asyncio.run(invoke(journal, messages_provider(), MESSAGES_MODEL))
    assert messages.log[-1].outcome == "lost_response" and messages.log[-1].response_id
    store, lease = journal
    assert store.inspect_run(lease.run_id)["blocked_reason"] == "unknown_exposure"


# Revoked and deleted sources


def test_revoked_and_deleted_sources_can_no_longer_be_read():
    service = FixtureService()
    first = service.add_source("brief", "Version one")
    second = service.change_source("brief", "Version two")
    service.add_source("notes", "Kept")
    assert service.revoke_source("brief", version=1) == first
    with pytest.raises(NotFound):
        service.read_source("brief", version=1)
    assert service.read_source("brief") == second
    # Revoking the latest version leaves no readable latest version.
    assert service.revoke_source("brief") == second
    with pytest.raises(NotFound):
        service.read_source("brief")
    state = service.snapshot()["sources"]
    assert state["brief"]["revoked"] == [1, 2] and len(state["brief"]["history"]) == 2
    assert "revoked" not in state["notes"]
    with pytest.raises(NotFound):
        service.revoke_source("brief", version=3)
    service.delete_source("notes")
    assert "notes" not in service.snapshot()["sources"]
    with pytest.raises(NotFound):
        service.read_source("notes")
    with pytest.raises(NotFound):
        service.delete_source("notes")
    # A source added again under a deleted id starts a new, readable history.
    service.delete_source("brief")
    assert service.add_source("brief", "Version one again").version == 1
    assert service.read_source("brief").content == "Version one again"
    with serve_http(service) as endpoint:
        service.revoke_source("brief")
        assert call_http(endpoint.port, "GET", "/sources/brief")[0] == 404
