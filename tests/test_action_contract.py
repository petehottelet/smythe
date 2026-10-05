"""External-action contract types: intent identity, keys, connector profiles and host checks."""

from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import math
import pickle
import subprocess
import typing
import warnings

import pytest

import smythe
import smythe.actions
from fakes import child_executable, offline_env
from smythe import OfflineProvider
from smythe._canonical import canonical_bytes, sha256_hex, sha256_prefixed
from smythe.actions import (
    ACTION_CONTRACT_VERSION,
    ACTION_STATES,
    ActionIntent,
    ActionStateError,
    ActionTarget,
    ApprovalRejected,
    ConnectorAmbiguous,
    ConnectorProfile,
    ConnectorReceipt,
    ConnectorRefused,
    ContentRef,
    ObservedObject,
    Principal,
    action_key,
    check_connector,
    check_verifier,
)
from smythe.optimize.contracts import sha256_prefixed as contracts_sha256_prefixed
from smythe.validation import _json_sha256
from smythe.workflow_store import WorkflowStateError

STORE = "0" * 32
BODY = "# Launch notes\n\nCafé opens at nine.\n".encode()
CONTENT = ContentRef.for_bytes(BODY, "text/markdown")


def _intent(**changes):
    values = {
        "operation": "publish",
        "target": ActionTarget("acme", "posts"),
        "parameters": {"title": "Launch notes", "tags": ["news", "cafe"], "draft": False},
        "content": CONTENT,
        "connector": "fixture-posts",
        "connector_version": "1",
    }
    values.update(changes)
    return ActionIntent(values.pop("operation"), values.pop("target"), **values)


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------


def test_canonical_digests_match_the_existing_canonical_json_forms():
    value = {"z": [1, 2.5, None, True], "a": {"é": "naïve", "b": "x"}, "n": -7}

    assert canonical_bytes(value) == (
        '{"a":{"b":"x","é":"naïve"},"n":-7,"z":[1,2.5,null,true]}'.encode()
    )
    assert sha256_hex(value) == _json_sha256(value, "value")
    assert sha256_prefixed(value) == contracts_sha256_prefixed(value) == (
        "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()
    )


def _cycle():
    value = {"items": []}
    value["items"].append(value)
    return value


@pytest.mark.parametrize("value", [
    math.nan, {"x": math.inf}, b"bytes", {1: "x"}, {True: "x"}, {None: "x"}, _cycle(),
    {"x": "\ud800"}, {"x": {1, 2}},
], ids=["nan", "infinity", "bytes", "int-key", "bool-key", "none-key", "cycle",
        "lone-surrogate", "set"])
def test_canonical_json_refuses_values_that_are_not_json_data(value):
    with pytest.raises(ValueError, match="not canonical JSON"):
        canonical_bytes(value)


# ---------------------------------------------------------------------------
# Intent identity
# ---------------------------------------------------------------------------


def test_intent_digest_is_pinned():
    intent = ActionIntent(
        "publish", ActionTarget("acme", "posts"), parameters={"title": "Launch"},
        content=ContentRef("a" * 64, 12, "text/plain"), connector="posts", connector_version="1",
    )
    canonical = (
        b'{"connector":"posts","connector_version":"1","content":{"media_type":"text/plain",'
        b'"sha256":"' + b"a" * 64 + b'","size_bytes":12},"contract_version":"smythe.action.v1",'
        b'"operation":"publish","parameters":{"title":"Launch"},'
        b'"target":{"collection":"posts","object_id":null,"tenant":"acme"}}'
    )
    # Pinned: approvals bind this digest, so its canonical form changes only
    # with the contract version.
    assert intent.digest() == hashlib.sha256(canonical).hexdigest() == (
        "20673861f5fe76adf6b152c4fbd4f8c206d5378952e4043f58373b443070b160"
    )


def test_intent_digest_is_stable_under_dict_reordering():
    first = _intent(parameters={"b": {"y": 2, "x": 1}, "a": [3, {"q": 1, "p": 2}]})
    again = _intent(parameters={"a": [3, {"p": 2, "q": 1}], "b": {"x": 1, "y": 2}})

    assert first == again
    assert first.digest() == again.digest()
    assert len(first.digest()) == 64 and int(first.digest(), 16) >= 0


def test_intent_round_trips_and_digest_does_not_depend_on_attempt_data():
    intent = _intent()
    data = intent.to_dict()

    # Nothing about an attempt, a time or a key is part of the intent.
    assert set(data) == {"contract_version", "operation", "target", "parameters", "content",
                         "connector", "connector_version"}
    for restored in (ActionIntent.from_dict(data), pickle.loads(pickle.dumps(intent)),
                     copy.deepcopy(intent), _intent()):
        assert restored == intent
        assert restored.digest() == intent.digest()
        assert hash(restored) == hash(intent)
    with pytest.raises(TypeError):
        _intent(attempt=2)


def test_intent_copies_its_parameters():
    parameters = {"tags": ["news"]}
    intent = _intent(parameters=parameters)
    digest = intent.digest()

    parameters["tags"].append("late")
    intent.parameters["tags"].append("later")

    assert intent.parameters == {"tags": ["news"]}
    assert intent.digest() == digest
    assert _intent(parameters=None).parameters == {}


@pytest.mark.parametrize("changes", [
    {"operation": "update"},
    {"target": ActionTarget("other-tenant", "posts")},
    {"target": ActionTarget("acme", "pages")},
    {"target": ActionTarget("acme", "posts", "obj_1")},
    {"parameters": {"title": "Launch notes!", "tags": ["news", "cafe"], "draft": False}},
    {"parameters": {"title": "Launch notes", "tags": ["cafe", "news"], "draft": False}},
    {"parameters": {"title": "Launch notes", "tags": ["news", "cafe"], "draft": 0}},
    {"parameters": {"title": "Launch notes", "tags": ["news", "cafe"]}},
    {"content": ContentRef("0" * 64, CONTENT.size_bytes, CONTENT.media_type)},
    {"content": ContentRef(CONTENT.sha256, CONTENT.size_bytes + 1, CONTENT.media_type)},
    {"content": ContentRef(CONTENT.sha256, CONTENT.size_bytes, "text/plain")},
    {"content": None},
    {"connector": "fixture-pages"},
    {"connector_version": "2"},
], ids=["operation", "tenant", "collection", "object_id", "parameter-value", "parameter-order",
        "parameter-type", "parameter-removed", "content-sha256", "content-size",
        "content-media-type", "no-content", "connector", "connector-version"])
def test_changing_any_intent_field_changes_the_digest(changes):
    assert _intent(**changes).digest() != _intent().digest()
    assert _intent(**changes) != _intent()


def test_intents_that_differ_only_in_parameters_are_unequal_and_hash_differently():
    first = _intent(parameters={"title": "Launch notes"})
    second = _intent(parameters={"title": "Launch notes!"})

    assert first != second and first.digest() != second.digest()
    assert hash(first) != hash(second)
    assert len({first, second, _intent(parameters={"title": "Launch notes"})}) == 2


def test_object_ids_name_different_intents():
    first = _intent(target=ActionTarget("acme", "posts", "obj_1"))
    assert first.digest() != _intent(target=ActionTarget("acme", "posts", "obj_2")).digest()


def test_changing_the_contract_version_changes_the_digest(monkeypatch):
    with pytest.raises(ValueError, match="Unknown action contract version"):
        _intent(contract_version="smythe.action.v2")
    current = _intent()
    monkeypatch.setattr(smythe.actions, "_CONTRACT_VERSIONS",
                        (ACTION_CONTRACT_VERSION, "smythe.action.v2"))

    later = _intent(contract_version="smythe.action.v2")

    assert later.to_dict()["contract_version"] == "smythe.action.v2"
    assert later.digest() != current.digest()


@pytest.mark.parametrize("parameters", [
    {"limit": math.nan}, {"limit": math.inf}, {"limit": -math.inf}, {"body": b"bytes"},
    {1: "one"}, {"nested": {2: "two"}}, _cycle(), {"items": [_cycle()]}, {"tags": ("a",)},
    {"tags": {"a"}}, {"text": "\ud800"}, ["not", "an", "object"], "text",
], ids=["nan", "infinity", "negative-infinity", "bytes", "int-key", "nested-int-key", "cycle",
        "nested-cycle", "tuple", "set", "lone-surrogate", "list", "string"])
def test_intent_rejects_parameters_that_are_not_a_json_object(parameters):
    with pytest.raises(ValueError, match="action parameters"):
        _intent(parameters=parameters)


@pytest.mark.parametrize("changes,error", [
    ({"operation": ""}, ValueError),
    ({"operation": "pub\nlish"}, ValueError),
    ({"connector": None}, ValueError),
    ({"connector_version": ""}, ValueError),
    ({"contract_version": None}, ValueError),
    ({"target": {"tenant": "acme", "collection": "posts"}}, TypeError),
    ({"content": CONTENT.to_dict()}, TypeError),
])
def test_intent_rejects_invalid_fields(changes, error):
    with pytest.raises(error):
        _intent(**changes)


def _without(data, key):
    return {name: value for name, value in data.items() if name != key}


@pytest.mark.parametrize("edit,match", [
    (lambda data: {**data, "contract_version": "smythe.action.v2"}, "contract version"),
    (lambda data: {**data, "contract_version": 1}, "contract version"),
    (lambda data: {**data, "attempt": 1}, "unknown keys"),
    (lambda data: _without(data, "content"), "missing keys"),
    (lambda data: {**data, "target": {**data["target"], "region": "eu"}}, "unknown keys"),
    (lambda data: {**data, "content": {**data["content"], "encoding": "gzip"}}, "unknown keys"),
    (lambda data: {**data, "parameters": [1]}, "JSON object"),
    # to_dict() always writes an object, so null is not its form.
    (lambda data: {**data, "parameters": None}, "JSON object"),
    (lambda data: ["not", "an", "object"], "JSON object"),
])
def test_intent_from_dict_rejects_unknown_versions_and_keys(edit, match):
    with pytest.raises(ValueError, match=match):
        ActionIntent.from_dict(edit(_intent().to_dict()))


# ---------------------------------------------------------------------------
# Action keys
# ---------------------------------------------------------------------------


def test_action_key_is_pinned_and_stable():
    key = action_key(STORE, "run-1", "publish", 1)

    assert key == action_key(STORE, "run-1", "publish", 1)
    assert key == hashlib.sha256(
        b'["smythe.action-key","' + STORE.encode() + b'","run-1","publish",1]'
    ).hexdigest() == "20fbf92a548ba75ef8f7542d2d596736fa010cb6616a41043dfb20f418c188bd"
    # Every attempt of the action sends this key; nothing about an attempt
    # or the intent is an input.
    assert list(inspect.signature(action_key).parameters) == [
        "store_id", "run_id", "slot", "sequence",
    ]


def test_action_key_differs_per_store_run_slot_and_sequence():
    keys = {
        action_key(STORE, "run-1", "publish", 1),
        action_key("1" * 32, "run-1", "publish", 1),
        action_key(STORE, "run-2", "publish", 1),
        action_key(STORE, "run-1", "announce", 1),
        action_key(STORE, "run-1", "publish", 2),
        action_key(STORE, "run-1", "publish", 0),
        # Field boundaries are unambiguous.
        action_key(STORE, "run-1p", "ublish", 1),
    }
    assert len(keys) == 7


@pytest.mark.parametrize("args", [
    ("", "run-1", "publish", 1), (STORE, "", "publish", 1), (STORE, "run-1", "", 1),
    (STORE, "run-1", "pub\x00lish", 1), (STORE, "run-1", "publish", -1),
    (STORE, "run-1", "publish", True), (STORE, "run-1", "publish", 1.0), (None, "run-1", "p", 1),
])
def test_action_key_rejects_invalid_parts(args):
    with pytest.raises(ValueError):
        action_key(*args)


# ---------------------------------------------------------------------------
# Targets and content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("args", [
    ("", "posts"), ("acme", ""), ("acme", "posts", ""), ("ac\nme", "posts"), (None, "posts"),
    ("acme", "posts", 7),
])
def test_target_rejects_invalid_names(args):
    with pytest.raises(ValueError):
        ActionTarget(*args)


def test_content_ref_identifies_bytes():
    assert CONTENT == ContentRef(hashlib.sha256(BODY).hexdigest(), len(BODY), "text/markdown")
    assert ContentRef.from_dict(CONTENT.to_dict()) == CONTENT
    with pytest.raises(TypeError, match="bytes"):
        ContentRef.for_bytes("text", "text/plain")


@pytest.mark.parametrize("args", [
    ("A" * 64, 1, "text/plain"), ("a" * 63, 1, "text/plain"), ("a" * 64, -1, "text/plain"),
    ("a" * 64, True, "text/plain"), ("a" * 64, 1.0, "text/plain"), ("a" * 64, 1, "text"),
    ("a" * 64, 1, ""),
])
def test_content_ref_rejects_invalid_fields(args):
    with pytest.raises(ValueError):
        ContentRef(*args)


# ---------------------------------------------------------------------------
# Connector profiles
# ---------------------------------------------------------------------------


def test_profile_to_dict_omits_defaults_and_round_trips():
    plain = ConnectorProfile("fixture-posts", "1", ["update", "publish"])

    assert plain.operations == ("publish", "update")
    assert plain.to_dict() == {"name": "fixture-posts", "version": "1",
                               "operations": ["publish", "update"]}
    assert ConnectorProfile.from_dict(plain.to_dict()) == plain

    declared = ConnectorProfile(
        "fixture-posts", "2", ("publish",), deduplicates=True, dedupe_window_s=86_400,
        lookup_by_key=True, lookup_by_object=True, visibility_bound_s=0, compensation="new_action",
        write_ceiling_nanousd=2_000_000, read_ceiling_nanousd=1_000, max_attempts=3,
        readback_reads=4, readback_interval_s=0.5, timeout_s=10,
    )
    assert declared.to_dict() == {
        "name": "fixture-posts", "version": "2", "operations": ["publish"],
        "deduplicates": True, "dedupe_window_s": 86_400.0, "lookup_by_key": True,
        "lookup_by_object": True, "visibility_bound_s": 0.0, "compensation": "new_action",
        "write_ceiling_nanousd": 2_000_000, "read_ceiling_nanousd": 1_000, "max_attempts": 3,
        "readback_reads": 4, "readback_interval_s": 0.5, "timeout_s": 10.0,
    }
    assert ConnectorProfile.from_dict(declared.to_dict()) == declared
    assert ConnectorProfile.from_dict(json.loads(canonical_bytes(declared.to_dict()))) == declared
    assert ConnectorProfile("fixture-posts", "1", ["publish"], timeout_s=30) == ConnectorProfile(
        "fixture-posts", "1", ["publish"])


def test_profile_settings_are_keyword_only():
    with pytest.raises(TypeError):
        ConnectorProfile("fixture-posts", "1", ["publish"], True)


@pytest.mark.parametrize("changes,match", [
    ({"max_attempts": True}, "max_attempts"),
    ({"readback_reads": False}, "readback_reads"),
    ({"write_ceiling_nanousd": True}, "write_ceiling_nanousd"),
    ({"read_ceiling_nanousd": False}, "read_ceiling_nanousd"),
    ({"timeout_s": True}, "timeout_s"),
    ({"readback_interval_s": False}, "readback_interval_s"),
    ({"visibility_bound_s": True}, "visibility_bound_s"),
    ({"deduplicates": True, "dedupe_window_s": True}, "dedupe_window_s"),
    ({"write_ceiling_nanousd": 1.0}, "write_ceiling_nanousd"),
    ({"max_attempts": 2.0}, "max_attempts"),
    ({"timeout_s": "30"}, "timeout_s"),
    ({"deduplicates": 1}, "deduplicates"),
    ({"lookup_by_key": "yes"}, "lookup_by_key"),
    ({"lookup_by_object": None}, "lookup_by_object"),
    ({"timeout_s": math.nan}, "timeout_s"),
    ({"timeout_s": math.inf}, "timeout_s"),
    ({"readback_interval_s": math.inf}, "readback_interval_s"),
    ({"visibility_bound_s": math.nan}, "visibility_bound_s"),
    ({"deduplicates": True, "dedupe_window_s": math.inf}, "dedupe_window_s"),
    ({"timeout_s": 10 ** 400}, "timeout_s"),
    ({"write_ceiling_nanousd": -1}, "write_ceiling_nanousd"),
    ({"read_ceiling_nanousd": -1}, "read_ceiling_nanousd"),
    ({"timeout_s": 0}, "timeout_s"),
    ({"readback_interval_s": -0.5}, "readback_interval_s"),
    ({"visibility_bound_s": -1}, "visibility_bound_s"),
    ({"deduplicates": True, "dedupe_window_s": 0}, "dedupe_window_s"),
    ({"max_attempts": 0}, "max_attempts"),
    ({"readback_reads": -1}, "readback_reads"),
    ({"dedupe_window_s": 60}, "needs deduplicates"),
    ({"operations": []}, "operations"),
    ({"operations": ()}, "operations"),
    ({"operations": "publish"}, "operations"),
    ({"operations": ["publish", "publish"]}, "operations"),
    ({"operations": ["publish", ""]}, "operation"),
    ({"compensation": "refund"}, "compensation"),
    ({"compensation": None}, "compensation"),
    ({"cancellation": "best_effort"}, "cancellation"),
    ({"name": ""}, "connector name"),
    ({"version": None}, "connector version"),
])
def test_profile_validation_rejects_invalid_declarations(changes, match):
    values = {"name": "fixture-posts", "version": "1", "operations": ["publish"], **changes}
    with pytest.raises(ValueError, match=match):
        ConnectorProfile(values.pop("name"), values.pop("version"), values.pop("operations"),
                         **values)


@pytest.mark.parametrize("data,match", [
    ({"name": "fixture-posts", "version": "1"}, "missing keys"),
    ({"name": "fixture-posts", "version": "1", "operations": ["publish"], "retries": 2},
     "unknown keys"),
    ({"name": "fixture-posts", "version": "1", "operations": ["publish"], "_": None},
     "unknown keys"),
    ({"name": "fixture-posts", "version": "1", "operations": ["publish"], "timeout_s": None},
     "timeout_s"),
    (["fixture-posts"], "JSON object"),
])
def test_profile_from_dict_rejects_unknown_or_invalid_data(data, match):
    with pytest.raises(ValueError, match=match):
        ConnectorProfile.from_dict(data)


# ---------------------------------------------------------------------------
# Receipts, observed objects and principals
# ---------------------------------------------------------------------------


OBJECT = ActionTarget("acme", "posts", "obj_1")


def test_receipt_round_trips():
    receipt = ConnectorReceipt("k" * 64, OBJECT, object_version=2, cost_nanousd=0,
                               deduplicated=True)
    assert ConnectorReceipt.from_dict(receipt.to_dict()) == receipt
    assert ConnectorReceipt.from_dict(ConnectorReceipt("k", OBJECT).to_dict()) == (
        ConnectorReceipt("k", OBJECT, None, None, False)
    )


@pytest.mark.parametrize("args,error", [
    (("k", ActionTarget("acme", "posts")), ValueError),
    (("k", OBJECT.to_dict()), TypeError),
    (("", OBJECT), ValueError),
    (("k", OBJECT, -1), ValueError),
    (("k", OBJECT, True), ValueError),
    (("k", OBJECT, 1, -5), ValueError),
    (("k", OBJECT, 1, 0.5), ValueError),
    (("k", OBJECT, 1, 0, 1), ValueError),
])
def test_receipt_rejects_invalid_fields(args, error):
    with pytest.raises(error):
        ConnectorReceipt(*args)


def test_receipt_from_dict_rejects_extra_keys_and_a_missing_object():
    data = ConnectorReceipt("k", OBJECT).to_dict()
    with pytest.raises(ValueError, match="unknown keys"):
        ConnectorReceipt.from_dict({**data, "body": "raw"})
    with pytest.raises(ValueError, match="object_id"):
        ConnectorReceipt.from_dict({**data, "target": {**data["target"], "object_id": None}})


def test_observed_object_digest_comes_from_its_bytes():
    observed = ObservedObject(OBJECT, BODY, "text/markdown", version=3, key="k")

    assert observed.content_ref == CONTENT
    assert BODY.decode() not in repr(observed)
    for args in ((ActionTarget("acme", "posts"), BODY, "text/markdown"),
                 (OBJECT, BODY, "markdown"), (OBJECT, BODY, "text/markdown", -1),
                 (OBJECT, BODY, "text/markdown", 1, "")):
        with pytest.raises(ValueError):
            ObservedObject(*args)
    with pytest.raises(TypeError, match="bytes"):
        ObservedObject(OBJECT, BODY.decode(), "text/markdown")


def test_principal_round_trips_and_requires_every_name():
    principal = Principal("https://idp.example", "editor-7", "acme")
    assert Principal.from_dict(principal.to_dict()) == principal
    with pytest.raises(ValueError, match="unknown keys"):
        Principal.from_dict({**principal.to_dict(), "role": "admin"})
    for args in (("", "editor-7", "acme"), ("issuer", None, "acme"), ("issuer", "editor-7", "")):
        with pytest.raises(ValueError):
            Principal(*args)


# ---------------------------------------------------------------------------
# Host connectors and verifiers
# ---------------------------------------------------------------------------


PROFILE = ConnectorProfile("fixture-posts", "1", ["publish"])


class _Connector:
    profile = PROFILE

    def submit(self, key, intent, payload, timeout_s):
        raise ConnectorRefused("not in this test")


class _LookupConnector(_Connector):
    profile = ConnectorProfile("fixture-posts", "1", ["publish"], lookup_by_key=True,
                               lookup_by_object=True)

    def lookup(self, key, target, timeout_s):
        return None

    def read(self, target, timeout_s):
        return None


class _AsyncSubmit(_Connector):
    async def submit(self, key, intent, payload, timeout_s):
        raise ConnectorRefused("never")


class _AsyncLookup(_LookupConnector):
    async def lookup(self, key, target, timeout_s):
        return None


class _AsyncUndeclaredRead(_Connector):
    async def read(self, target, timeout_s):
        return None


class _ProviderConnector(OfflineProvider):
    profile = PROFILE

    def submit(self, key, intent, payload, timeout_s):
        raise ConnectorRefused("never")


class _NoProfile:
    def submit(self, key, intent, payload, timeout_s):
        raise ConnectorRefused("never")


class _DictProfile(_Connector):
    profile = PROFILE.to_dict()


class _Awaitable:
    def __await__(self):
        yield


async def _async_function(*args):
    return None


@pytest.mark.parametrize("candidate,match", [
    (OfflineProvider(), "Provider"),
    (_ProviderConnector(), "Provider"),
    (_async_function, "coroutine function"),
    (_AsyncSubmit(), "submit.. must be synchronous"),
    (_AsyncLookup(), "lookup.. must be synchronous"),
    (_AsyncUndeclaredRead(), "read.. must be synchronous"),
    (_Awaitable(), "awaitable"),
    (_Connector, "instance"),
    (_NoProfile(), "ConnectorProfile"),
    (_DictProfile(), "ConnectorProfile"),
    (type("_NoSubmit", (), {"profile": PROFILE})(), "submit"),
    (type("_NoLookup", (), {"profile": _LookupConnector.profile, "submit": _Connector.submit,
                            "read": _LookupConnector.read})(), "lookup"),
    (type("_NoRead", (), {"profile": _LookupConnector.profile, "submit": _Connector.submit,
                          "lookup": _LookupConnector.lookup})(), "read"),
])
def test_connector_check_refuses_providers_and_async_code(candidate, match):
    with pytest.raises(TypeError, match=match):
        check_connector(candidate)


def test_connector_check_returns_the_profile_and_needs_only_declared_lookups():
    assert check_connector(_Connector()) is PROFILE
    assert check_connector(_LookupConnector()) is _LookupConnector.profile


def test_connector_check_closes_a_rejected_coroutine():
    coroutine = _async_function()
    with pytest.raises(TypeError, match="awaitable"):
        check_connector(coroutine)
    assert coroutine.cr_frame is None  # closed, so it never warns as unawaited


class _AsyncProfile(_Connector):
    @property
    async def profile(self):
        return PROFILE


def test_connector_check_closes_the_coroutine_an_async_profile_returns(monkeypatch):
    """The refused coroutine used to be dropped unawaited, which warns."""
    returned = []

    class Recorded(_Connector):
        @property
        def profile(self):
            returned.append(_async_function())
            return returned[-1]

    with pytest.raises(TypeError, match="ConnectorProfile, not coroutine"):
        check_connector(Recorded())
    assert returned[0].cr_frame is None


def test_an_async_profile_property_does_not_warn_as_never_awaited():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="ConnectorProfile"):
            check_connector(_AsyncProfile())
        gc.collect()
    assert not [warning for warning in caught if "never awaited" in str(warning.message)]


class _AsyncSubmitProperty(_Connector):
    @property
    async def submit(self):
        return None


def test_an_async_method_property_is_refused_without_an_unawaited_coroutine():
    class _AsyncVerifyProperty(_Verifier):
        @property
        async def verify(self):
            return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match=r"has no submit\(\) method"):
            check_connector(_AsyncSubmitProperty())
        with pytest.raises(TypeError, match=r"has no verify\(\) method"):
            check_verifier(_AsyncVerifyProperty())
        gc.collect()
    assert not [warning for warning in caught if "never awaited" in str(warning.message)]


class _Verifier:
    identity = "example-hmac/1"

    def verify(self, assertion, intent_sha256, now_ns):
        raise ApprovalRejected("not in this test")

    def still_authorized(self, principal, intent, now_ns):
        return False


class _AsyncVerify(_Verifier):
    async def verify(self, assertion, intent_sha256, now_ns):
        raise ApprovalRejected("never")


class _AsyncStillAuthorized(_Verifier):
    async def still_authorized(self, principal, intent, now_ns):
        return True


class _AsyncCallVerify(_Verifier):
    class _Call:
        async def __call__(self, *args):
            return None

    verify = _Call()


class _ProviderVerifier(OfflineProvider, _Verifier):
    pass


@pytest.mark.parametrize("candidate,match", [
    (OfflineProvider(), "Provider"),
    (_ProviderVerifier(), "Provider"),
    (_Awaitable(), "awaitable"),
    (_async_function, "coroutine function"),
    (_AsyncVerify(), "verify.. must be synchronous"),
    (_AsyncCallVerify(), "verify.. must be synchronous"),
    (_AsyncStillAuthorized(), "still_authorized.. must be synchronous"),
    (_Verifier, "instance"),
    (type("_NoVerify", (), {"identity": "x", "still_authorized": _Verifier.still_authorized})(),
     "verify"),
    (type("_NoIdentity", (), {"verify": _Verifier.verify,
                              "still_authorized": _Verifier.still_authorized})(), "identity"),
    (type("_BlankIdentity", (_Verifier,), {"identity": " "})(), "identity"),
])
def test_verifier_check_refuses_providers_awaitables_and_async_code(candidate, match):
    with pytest.raises(TypeError, match=match):
        check_verifier(candidate)


def test_verifier_check_closes_a_rejected_coroutine_and_returns_the_identity():
    coroutine = _async_function()
    with pytest.raises(TypeError, match="awaitable"):
        check_verifier(coroutine)
    assert coroutine.cr_frame is None
    assert check_verifier(_Verifier()) == "example-hmac/1"


@pytest.mark.parametrize("identity", ["example-hmac/1\n", "example\x00hmac", "\tx", "x" * 513],
                         ids=["newline", "nul", "tab", "too-long"])
def test_verifier_identity_is_refused_like_other_names(identity):
    """An identity with control characters used to pass, unlike every other name."""
    with pytest.raises(TypeError, match="identity"):
        check_verifier(type("_Odd", (_Verifier,), {"identity": identity})())


def test_verifier_check_closes_the_coroutine_an_async_identity_returns():
    returned = []

    class Recorded(_Verifier):
        @property
        def identity(self):
            returned.append(_async_function())
            return returned[-1]

    with pytest.raises(TypeError, match="identity"):
        check_verifier(Recorded())
    assert returned[0].cr_frame is None


# ---------------------------------------------------------------------------
# Vocabulary and module boundary
# ---------------------------------------------------------------------------


def test_action_vocabulary_and_errors():
    assert ACTION_CONTRACT_VERSION == "smythe.action.v1"
    assert ACTION_STATES == ("not_dispatched", "dispatched", "outcome_unknown", "observed",
                             "reconciled", "compensation_required", "canceled")
    assert issubclass(ActionStateError, WorkflowStateError)
    for error in (ConnectorRefused, ConnectorAmbiguous, ApprovalRejected):
        assert issubclass(error, Exception)
        assert not issubclass(error, (ValueError, OSError, WorkflowStateError))
    assert not issubclass(ConnectorRefused, ConnectorAmbiguous)
    assert not issubclass(ConnectorAmbiguous, ConnectorRefused)


_PUBLIC_TYPES = [getattr(smythe.actions, name) for name in smythe.actions.__all__
                 if inspect.isclass(getattr(smythe.actions, name))]


@pytest.mark.parametrize("kind", _PUBLIC_TYPES, ids=lambda kind: kind.__name__)
def test_public_types_resolve_their_type_hints(kind):
    typing.get_type_hints(kind)
    for member in vars(kind).values():
        if isinstance(member, property):
            member = member.fget
        member = getattr(member, "__func__", member)
        if inspect.isfunction(member):
            typing.get_type_hints(member)


def test_import_smythe_does_not_load_the_actions_module():
    assert not set(smythe.actions.__all__) & set(smythe.__all__)
    code = (
        "import sys\n"
        "import smythe\n"
        "assert 'smythe.actions' not in sys.modules\n"
        "import smythe.actions\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [child_executable(), "-c", code], env=offline_env(), capture_output=True,
        text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
