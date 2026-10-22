"""Delivery acceptance policies, records, validator registration and checkpoint compatibility."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import subprocess
import typing
from collections.abc import Sequence

import pytest

import smythe.acceptance
import smythe.validation
from fakes import child_executable, offline_env
from smythe import (
    AcceptanceDecision,
    AcceptancePolicy,
    DeliveryAcceptance,
    OfflineProvider,
    SwarmResult,
    Task,
    ValidationReport,
    ValidationSubject,
    ValidatorIdentity,
)
from smythe.acceptance import (
    DELIVERY_ACCEPTANCE_VERSION,
    AcceptanceArtifact,
    AcceptanceSource,
    ValidatorResult,
    check_validator,
    task_sha256,
    text_sha256,
)
from smythe.checkpoint import (
    STRICT_CHECKPOINT_VERSION,
    SUPPORTED_CHECKPOINT_VERSIONS,
    build_state,
    delivery_acceptance_from_state,
)
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.registry import Registry
from smythe.validation import hard_finding

OUTPUT = "## Summary\nThe final report."
POLICY_HASH = "a" * 64


class RequiredHeading:
    """A local validator: the delivered text must contain a heading."""

    def __init__(self, heading: str = "## Summary", *, name: str = "required-heading",
                 version: str = "1") -> None:
        self.heading = heading
        self.identity = ValidatorIdentity.for_config(name, version, {"heading": heading})

    def validate(self, subject: ValidationSubject) -> ValidationReport:
        if self.heading in subject.text():
            return ValidationReport("output", validator=self.identity)
        return ValidationReport("output", (
            hard_finding("heading_missing", f"output has no {self.heading!r} heading"),
        ), validator=self.identity)


def _record(decision="accepted", **overrides) -> DeliveryAcceptance:
    validator = RequiredHeading().identity
    fields = {
        "decision": decision,
        "policy_sha256": POLICY_HASH,
        "output_sha256": text_sha256(OUTPUT),
        "strategy": "deliverable",
        "sources": (AcceptanceSource("draft", 2),),
        "used_fallback": False,
        "task_sha256": task_sha256(Task("Write the report")),
        "validators": (ValidatorResult(
            validator, "passed" if decision == "accepted" else "rejected",
        ),),
        "reasons": () if decision == "accepted" else ("output has no '## Summary' heading",),
    }
    fields.update(overrides)
    return DeliveryAcceptance(**fields)


# ---------------------------------------------------------------------------
# Validation subjects
# ---------------------------------------------------------------------------


def test_acceptance_types_import_without_an_image_library():
    code = (
        "import sys\n"
        "sys.modules['PIL'] = None\n"
        "import smythe.acceptance\n"
        "assert smythe.acceptance.AcceptancePolicy().mode == 'advisory'\n"
        "assert not [m for m in sys.modules if m.startswith('PIL') and sys.modules[m]]\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [child_executable(), "-c", code], env=offline_env(), capture_output=True,
        text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


_PUBLIC_TYPES = [
    *(getattr(module, name) for module in (smythe.acceptance, smythe.validation)
      for name in module.__all__ if inspect.isclass(getattr(module, name))),
    SwarmResult,
]


@pytest.mark.parametrize("kind", _PUBLIC_TYPES, ids=lambda kind: kind.__name__)
def test_public_types_resolve_their_type_hints(kind):
    """Tools that read annotations at runtime can resolve every field and method."""
    typing.get_type_hints(kind)
    for member in vars(kind).values():
        if isinstance(member, property):
            member = member.fget
        member = getattr(member, "__func__", member)
        if inspect.isfunction(member):
            typing.get_type_hints(member)


def test_policy_type_hints_name_the_validator_protocol():
    hints = typing.get_type_hints(AcceptancePolicy)
    assert hints["final_validators"] == Sequence[smythe.acceptance.Validator]


def test_subject_binds_bytes_to_their_digest():
    subject = ValidationSubject.from_text(OUTPUT)
    assert subject.data == OUTPUT.encode("utf-8")
    assert subject.sha256 == text_sha256(OUTPUT) == hashlib.sha256(OUTPUT.encode()).hexdigest()
    assert subject.text() == OUTPUT
    assert ValidationSubject(bytearray(b"abc")).data == b"abc"
    with pytest.raises(TypeError):
        ValidationSubject("text is not bytes")


# ---------------------------------------------------------------------------
# Acceptance policy
# ---------------------------------------------------------------------------


def test_default_policy_is_advisory_and_records_nothing_strict():
    policy = AcceptancePolicy()
    assert policy.mode == "advisory"
    assert policy.to_dict() == {
        "version": 1, "mode": "advisory", "final_validators": [],
        "on_exhausted": "rejected", "on_invalid": "indeterminate",
        "allow_gate_acceptance": True,
    }


def test_policy_hash_is_stable_and_pinned():
    first = AcceptancePolicy(mode="strict", final_validators=(RequiredHeading(),))
    again = AcceptancePolicy(mode="strict", final_validators=[RequiredHeading()])
    canonical = json.dumps(first.to_dict(), sort_keys=True, separators=(",", ":")).encode()

    assert first.sha256 == again.sha256 == hashlib.sha256(canonical).hexdigest()
    assert first.validator_identities == (RequiredHeading().identity,)
    # Pinned: a change to the canonical policy form changes every saved
    # policy identity, so it must be deliberate.
    pinned = (b'{"allow_gate_acceptance":true,"final_validators":[],"mode":"strict",'
              b'"on_exhausted":"rejected","on_invalid":"indeterminate","version":1}')
    assert AcceptancePolicy(mode="strict").sha256 == hashlib.sha256(pinned).hexdigest() == (
        "8a51880d163fc401c0ded4dc951ccdf0df72f7bd1b15787b0beb85cec092111f"
    )


def test_policy_hash_ignores_validator_order_but_tracks_every_setting():
    heading = RequiredHeading()
    sources = RequiredHeading("## Sources", name="required-sources")
    base = AcceptancePolicy(mode="strict", final_validators=(heading, sources))

    assert base.sha256 == AcceptancePolicy(mode="strict", final_validators=(sources, heading)).sha256
    changed = [
        AcceptancePolicy(mode="strict", final_validators=(heading,)),
        AcceptancePolicy(mode="strict", final_validators=(heading, RequiredHeading(
            "## Sources", name="required-sources", version="2"))),
        AcceptancePolicy(mode="strict", final_validators=(heading, RequiredHeading(
            "## References", name="required-sources"))),
        AcceptancePolicy(mode="strict", final_validators=(heading, sources),
                         on_exhausted="needs_review"),
        AcceptancePolicy(mode="strict", final_validators=(heading, sources),
                         on_invalid="needs_review"),
        AcceptancePolicy(mode="strict", final_validators=(heading, sources),
                         allow_gate_acceptance=False),
        AcceptancePolicy(mode="strict", final_validators=(heading, sources),
                         allow_generated_gates=True),
    ]
    assert len({base.sha256, *(policy.sha256 for policy in changed)}) == len(changed) + 1


def test_counting_generated_gates_is_off_and_omitted_by_default():
    policy = AcceptancePolicy(mode="strict")
    assert policy.allow_generated_gates is False
    assert "allow_generated_gates" not in policy.to_dict()
    allowed = AcceptancePolicy(mode="strict", allow_generated_gates=True)
    assert allowed.to_dict() == {**policy.to_dict(), "allow_generated_gates": True}


def test_policy_snapshots_validator_identities_at_construction():
    validator = RequiredHeading()
    policy = AcceptancePolicy(mode="strict", final_validators=(validator,))
    before = policy.sha256
    validator.identity = ValidatorIdentity.for_config("required-heading", "9")
    assert policy.sha256 == before


@pytest.mark.parametrize("kwargs,match", [
    ({"mode": "lenient"}, "mode"),
    ({"mode": "strict", "on_exhausted": "accepted"}, "on_exhausted"),
    ({"mode": "strict", "on_exhausted": "indeterminate"}, "on_exhausted"),
    ({"mode": "strict", "on_invalid": "rejected"}, "on_invalid"),
    ({"mode": "strict", "allow_gate_acceptance": 1}, "allow_gate_acceptance"),
    ({"mode": "strict", "allow_gate_acceptance": False}, "never accept"),
    ({"final_validators": (RequiredHeading(),)}, "advisory"),
    ({"on_exhausted": "needs_review"}, "advisory"),
    ({"allow_gate_acceptance": False}, "advisory"),
    ({"allow_generated_gates": True}, "advisory"),
    ({"mode": "strict", "allow_generated_gates": 1}, "allow_generated_gates"),
    ({"mode": "strict", "final_validators": (RequiredHeading(),), "allow_gate_acceptance": False,
      "allow_generated_gates": True}, "allow_gate_acceptance"),
    ({"mode": "strict", "final_validators": (RequiredHeading(), RequiredHeading("## Other"))},
     "duplicate"),
])
def test_policy_rejects_settings_it_cannot_honor(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AcceptancePolicy(**kwargs)


def test_policy_requires_a_sequence_of_validators():
    with pytest.raises(TypeError, match="final_validators"):
        AcceptancePolicy(mode="strict", final_validators=RequiredHeading())


class _AsyncValidate:
    identity = ValidatorIdentity.for_config("async", "1")

    async def validate(self, subject):
        return ValidationReport("output")


class _Awaitable:
    identity = ValidatorIdentity.for_config("awaitable", "1")

    def validate(self, subject):
        return ValidationReport("output")

    def __await__(self):
        yield


class _AsyncCall:
    async def __call__(self, subject):
        return ValidationReport("output")


class _AsyncGeneratorCall:
    async def __call__(self, subject):
        yield ValidationReport("output")


class _SyncCall:
    def __call__(self, subject):
        return ValidationReport("output")


class _AsyncCallValidate:
    """``validate`` is an object whose ``__call__`` is a coroutine function."""

    identity = ValidatorIdentity.for_config("async-call", "1")
    validate = _AsyncCall()


class _AsyncGeneratorCallValidate:
    identity = ValidatorIdentity.for_config("async-generator-call", "1")
    validate = _AsyncGeneratorCall()


class _NoIdentity:
    def validate(self, subject):
        return ValidationReport("output")


class _PlainIdentity(_NoIdentity):
    identity = {"name": "plain", "version": "1"}


async def _async_function(subject):
    return ValidationReport("output")


class _ProviderValidator(OfflineProvider):
    identity = ValidatorIdentity.for_config("provider", "1")

    def validate(self, subject):
        return ValidationReport("output")


@pytest.mark.parametrize("candidate,match", [
    (OfflineProvider(), "Provider"),
    (_ProviderValidator(), "Provider"),
    (_async_function, "coroutine function"),
    (_AsyncValidate(), "synchronous"),
    (_AsyncCallValidate(), "synchronous"),
    (_AsyncGeneratorCallValidate(), "synchronous"),
    (_Awaitable(), "awaitable"),
    (RequiredHeading, "instance"),
    (_NoIdentity(), "identity"),
    (_PlainIdentity(), "identity"),
    (object(), "validate"),
])
def test_registration_structurally_rejects_non_local_validators(candidate, match):
    with pytest.raises(TypeError, match=match):
        check_validator(candidate)
    with pytest.raises(TypeError, match=match):
        AcceptancePolicy(mode="strict", final_validators=(candidate,))


def test_registration_closes_a_rejected_coroutine():
    coroutine = _async_function(ValidationSubject(b""))
    with pytest.raises(TypeError, match="awaitable"):
        check_validator(coroutine)
    assert coroutine.cr_frame is None  # closed, so it never warns as unawaited


def test_registration_rejects_an_asyncio_future():
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(TypeError, match="awaitable"):
            check_validator(loop.create_future())
    finally:
        loop.close()


def test_registration_returns_the_validator_identity():
    validator = RequiredHeading()
    assert check_validator(validator) == validator.identity


def test_registration_accepts_a_synchronous_callable_validate():
    class CallableValidate:
        identity = ValidatorIdentity.for_config("sync-call", "1")
        validate = _SyncCall()

    class AsyncCallableValidator(RequiredHeading):
        """Only ``validate`` is called, so the validator's own ``__call__`` does not matter."""

        async def __call__(self):
            return None

    assert check_validator(CallableValidate()) == CallableValidate.identity
    validator = AsyncCallableValidator()
    assert check_validator(validator) == validator.identity


# ---------------------------------------------------------------------------
# Delivery acceptance records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decision", ["accepted", "rejected", "needs_review", "indeterminate"])
def test_record_round_trips_through_json(decision):
    record = _record(decision)
    data = json.loads(json.dumps(record.to_dict()))

    assert set(data) == {
        "version", "decision", "reasons", "output_sha256", "strategy", "sources",
        "used_fallback", "task_sha256", "policy_sha256", "validators", "scope",
    }
    assert data["version"] == DELIVERY_ACCEPTANCE_VERSION == 1
    assert data["decision"] == decision
    assert data["sources"] == [{"node_id": "draft", "execution_generation": 2}]
    assert DeliveryAcceptance.from_dict(data) == record
    assert DeliveryAcceptance.from_dict(data).accepted is (decision == "accepted")


def test_not_evaluated_record_needs_no_output_yet():
    record = DeliveryAcceptance(decision="not_evaluated", policy_sha256=POLICY_HASH)
    assert record.decision is AcceptanceDecision.NOT_EVALUATED
    assert record.output_sha256 is None and record.strategy is None
    assert DeliveryAcceptance.from_dict(record.to_dict()) == record


@pytest.mark.parametrize("version", [0, 2, "1", 1.0, True, None])
def test_record_reader_rejects_unknown_versions(version):
    data = _record().to_dict()
    data["version"] = version
    with pytest.raises(ValueError, match="version"):
        DeliveryAcceptance.from_dict(data)


def test_record_reader_rejects_a_missing_version():
    data = _record().to_dict()
    del data["version"]
    with pytest.raises(ValueError, match="version"):
        DeliveryAcceptance.from_dict(data)


@pytest.mark.parametrize("decision", ["approved", "ACCEPTED", "", None, True, "passed"])
def test_record_reader_rejects_unknown_decisions(decision):
    data = _record().to_dict()
    data["decision"] = decision
    with pytest.raises(ValueError, match="decision"):
        DeliveryAcceptance.from_dict(data)


@pytest.mark.parametrize("change", [
    lambda data: data.update(accepted=True),
    lambda data: data.pop("policy_sha256"),
    lambda data: data.update(used_fallback="no"),
    lambda data: data.update(scope="everything"),
    lambda data: data.update(scope=[]),
    lambda data: data.update(scope=["output"]),
    lambda data: data.update(scope={"output": True}),
    lambda data: data.update(reasons="none"),
    lambda data: data.update(sources=[{"node_id": "draft"}]),
    lambda data: data.update(sources=[{"node_id": "draft", "execution_generation": -1}]),
    lambda data: data.update(sources=[{"node_id": "draft", "execution_generation": True}]),
    lambda data: data.update(validators=[{"validator": {}, "outcome": "passed"}]),
    lambda data: data["validators"][0].update(outcome="maybe"),
    lambda data: data.update(output_sha256="not-a-digest"),
    lambda data: data.update(task_sha256=""),
])
def test_record_reader_rejects_malformed_records(change):
    data = _record().to_dict()
    change(data)
    with pytest.raises(ValueError):
        DeliveryAcceptance.from_dict(data)


def test_record_reader_rejects_non_objects():
    for value in (None, [], "accepted"):
        with pytest.raises(ValueError):
            DeliveryAcceptance.from_dict(value)


def test_accepted_record_cannot_carry_a_failing_validator():
    identity = RequiredHeading().identity
    for outcome in ("rejected", "error"):
        with pytest.raises(ValueError, match="accepted"):
            _record("accepted", validators=(ValidatorResult(identity, outcome),))


def test_decided_records_bind_output_strategy_and_reasons():
    with pytest.raises(ValueError, match="output_sha256"):
        _record("rejected", output_sha256=None)
    with pytest.raises(ValueError, match="strategy"):
        _record("accepted", strategy=None)
    with pytest.raises(ValueError, match="reason"):
        _record("indeterminate", reasons=())
    with pytest.raises(ValueError, match="duplicate"):
        _record(sources=(AcceptanceSource("draft"), AcceptanceSource("draft", 1)))
    identity = RequiredHeading().identity
    with pytest.raises(ValueError, match="duplicate"):
        _record(validators=(ValidatorResult(identity, "passed"),) * 2)


PNG_SHA256 = hashlib.sha256(b"\x89PNG").hexdigest()


def test_record_binds_delivered_files_only_when_there_are_some():
    artifact = AcceptanceArtifact("draft", "draft_00.png", PNG_SHA256, 4, "image/png")
    record = _record(artifacts=(artifact,))
    data = json.loads(json.dumps(record.to_dict()))

    assert data["artifacts"] == [{"node_id": "draft", "name": "draft_00.png",
                                  "sha256": PNG_SHA256, "size_bytes": 4, "mime_type": "image/png"}]
    assert DeliveryAcceptance.from_dict(data) == record
    assert "artifacts" not in _record().to_dict()
    with pytest.raises(ValueError, match="sources"):
        _record(artifacts=(AcceptanceArtifact("other", "a.png", PNG_SHA256, 4, "image/png"),))


@pytest.mark.parametrize("change", [
    lambda item: item.update(sha256="not-a-digest"),
    lambda item: item.update(size_bytes=-1),
    lambda item: item.update(size_bytes=True),
    lambda item: item.update(name=""),
    lambda item: item.update(mime_type=None),
    lambda item: item.update(path="/absolute/draft_00.png"),
    lambda item: item.pop("node_id"),
])
def test_record_reader_rejects_malformed_artifacts(change):
    artifact = AcceptanceArtifact("draft", "draft_00.png", PNG_SHA256, 4, "image/png")
    data = _record(artifacts=(artifact,)).to_dict()
    change(data["artifacts"][0])
    with pytest.raises(ValueError):
        DeliveryAcceptance.from_dict(data)
    data["artifacts"] = {}
    with pytest.raises(ValueError, match="artifacts"):
        DeliveryAcceptance.from_dict(data)


def test_constructing_an_unknown_record_version_fails():
    with pytest.raises(ValueError, match="version"):
        _record(version=2)


@pytest.mark.parametrize("scope", [[], ["output"], {"output": True}, None])
def test_record_rejects_a_scope_that_is_not_a_known_name(scope):
    with pytest.raises(ValueError, match="scope"):
        _record(scope=scope)


def test_task_hash_follows_the_task_snapshot():
    task = Task("Write the report", constraints=["cite sources"], context={"n": 1})
    assert task_sha256(None) is None
    assert task_sha256(task) == task_sha256(Task(
        "Write the report", constraints=["cite sources"], context={"n": 1},
    ))
    assert task_sha256(task) != task_sha256(Task(
        "Write the report", constraints=["cite sources"], context={"n": 2},
    ))


# ---------------------------------------------------------------------------
# SwarmResult
# ---------------------------------------------------------------------------


def _graph() -> ExecutionGraph:
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="Draft", status=NodeStatus.COMPLETED, result=OUTPUT),
    ])


def test_swarm_result_positional_construction_keeps_its_field_order():
    graph = _graph()
    legacy = SwarmResult("done", graph, [], 0.5, "run-1", True, False,
                         "execution_and_synthesis", None, {"elapsed_ms": 1.0})
    assert legacy.timing == {"elapsed_ms": 1.0}
    assert (legacy.acceptance, legacy.rejected_output, legacy.accepted) == (None, None, False)

    record = _record("rejected")
    strict = SwarmResult("", graph, [], 0.5, "run-1", True, False,
                         "execution_and_synthesis", None, None, record, OUTPUT)
    assert strict.acceptance is record
    assert strict.rejected_output == OUTPUT
    assert strict.accepted is False


@pytest.mark.parametrize("decision,accepted", [
    ("accepted", True), ("rejected", False), ("needs_review", False), ("indeterminate", False),
])
def test_swarm_result_is_accepted_only_for_an_accepted_record(decision, accepted):
    result = SwarmResult(OUTPUT, _graph(), acceptance=_record(decision))
    assert result.accepted is accepted


def test_swarm_result_accepted_is_read_only():
    result = SwarmResult(OUTPUT, _graph(), acceptance=_record())
    with pytest.raises(AttributeError):
        result.accepted = False


def test_swarm_result_projects_to_json():
    record = _record("rejected")
    result = SwarmResult("", _graph(), execution_id="run-1", acceptance=record,
                         rejected_output=OUTPUT, timing={"elapsed_ms": 2.5})

    data = json.loads(json.dumps(result.to_dict()))

    assert data["output"] == "" and data["rejected_output"] == OUTPUT
    assert data["delivery_acceptance"] == record.to_dict()
    # JSON uses one distinct key for the decision, never a bare "accepted".
    assert "acceptance" not in data and "accepted" not in data
    assert data["execution_id"] == "run-1"
    assert data["timing"] == {"elapsed_ms": 2.5}
    assert data["cost_scope"] == "execution_and_synthesis"
    assert [node["id"] for node in data["graph"]["nodes"]] == ["draft"]


def test_legacy_swarm_result_projects_without_a_record():
    data = SwarmResult(OUTPUT, _graph()).to_dict()
    assert data["output"] == OUTPUT
    assert data["delivery_acceptance"] is None
    assert data["rejected_output"] is None


# ---------------------------------------------------------------------------
# Checkpoint compatibility, with fixtures generated here
# ---------------------------------------------------------------------------


def _checkpoint_state(version: int, *, status: str = "completed",
                      acceptance: DeliveryAcceptance | None = None) -> dict:
    state = build_state(
        execution_id="run1", status=status, model="m", graph=_graph(), registry=Registry(),
        task=None, max_budget_usd=None, node_costs={},
        output=OUTPUT if status == "completed" else None,
        delivery_acceptance=acceptance,
    )
    state["version"] = version
    if version == 1:
        del state["control"]  # version 1 predates run control state
    return json.loads(json.dumps(state))


def test_supported_checkpoint_versions_add_the_strict_version():
    assert SUPPORTED_CHECKPOINT_VERSIONS == (1, 2, 3, 4, 5)
    assert STRICT_CHECKPOINT_VERSION == 5


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("status", ["completed", "running", "failed"])
def test_legacy_checkpoints_project_no_acceptance(version, status):
    assert delivery_acceptance_from_state(_checkpoint_state(version, status=status)) is None


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_legacy_checkpoints_ignore_a_delivery_acceptance_block(version):
    state = _checkpoint_state(version)
    state["delivery_acceptance"] = _record().to_dict()
    assert delivery_acceptance_from_state(state) is None


def test_strict_checkpoint_carries_its_record():
    record = _record("needs_review")
    state = _checkpoint_state(5, acceptance=record)
    assert state["requires"] == ["delivery_acceptance"]
    assert state["delivery_acceptance"] == record.to_dict()
    assert delivery_acceptance_from_state(state) == record


def _pending_record() -> DeliveryAcceptance:
    return DeliveryAcceptance("not_evaluated", POLICY_HASH)


@pytest.mark.parametrize("status", ["running", "failed"])
def test_unfinished_strict_checkpoint_carries_an_undecided_record(status):
    record = _pending_record()
    state = _checkpoint_state(5, status=status, acceptance=record)
    assert state["requires"] == ["delivery_acceptance"]
    assert delivery_acceptance_from_state(state) == record


@pytest.mark.parametrize("status", ["completed", "running"])
def test_version_5_without_requirements_has_no_record(status):
    state = _checkpoint_state(5, status=status)
    state["requires"] = []
    assert delivery_acceptance_from_state(state) is None


@pytest.mark.parametrize("status", ["completed", "running"])
def test_version_5_must_list_its_requirements(status):
    state = _checkpoint_state(5, status=status)
    state.pop("requires", None)
    with pytest.raises(ValueError, match="requires"):
        delivery_acceptance_from_state(state)


@pytest.mark.parametrize("requires", ["delivery_acceptance", [1], {"delivery_acceptance": 1}, None])
def test_version_5_requirements_must_be_a_list_of_names(requires):
    state = _checkpoint_state(5, acceptance=_record())
    state["requires"] = requires
    with pytest.raises(ValueError, match="requires"):
        delivery_acceptance_from_state(state)


@pytest.mark.parametrize("status", ["completed", "running", "failed"])
def test_version_5_listing_an_unknown_requirement_is_refused(status):
    record = _record() if status == "completed" else _pending_record()
    state = _checkpoint_state(5, status=status, acceptance=record)
    state["requires"] = ["delivery_acceptance", "attempt_allowance"]
    with pytest.raises(ValueError, match="attempt_allowance"):
        delivery_acceptance_from_state(state)


@pytest.mark.parametrize("status", ["completed", "running"])
def test_completed_strict_checkpoint_requires_its_record(status):
    state = _checkpoint_state(5, status=status)
    state["requires"] = ["delivery_acceptance"]
    with pytest.raises(ValueError, match="delivery_acceptance"):
        delivery_acceptance_from_state(state)


def test_a_record_must_be_listed_as_required():
    state = _checkpoint_state(5, acceptance=_record())
    state["requires"] = []
    with pytest.raises(ValueError, match="requires"):
        delivery_acceptance_from_state(state)


def test_completed_strict_checkpoint_record_holds_a_decision():
    state = _checkpoint_state(5, acceptance=_record())
    state["delivery_acceptance"] = _pending_record().to_dict()
    with pytest.raises(ValueError, match="decision"):
        delivery_acceptance_from_state(state)
    with pytest.raises(ValueError, match="decision"):
        build_state(
            execution_id="run1", status="completed", model="m", graph=_graph(),
            registry=Registry(), task=None, max_budget_usd=None, node_costs={}, output=OUTPUT,
            delivery_acceptance=_pending_record(),
        )


@pytest.mark.parametrize("version", [0, 6, "5", None])
def test_acceptance_reader_rejects_unknown_checkpoint_versions(version):
    state = _checkpoint_state(4)
    state["version"] = version
    with pytest.raises(ValueError, match="version"):
        delivery_acceptance_from_state(state)


def test_build_state_writes_the_strict_version_only_with_a_record():
    record = _record()
    plain = build_state(
        execution_id="run1", status="completed", model="m", graph=_graph(),
        registry=Registry(), task=None, max_budget_usd=None, node_costs={}, output=OUTPUT,
    )
    strict = build_state(
        execution_id="run1", status="completed", model="m", graph=_graph(),
        registry=Registry(), task=None, max_budget_usd=None, node_costs={}, output=OUTPUT,
        delivery_acceptance=record,
    )
    assert plain["version"] == 4 and "delivery_acceptance" not in plain
    assert "requires" not in plain
    assert strict["version"] == 5 and strict["delivery_acceptance"] == record.to_dict()
    assert strict["requires"] == ["delivery_acceptance"]


def test_build_state_refuses_a_record_for_other_output():
    with pytest.raises(ValueError, match="output"):
        build_state(
            execution_id="run1", status="completed", model="m", graph=_graph(),
            registry=Registry(), task=None, max_budget_usd=None, node_costs={},
            output="different text", delivery_acceptance=_record(),
        )
    with pytest.raises(TypeError, match="DeliveryAcceptance"):
        build_state(
            execution_id="run1", status="completed", model="m", graph=_graph(),
            registry=Registry(), task=None, max_budget_usd=None, node_costs={},
            output=OUTPUT, delivery_acceptance=_record().to_dict(),
        )
