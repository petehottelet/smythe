"""Artifact lineage policies and records, and a durable run's opt-in to them."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import inspect
import json
import typing

import pytest

from smythe import (
    AcceptancePolicy, ArtifactLineage, ContextSource, LineagePolicy, OfflineProvider,
    SimpleArchitect, SQLiteWorkflowStore, Swarm, Task, TextValidator, ValidatorIdentity,
    WorkflowBindingError,
)
from smythe._canonical import canonical_bytes, sha256_hex
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.lineage import LINEAGE_REUSE_KEY, validator_set_sha256
from smythe.workflow import _canonical
from smythe.workflow_store import WorkflowConflictError

# The persisted recipe digests of two durable configurations before lineage
# existed. A configuration without lineage must keep these exact bytes.
DEFAULT_RECIPE_SHA256 = "9b477d1c599b4e260b8b8b5f8b684a19f365a76716b6c67e706affbb3303b243"
STRICT_RECIPE_SHA256 = "a88bba09fecc7bdd55ac4c531699b7f50521203f5d1e6c2252c9e8b31338aca7"


@pytest.fixture
def store(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as value:
        yield value


def strict() -> AcceptancePolicy:
    return AcceptancePolicy(mode="strict", final_validators=[TextValidator()])


def durable(store, **options) -> Swarm:
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **options)


def no_provider_calls(monkeypatch) -> None:
    async def refuse(*args, **kwargs):
        pytest.fail("A provider was called")

    monkeypatch.setattr(OfflineProvider, "complete", refuse)
    monkeypatch.setattr(OfflineProvider, "chat", refuse)


def digest(label: str) -> str:
    return sha256_hex(label)


VALIDATORS = [ValidatorIdentity.for_config("smythe.text-spec", "1", {}),
              ValidatorIdentity.for_config("contains", "2", {"needle": "x"})]


def node(node_id: str, *, depends_on=(), kind="model", sources=(), context=True, tool=None,
         component=None, agent=True, coverage="declared", **extra) -> dict:
    inputs = {"sources": [dict(source) for source in sources]}
    if context:
        inputs["context"] = {"snapshot_key": f"context/{node_id}/0", "assembly_version": 1}
    if tool is not None:
        inputs["tool"] = dict(tool)
    if component is not None:
        inputs["component"] = dict(component)
    if agent:
        inputs["agent_sha256"] = digest("agent " + node_id)
    return {"id": node_id, "generation": 0, "kind": kind, "depends_on": list(depends_on),
            "result_sha256": digest("result " + node_id), "inputs": inputs,
            "coverage": coverage, **extra}


SOURCE = {"source_id": "handbook", "version": "v3", "sha256": digest("handbook v3")}
TOOL = {"name": "catalog", "version": "1", "decoder_version": "text-utf8-v1"}


def record_data(**changes) -> dict:
    data = {
        "version": 1,
        "run": {"store_id": "store-1", "run_id": "run-1", "recipe_sha256": digest("recipe"),
                "task_sha256": digest("task")},
        "access_scope": "local",
        "artifact": {
            "relative_path": "run-1/document.md", "sha256": digest("document"), "size_bytes": 120,
            "media_type": "text/markdown; charset=utf-8",
            "renderer": {"name": "markdown_renderer", "version": "1",
                         "spec_sha256": digest("renderer")},
        },
        "acceptance": {
            "operation_key": "acceptance/" + digest("inputs"), "record_sha256": digest("record"),
            "decision": "accepted", "policy_sha256": digest("policy"),
            "validator_set_sha256": validator_set_sha256(VALIDATORS),
            "validators": [identity.to_dict() for identity in sorted(
                VALIDATORS, key=lambda item: (item.name, item.version, item.config_sha256))],
        },
        "nodes": [
            node("read", kind="read", context=False, tool=TOOL, sources=[SOURCE], agent=False),
            node("extract", depends_on=["read"], sources=[SOURCE]),
            node("write", depends_on=["extract"],
                 component={"name": "brief-writer", "version": "2"}),
        ],
        "coverage": "complete",
    }
    data.update(changes)
    return data


def reordered(value):
    """*value* with every object's keys in reverse order."""
    if isinstance(value, dict):
        return {key: reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reordered(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# LineagePolicy
# ---------------------------------------------------------------------------


def test_policy_defaults_round_trip_with_a_canonical_sha():
    policy = LineagePolicy()
    assert policy.to_dict() == {"version": 1, "access_scope": "local", "reuse": True,
                                "undeclared_coverage": "review"}
    assert LineagePolicy.from_dict(policy.to_dict()) == policy
    assert policy.sha256 == sha256_hex(policy.to_dict())
    custom = LineagePolicy(access_scope="tenant-a", reuse=False, undeclared_coverage="regenerate")
    again = LineagePolicy.from_dict(reordered(custom.to_dict()))
    assert again == custom and again.sha256 == custom.sha256 != policy.sha256


def test_policy_is_frozen_and_keyword_only():
    policy = LineagePolicy()
    with pytest.raises(FrozenInstanceError):
        policy.reuse = False
    with pytest.raises(TypeError):
        LineagePolicy("tenant-a")


class Text(str):
    pass


@pytest.mark.parametrize("scope", ["local", "tenant-a", "team.alpha:prod", "A" * 64, "x", "7_b"])
def test_scope_labels_accept_short_declared_names(scope):
    assert LineagePolicy(access_scope=scope).access_scope == scope


@pytest.mark.parametrize("scope", [
    "", " ", "-lead", ".hidden", "has space", "a" * 65, "tab\there", "slash/scope", "back\\slash",
    "caf\u00e9", "line\n", 7, None, True, Text("local"),
])
def test_scope_labels_are_validated(scope):
    with pytest.raises(WorkflowBindingError, match="access_scope"):
        LineagePolicy(access_scope=scope)


@pytest.mark.parametrize("field,value", [
    ("reuse", 1), ("reuse", "yes"), ("reuse", None), ("undeclared_coverage", "skip"),
    ("undeclared_coverage", ""), ("undeclared_coverage", None), ("undeclared_coverage", 1),
    ("version", 2), ("version", True), ("version", "1"),
])
def test_policy_settings_are_validated(field, value):
    with pytest.raises(WorkflowBindingError, match=field.split("_")[0]):
        LineagePolicy(**{field: value})


def test_policy_rejects_subclasses_and_impostors(store):
    class Derived(LineagePolicy):
        pass

    with pytest.raises(WorkflowBindingError, match="built-in"):
        Derived()
    for impostor in (LineagePolicy().to_dict(), "local", True):
        with pytest.raises(WorkflowBindingError, match="LineagePolicy"):
            durable(store, acceptance=strict(), lineage=impostor)


@pytest.mark.parametrize("change", ["missing", "extra", "version", "bool-version", "list"])
def test_policy_description_has_an_exact_versioned_schema(change):
    value = LineagePolicy().to_dict()
    if change == "missing":
        del value["reuse"]
    elif change == "extra":
        value["extra"] = 1
    elif change == "version":
        value["version"] = 2
    elif change == "bool-version":
        value["version"] = True
    else:
        value = list(value.items())
    with pytest.raises(WorkflowBindingError):
        LineagePolicy.from_dict(value)


# ---------------------------------------------------------------------------
# ArtifactLineage
# ---------------------------------------------------------------------------


def test_record_round_trips_with_a_canonical_sha_stable_across_key_order():
    data = record_data()
    record = ArtifactLineage.from_dict(data)
    assert record.to_dict() == data
    assert record.sha256 == sha256_hex(data)
    shuffled = ArtifactLineage.from_dict(reordered(data))
    assert shuffled == record and shuffled.sha256 == record.sha256
    assert canonical_bytes(shuffled.to_dict()) == canonical_bytes(data)
    assert ArtifactLineage.from_dict(json.loads(json.dumps(data))).sha256 == record.sha256
    assert ArtifactLineage(**{key: value for key, value in data.items()}) == record
    assert record.coverage == "complete" and record.access_scope == "local"
    assert [item["id"] for item in record.nodes] == ["read", "extract", "write"]


def test_validator_set_digest_ignores_order_and_repeats():
    expected = validator_set_sha256(VALIDATORS)
    assert validator_set_sha256(reversed(VALIDATORS)) == expected
    assert validator_set_sha256([*VALIDATORS, VALIDATORS[0]]) == expected
    assert validator_set_sha256(VALIDATORS[:1]) != expected
    # The record lists the set once, in its canonical order, whatever order it was given in.
    data = record_data()
    data["acceptance"]["validators"].reverse()
    record = ArtifactLineage.from_dict(data)
    assert record.to_dict() == record_data()


@pytest.mark.parametrize("version", [2, 0, True, "1", None])
def test_record_reader_rejects_unknown_versions(version):
    with pytest.raises(ValueError, match="version"):
        ArtifactLineage.from_dict(record_data(version=version))
    data = record_data()
    del data["version"]
    with pytest.raises(ValueError):
        ArtifactLineage.from_dict(data)


def successor_data() -> dict:
    data = record_data(predecessor={"store_id": "store-1", "run_id": "run-0",
                                    "lineage_sha256": digest("lineage run-0")},
                       repair={"plan_sha256": digest("plan"), "reason": "source handbook changed"})
    data["nodes"][1]["reused_from"] = {"run_id": "run-0", "generation": 0}
    data["nodes"][2]["coverage"] = "undeclared"
    del data["nodes"][2]["inputs"]["component"]
    data["coverage"] = "partial"
    return data


# Each place a key can go, as a path from the record's top level.
PLACES = {
    "record": (), "run": ("run",), "artifact": ("artifact",), "renderer": ("artifact", "renderer"),
    "acceptance": ("acceptance",), "node": ("nodes", 0), "inputs": ("nodes", 1, "inputs"),
    "source": ("nodes", 1, "inputs", "sources", 0), "context": ("nodes", 1, "inputs", "context"),
    "tool": ("nodes", 0, "inputs", "tool"), "component": ("nodes", 2, "inputs", "component"),
    "reused-from": ("nodes", 1, "reused_from"), "predecessor": ("predecessor",),
    "repair": ("repair",),
}


def at(data, path):
    for step in path:
        data = data[step]
    return data


@pytest.mark.parametrize("place", sorted(PLACES))
def test_record_reader_rejects_unknown_keys_everywhere(place):
    data = successor_data()
    if place == "component":
        data = record_data()
    at(data, PLACES[place])["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        ArtifactLineage.from_dict(data)


@pytest.mark.parametrize("path,key", [
    (("run",), "task_sha256"), (("artifact",), "renderer"), (("acceptance",), "validators"),
    (("nodes", 0), "coverage"), (("nodes", 0, "inputs"), "sources"),
    (("nodes", 1, "inputs", "context"), "assembly_version"),
    (("nodes", 0, "inputs", "tool"), "decoder_version"),
    ((), "coverage"), ((), "nodes"),
])
def test_record_reader_rejects_missing_keys(path, key):
    data = record_data()
    del at(data, path)[key]
    with pytest.raises(ValueError, match="missing keys"):
        ArtifactLineage.from_dict(data)


def test_optional_blocks_are_omitted_when_absent_and_never_null():
    record = ArtifactLineage.from_dict(record_data())
    assert "predecessor" not in record.to_dict() and "repair" not in record.to_dict()
    assert record.predecessor is None and record.repair is None
    assert set(record.to_dict()["nodes"][0]["inputs"]) == {"sources", "tool"}
    successor = ArtifactLineage.from_dict(successor_data())
    assert successor.to_dict() == successor_data() and successor.coverage == "partial"
    assert successor.predecessor["run_id"] == "run-0"
    for change in ({"predecessor": None}, {"repair": None}):
        with pytest.raises(ValueError):
            ArtifactLineage.from_dict(record_data(**change))
    data = record_data()
    data["nodes"][1]["inputs"]["context"] = None
    with pytest.raises(ValueError):
        ArtifactLineage.from_dict(data)


def test_a_run_without_a_published_document_has_no_artifact_block():
    record = ArtifactLineage.from_dict(record_data(artifact=None))
    assert record.artifact is None and record.to_dict()["artifact"] is None


def set_at(path, value):
    def change(data):
        at(data, path[:-1])[path[-1]] = value
    return change


@pytest.mark.parametrize("change", [
    pytest.param(set_at(("run", "recipe_sha256"), "A" * 64), id="uppercase-digest"),
    pytest.param(set_at(("run", "run_id"), ""), id="empty-run-id"),
    pytest.param(set_at(("access_scope",), "two words"), id="scope"),
    pytest.param(set_at(("artifact", "relative_path"), "/abs/document.md"), id="absolute-path"),
    pytest.param(set_at(("artifact", "relative_path"), "C:/runs/document.md"), id="drive-path"),
    pytest.param(set_at(("artifact", "relative_path"), "run-1\\document.md"), id="backslash-path"),
    pytest.param(set_at(("artifact", "relative_path"), "../document.md"), id="parent-path"),
    pytest.param(set_at(("artifact", "relative_path"), "run-1//document.md"), id="empty-segment"),
    pytest.param(set_at(("artifact", "size_bytes"), -1), id="negative-size"),
    pytest.param(set_at(("artifact", "size_bytes"), True), id="bool-size"),
    pytest.param(set_at(("artifact", "renderer", "version"), 1), id="renderer-version-number"),
    pytest.param(set_at(("acceptance", "decision"), "not_evaluated"), id="undecided"),
    pytest.param(set_at(("acceptance", "decision"), ["accepted"]), id="decision-list"),
    pytest.param(set_at(("acceptance", "validator_set_sha256"), digest("other")),
                 id="validator-set"),
    pytest.param(set_at(("acceptance", "validators"), [{"name": "x"}]), id="validator-identity"),
    pytest.param(set_at(("nodes",), []), id="no-nodes"),
    pytest.param(set_at(("nodes", 1, "id"), "read"), id="duplicate-node"),
    pytest.param(set_at(("nodes", 1, "depends_on"), ["missing"]), id="unknown-dependency"),
    pytest.param(set_at(("nodes", 1, "depends_on"), ["extract"]), id="self-dependency"),
    pytest.param(set_at(("nodes", 2, "depends_on"), ["extract", "extract"]),
                 id="repeated-dependency"),
    pytest.param(set_at(("nodes", 1, "generation"), -1), id="negative-generation"),
    pytest.param(set_at(("nodes", 1, "kind"), "Model"), id="kind"),
    pytest.param(set_at(("nodes", 1, "coverage"), "partial"), id="node-coverage"),
    pytest.param(set_at(("nodes", 1, "result_sha256"), "short"), id="result-digest"),
    pytest.param(set_at(("nodes", 1, "inputs", "sources"), [SOURCE, SOURCE]), id="repeated-source"),
    pytest.param(set_at(("nodes", 1, "inputs", "sources", 0, "version"), ""), id="empty-version"),
    pytest.param(set_at(("nodes", 1, "inputs", "context", "assembly_version"), 0), id="assembly-0"),
    pytest.param(set_at(("nodes", 1, "inputs", "agent_sha256"), None), id="null-agent"),
    pytest.param(set_at(("coverage",), "partial"), id="partial-but-all-declared"),
    pytest.param(set_at(("nodes", 1, "coverage"), "undeclared"), id="complete-with-undeclared"),
    pytest.param(set_at(("nodes", 1, "reused_from"), {"run_id": "run-0", "generation": 0}),
                 id="reused-without-predecessor"),
])
def test_record_values_are_validated(change):
    data = record_data()
    change(data)
    with pytest.raises(ValueError):
        ArtifactLineage.from_dict(data)


def test_a_declared_node_names_what_it_was_given():
    data = record_data()
    del data["nodes"][1]["inputs"]["context"]
    with pytest.raises(ValueError, match="declared"):
        ArtifactLineage.from_dict(data)
    data["nodes"][1]["coverage"] = "undeclared"
    data["coverage"] = "partial"
    assert ArtifactLineage.from_dict(data).coverage == "partial"


@pytest.mark.parametrize("change", [
    pytest.param(set_at(("nodes", 1, "reused_from", "run_id"), "run-9"), id="other-run"),
    pytest.param(set_at(("nodes", 1, "result_sha256"), None), id="reused-without-result"),
    pytest.param(set_at(("predecessor", "store_id"), "store-2"), id="other-store"),
    pytest.param(set_at(("predecessor", "run_id"), "run-1"), id="own-run"),
    pytest.param(set_at(("repair", "reason"), ""), id="empty-reason"),
    pytest.param(set_at(("repair", "reason"), "x" * 257), id="long-reason"),
])
def test_reuse_names_the_same_store_predecessor(change):
    data = successor_data()
    change(data)
    with pytest.raises(ValueError):
        ArtifactLineage.from_dict(data)


def test_record_is_immutable_and_its_dict_is_detached():
    record = ArtifactLineage.from_dict(record_data())
    with pytest.raises(FrozenInstanceError):
        record.coverage = "partial"
    with pytest.raises(TypeError):
        record.run["run_id"] = "other"
    with pytest.raises((TypeError, AttributeError)):
        record.nodes[0]["inputs"]["sources"].append(SOURCE)
    data = record.to_dict()
    data["nodes"][0]["inputs"]["sources"].clear()
    data["run"]["run_id"] = "other"
    assert record.to_dict() == record_data()
    changed = replace(record, access_scope="tenant-b")
    assert changed.access_scope == "tenant-b" and changed.nodes == record.nodes
    assert changed.sha256 != record.sha256


def test_a_256_node_record_stays_under_a_fixed_byte_cap_and_holds_no_source_text():
    sources = [ContextSource(f"source-{index}", f"SECRET passage {index} " * 200, version="v12")
               for index in range(4)]
    refs = [{"source_id": item.source_id, "version": item.version, "sha256": item.sha256}
            for item in sources]
    nodes = []
    for index in range(256):
        node_id = f"node-{index:03d}"
        reads = index % 2 == 0
        nodes.append(node(
            node_id, depends_on=[f"node-{index - 1:03d}"] if index else [],
            kind="read" if reads else "model", sources=refs[index % 4:index % 4 + 2],
            tool=TOOL if reads else None,
            component={"name": "extractor", "version": "3"} if index % 5 == 0 else None,
        ))
    record = ArtifactLineage.from_dict(record_data(nodes=nodes))
    encoded = canonical_bytes(record.to_dict())
    # Every node's entry is ids, versions and digests: well under 1 KiB each.
    assert len(encoded) <= 256 * 1024
    assert b"SECRET" not in encoded and b"passage" not in encoded
    # A source given with its text is not a source reference.
    data = record_data()
    data["nodes"][1]["inputs"]["sources"] = [sources[0].to_dict()]
    with pytest.raises(ValueError, match="unknown keys"):
        ArtifactLineage.from_dict(data)


@pytest.mark.parametrize("kind", [LineagePolicy, ArtifactLineage], ids=lambda kind: kind.__name__)
def test_public_types_resolve_their_type_hints(kind):
    typing.get_type_hints(kind)
    for member in vars(kind).values():
        if isinstance(member, property):
            member = member.fget
        member = getattr(member, "__func__", member)
        if inspect.isfunction(member):
            typing.get_type_hints(member)


# ---------------------------------------------------------------------------
# Durable runs opt in
# ---------------------------------------------------------------------------


def test_swarm_lineage_needs_a_durable_store_before_any_call(tmp_path, monkeypatch):
    no_provider_calls(monkeypatch)
    with pytest.raises(WorkflowBindingError, match="run_store"):
        Swarm(model="offline", provider=OfflineProvider(), acceptance=strict(),
              lineage=LineagePolicy())
    path = tmp_path / "graph.yaml"
    path.write_text("nodes:\n  - id: first\n    label: First\n", encoding="utf-8")
    with pytest.raises(WorkflowBindingError, match="run_store"):
        Swarm.from_yaml(str(path), acceptance=strict(), lineage=LineagePolicy())


@pytest.mark.parametrize("acceptance", [None, AcceptancePolicy()], ids=["none", "advisory"])
def test_lineage_needs_a_strict_acceptance_policy(store, monkeypatch, acceptance):
    no_provider_calls(monkeypatch)
    with pytest.raises(WorkflowBindingError, match="strict acceptance"):
        durable(store, acceptance=acceptance, lineage=LineagePolicy())
    # The durable runtime refuses it too, whatever builds it.
    swarm = durable(store, acceptance=strict(), lineage=LineagePolicy())
    with pytest.raises(WorkflowBindingError, match="strict acceptance"):
        swarm._workflow_runtime(acceptance=AcceptancePolicy())
    assert store.list_runs() == []


def test_recipes_without_lineage_keep_their_bytes(store):
    plain = durable(store).plan(Task("Report")).run_ref["run_id"]
    checked = durable(store, acceptance=strict()).plan(Task("Report")).run_ref["run_id"]
    for run_id, expected in ((plain, DEFAULT_RECIPE_SHA256), (checked, STRICT_RECIPE_SHA256)):
        run = store.load_run(run_id)
        assert run["config_sha256"] == expected and "lineage" not in run["config"]
        assert (run["config"]["type"], run["config"]["version"]) == ("text_workflow", 1)
    assert store.load_run(plain)["config"]["verifier"] == "token-v1"


def test_recipe_records_the_configured_policy_as_identity(store):
    policy = LineagePolicy(access_scope="tenant-a")
    swarm = durable(store, acceptance=strict(), lineage=policy)
    run_id = swarm.plan(Task("Report")).run_ref["run_id"]
    config = store.load_run(run_id)["config"]
    assert config["lineage"] == policy.to_dict()
    without = {key: value for key, value in config.items() if key != "lineage"}
    assert _canonical(without) == _canonical(store.load_run(
        durable(store, acceptance=strict()).plan(Task("Report")).run_ref["run_id"])["config"])
    # The runtime holds its own copy, which a change to the caller's object cannot reach.
    runtime = swarm._workflow_runtime()
    object.__setattr__(policy, "access_scope", "tenant-b")
    assert runtime.recipe["lineage"]["access_scope"] == "tenant-a"
    assert swarm._workflow_runtime().lineage.access_scope == "tenant-a"


def test_a_lineage_run_executes_and_resumes_under_its_policy(store, monkeypatch):
    swarm = durable(store, acceptance=strict(), lineage=LineagePolicy())
    result = swarm.execute(Task("Report"))
    assert result.accepted and result.output == "offline: Report"
    calls = store.inspect_run(result.execution_id)["calls"]
    no_provider_calls(monkeypatch)
    again = durable(store, acceptance=strict(), lineage=LineagePolicy()).resume(result.execution_id)
    assert again.output == result.output and again.accepted
    assert store.inspect_run(result.execution_id)["calls"] == calls


@pytest.mark.parametrize("saved,changed", [
    pytest.param(LineagePolicy(), None, id="removed"),
    pytest.param(None, LineagePolicy(), id="added"),
    pytest.param(LineagePolicy(), LineagePolicy(access_scope="tenant-b"), id="scope"),
    pytest.param(LineagePolicy(), LineagePolicy(reuse=False), id="reuse"),
    pytest.param(LineagePolicy(), LineagePolicy(undeclared_coverage="regenerate"), id="coverage"),
])
@pytest.mark.parametrize("entry", ["resume", "execute-planned"])
def test_an_in_flight_run_keeps_its_lineage_policy(store, monkeypatch, saved, changed, entry):
    planned = durable(store, acceptance=strict(), lineage=saved).plan(Task("Report"))
    run_id = planned.run_ref["run_id"]
    checkpoint = deepcopy(store.get_checkpoint(run_id))
    no_provider_calls(monkeypatch)
    other = durable(store, acceptance=strict(), lineage=changed)
    with pytest.raises(WorkflowConflictError, match="lineage"):
        if entry == "resume":
            other.resume(run_id)
        else:
            other.execute(planned)
    assert store.get_checkpoint(run_id) == checkpoint
    assert store.inspect_run(run_id)["calls"] == []


def test_a_completed_lineage_run_is_re_evaluated_only_under_its_policy(store, monkeypatch):
    run_id = durable(store, acceptance=strict(), lineage=LineagePolicy()).execute(
        Task("Report")).execution_id
    no_provider_calls(monkeypatch)
    with pytest.raises(WorkflowConflictError, match="lineage"):
        durable(store, acceptance=strict()).reevaluate(run_id)
    again = durable(store, acceptance=strict(), lineage=LineagePolicy()).reevaluate(run_id)
    assert again.accepted


def test_lineage_comes_only_from_configuration(store):
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="write", label="Write", metadata={"lineage": LineagePolicy(reuse=False).to_dict()}),
    ])
    graph.task = Task("Report", context={"lineage": {"access_scope": "tenant-b"}})
    result = durable(store, acceptance=strict()).execute(graph)
    assert "lineage" not in store.load_run(result.execution_id)["config"]
    policy = LineagePolicy(access_scope="tenant-a")
    run_id = durable(store, acceptance=strict(), lineage=policy).execute(graph).execution_id
    assert store.load_run(run_id)["config"]["lineage"] == policy.to_dict()


def test_a_new_graph_cannot_carry_reused_results(store, monkeypatch):
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="write", label="Write", metadata={LINEAGE_REUSE_KEY: {"run_id": "run-0"}}),
    ])
    no_provider_calls(monkeypatch)
    with pytest.raises(WorkflowBindingError, match="inherit execution"):
        durable(store, acceptance=strict(), lineage=LineagePolicy()).execute(graph)
    assert [store.inspect_run(run["run_id"])["call_count"] for run in store.list_runs()] == [0]


def test_yaml_graphs_take_lineage_from_the_argument(store, tmp_path):
    path = tmp_path / "graph.yaml"
    path.write_text("nodes:\n  - id: write\n    label: Write\n", encoding="utf-8")
    policy = LineagePolicy(access_scope="tenant-a")
    swarm = Swarm.from_yaml(str(path), model="offline", provider=OfflineProvider(), run_store=store,
                            acceptance=strict(), lineage=policy)
    result = asyncio.run(swarm.execute_async())
    assert result.accepted
    assert store.load_run(result.execution_id)["config"]["lineage"] == policy.to_dict()
