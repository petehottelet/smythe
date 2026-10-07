"""Reuse decisions: what a recorded artifact's lineage calls for under the current state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import re
import typing

import pytest

import smythe
import smythe.reuse
from fakes import release_price_version
from fixture_service import FixtureService
from smythe import (
    ArtifactLineage, CurrentState, ItemDecision, LineagePolicy, SourceStatus, ValidatorIdentity,
    WorkflowBindingError,
)
from smythe._canonical import sha256_hex
from smythe.context import CONTEXT_ASSEMBLY_VERSION
from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, Node
from smythe.lineage import validator_set_sha256
from smythe.reuse import ACTIONS, REASONS, NodeDecision, evaluate
from smythe.verifier import TOKEN_VERIFIER_IDENTITY
from smythe.workflow_artifact import PublishedArtifact, artifact_integrity


def digest(label: str) -> str:
    return sha256_hex(label)


VALIDATORS = (ValidatorIdentity.for_config("smythe.text-spec", "1", {}),
              ValidatorIdentity.for_config("contains", "2", {"needle": "x"}))
POLICY = digest("policy")
TOOL = {"name": "catalog", "version": "1", "decoder_version": "text-utf8-v1"}
WRITER = {"name": "brief-writer", "version": "2"}
SOURCES = {"a": {"source_id": "a", "version": "v1", "sha256": digest("a v1")},
           "b": {"source_id": "b", "version": "v1", "sha256": digest("b v1")}}
NODES = ["read_a", "extract_a", "read_b", "extract_b", "write"]
AGENTS = frozenset(digest("agent " + node_id) for node_id in ("extract_a", "extract_b", "write"))


def node(node_id, *, depends_on=(), kind="model", sources=(), context=True, tool=None,
         component=None, agent=True, coverage="declared", result=True) -> dict:
    inputs = {"sources": [dict(SOURCES[name]) for name in sources]}
    if context:
        inputs["context"] = {"snapshot_key": f"context/{node_id}/0", "assembly_version": 1}
    if tool is not None:
        inputs["tool"] = dict(tool)
    if component is not None:
        inputs["component"] = dict(component)
    if agent:
        inputs["agent_sha256"] = digest("agent " + node_id)
    return {"id": node_id, "generation": 0, "kind": kind, "depends_on": list(depends_on),
            "result_sha256": digest("result " + node_id) if result else None,
            "inputs": inputs, "coverage": coverage}


def branch(name: str) -> list[dict]:
    """A source's read step and the extraction that depends on it."""
    return [node(f"read_{name}", kind="read", context=False, tool=TOOL, sources=[name],
                 agent=False),
            node(f"extract_{name}", depends_on=[f"read_{name}"], sources=[name])]


def record_data(*, gate=False, **changes) -> dict:
    nodes = [*branch("a"), *branch("b"),
             node("write", depends_on=["extract_a", "extract_b"], component=WRITER)]
    if gate:
        # A gate that judges extract_b, and so depends on it.
        nodes.append(node("check", kind="gate", depends_on=["extract_b"], context=False,
                          component={"name": "fact-check", "version": "1"}, agent=False))
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
            "decision": "accepted", "policy_sha256": POLICY,
            "validator_set_sha256": validator_set_sha256(VALIDATORS),
            "validators": [identity.to_dict() for identity in sorted(
                VALIDATORS, key=lambda item: (item.name, item.version, item.config_sha256))],
        },
        "nodes": nodes,
        "coverage": "complete",
    }
    data.update(changes)
    return data


def lineage(**changes) -> ArtifactLineage:
    return ArtifactLineage.from_dict(record_data(**changes))


class Catalog:
    """Describes sources from a dict, recording each source it is asked about."""

    def __init__(self, sources=None):
        self.sources = {name: SourceStatus(version=ref["version"], sha256=ref["sha256"])
                        for name, ref in SOURCES.items()} if sources is None else sources
        self.calls: list[str] = []

    def describe(self, source_id):
        self.calls.append(source_id)
        return self.sources.get(source_id)


def state(catalog=None, **changes) -> CurrentState:
    options = {
        "catalog": Catalog() if catalog is None else catalog, "validators": VALIDATORS,
        "policy_sha256": POLICY,
        "tools": {"catalog": {"version": "1", "decoder_version": "text-utf8-v1"}},
        "components": {"brief-writer": "2", "fact-check": "1"}, "agents": AGENTS,
        "integrity": {"run-1": "verified"},
    }
    options.update(changes)
    return CurrentState(**options)


def actions(decision: ItemDecision) -> dict[str, str]:
    return {item.id: item.action for item in decision.nodes}


def reasons(decision: ItemDecision) -> dict[str, tuple[str, ...]]:
    return {item.id: item.reasons for item in decision.nodes}


def changed(source_id: str, *, version="v2", same_bytes=False) -> Catalog:
    catalog = Catalog()
    ref = SOURCES[source_id]
    catalog.sources[source_id] = SourceStatus(
        version=version, sha256=ref["sha256"] if same_bytes else digest(f"{source_id} {version}"))
    return catalog


# ---------------------------------------------------------------------------
# Rules, one at a time
# ---------------------------------------------------------------------------


def test_unchanged_inputs_are_reused_with_full_authority():
    decision = evaluate(lineage(), state())
    assert decision.action == "reuse" and decision.reasons == ()
    assert actions(decision) == dict.fromkeys(NODES, "reuse")
    assert all(item.reasons == () for item in decision.nodes)
    assert decision.to_dict()["authority"] == {
        "access_scope": "local", "recipe_sha256": digest("recipe"), "policy_sha256": POLICY,
        "validator_set_sha256": validator_set_sha256(VALIDATORS),
        "sources": [SOURCES["a"], SOURCES["b"]],
    }


def test_one_changed_source_regenerates_its_branch_and_reuses_the_sibling():
    decision = evaluate(lineage(), state(changed("b")))
    assert decision.action == "regenerate"
    assert actions(decision) == {"read_a": "reuse", "extract_a": "reuse", "read_b": "regenerate",
                                 "extract_b": "regenerate", "write": "regenerate"}
    assert reasons(decision) == {
        "read_a": (), "extract_a": (),
        "read_b": ("source_changed", "source_version_changed"),
        "extract_b": ("dependency_changed", "source_changed", "source_version_changed"),
        "write": ("dependency_changed",),
    }
    assert decision.reasons == ("dependency_changed", "source_changed", "source_version_changed")
    # Reuse is authorized only under the versions the reused nodes read.
    assert decision.to_dict()["authority"]["sources"] == [SOURCES["a"]]


def test_same_bytes_under_a_new_version_regenerate_with_the_version_reason():
    decision = evaluate(lineage(), state(changed("a", same_bytes=True)))
    assert decision.action == "regenerate"
    assert reasons(decision)["read_a"] == ("source_version_changed",)
    assert actions(decision)["read_b"] == "reuse"
    # A version where none was recorded is a new version too.
    data = record_data()
    data["nodes"][0]["inputs"]["sources"][0]["version"] = None
    data["nodes"][1]["inputs"]["sources"][0]["version"] = None
    assert reasons(evaluate(ArtifactLineage.from_dict(data), state()))["read_a"] == (
        "source_version_changed",)


def test_changed_bytes_under_the_same_version_regenerate():
    catalog = Catalog()
    catalog.sources["a"] = SourceStatus(version="v1", sha256=digest("a edited in place"))
    decision = evaluate(lineage(), state(catalog))
    assert reasons(decision)["read_a"] == ("source_changed",)
    assert decision.action == "regenerate"


def test_a_versioned_service_source_rewritten_with_identical_bytes_regenerates():
    service = FixtureService()
    first = service.add_source("a", "The handbook.")

    class ServiceCatalog:
        def describe(self, source_id):
            record = service.read_source(source_id)
            return SourceStatus(version=str(record.version), sha256=record.sha256)

    data = record_data()
    for item in data["nodes"][:2]:
        item["inputs"]["sources"] = [{"source_id": "a", "version": "1", "sha256": first.sha256}]
    data["nodes"] = data["nodes"][:2]
    recorded = ArtifactLineage.from_dict(data)
    current = state(ServiceCatalog(), components={})
    assert evaluate(recorded, current).action == "reuse"
    again = service.change_source("a", "The handbook.")
    assert again.sha256 == first.sha256 and again.version == 2
    decision = evaluate(recorded, current)
    assert decision.action == "regenerate"
    assert reasons(decision) == {"read_a": ("source_version_changed",),
                                 "extract_a": ("dependency_changed", "source_version_changed")}


@pytest.mark.parametrize("status,code", [
    (SourceStatus(available=False), "source_unavailable"),
    (SourceStatus(version="v1", sha256=SOURCES["b"]["sha256"], available=False),
     "source_unavailable"),
    (SourceStatus(available=False, tombstone_reason="withdrawn by its owner"), "source_tombstoned"),
    (None, "source_unavailable"),
], ids=["unavailable", "unavailable-same-version", "tombstoned", "unknown"])
def test_unavailable_or_tombstoned_sources_need_review(status, code):
    catalog = Catalog()
    catalog.sources["b"] = status
    decision = evaluate(lineage(), state(catalog))
    assert decision.action == "review"
    assert reasons(decision)["read_b"] == (code,)
    assert reasons(decision)["write"] == ("dependency_review",)
    assert actions(decision) == {"read_a": "reuse", "extract_a": "reuse", "read_b": "review",
                                 "extract_b": "review", "write": "review"}


IDENTITY_CHANGES = {
    "tool-version": ({"tools": {"catalog": {"version": "2", "decoder_version": "text-utf8-v1"}}},
                     {"read_a": {"tool_changed"}, "read_b": {"tool_changed"}}),
    "decoder": ({"tools": {"catalog": {"version": "1", "decoder_version": "text-utf8-v2"}}},
                {"read_a": {"decoder_changed"}, "read_b": {"decoder_changed"}}),
    "tool-and-decoder": ({"tools": {"catalog": {"version": "2", "decoder_version": "v2"}}},
                         {"read_a": {"tool_changed", "decoder_changed"},
                          "read_b": {"tool_changed", "decoder_changed"}}),
    "tool-removed": ({"tools": {}}, {"read_a": {"tool_changed"}, "read_b": {"tool_changed"}}),
    "component": ({"components": {"brief-writer": "3"}}, {"write": {"component_changed"}}),
    "context-assembly": ({"context_assembly_version": 2},
                         dict.fromkeys(["extract_a", "extract_b", "write"],
                                       {"context_assembly_changed"})),
    "no-context-assembly": ({"context_assembly_version": None},
                            dict.fromkeys(["extract_a", "extract_b", "write"],
                                          {"context_assembly_changed"})),
    "agent": ({"agents": (AGENTS - {digest("agent extract_a")}) | {digest("another agent")}},
              {"extract_a": {"agent_changed"}}),
    "agent-removed": ({"agents": AGENTS - {digest("agent extract_b")}},
                      {"extract_b": {"agent_changed"}}),
    "task": ({"tasks": {digest("task"): digest("revised task")}},
             dict.fromkeys(NODES, {"task_changed"})),
}


@pytest.mark.parametrize("name", sorted(IDENTITY_CHANGES))
def test_changed_tools_decoders_components_context_agents_or_task_regenerate(name):
    changes, expected = IDENTITY_CHANGES[name]
    decision = evaluate(lineage(), state(**changes))
    assert decision.action == "regenerate"
    for node_id, codes in expected.items():
        assert codes <= set(reasons(decision)[node_id])
        assert actions(decision)[node_id] == "regenerate"
    # Descendants of a changed node regenerate; nothing else does.
    regenerated = set(expected)
    regenerated |= {"extract_a"} if "read_a" in expected else set()
    regenerated |= {"extract_b"} if "read_b" in expected else set()
    regenerated |= {"write"} if regenerated else set()
    assert {key for key, value in actions(decision).items() if value == "regenerate"} == regenerated
    for node_id in regenerated - set(expected):
        assert reasons(decision)[node_id] == ("dependency_changed",)


def test_an_unchanged_task_override_changes_nothing():
    decision = evaluate(lineage(), state(tasks={digest("task"): digest("task"),
                                               digest("other"): digest("revised")}))
    assert decision.action == "reuse"


@pytest.mark.parametrize("changes,codes", [
    ({"validators": (VALIDATORS[0], ValidatorIdentity.for_config("contains", "3", {"needle": "x"}))},
     ("validator_set_changed",)),
    ({"validators": (VALIDATORS[0], ValidatorIdentity.for_config("contains", "2", {"needle": "y"}))},
     ("validator_set_changed",)),
    ({"policy_sha256": digest("stricter policy")}, ("policy_changed",)),
    ({"validators": VALIDATORS[:1], "policy_sha256": digest("stricter policy")},
     ("policy_changed", "validator_set_changed")),
], ids=["validator-version", "validator-config", "policy", "both"])
def test_a_changed_validator_set_or_policy_needs_recheck(changes, codes):
    current = state(**changes)
    decision = evaluate(lineage(), current)
    assert decision.action == "recheck" and decision.reasons == codes
    # Nothing is computed again: every node keeps its result for the new decision.
    assert actions(decision) == dict.fromkeys(NODES, "recheck")
    authority = decision.to_dict()["authority"]
    assert authority["validator_set_sha256"] == current.validator_set_sha256
    assert authority["policy_sha256"] == current.policy_sha256
    assert authority["sources"] == [SOURCES["a"], SOURCES["b"]]


def test_validator_order_and_repeats_do_not_change_the_set():
    current = state(validators=[VALIDATORS[1], VALIDATORS[0], VALIDATORS[1]])
    assert current.validator_set_sha256 == validator_set_sha256(VALIDATORS)
    assert current.validators == tuple(sorted(VALIDATORS, key=lambda item: item.name))
    assert evaluate(lineage(), current).action == "reuse"


@pytest.mark.parametrize("status,code", [("missing", "artifact_missing"),
                                         ("changed", "artifact_changed")])
def test_a_missing_or_changed_artifact_is_rendered_again(status, code):
    decision = evaluate(lineage(), state(integrity={"run-1": status}))
    assert decision.action == "rerender" and decision.reasons == (code,)
    assert actions(decision) == dict.fromkeys(NODES, "rerender")
    assert decision.to_dict()["authority"]["sources"] == [SOURCES["a"], SOURCES["b"]]


@pytest.mark.parametrize("damage", ["truncate", "flip", "delete"])
def test_a_damaged_published_file_is_rendered_again(tmp_path, damage):
    data = b"# Brief\n\nThe accepted text.\n"
    (tmp_path / "document.md").write_bytes(data)
    record = PublishedArtifact(
        sequence=0, directory=str(tmp_path), name="document.md",
        sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), media_type="text/markdown",
        renderer={"name": "markdown_renderer"}, synthesis="synthesis", output_sha256=digest("out"))
    assert artifact_integrity(record)["status"] == "verified"
    if damage == "truncate":
        (tmp_path / "document.md").write_bytes(data[:10])
    elif damage == "flip":
        (tmp_path / "document.md").write_bytes(bytes([data[0] ^ 1]) + data[1:])
    else:
        (tmp_path / "document.md").unlink()
    status = artifact_integrity(record)["status"]
    decision = evaluate(lineage(), state(integrity={"run-1": status}))
    code = "artifact_missing" if damage == "delete" else "artifact_changed"
    assert (decision.action, decision.reasons) == ("rerender", (code,))


@pytest.mark.parametrize("integrity,code", [
    ({"run-1": "unsafe"}, "artifact_unsafe"), ({"run-1": "unreadable"}, "artifact_unreadable"),
    ({"run-1": "not_checked"}, "artifact_unchecked"), ({}, "artifact_unchecked"),
    ({"run-2": "verified"}, "artifact_unchecked"),
], ids=["unsafe", "unreadable", "not-checked", "absent", "another-run"])
def test_an_artifact_that_cannot_be_checked_needs_review(integrity, code):
    decision = evaluate(lineage(), state(integrity=integrity))
    assert decision.action == "review" and decision.reasons == (code,)


def test_a_run_without_a_document_needs_no_integrity_finding():
    assert evaluate(lineage(artifact=None), state(integrity={})).action == "reuse"


@pytest.mark.parametrize("status", ["missing", "changed"])
def test_corruption_with_a_source_change_regenerates_never_rerenders(status):
    decision = evaluate(lineage(), state(changed("a"), integrity={"run-1": status}))
    assert decision.action == "regenerate"
    assert f"artifact_{status}" in decision.reasons
    assert "rerender" not in actions(decision).values()
    assert actions(decision)["read_b"] == "reuse"


@pytest.mark.parametrize("decision_value", ["rejected", "needs_review", "indeterminate"])
def test_a_decision_other_than_accepted_needs_review(decision_value):
    data = record_data()
    data["acceptance"]["decision"] = decision_value
    decision = evaluate(ArtifactLineage.from_dict(data), state(
        validators=VALIDATORS[:1], integrity={"run-1": "missing"}))
    assert decision.action == "review"
    assert decision.reasons == ("artifact_missing", "decision_not_accepted",
                                "validator_set_changed")


@pytest.mark.parametrize("coverage,expected", [("review", "review"),
                                               ("regenerate", "regenerate")])
def test_a_missing_record_follows_the_undeclared_coverage_policy(coverage, expected):
    for reuse in (True, False):
        current = state(lineage=LineagePolicy(undeclared_coverage=coverage, reuse=reuse))
        decision = evaluate(None, current)
        assert decision.action == expected and decision.reasons == ("lineage_missing",)
        assert decision.nodes == ()
        assert decision.to_dict()["authority"] == {
            "access_scope": "local", "recipe_sha256": None, "policy_sha256": POLICY,
            "validator_set_sha256": validator_set_sha256(VALIDATORS), "sources": []}
        assert current.catalog.calls == []


def test_a_missing_record_needs_review_by_default():
    assert evaluate(None, state()).action == "review"


def partial() -> ArtifactLineage:
    data = record_data(coverage="partial")
    extract_b = data["nodes"][3]
    extract_b["coverage"] = "undeclared"
    del extract_b["inputs"]["context"]
    return ArtifactLineage.from_dict(data)


def test_partial_coverage_needs_review_by_default():
    decision = evaluate(partial(), state())
    assert decision.action == "review"
    assert reasons(decision)["extract_b"] == ("coverage_undeclared",)
    assert reasons(decision)["write"] == ("dependency_review",)
    assert actions(decision) == {"read_a": "reuse", "extract_a": "reuse", "read_b": "reuse",
                                 "extract_b": "review", "write": "review"}
    assert decision.reasons == ("coverage_partial", "coverage_undeclared", "dependency_review")


def test_partial_coverage_regenerates_when_the_policy_says_so():
    decision = evaluate(partial(), state(lineage=LineagePolicy(undeclared_coverage="regenerate")))
    assert decision.action == "regenerate"
    assert actions(decision) == {"read_a": "reuse", "extract_a": "reuse", "read_b": "reuse",
                                 "extract_b": "regenerate", "write": "regenerate"}
    assert decision.reasons == ("coverage_partial", "coverage_undeclared", "dependency_changed")


def test_nodes_that_did_not_complete_regenerate():
    data = record_data()
    data["nodes"][1]["result_sha256"] = None
    decision = evaluate(ArtifactLineage.from_dict(data), state())
    assert decision.action == "regenerate"
    assert reasons(decision)["extract_a"] == ("node_incomplete",)
    assert reasons(decision)["write"] == ("dependency_changed",)
    assert actions(decision)["read_a"] == "reuse"


DIAMOND = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"], "e": [], "f": ["e", "c"],
           "g": ["d", "f"]}


@pytest.mark.parametrize("changed_id", sorted(DIAMOND))
def test_regeneration_reaches_exactly_the_nodes_a_run_resets(changed_id):
    data = record_data(artifact=None, nodes=[
        node(node_id, depends_on=depends_on, result=node_id != changed_id)
        for node_id, depends_on in DIAMOND.items()])
    decision = evaluate(ArtifactLineage.from_dict(data), state(
        agents={digest("agent " + node_id) for node_id in DIAMOND}))
    graph = ExecutionGraph(topology=[], nodes=[
        Node(label=node_id, id=node_id, depends_on=list(depends_on))
        for node_id, depends_on in DIAMOND.items()])
    target = next(item for item in graph.nodes if item.id == changed_id)
    reset = {item.id for item in ExecutorBase._affected_nodes(None, target, graph)}
    assert {key for key, value in actions(decision).items() if value == "regenerate"} == reset
    assert {key for key, value in actions(decision).items() if value == "reuse"} == (
        set(DIAMOND) - reset)


def test_a_gate_pair_in_an_affected_item_is_never_reused():
    decision = evaluate(lineage(gate=True), state(changed("a")))
    assert decision.action == "regenerate"
    assert actions(decision) == {"read_a": "regenerate", "extract_a": "regenerate",
                                 "read_b": "reuse", "extract_b": "regenerate",
                                 "write": "regenerate", "check": "regenerate"}
    assert reasons(decision)["extract_b"] == ("gate_pair",)
    assert reasons(decision)["check"] == ("dependency_changed", "gate_pair")
    assert "gate_pair" in decision.reasons
    assert decision.to_dict()["authority"]["sources"] == [SOURCES["b"]]


def test_a_gate_pair_in_an_item_under_review_is_never_reused():
    catalog = Catalog()
    catalog.sources["a"] = SourceStatus(available=False, tombstone_reason="withdrawn")
    decision = evaluate(lineage(gate=True), state(catalog))
    assert decision.action == "review"
    assert actions(decision)["extract_b"] == actions(decision)["check"] == "regenerate"
    assert actions(decision)["read_b"] == "reuse"


GATE_VALIDATOR = ValidatorIdentity.for_config("contains", "1", {"needle": "needle"})


def judged(verifier="validator-v1", validators=(GATE_VALIDATOR,)) -> ArtifactLineage:
    """A record whose gate names the gate verifier that judged it."""
    data = record_data(gate=True)
    data["nodes"][-1]["inputs"]["verifier"] = {
        "name": verifier, "validators": [item.to_dict() for item in validators]}
    return ArtifactLineage.from_dict(data)


@pytest.mark.parametrize("changes", [
    {"gate_validators": [ValidatorIdentity.for_config("contains", "2", {"needle": "absent"})]},
    {"gate_validators": [ValidatorIdentity.for_config("contains", "1", {"needle": "absent"})]},
    {"gate_verifier": TOKEN_VERIFIER_IDENTITY, "gate_validators": []},
    {"gate_validators": [GATE_VALIDATOR, ValidatorIdentity.for_config("other", "1")]},
], ids=["version", "configuration", "token-verifier", "added"])
def test_a_changed_gate_validator_is_never_reused(changes):
    current = {"gate_verifier": "validator-v1", "gate_validators": [GATE_VALIDATOR]}
    assert evaluate(judged(), state(**current)).action == "reuse"

    decision = evaluate(judged(), state(**{**current, **changes}))

    # The gate and the node it judged regenerate, and so does what depends on them.
    assert decision.action == "regenerate"
    assert actions(decision) == {"read_a": "reuse", "extract_a": "reuse", "read_b": "reuse",
                                 "extract_b": "regenerate", "write": "regenerate",
                                 "check": "regenerate"}
    assert reasons(decision)["extract_b"] == ("gate_validator_changed",)
    assert reasons(decision)["check"] == ("dependency_changed", "gate_validator_changed")
    assert reasons(decision)["write"] == ("dependency_changed",)
    assert decision.reasons == ("dependency_changed", "gate_validator_changed")
    # A gate judged by TokenVerifier is compared the same way.
    token = judged(TOKEN_VERIFIER_IDENTITY, ())
    assert evaluate(token, state()).action == "reuse"
    assert "gate_validator_changed" in evaluate(token, state(**current)).reasons


@pytest.mark.parametrize("coverage", ["review", "regenerate"])
def test_a_gate_recorded_without_its_verifier_is_unknown_under_a_gate_validator(coverage):
    # A record written before records named their gates' verifier.
    record, policy = lineage(gate=True), LineagePolicy(undeclared_coverage=coverage)
    assert "verifier" not in record.nodes[-1]["inputs"]
    assert evaluate(record, state(lineage=policy)).action == "reuse"

    decision = evaluate(record, state(gate_verifier="validator-v1",
                                      gate_validators=[GATE_VALIDATOR], lineage=policy))

    assert decision.action == coverage
    assert reasons(decision)["extract_b"] == ("gate_validator_unknown",)
    assert "gate_validator_unknown" in reasons(decision)["check"]
    assert actions(decision)["check"] == actions(decision)["extract_b"] == coverage
    assert actions(decision)["read_b"] == "reuse"


@pytest.mark.parametrize("changes,expected", [
    ({}, "reuse"), ({"policy_sha256": digest("stricter policy")}, "recheck"),
    ({"integrity": {"run-1": "missing"}}, "rerender"),
])
def test_gates_in_items_that_compute_nothing_keep_the_item_action(changes, expected):
    decision = evaluate(lineage(gate=True), state(**changes))
    assert decision.action == expected
    assert set(actions(decision).values()) == {expected}


REUSE_OFF_CASES = {
    "unchanged": ({}, "regenerate"),
    "source": ({"catalog": changed("b")}, "regenerate"),
    "artifact": ({"integrity": {"run-1": "changed"}}, "regenerate"),
    "validator": ({"validators": VALIDATORS[:1]}, "recheck"),
    "tombstone": ({"catalog": Catalog({**Catalog().sources, "b": SourceStatus(
        available=False, tombstone_reason="withdrawn")})}, "review"),
    "partial": ({}, "review"),
}


@pytest.mark.parametrize("name", sorted(REUSE_OFF_CASES))
def test_with_reuse_off_nothing_is_reused_or_rendered_again(name):
    changes, expected = REUSE_OFF_CASES[name]
    record = partial() if name == "partial" else lineage(gate=True)
    decision = evaluate(record, state(lineage=LineagePolicy(reuse=False), **deepcopy(changes)))
    assert decision.action == expected
    assert {decision.action, *actions(decision).values()}.isdisjoint({"reuse", "rerender"})
    if expected == "recheck":
        # A recheck makes a new decision about the recorded output; it is unchanged.
        assert set(actions(decision).values()) == {"recheck"}
        assert "reuse_disabled" not in decision.reasons
    else:
        assert "reuse_disabled" in decision.reasons
    # Every node that is computed again says why.
    assert all(reasons(decision)[key] for key, value in actions(decision).items()
               if value == "regenerate")
    if expected != "review":
        assert decision.to_dict()["authority"]["sources"] == (
            [] if expected == "regenerate" else [SOURCES["a"], SOURCES["b"]])


def test_reuse_off_regenerates_every_node_of_a_changed_item():
    decision = evaluate(lineage(), state(changed("b"), lineage=LineagePolicy(reuse=False)))
    assert set(actions(decision).values()) == {"regenerate"}
    assert reasons(decision)["read_a"] == ("reuse_disabled",)
    assert reasons(decision)["read_b"] == ("source_changed", "source_version_changed")


# ---------------------------------------------------------------------------
# Order, scope and determinism
# ---------------------------------------------------------------------------


def test_actions_follow_their_order_of_precedence():
    assert ACTIONS == ("reuse", "recheck", "rerender", "regenerate", "review")
    catalog = changed("a")
    catalog.sources["b"] = SourceStatus(available=False)
    everything = state(catalog, validators=VALIDATORS[:1], integrity={"run-1": "changed"})
    assert evaluate(lineage(), everything).action == "review"
    assert evaluate(lineage(), replace(everything, catalog=changed("a"))).action == "regenerate"
    assert evaluate(lineage(), replace(everything, catalog=Catalog())).action == "rerender"
    assert evaluate(lineage(), replace(everything, catalog=Catalog(),
                                       integrity={"run-1": "verified"})).action == "recheck"


@pytest.mark.parametrize("record_scope,state_scope", [("tenant-a", "local"),
                                                      ("local", "tenant-a"),
                                                      ("Tenant-A", "tenant-a")])
def test_another_scope_is_invisible_even_when_every_digest_matches(record_scope, state_scope):
    current = state(lineage=LineagePolicy(access_scope=state_scope))
    assert evaluate(lineage(access_scope=record_scope), current) is None
    # Nothing was compared: the catalog was never asked about a source.
    assert current.catalog.calls == []
    same = state(lineage=LineagePolicy(access_scope=record_scope))
    decision = evaluate(lineage(access_scope=record_scope), same)
    assert decision.action == "reuse"
    assert decision.authority["access_scope"] == record_scope


def test_each_source_is_described_once_in_record_order():
    current = state()
    evaluate(lineage(), current)
    assert current.catalog.calls == ["a", "b"]


def test_decisions_are_deterministic_json_data():
    record = lineage(gate=True)
    first = evaluate(record, state(changed("b"), integrity={"run-1": "missing"})).to_dict()
    reordered = state(changed("b"), integrity={"run-1": "missing"}, agents=list(AGENTS)[::-1],
                      components={"fact-check": "1", "brief-writer": "2"},
                      validators=tuple(reversed(VALIDATORS)))
    second = evaluate(ArtifactLineage.from_dict(record.to_dict()), reordered).to_dict()
    assert first == second
    assert json.loads(json.dumps(first)) == first
    assert first["nodes"][0] == {"id": "read_a", "action": "reuse", "reasons": []}
    assert list(first) == ["action", "reasons", "nodes", "authority"]
    assert sha256_hex(first) == sha256_hex(second)


def test_a_new_price_version_never_invalidates(monkeypatch):
    before = evaluate(lineage(), state()).to_dict()
    release_price_version(monkeypatch, version="test-prices-2026-10-05")
    assert evaluate(lineage(), state()).to_dict() == before


def test_evaluation_leaves_the_record_unchanged():
    record = lineage(gate=True)
    data = record.to_dict()
    evaluate(record, state(changed("a")))
    assert record.to_dict() == data


# ---------------------------------------------------------------------------
# The catalog, the state and the result types
# ---------------------------------------------------------------------------


def test_a_catalog_declared_metered_is_refused():
    class Metered(Catalog):
        metered = True

    with pytest.raises(ValueError, match="metered"):
        state(Metered())
    for value in (None, 1, "no"):
        catalog = Catalog()
        catalog.metered = value
        with pytest.raises(ValueError, match="metered"):
            state(catalog)
    free = Catalog()
    free.metered = False
    assert evaluate(lineage(), state(free)).action == "reuse"


def test_a_catalog_needs_a_synchronous_describe():
    class Async:
        async def describe(self, source_id):
            return None

    for catalog in (object(), Async(), Catalog, {"a": SourceStatus(sha256=digest("a"))}):
        with pytest.raises(TypeError, match="catalog"):
            state(catalog)


@pytest.mark.parametrize("answer", ["dict", "coroutine", "tuple"])
def test_a_catalog_must_answer_with_a_source_status(answer):
    async def later():
        return SourceStatus(version="v1", sha256=SOURCES["a"]["sha256"])

    pending = []

    class Odd(Catalog):
        def describe(self, source_id):
            if answer == "coroutine":
                pending.append(later())
                return pending[-1]
            if answer == "dict":
                return {"version": "v1", "sha256": SOURCES[source_id]["sha256"]}
            return ("v1", SOURCES[source_id]["sha256"])

    with pytest.raises(TypeError, match="SourceStatus"):
        evaluate(lineage(), state(Odd()))
    # A refused coroutine is closed, so it is never reported as unawaited.
    assert all(item.cr_frame is None for item in pending)


@pytest.mark.parametrize("options,error", [
    ({"tombstone_reason": "withdrawn"}, ValueError),
    ({"available": 1, "sha256": digest("a")}, TypeError),
    ({"version": "v1"}, ValueError),
    ({"sha256": "A" * 64}, ValueError),
    ({"sha256": digest("a"), "version": ""}, ValueError),
    ({"sha256": digest("a"), "version": "v\n1"}, ValueError),
    ({"available": False, "tombstone_reason": ""}, ValueError),
], ids=["tombstoned-but-available", "available-not-bool", "available-without-digest",
        "bad-digest", "empty-version", "control-character", "empty-tombstone-reason"])
def test_source_statuses_are_validated(options, error):
    with pytest.raises(error):
        SourceStatus(**options)


def test_source_statuses_are_frozen_and_keyword_only():
    status = SourceStatus(version="v1", sha256=digest("a"))
    with pytest.raises(FrozenInstanceError):
        status.version = "v2"
    with pytest.raises(TypeError):
        SourceStatus("v1", digest("a"))
    assert SourceStatus(available=False).sha256 is None


@pytest.mark.parametrize("changes,error", [
    ({"validators": ["smythe.text-spec"]}, TypeError),
    ({"validators": VALIDATORS[0]}, TypeError),
    ({"policy_sha256": "policy"}, ValueError),
    ({"lineage": None}, WorkflowBindingError),
    ({"lineage": LineagePolicy().to_dict()}, WorkflowBindingError),
    ({"tools": {"catalog": {"version": "1"}}}, ValueError),
    ({"tools": {"catalog": "1"}}, ValueError),
    ({"tools": [TOOL]}, TypeError),
    ({"components": {"brief-writer": 2}}, ValueError),
    ({"context_assembly_version": True}, ValueError),
    ({"context_assembly_version": 0}, ValueError),
    ({"agents": ["agent"]}, ValueError),
    ({"agents": digest("agent write")}, TypeError),
    ({"agents": {"write": digest("agent write")}}, TypeError),
    ({"tasks": {digest("task"): None}}, ValueError),
    ({"tasks": {"task": digest("task")}}, ValueError),
    ({"integrity": {"run-1": "fine"}}, ValueError),
    ({"integrity": ["verified"]}, TypeError),
    ({"gate_verifier": ""}, ValueError),
    ({"gate_verifier": None}, ValueError),
    ({"gate_validators": ["contains"]}, TypeError),
    ({"gate_validators": GATE_VALIDATOR}, TypeError),
], ids=lambda value: re.sub(r"\W+", "-", str(value))[:40])
def test_the_current_state_is_validated(changes, error):
    with pytest.raises(error):
        state(**changes)


def test_the_current_state_is_detached_and_read_only():
    components = {"brief-writer": "2"}
    current = state(components=components)
    components["brief-writer"] = "3"
    assert current.components["brief-writer"] == "2"
    with pytest.raises(TypeError):
        current.components["brief-writer"] = "3"
    with pytest.raises(FrozenInstanceError):
        current.policy_sha256 = digest("other")
    assert current.access_scope == "local"
    assert isinstance(current.validators, tuple) and isinstance(current.agents, frozenset)
    defaults = CurrentState(catalog=Catalog(), validators=(), policy_sha256=POLICY)
    assert defaults.context_assembly_version == CONTEXT_ASSEMBLY_VERSION
    assert defaults.lineage == LineagePolicy()
    assert (dict(defaults.tools), dict(defaults.components), defaults.agents,
            dict(defaults.tasks), dict(defaults.integrity)) == ({}, {}, frozenset(), {}, {})
    # Gates are judged by TokenVerifier unless the state names a gate validator.
    assert (defaults.gate_verifier, defaults.gate_validators) == (TOKEN_VERIFIER_IDENTITY, ())
    validators = [ValidatorIdentity.for_config("b", "1"), GATE_VALIDATOR, GATE_VALIDATOR]
    assert state(gate_validators=validators).gate_validators == tuple(sorted(
        set(validators), key=lambda item: item.name))


def test_evaluate_refuses_anything_but_a_record_and_a_state():
    with pytest.raises(TypeError, match="ArtifactLineage"):
        evaluate(record_data(), state())
    with pytest.raises(TypeError, match="CurrentState"):
        evaluate(lineage(), {"catalog": Catalog()})


def test_decisions_are_read_only():
    decision = evaluate(lineage(), state(changed("b")))
    with pytest.raises(FrozenInstanceError):
        decision.action = "reuse"
    with pytest.raises(TypeError):
        decision.authority["access_scope"] = "tenant-a"
    with pytest.raises(TypeError):
        decision.authority["sources"][0]["version"] = "v9"
    assert isinstance(decision.nodes[0], NodeDecision)
    again = evaluate(lineage(), state(changed("b")))
    assert again == decision and hash(again) == hash(decision)


def test_every_reason_code_calls_for_a_known_action():
    assert set(REASONS.values()) <= {*ACTIONS, None}
    assert all(re.fullmatch(r"[a-z]+(_[a-z]+)+", code) for code in REASONS)
    with pytest.raises(TypeError):
        REASONS["source_changed"] = "reuse"


_PUBLIC_TYPES = [getattr(smythe.reuse, name) for name in smythe.reuse.__all__
                 if isinstance(getattr(smythe.reuse, name), type)]


@pytest.mark.parametrize("kind", _PUBLIC_TYPES, ids=lambda kind: kind.__name__)
def test_public_types_resolve_their_type_hints(kind):
    typing.get_type_hints(kind)


def test_public_names_are_exported():
    for name in ("CurrentState", "ItemDecision", "SourceCatalog", "SourceStatus"):
        assert name in smythe.__all__ and name in smythe.reuse.__all__
        assert getattr(smythe, name) is getattr(smythe.reuse, name)


def test_the_lineage_guide_documents_every_reason_code():
    guide = (Path(__file__).resolve().parents[1] / "docs/lineage.md").read_text(encoding="utf-8")
    section = guide.split("\n### Reason codes\n", 1)[1].split("\n#", 1)[0]
    documented = dict(re.findall(r"^\| `([a-z_]+)` \| (`[a-z_]+`) \|", section, re.MULTILINE))
    assert set(documented) == set(REASONS)
    for code, action in REASONS.items():
        assert documented[code] == (f"`{action}`" if action else "`undeclared_coverage`")
