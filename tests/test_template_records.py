"""Versioned template records, their digests, and the append-only catalog."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import typing

import pytest

import smythe
from fakes import kill_and_reap, launch_child, wait_for_barrier
from smythe._canonical import canonical_bytes, sha256_hex
from smythe.constrained_planner import ConstrainedArchitect, SubGraphTemplate
from smythe.constrained_prompts import CONSTRAINED_SYSTEM_PROMPT
from smythe.distill import DistillationError, distill_template
from smythe.graph import FailurePolicy, NodeStatus, structure_sha256
from smythe.provider import CompletionResult, OfflineProvider
from smythe.swarm import Swarm
from smythe.synthesizer import SynthesisStrategy, Synthesizer
from smythe.task import Task
from smythe.templates import (
    MAX_FAILURE_EXAMPLES, TemplateCatalog, TemplateRecord, TemplateRecordError, distill_record,
)
from smythe.validation import ValidatorIdentity
from smythe.workflow_binding import LocalOnly
from smythe.workflow_store import SQLiteWorkflowStore, WorkflowConflictError
from test_distill import (
    JUDGE, WRITER, _completed_gated_graph, _completed_graph, _PersonaScript, _Selects,
)

LENGTH = ValidatorIdentity.for_config("memo-length", "1", {"max_words": 500}).to_dict()
SECTIONS = {
    "applicability": {"task_families": ["memo"], "keywords": ["quarter", "summary"],
                      "required_context_keys": ["period"], "excluded_task_families": ["legal"]},
    "prerequisites": {"capabilities": ["review", "writing"],
                      "skills": [{"name": "memo-style", "version": "2"}],
                      "tools": [{"name": "search", "version": "1",
                                 "schema_sha256": sha256_hex({"type": "object"})}],
                      "validators": [LENGTH]},
    "expected_outputs": [{"node": "draft", "description": "The approved memo",
                          "validators": [LENGTH]}],
    "failure_examples": [{"category": "verification_exhausted",
                          "evidence_sha256": sha256_hex("rejected memo")}],
    "tests": [{"name": "memo-fixture", "sha256": sha256_hex("fixture")}],
    "source_runs": [{"execution_id": "run-1", "structure_sha256": sha256_hex("shape"),
                     "acceptance": "accepted"}],
    "parent_version": sha256_hex("parent"),
}
# A rooted POSIX path, a drive path or a home-relative path.
ROOTED = re.compile(r"""(?:^|[\s"'(=])(?:/(?:home|Users|tmp|var|root|private)/|[A-Za-z]:[\\/]|~[\\/])""")


def full_record() -> TemplateRecord:
    """The gated memo record with every optional section."""
    graph, registry = _completed_gated_graph()
    graph.nodes[0].label = "Draft the quarterly memo"
    data = distill_record(graph, name="memo", registry=registry,
                          source_task=Task("Write the quarterly memo")).to_dict()
    return TemplateRecord.from_dict({**data, **deepcopy(SECTIONS)})


def reversed_keys(value):
    if isinstance(value, dict):
        return {key: reversed_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reversed_keys(item) for item in value]
    return value


def all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_keys(item)


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


def test_record_round_trips_byte_for_byte():
    record = full_record()
    data = record.to_dict()
    text = canonical_bytes(data)

    again = TemplateRecord.from_dict(json.loads(text))

    assert canonical_bytes(again.to_dict()) == text
    assert json.dumps(again.to_dict()) == json.dumps(data)
    assert again == record and hash(again) == hash(record)
    assert (again.template_version, again.record_sha256) == (
        record.template_version, record.record_sha256)
    assert record.record_sha256 == hashlib.sha256(text).hexdigest()
    assert list(data) == ["type", "version", "name", "description", "blueprint", *SECTIONS,
                          "review"]
    assert (data["type"], data["version"]) == ("smythe.template_record", 1)
    assert set(data["blueprint"][0]) == {
        "id", "label", "depends_on", "capabilities", "profile", "verifies", "max_regenerations",
        "failure_policy", "max_retries", "timeout_s", "attach_dep_artifacts", "metadata"}


def test_records_are_read_only_and_detached_from_their_input():
    blueprint = full_record().to_dict()["blueprint"]
    record = TemplateRecord(name="memo", description="A memo", blueprint=blueprint)
    blueprint[0]["label"] = "changed"
    assert record.blueprint[0]["label"] == "Draft the quarterly memo"
    with pytest.raises(TypeError):
        record.blueprint[0]["label"] = "changed"
    with pytest.raises(AttributeError):
        record.name = "other"
    record.to_dict()["blueprint"][0]["label"] = "changed"
    assert record.blueprint[0]["label"] == "Draft the quarterly memo"


def test_empty_sections_are_omitted():
    record = TemplateRecord(
        name="memo", description="A memo", blueprint=full_record().to_dict()["blueprint"],
        applicability={key: [] for key in SECTIONS["applicability"]},
        prerequisites={key: [] for key in SECTIONS["prerequisites"]},
        expected_outputs=[], failure_examples=(), tests=[], source_runs=[], parent_version=None,
        review={"label_findings": []},
    )
    assert set(record.to_dict()) == {"type", "version", "name", "description", "blueprint"}
    assert (record.applicability, record.review, record.tests) == (None, None, ())


def _set(path, value):
    def change(data):
        *parents, last = path
        for key in parents:
            data = data[key]
        data[last] = value
    return change


def _delete(*path):
    def change(data):
        *parents, last = path
        for key in parents:
            data = data[key]
        del data[last]
    return change


@pytest.mark.parametrize("change", [
    _set(["extra"], 1),
    _set(["blueprint", 0, "mcp_servers"], []),
    _set(["blueprint", 0, "model"], "a-model"),
    _set(["blueprint", 0, "agent_id"], "a1"),
    _set(["blueprint", 0, "metadata", "model"], "a-model"),
    _set(["blueprint", 0, "profile", "mcp_servers"], []),
    _set(["applicability", "url"], "https://example.com"),
    _set(["prerequisites", "tools", 0, "allowed"], True),
    _set(["expected_outputs", 0, "acceptance"], {}),
    _set(["source_runs", 0, "env"], {}),
    _set(["review", "budget"], 1),
    _delete("blueprint"),
    _delete("blueprint", 0, "verifies"),
    _delete("prerequisites", "skills"),
], ids=["top", "node-mcp", "node-model", "node-agent", "metadata-model", "profile-mcp",
        "applicability-url", "tool-grant", "output-acceptance", "run-env", "review-budget",
        "no-blueprint", "no-verifies", "no-skills"])
def test_unknown_and_missing_keys_are_rejected(change):
    data = full_record().to_dict()
    change(data)
    with pytest.raises(TemplateRecordError, match="keys"):
        TemplateRecord.from_dict(data)


@pytest.mark.parametrize("version", [0, 2, "1", True, 1.0, None])
def test_unknown_record_versions_are_rejected(version):
    data = full_record().to_dict()
    data["version"] = version
    with pytest.raises(TemplateRecordError, match="Unsupported template record version"):
        TemplateRecord.from_dict(data)
    del data["version"]
    with pytest.raises(TemplateRecordError, match="Unsupported template record version"):
        TemplateRecord.from_dict(data)


@pytest.mark.parametrize("data", [
    None, [], {"type": "smythe.lineage", "version": 1}, {"version": 1},
], ids=["none", "list", "other-type", "no-type"])
def test_other_documents_are_not_template_records(data):
    with pytest.raises(TemplateRecordError, match="type 'smythe.template_record'"):
        TemplateRecord.from_dict(data)


@pytest.mark.parametrize("key,value", [
    ("tests", []), ("parent_version", None), ("review", {"label_findings": []}),
    ("applicability", {key: [] for key in SECTIONS["applicability"]}),
])
def test_present_but_empty_sections_are_rejected(key, value):
    data = full_record().to_dict()
    data[key] = value
    with pytest.raises(TemplateRecordError, match="omitted when empty"):
        TemplateRecord.from_dict(data)


@pytest.mark.parametrize("change,match", [
    (_set(["blueprint", 1, "verifies"], "ghost"), "unknown nodes"),
    (_set(["blueprint", 1, "verifies"], "check"), "verifies itself"),
    (_set(["blueprint", 1, "depends_on"], ["ghost"]), "unknown nodes"),
    (_set(["blueprint", 0, "depends_on"], ["draft"]), "depends on"),
    (_set(["blueprint", 0, "depends_on"], ["check"]), "dependency cycle"),
    (_set(["blueprint", 1, "id"], "draft"), "each id once"),
    (_set(["blueprint"], []), "at least one node"),
    (_set(["blueprint", 0, "failure_policy"], "ignore"), "failure_policy must be one of"),
    (_set(["blueprint", 0, "max_retries"], -1), "non-negative integer"),
    (_set(["blueprint", 1, "max_regenerations"], True), "non-negative integer"),
    (_set(["blueprint", 0, "timeout_s"], 0), "positive finite"),
    (_set(["blueprint", 0, "timeout_s"], float("inf")), "positive finite"),
    (_set(["blueprint", 0, "attach_dep_artifacts"], 1), "true or false"),
    (_set(["blueprint", 0, "metadata", "role"], 5), "must be a string"),
    (_set(["name"], " "), "must not be blank"),
    (_set(["expected_outputs", 0, "node"], "ghost"), "unknown nodes"),
    (_set(["review", "label_findings", 0, "node"], "ghost"), "unknown nodes"),
    (_set(["prerequisites", "validators", 0, "config_sha256"], "x"), "config_sha256"),
    (_set(["source_runs", 0, "acceptance"], "great"), "must be one of"),
    (_set(["tests", 0, "sha256"], "A" * 64), "hexadecimal"),
    (_set(["failure_examples"], [{"category": "c", "evidence_sha256": "0" * 64}] * (
        MAX_FAILURE_EXAMPLES + 1)), "at most"),
])
def test_malformed_values_are_rejected(change, match):
    data = full_record().to_dict()
    change(data)
    with pytest.raises(TemplateRecordError, match=match):
        TemplateRecord.from_dict(data)


def test_records_are_equal_exactly_when_their_digests_are():
    """90 and 90.0 are equal numbers but different canonical JSON."""
    data = full_record().to_dict()
    data["blueprint"][0]["timeout_s"] = 90
    whole = TemplateRecord.from_dict(data)
    data["blueprint"][0]["timeout_s"] = 90.0
    decimal = TemplateRecord.from_dict(data)

    assert whole.record_sha256 != decimal.record_sha256
    assert whole.template_version != decimal.template_version
    assert whole != decimal and not whole == decimal
    assert len({whole, decimal}) == 2
    assert decimal == TemplateRecord.from_dict(json.loads(canonical_bytes(decimal.to_dict())))
    assert whole != data and whole.__eq__(data) is NotImplemented


@pytest.mark.parametrize("change", [
    _set(["name"], "memo \ud800"),
    _set(["description"], "A memo \udfff"),
    _set(["blueprint", 0, "label"], "Draft \ud800 the memo"),
    _set(["blueprint", 0, "profile", "persona"], "You write \ud83d."),
    _set(["blueprint", 0, "metadata", "role"], "\udc80"),
    _set(["prerequisites", "validators", 0, "name"], "length \ud800"),
    _set(["expected_outputs", 0, "description"], "\ud800"),
], ids=["name", "description", "label", "persona", "role", "validator", "output"])
def test_text_utf8_cannot_encode_is_a_record_error(change):
    """A lone surrogate has no UTF-8 form, so no digest can be computed."""
    data = full_record().to_dict()
    change(data)
    with pytest.raises(TemplateRecordError, match="UTF-8"):
        TemplateRecord.from_dict(data)
    with pytest.raises(TemplateRecordError, match="UTF-8"):
        TemplateRecord(**{key: value for key, value in data.items()
                          if key not in ("type", "version")})


def test_distillation_refuses_text_utf8_cannot_encode():
    graph, registry = _completed_graph()
    graph.nodes[1].label = "Red-team \ud800 the findings"
    for distill in (distill_record, distill_template):
        with pytest.raises(DistillationError, match="UTF-8"):
            distill(graph, name="t", registry=registry)
    graph.nodes[1].label = "Red-team the findings"
    with pytest.raises(DistillationError, match="UTF-8"):
        distill_record(graph, name="t", registry=registry, description="About \udfff")


def test_a_record_cannot_be_its_own_parent():
    record = full_record()
    with pytest.raises(TemplateRecordError, match="its parent"):
        TemplateRecord.from_dict({**record.to_dict(), "parent_version": record.template_version})


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def test_template_version_ignores_key_order_set_order_and_evidence():
    record = full_record()
    data = record.to_dict()
    assert TemplateRecord.from_dict(reversed_keys(data)).record_sha256 == record.record_sha256
    shuffled = deepcopy(data)
    for values in (*shuffled["applicability"].values(), *shuffled["prerequisites"].values()):
        values.reverse()
    shuffled["prerequisites"]["capabilities"].append("review")
    assert TemplateRecord.from_dict(shuffled) == record

    evidence = {
        "source_runs": [*SECTIONS["source_runs"], {"execution_id": "run-2",
                        "structure_sha256": sha256_hex("other"), "acceptance": "rejected"}],
        "failure_examples": [{"category": "timeout", "evidence_sha256": sha256_hex("late")}],
        "tests": [{"name": "another", "sha256": sha256_hex("another")}],
        "parent_version": sha256_hex("another parent"),
        "review": {"label_findings": [{"node": "check", "contains_goal": False,
                                       "words": ["memo", "quarterly"]}]},
    }
    for key, value in evidence.items():
        changed = TemplateRecord.from_dict({**data, key: value})
        assert changed.template_version == record.template_version, key
        assert changed.record_sha256 != record.record_sha256, key
    bare = {key: value for key, value in data.items() if key not in evidence}
    assert TemplateRecord.from_dict(bare).template_version == record.template_version


@pytest.mark.parametrize("change", [
    _set(["blueprint", 1, "verifies"], None),
    _set(["blueprint", 1, "max_regenerations"], 1),
    _set(["blueprint", 0, "failure_policy"], "halt"),
    _set(["blueprint", 0, "max_retries"], 2),
    _set(["blueprint", 0, "timeout_s"], 120.0),
    _set(["blueprint", 1, "attach_dep_artifacts"], False),
    _set(["blueprint", 1, "depends_on"], []),
    _set(["blueprint", 1, "metadata"], {}),
    _set(["blueprint", 0, "capabilities"], ["drafting"]),
    _set(["blueprint", 0, "label"], "Draft the memo"),
    _set(["blueprint", 0, "profile", "persona"], "You write tersely."),
    _set(["description"], "Another description"),
    _set(["applicability", "excluded_task_families"], ["legal", "tax"]),
    _set(["applicability", "keywords"], ["quarter"]),
    _set(["prerequisites", "validators"], []),
    _set(["expected_outputs", 0, "validators"], []),
], ids=["gate", "regenerations", "failure-policy", "retries", "timeout", "attachments", "edge",
        "role", "capabilities", "label", "persona", "description", "excluded-family", "keyword",
        "prerequisite-validator", "output-validator"])
def test_template_version_changes_with_any_behavior(change):
    record = full_record()
    data = record.to_dict()
    change(data)
    assert TemplateRecord.from_dict(data).template_version != record.template_version


def test_structure_digest_is_shared_by_runs_of_one_template_on_different_goals():
    graph, registry = _completed_gated_graph()
    template = distill_template(graph, name="memo", registry=registry)
    architect = ConstrainedArchitect(provider=_Selects("memo"), templates=[template], max_retries=0)
    first, _ = architect.plan(Task("Summarize the quarter"))
    second, _ = architect.plan(Task("Review the annual budget"))
    for node in second.nodes:
        node.status, node.result = NodeStatus.COMPLETED, "an answer"
        node.metadata.update(model="a-model", cost_usd=0.5)
    second.nodes.reverse()

    assert [node.label for node in first.nodes] != [node.label for node in second.nodes]
    assert structure_sha256(first) == structure_sha256(second)
    assert re.fullmatch(r"[0-9a-f]{64}", structure_sha256(first))


def _node_change(name, value, index=1):
    def change(graph):
        setattr(graph.nodes[index], name, value)
    return change


@pytest.mark.parametrize("change", [
    _node_change("verifies", None), _node_change("depends_on", []),
    _node_change("required_capabilities", ["judging"]),
    _node_change("failure_policy", FailurePolicy.HALT), _node_change("max_retries", 2),
    _node_change("timeout_s", 30.0), _node_change("max_tool_iterations", 4),
    _node_change("attach_dep_artifacts", False), _node_change("max_regenerations", 1),
    _node_change("metadata", {"role": "judge"}), _node_change("id", "judge"),
], ids=["gate", "edge", "capabilities", "failure-policy", "retries", "timeout",
        "tool-iterations", "attachments", "regenerations", "role", "id"])
def test_structure_digest_changes_with_the_shape(change):
    graph, _ = _completed_gated_graph()
    before = structure_sha256(graph)
    change(graph)
    assert structure_sha256(graph) != before


# ---------------------------------------------------------------------------
# Distillation and building
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", [
    _completed_graph, lambda: _completed_graph(with_agents=False), _completed_gated_graph,
], ids=["agents", "no-agents", "gated"])
def test_records_build_the_nodes_distill_template_builds(fixture):
    graph, registry = fixture()
    expected = distill_template(graph, name="t", registry=registry)
    record = TemplateRecord.from_dict(json.loads(json.dumps(
        distill_record(graph, name="t", registry=registry).to_dict())))
    template = record.to_template()

    assert isinstance(template, SubGraphTemplate)
    assert (template.name, template.description) == (expected.name, expected.description)

    def shape(nodes, built):
        return [({key: value for key, value in vars(node).items() if key != "agent_id"},
                 None if node.agent_id is None else vars(built.get(node.agent_id).profile))
                for node in nodes]

    nodes, built = template.builder(Task("A new goal"), depth=2)
    assert shape(nodes, built) == shape(*expected.builder(Task("A new goal"), depth=2))
    for source, node in zip(graph.nodes, nodes, strict=True):
        for name in ("id", "depends_on", "required_capabilities", "failure_policy", "max_retries",
                     "timeout_s", "attach_dep_artifacts", "verifies", "max_regenerations"):
            assert getattr(node, name) == getattr(source, name)
            assert type(getattr(node, name)) is type(getattr(source, name))
        assert node.label == f"{source.label}\n\nApply this step to: A new goal"
        assert (node.status, node.result) == (NodeStatus.PENDING, None)
        assert node.metadata == {key: source.metadata[key] for key in ("role",)
                                 if key in source.metadata}


def test_a_gated_record_keeps_its_gate_through_the_catalog(tmp_path):
    """The approved memo is delivered, not the judge's verdict."""
    graph, registry = _completed_gated_graph()
    record = distill_record(graph, name="memo", registry=registry)
    draft, check = record.to_dict()["blueprint"]
    assert (check["verifies"], check["depends_on"], check["max_regenerations"]) == (
        "draft", ["draft"], 2)
    assert (draft["failure_policy"], draft["max_retries"], draft["timeout_s"]) == ("retry", 3, 90.0)

    catalog = TemplateCatalog(tmp_path)
    stored = catalog.get(catalog.put(record))
    provider = _PersonaScript({
        WRITER: ["first memo", "THE REVISED MEMO"],
        JUDGE: ["FAIL: cites no sources", "PASS"],
    })
    swarm = Swarm(
        provider=provider,
        architect=ConstrainedArchitect(provider=_Selects("memo"), templates=[stored.to_template()],
                                       max_retries=0),
        synthesizer=Synthesizer(SynthesisStrategy.DELIVERABLE),
        artifact_dir=None,
    )

    result = swarm.execute(Task(goal="Summarize the quarter"))

    assert result.output == "THE REVISED MEMO"
    assert provider.runs == {WRITER: 2, JUDGE: 2}


def test_label_review_names_labels_that_contain_the_source_goal():
    graph, registry = _completed_graph()
    graph.nodes[0].label = "Research the market for SOLAR chargers   in Europe"
    graph.task = Task("Solar chargers in Europe")

    record = distill_record(graph, name="t", registry=registry)

    assert record.to_dict()["review"] == {"label_findings": [
        {"node": "research", "contains_goal": True, "words": ["chargers", "europe", "solar"]},
    ]}
    explicit = distill_record(graph, name="t", registry=registry,
                              source_task=Task("Red-team the findings on chargers"))
    assert explicit.to_dict()["review"] == {"label_findings": [
        {"node": "review", "contains_goal": False, "words": ["findings", "team"]},
    ]}
    # Labels are only flagged: each build still points them at the new task.
    nodes, _ = record.to_template().builder(Task("Assess widgets"))
    assert nodes[0].label == (
        "Research the market for SOLAR chargers   in Europe\n\nApply this step to: Assess widgets")
    graph.task = None
    unreviewed = distill_record(graph, name="t", registry=registry)
    assert "review" not in unreviewed.to_dict()
    assert unreviewed.template_version == record.template_version


def test_review_names_labels_personas_and_descriptions_holding_a_path():
    graph, registry = _completed_graph()
    research, review = graph.nodes
    research.label = r"Summarize D:\exports\q3.csv"
    registry.get(review.agent_id).profile.persona = "Read ~/notes before you start."

    record = distill_record(graph, name="t", registry=registry,
                            description="Reports kept under /srv/reports")

    assert record.to_dict()["review"] == {"path_findings": [
        {"node": None, "field": "description"},
        {"node": "research", "field": "label"},
        {"node": "review", "field": "persona"},
    ]}
    # Flagged, not changed, and kept out of the template version.
    assert record.blueprint[0]["label"] == r"Summarize D:\exports\q3.csv"
    unreviewed = TemplateRecord.from_dict(
        {key: value for key, value in record.to_dict().items() if key != "review"})
    assert unreviewed.template_version == record.template_version
    # Summarized from the labels, the default description shows the path too.
    described = distill_record(graph, name="t", registry=registry)
    assert {"node": None, "field": "description"} in described.review["path_findings"]
    graph.task = Task(r"Summarize D:\exports\q3.csv")
    both = distill_record(graph, name="t", registry=registry, description="Quarterly")
    assert both.to_dict()["review"] == {
        "label_findings": [{"node": "research", "contains_goal": True,
                            "words": ["exports", "summarize"]}],
        "path_findings": [{"node": "research", "field": "label"},
                          {"node": "review", "field": "persona"}]}


@pytest.mark.parametrize("text,flagged", [
    (r"Read E:\exports\q3.csv", True), ("Load D:/data/x.csv", True),
    ("Open /srv/notes", True), ("/etc/hosts first", True), ("(see /tmp/x)", True),
    ("Read ~/notes", True), (r"Read ~\notes", True), ("Read ~pat/notes", True),
    (r"Copy \\server\share\q3.csv", True), ("Use $HOME/notes", True),
    (r"Use %UserProfile%\notes", True),
    ("Answer PASS/FAIL", False), ("Agree and/or object", False), ("Keep 1/2 of it", False),
    ("Cite https://example.com/a/b", False), ("Edit ./src/app.py", False),
    ("Edit ../up.txt", False), ("Wait ~5 minutes", False), ("Option C: none", False),
    ("Write w/o sources over TCP/IP", False), ("Close </b> tags", False),
    ("Grade the $HOMEWORK", False),
])
def test_path_review_finds_rooted_and_home_paths_only(text, flagged):
    graph, registry = _completed_graph()
    graph.nodes[0].label = text
    record = distill_record(graph, name="t", registry=registry, description="A review")
    findings = (record.review or {}).get("path_findings", ())
    assert ({"node": "research", "field": "label"} in findings) is flagged


@pytest.mark.parametrize("review,match", [
    ({"label_findings": [], "path_findings": [{"node": "draft", "field": "label"}]},
     "omitted when empty"),
    ({"path_findings": [{"node": None, "field": "label"}]}, "names a node"),
    ({"path_findings": [{"node": "draft", "field": "description"}]}, "names a node"),
    ({"path_findings": [{"node": "ghost", "field": "label"}]}, "unknown nodes"),
    ({"path_findings": [{"node": "draft", "field": "name"}]}, "must be one of"),
])
def test_malformed_path_findings_are_rejected(review, match):
    data = full_record().to_dict()
    data["review"] = review
    with pytest.raises(TemplateRecordError, match=match):
        TemplateRecord.from_dict(data)


def test_distillation_refuses_a_graph_that_is_not_a_valid_record():
    graph, registry = _completed_graph()
    graph.nodes[1].depends_on = ["ghost"]
    for distill in (distill_record, distill_template):
        with pytest.raises(DistillationError, match="'review' names unknown nodes"):
            distill(graph, name="t", registry=registry)


@pytest.mark.parametrize("change", [
    _node_change("depends_on", ["ghost"]), _node_change("depends_on", [" "]),
    _node_change("depends_on", ["review"]), _node_change("depends_on", ["review"], index=0),
    _node_change("id", "research"), _node_change("id", " "), _node_change("timeout_s", 0.0),
    _node_change("timeout_s", float("nan")), _node_change("max_retries", -1),
    _node_change("max_retries", 2.0), _node_change("max_regenerations", -1),
    _node_change("metadata", {"role": 7}), _node_change("label", None),
    _node_change("label", "Red-team \ud800"),
], ids=["dangling-edge", "blank-edge", "self-edge", "cycle", "duplicate-id", "blank-id",
        "zero-timeout", "nan-timeout", "negative-retries", "float-retries",
        "negative-regenerations", "non-text-role", "non-text-label", "no-utf8-form"])
def test_distill_template_refuses_a_graph_it_cannot_write_as_a_record(change):
    """Without a description too, which is summarized from the graph's shape."""
    for description in (None, "A review"):
        graph, registry = _completed_graph()
        change(graph)
        for distill in (distill_record, distill_template):
            with pytest.raises(DistillationError, match="cannot distill this graph"):
                distill(graph, name="t", registry=registry, description=description)
    graph, registry = _completed_graph()
    with pytest.raises(DistillationError, match="must not be blank"):
        distill_template(graph, name=" ", registry=registry)


def test_local_only_pins_the_record_version_in_a_durable_recipe(tmp_path, monkeypatch):
    async def complete(self, system, prompt, model):
        if system == CONSTRAINED_SYSTEM_PROMPT:
            return CompletionResult(json.dumps([{"template": "learned"}]), cost_usd=0)
        return CompletionResult("Step done", cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    graph, registry = _completed_graph()
    first = distill_record(graph, name="learned", registry=registry)
    changed = first.to_dict()
    changed["blueprint"][0]["max_retries"] = 2
    second = TemplateRecord.from_dict(changed)
    declaration = first.local_only()
    assert declaration == LocalOnly(object, "template:learned", first.template_version,
                                    role="template_builder")

    def durable(record, store):
        provider = OfflineProvider()
        template = SubGraphTemplate(record.name, record.description, record.local_only())
        return Swarm(model="offline", provider=provider, run_store=store,
                     architect=ConstrainedArchitect(provider, [template], model="offline"))

    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        recipes = [durable(record, store)._workflow_runtime().recipe
                   for record in (first, first, second)]
        assert recipes[0] == recipes[1] != recipes[2]
        builders = [recipe["components"]["architect"]["templates"][0]["builder"]
                    for recipe in recipes]
        assert builders[0] == {"type": "local_only", "role": "template_builder",
                               "identity": "template:learned", "version": first.template_version}
        assert builders[2]["version"] == second.template_version

        run_id = durable(first, store).plan(Task("Assess gadgets")).run_ref["run_id"]
        with pytest.raises(WorkflowConflictError, match="saved recipe"):
            durable(second, store).resume(run_id)
        catalog = TemplateCatalog(tmp_path / "catalog")
        catalog.put(first)
        result = durable(catalog.get(first.template_version), store).resume(run_id)

    assert [node.id for node in result.graph.nodes] == ["learned-0-research", "learned-0-review"]
    assert result.output == "Step done"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def snapshot(root: Path):
    return {path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_catalog_put_is_idempotent_and_never_replaces_a_record(tmp_path):
    catalog = TemplateCatalog(tmp_path)
    record = full_record()
    version = catalog.put(record)
    stored = tmp_path / "records" / f"{record.record_sha256}.json"
    assert stored.read_bytes() == canonical_bytes(record.to_dict())
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == record.record_sha256
    assert (tmp_path / "versions" / version).read_bytes() == record.record_sha256.encode()
    before = snapshot(tmp_path)

    assert catalog.put(TemplateRecord.from_dict(record.to_dict())) == version
    other = TemplateRecord.from_dict({**record.to_dict(),
                                      "tests": [{"name": "other", "sha256": sha256_hex("o")}]})
    assert other.template_version == version and other.record_sha256 != record.record_sha256
    with pytest.raises(TemplateRecordError, match="different record"):
        catalog.put(other)

    assert snapshot(tmp_path) == before
    assert TemplateCatalog(tmp_path).get(version) == record
    newer = TemplateRecord.from_dict({**record.to_dict(), "parent_version": version,
                                      "description": "A shorter memo"})
    assert catalog.versions("memo") == [version]
    assert catalog.put(newer) == newer.template_version
    assert catalog.versions("memo") == sorted([version, newer.template_version])
    assert catalog.versions("other") == []
    with pytest.raises(KeyError):
        catalog.get(sha256_hex("missing"))
    with pytest.raises(TemplateRecordError, match="64 lowercase"):
        catalog.get(f"../records/{record.record_sha256}")
    assert TemplateCatalog(tmp_path / "absent").versions("memo") == []


def _edit_record(field_change):
    def tamper(root, record):
        path = root / "records" / f"{record.record_sha256}.json"
        data = json.loads(path.read_bytes())
        field_change(data)
        path.write_bytes(canonical_bytes(data))
    return tamper


def _repoint(root, record):
    other = TemplateRecord.from_dict({**record.to_dict(), "description": "Another memo"})
    TemplateCatalog(root).put(other)
    (root / "versions" / record.template_version).write_bytes(other.record_sha256.encode())


def _reformat(root, record):
    path = root / "records" / f"{record.record_sha256}.json"
    path.write_text(json.dumps(json.loads(path.read_bytes()), indent=1), encoding="utf-8")


@pytest.mark.parametrize("tamper", [
    _edit_record(_set(["blueprint", 1, "verifies"], None)),
    _edit_record(_set(["source_runs", 0, "acceptance"], "rejected")),
    _edit_record(_set(["source_runs", 0, "execution_id"], "run-9")),
    _reformat, _repoint,
], ids=["gate", "evidence", "run-id", "formatting", "version-file"])
def test_a_tampered_catalog_file_is_detected(tmp_path, tamper):
    catalog = TemplateCatalog(tmp_path)
    record = full_record()
    catalog.put(record)
    tamper(tmp_path, record)
    before = snapshot(tmp_path)
    with pytest.raises(TemplateRecordError):
        catalog.get(record.template_version)
    with pytest.raises(TemplateRecordError):
        catalog.versions("memo")
    with pytest.raises(TemplateRecordError):
        catalog.put(record)
    assert snapshot(tmp_path) == before


def test_a_missing_record_file_is_detected_and_put_publishes_it_again(tmp_path):
    catalog = TemplateCatalog(tmp_path)
    record = full_record()
    catalog.put(record)
    (tmp_path / "records" / f"{record.record_sha256}.json").unlink()
    with pytest.raises(TemplateRecordError, match="missing record"):
        catalog.get(record.template_version)
    assert catalog.put(record) == record.template_version
    assert catalog.get(record.template_version) == record


def _race(monkeypatch, root, folder, rival):
    """Have another put of *rival* finish first, when this put first publishes into *folder*."""
    publish = smythe.templates.atomic_publish_bytes
    raced, lost = [], []

    def publish_after_a_rival(path, data):
        if not raced and path.parent.name == folder:
            raced.append(path.name)
            TemplateCatalog(root).put(rival)
        try:
            publish(path, data)
        except FileExistsError:
            lost.append(path.name)
            raise

    monkeypatch.setattr("smythe.templates.atomic_publish_bytes", publish_after_a_rival)
    return lost


@pytest.mark.parametrize("folder", ["records", "versions"])
def test_a_put_that_loses_a_race_for_the_same_record_succeeds(tmp_path, monkeypatch, folder):
    record = full_record()
    lost = _race(monkeypatch, tmp_path, folder, record)

    assert TemplateCatalog(tmp_path).put(record) == record.template_version

    assert lost == [f"{record.record_sha256}.json" if folder == "records"
                    else record.template_version]
    assert TemplateCatalog(tmp_path).get(record.template_version) == record


def test_a_put_that_loses_a_race_to_another_record_of_its_version_is_refused(
        tmp_path, monkeypatch):
    record = full_record()
    rival = TemplateRecord.from_dict({**record.to_dict(),
                                      "tests": [{"name": "rival", "sha256": sha256_hex("r")}]})
    assert rival.template_version == record.template_version
    lost = _race(monkeypatch, tmp_path, "versions", rival)

    with pytest.raises(TemplateRecordError, match="different record"):
        TemplateCatalog(tmp_path).put(record)

    assert lost == [record.template_version]
    assert TemplateCatalog(tmp_path).get(record.template_version) == rival


LAYOUT_1 = b'{"layout":1,"type":"smythe.template_catalog"}'


def test_the_catalog_records_its_layout(tmp_path):
    catalog = TemplateCatalog(tmp_path)
    record = full_record()
    version = catalog.put(record)
    assert (tmp_path / "layout.json").read_bytes() == LAYOUT_1
    assert smythe.templates.TEMPLATE_CATALOG_LAYOUT == 1

    # A catalog written before the layout file existed is read as layout 1.
    (tmp_path / "layout.json").unlink()
    assert catalog.get(version) == record and catalog.versions("memo") == [version]
    assert catalog.put(record) == version
    assert (tmp_path / "layout.json").read_bytes() == LAYOUT_1


@pytest.mark.parametrize("marker,match", [
    (b'{"layout":2,"type":"smythe.template_catalog"}', "layout 2"),
    (b'{"layout":1}', "layout 1"),
    (b"\xff not json", "layout None"),
    (b"[2]", "layout None"),
])
def test_a_catalog_in_another_layout_is_refused(tmp_path, marker, match):
    catalog = TemplateCatalog(tmp_path)
    record = full_record()
    version = catalog.put(record)
    (tmp_path / "layout.json").write_bytes(marker)
    newer = TemplateRecord.from_dict({**record.to_dict(), "description": "A shorter memo"})
    before = snapshot(tmp_path)

    for read in (lambda: catalog.get(version), lambda: catalog.versions("memo"),
                 lambda: catalog.put(record), lambda: TemplateCatalog(tmp_path).put(newer)):
        with pytest.raises(TemplateRecordError, match=f"has {match}; this release reads layout 1"):
            read()

    assert snapshot(tmp_path) == before


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_catalog_directories_and_files_are_owner_only(tmp_path):
    root = tmp_path / "catalog"
    record = full_record()
    version = TemplateCatalog(root).put(record)
    for path, mode in ((root, 0o700), (root / "records", 0o700), (root / "versions", 0o700),
                       (root / "records" / f"{record.record_sha256}.json", 0o600),
                       (root / "versions" / version, 0o600), (root / "layout.json", 0o600)):
        assert stat.S_IMODE(path.stat().st_mode) == mode, path


def test_records_carry_no_paths_or_execution_settings(tmp_path):
    home = str(Path.home())
    graph, registry = _completed_gated_graph()
    for node in graph.nodes:
        node.metadata.update(artifacts=[str(tmp_path / "draft.png")], model="a-model",
                             task_context={"input": home}, mcp_servers=["files"])
    graph.run_ref = {"version": 1, "store_id": "store", "run_id": "run",
                     "recipe_sha256": sha256_hex("recipe")}
    graph.task = Task("Summarize the quarter", context={"input": str(tmp_path / "q3.csv")})
    record = TemplateRecord.from_dict({
        **distill_record(graph, name="memo", registry=registry).to_dict(), **deepcopy(SECTIONS)})
    catalog = TemplateCatalog(tmp_path / "catalog")
    version = catalog.put(record)

    files = sorted(path for path in (tmp_path / "catalog").rglob("*") if path.is_file())
    assert [path.relative_to(tmp_path / "catalog").as_posix() for path in files] == [
        "layout.json", f"records/{record.record_sha256}.json", f"versions/{version}"]
    texts = [path.read_text(encoding="utf-8") for path in files]
    for text in (*texts, json.dumps(catalog.get(version).to_dict())):
        for path in (str(tmp_path), str(tmp_path.resolve()), home):
            assert path not in text and json.dumps(path)[1:-1] not in text
        assert ROOTED.search(text) is None, text
    assert not set(all_keys(record.to_dict())) & {
        "mcp_servers", "model", "env", "url", "command", "allowed_tools", "evaluator", "budget",
        "agent_id", "artifacts", "cost_usd", "task_context", "run_ref", "context"}


KILLED_PUT = r"""
import json
import os
import sys
import time
from pathlib import Path

from fakes import write_barrier
from smythe.templates import TemplateCatalog, TemplateRecord

root, folder, source, hold = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
record = TemplateRecord.from_dict(json.loads(source.read_text(encoding="utf-8")))
link, links = os.link, []


def held(temporary, destination, *args, **kwargs):
    links.append(destination)
    if len(links) == hold:
        write_barrier(folder / "linking.json", {
            "temporary": Path(temporary).name, "destination": Path(destination).name,
            "size": Path(temporary).stat().st_size})
        time.sleep(600)
    link(temporary, destination, *args, **kwargs)


os.link = held
TemplateCatalog(root).put(record)
write_barrier(folder / "published.json", {})
"""


@pytest.mark.parametrize("hold", [1, 2], ids=["record", "version"])
def test_a_put_killed_before_a_file_is_linked_leaves_no_partial_record(tmp_path, hold):
    record = full_record()
    source = tmp_path / "record.json"
    source.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    root = tmp_path / "catalog"
    process = launch_child(KILLED_PUT, root, tmp_path, source, hold)
    try:
        barrier = wait_for_barrier(process, tmp_path / "linking.json")
        reaped = kill_and_reap(process)
    finally:
        kill_and_reap(process)

    assert reaped.killed and reaped.returncode != 0
    assert not (tmp_path / "published.json").exists()
    data = canonical_bytes(record.to_dict())
    held_file, size = ((f"{record.record_sha256}.json", len(data)) if hold == 1
                       else (record.template_version, 64))
    assert (barrier["destination"], barrier["size"]) == (held_file, size)
    # The killed put wrote its bytes in full, under a temporary name only.
    assert barrier["temporary"].startswith(f".{held_file}.")
    published = sorted(path.name for path in root.rglob("*")
                       if path.is_file() and not path.name.startswith("."))
    assert published == ([] if hold == 1 else [f"{record.record_sha256}.json"])

    catalog = TemplateCatalog(root)
    assert catalog.versions(record.name) == []
    with pytest.raises(KeyError):
        catalog.get(record.template_version)
    assert catalog.put(record) == record.template_version
    assert catalog.get(record.template_version) == record
    assert catalog.versions(record.name) == [record.template_version]


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_names_are_exported():
    for name in ("TemplateCatalog", "TemplateRecord", "TemplateRecordError", "distill_record"):
        assert name in smythe.__all__ and getattr(smythe, name) is getattr(smythe.templates, name)
    assert smythe.structure_sha256 is structure_sha256
    assert issubclass(TemplateRecordError, ValueError)


@pytest.mark.parametrize("kind", [TemplateRecord, TemplateCatalog], ids=lambda kind: kind.__name__)
def test_public_types_resolve_their_type_hints(kind):
    typing.get_type_hints(kind)
    for member in vars(kind).values():
        member = getattr(member, "__func__", member)
        if callable(member) and hasattr(member, "__annotations__"):
            typing.get_type_hints(member)
    typing.get_type_hints(distill_record)
    typing.get_type_hints(structure_sha256)
