"""Template records validated as untrusted candidates before they are stored."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import typing

import pytest

import smythe
from smythe._canonical import canonical_bytes, sha256_hex
from smythe.constrained_planner import ConstrainedArchitect
from smythe.distill import distill_template
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.loader import (
    MODEL_PLAN_MAX_DEPTH, MODEL_PLAN_MAX_GATES, MODEL_PLAN_MAX_NODES, MODEL_PLAN_MAX_REGENERATIONS,
    MODEL_PLAN_MAX_RETRIES, MODEL_PLAN_MIN_TIMEOUT_S, build_graph_from_model_output,
)
from smythe.swarm import Swarm
from smythe.synthesizer import SynthesisStrategy, Synthesizer
from smythe.task import Task
from smythe.templates import (
    TemplateCatalog, TemplateLimits, TemplateRecord, TemplateRecordError, TemplateValidationError,
    distill_record, validate_template_record,
)
from smythe.validation import ValidatorIdentity
from test_distill import JUDGE, WRITER, _completed_gated_graph, _PersonaScript, _Selects

LENGTH = ValidatorIdentity.for_config("memo-length", "1", {"max_words": 500}).to_dict()
TONE = ValidatorIdentity.for_config("memo-tone", "1", {"register": "plain"}).to_dict()


def source_record() -> TemplateRecord:
    """The gated memo: a judge gates the writer's draft, which two validators check."""
    graph, registry = _completed_gated_graph()
    data = distill_record(graph, name="memo", registry=registry).to_dict()
    data["expected_outputs"] = [{"node": "draft", "description": "The approved memo",
                                 "validators": [LENGTH, TONE]}]
    data["prerequisites"] = {
        "capabilities": ["review", "writing"], "skills": [{"name": "memo-style", "version": "2"}],
        "tools": [{"name": "search", "version": "1",
                   "schema_sha256": sha256_hex({"type": "object"})}],
        "validators": [LENGTH]}
    # A run's recorded acceptance decision is evidence, not authority.
    data["source_runs"] = [{"execution_id": "run-1", "structure_sha256": sha256_hex("shape"),
                            "acceptance": "accepted"}]
    return TemplateRecord.from_dict(data)


def listing(root):
    """Every path under *root*, with each file's SHA-256."""
    return {path.relative_to(root).as_posix():
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            for path in sorted(root.rglob("*"))}


def _set(path, value):
    def change(data):
        *parents, last = path
        for key in parents:
            data = data[key]
        data[last] = value
    return change


def _node(node_id, **fields):
    """A plain blueprint node; *fields* replace its defaults."""
    return {"id": node_id, "label": f"Do {node_id}", "depends_on": [], "capabilities": [],
            "profile": None, "verifies": None, "max_regenerations": 0, "failure_policy": "halt",
            "max_retries": 1, "timeout_s": None, "attach_dep_artifacts": False, "metadata": {},
            **fields}


def _add(*nodes):
    def change(data):
        data["blueprint"].extend(nodes)
    return change


def _retarget(data):
    data["blueprint"].append(_node("outline"))
    data["blueprint"][1].update(verifies="outline", depends_on=["draft", "outline"])


def _remove_gate(data):
    del data["blueprint"][1]


def _drop_outputs(data):
    del data["expected_outputs"]


# Each candidate changes the source record's data, names the parent's version,
# and must be refused naming this node and field.
HARMFUL = {
    "node-mcp": (_set(["blueprint", 0, "mcp_servers"], [{"name": "files", "command": "serve"}]),
                 "draft", "mcp_servers"),
    "node-model": (_set(["blueprint", 0, "model"], "a-model"), "draft", "model"),
    "node-env": (_set(["blueprint", 1, "env"], {"TOKEN": "x"}), "check", "env"),
    "node-url": (_set(["blueprint", 0, "url"], "https://example.com"), "draft", "url"),
    "node-tools": (_set(["blueprint", 0, "tools"], ["search"]), "draft", "tools"),
    "node-command": (_set(["blueprint", 1, "command"], "rm"), "check", "command"),
    "agent-mcp": (_set(["blueprint", 1, "profile", "mcp_servers"], []), "check",
                  "profile.mcp_servers"),
    "agent-model": (_set(["blueprint", 0, "profile", "model"], "a-model"), "draft",
                    "profile.model"),
    "agent-env": (_set(["blueprint", 0, "profile", "env"], {}), "draft", "profile.env"),
    "agent-url": (_set(["blueprint", 0, "profile", "url"], "https://example.com"), "draft",
                  "profile.url"),
    "agent-tools": (_set(["blueprint", 0, "profile", "allowed_tools"], ["search"]), "draft",
                    "profile.allowed_tools"),
    "plan-agent-model": (_set(["blueprint", 0, "agent"], {"name": "Writer", "model": "m"}),
                         "draft", "agent.model"),
    "metadata-model": (_set(["blueprint", 0, "metadata", "model"], "a-model"), "draft",
                       "metadata.model"),
    "acceptance": (_set(["acceptance"], {"mode": "advisory"}), None, "acceptance"),
    "evaluator": (_set(["evaluator"], {"name": "lenient"}), None, "evaluator"),
    "budget": (_set(["budget"], {"max_usd": 100}), None, "budget"),
    "output-acceptance": (_set(["expected_outputs", 0, "acceptance"], {}), None,
                          "expected_outputs[0].acceptance"),
    "validator-evaluator": (_set(["expected_outputs", 0, "validators", 0, "evaluator"], "x"),
                            None, "expected_outputs[0].validators[0].evaluator"),
    "prerequisite-url": (_set(["prerequisites", "tools", 0, "url"], "https://example.com"), None,
                         "prerequisites.tools[0].url"),
    "source-run-evaluator": (_set(["source_runs", 0, "evaluator"], {"name": "lenient"}), None,
                             "source_runs[0].evaluator"),
    "run-acceptance-elsewhere": (_set(["tests"], [{"name": "t", "acceptance": "accepted"}]),
                                 None, "tests[0].acceptance"),
    "gate-dropped": (_set(["blueprint", 1, "verifies"], None), "check", "verifies"),
    "gate-removed": (_remove_gate, "check", "verifies"),
    "gate-retargeted": (_retarget, "check", "verifies"),
    "regenerations-lowered": (_set(["blueprint", 1, "max_regenerations"], 1), "check",
                              "max_regenerations"),
    "regenerations-above-cap": (_set(["blueprint", 1, "max_regenerations"], 3), "check",
                                "max_regenerations"),
    "retries-above-cap": (_set(["blueprint", 0, "max_retries"], 4), "draft", "max_retries"),
    "timeout-below-minimum": (_set(["blueprint", 0, "timeout_s"], 5.0), "draft", "timeout_s"),
    "gate-before-target": (_set(["blueprint", 1, "depends_on"], []), "check", "depends_on"),
    "second-gate": (_add(_node("audit", verifies="draft", depends_on=["draft"])), "audit",
                    "verifies"),
    "target-skips": (_set(["blueprint", 0, "failure_policy"], "skip"), "draft",
                     "failure_policy"),
    "output-validator-removed": (_set(["expected_outputs", 0, "validators"], [LENGTH]), "draft",
                                 "expected_outputs.validators"),
    "outputs-removed": (_drop_outputs, "draft", "expected_outputs.validators"),
    "prerequisite-tool-removed": (_set(["prerequisites", "tools"], []), None,
                                  "prerequisites.tools"),
    "prerequisite-validator-removed": (_set(["prerequisites", "validators"], []), None,
                                       "prerequisites.validators"),
    "too-many-nodes": (_add(*(_node(f"extra-{index}") for index in range(7))), None, "blueprint"),
    "plan-id": (_add(_node("extra step")), "extra step", "id"),
    "reserved-id": (_add(_node("__synthesis__")), "__synthesis__", "id"),
    "blank-label": (_set(["blueprint", 0, "label"], "  "), "draft", "label"),
    "blank-agent": (_set(["blueprint", 0, "profile", "name"], ""), "draft", "profile.name"),
}


@pytest.fixture(scope="module")
def stored_source(tmp_path_factory):
    """One catalog holding the source; a refused candidate leaves it unchanged."""
    root = tmp_path_factory.mktemp("catalog")
    source = source_record()
    TemplateCatalog(root).put(source)
    return root, source


@pytest.mark.parametrize("name", list(HARMFUL))
def test_harmful_candidates_are_refused_and_write_nothing(stored_source, name):
    change, node, field = HARMFUL[name]
    tmp_path, source = stored_source
    catalog = TemplateCatalog(tmp_path)
    data = source.to_dict()
    data["parent_version"] = source.template_version
    change(data)
    before = listing(tmp_path)

    attempts = [lambda: catalog.put(deepcopy(data)),
                lambda: catalog.put(deepcopy(data), parent=source),
                lambda: validate_template_record(deepcopy(data), parent=source)]
    try:
        record = TemplateRecord.from_dict(deepcopy(data))
    except TemplateRecordError:
        record = None  # A key no record holds: only the data can carry it.
    else:
        attempts += [lambda: catalog.put(record),
                     lambda: validate_template_record(record, parent=source)]
    for attempt in attempts:
        with pytest.raises(TemplateValidationError) as caught:
            attempt()
        assert (caught.value.node, caught.value.field) == (node, field)
        assert field in str(caught.value)
        assert node is None or repr(node) in str(caught.value)

    assert listing(tmp_path) == before
    assert catalog.versions("memo") == [source.template_version]


def test_validation_errors_are_record_errors():
    assert issubclass(TemplateValidationError, TemplateRecordError)
    assert issubclass(TemplateValidationError, ValueError)
    error = TemplateValidationError("refused", node="draft", field="model")
    assert (str(error), error.node, error.field) == ("refused", "draft", "model")


@pytest.mark.parametrize("change,match", [
    (_set(["blueprint", 0, "agent_id"], "a1"), "unknown keys"),
    (_set(["version"], 2), "Unsupported template record version"),
    (_set(["blueprint", 0, "depends_on"], ["check"]), "dependency cycle"),
], ids=["unknown-key", "version", "cycle"])
def test_data_no_record_can_hold_is_refused(change, match):
    data = source_record().to_dict()
    change(data)
    with pytest.raises(TemplateValidationError, match=match) as caught:
        validate_template_record(data)
    assert (caught.value.node, caught.value.field) == (None, None)


def test_deeply_nested_data_is_refused():
    data = source_record().to_dict()
    nested = []
    for _ in range(5000):
        nested = [nested]
    data["tests"] = nested
    with pytest.raises(TemplateValidationError):
        validate_template_record(data)


def test_a_record_deeper_than_the_model_plan_limit_is_refused():
    data = source_record().to_dict()
    chain = [_node(f"step-{index}", depends_on=[f"step-{index - 1}"] if index else ["check"])
             for index in range(4)]
    data["blueprint"].extend(chain)
    with pytest.raises(TemplateValidationError, match="6 levels deep; the limit is 5") as caught:
        validate_template_record(data)
    assert (caught.value.node, caught.value.field) == (None, "blueprint")
    validate_template_record(data, limits=TemplateLimits(max_depth=6))


def test_a_faithful_or_stronger_candidate_is_accepted(tmp_path):
    source = source_record()
    catalog = TemplateCatalog(tmp_path)
    catalog.put(source)
    data = source.to_dict()
    data["parent_version"] = source.template_version
    draft, check = data["blueprint"]
    draft.update(label="Draft the memo in plain words", failure_policy="halt", max_retries=0)
    check.update(failure_policy="retry", max_retries=2, timeout_s=MODEL_PLAN_MIN_TIMEOUT_S)
    data["blueprint"].append(_node("notes", depends_on=["check"]))
    data["expected_outputs"][0]["validators"].append(
        ValidatorIdentity.for_config("memo-sources", "1").to_dict())
    data["expected_outputs"].append({"node": "notes", "description": "Notes", "validators": []})
    data["prerequisites"]["tools"].append(
        {"name": "fetch", "version": "1", "schema_sha256": sha256_hex({})})

    version = catalog.put(deepcopy(data))

    stored = catalog.get(version)
    assert stored == TemplateRecord.from_dict(data) == validate_template_record(data, parent=source)
    assert stored.parent_version == source.template_version
    assert sorted(catalog.versions("memo")) == sorted([source.template_version, version])
    # The same candidate checked again is the same record.
    assert validate_template_record(stored, parent=source) is stored
    assert validate_template_record(source) is source
    assert validate_template_record(source, parent=source) is source


def test_a_parent_the_catalog_does_not_hold_is_not_compared(tmp_path):
    source = source_record()
    data = source.to_dict()
    data["parent_version"] = source.template_version
    data["blueprint"][1]["verifies"] = None
    catalog = TemplateCatalog(tmp_path)
    # Without the parent the candidate is an ungated record, which the limits allow.
    assert catalog.get(catalog.put(data)).parent_version == source.template_version
    with pytest.raises(TemplateValidationError, match="no longer gates 'draft'"):
        validate_template_record(data, parent=source)


def test_a_new_version_of_a_stored_template_must_name_its_parent(tmp_path):
    """A later version is compared with a stored version of its name, so it cannot drop a gate."""
    source = source_record()
    other = TemplateRecord(name="notes", description="Notes", blueprint=[_node("notes")])
    catalog = TemplateCatalog(tmp_path)
    catalog.put(source)
    catalog.put(other)  # The first version of a name needs no parent.
    gateless = source.to_dict()
    _remove_gate(gateless)
    faithful = {**source.to_dict(), "description": "A shorter memo"}
    before = listing(tmp_path)

    for data in (gateless, faithful):
        for parent_version, problem in ((None, "is not set"),
                                        (sha256_hex("missing"), "not a version of template"),
                                        (other.template_version, "not a version of template")):
            candidate = dict(data, **({"parent_version": parent_version} if parent_version else {}))
            for attempt in (lambda: catalog.put(deepcopy(candidate)),
                            lambda: catalog.put(TemplateRecord.from_dict(deepcopy(candidate))),
                            lambda: catalog.put(deepcopy(candidate), parent=source)):
                with pytest.raises(TemplateValidationError, match=problem) as caught:
                    attempt()
                assert (caught.value.node, caught.value.field) == (None, "parent_version")
                assert "'memo'" in str(caught.value)

    assert listing(tmp_path) == before
    assert catalog.versions("memo") == [source.template_version]
    # Named, the gateless version is compared with its parent; a stored record is stored again.
    with pytest.raises(TemplateValidationError, match="no longer gates 'draft'"):
        catalog.put(dict(gateless, parent_version=source.template_version))
    assert catalog.put(source) == source.template_version
    assert listing(tmp_path) == before
    version = catalog.put(dict(faithful, parent_version=source.template_version))
    assert sorted(catalog.versions("memo")) == sorted([source.template_version, version])


def test_a_candidate_compared_with_another_parent_is_refused():
    source = source_record()
    other = TemplateRecord.from_dict({**source.to_dict(), "description": "Another memo"})
    data = {**source.to_dict(), "description": "A third memo",
            "parent_version": other.template_version}
    with pytest.raises(TemplateValidationError, match="parent_version") as caught:
        validate_template_record(data, parent=source)
    assert (caught.value.node, caught.value.field) == (None, "parent_version")
    validate_template_record(data, parent=other)
    # A candidate that names no parent may be compared with its source.
    del data["parent_version"]
    validate_template_record(data, parent=source)


def _large_record() -> TemplateRecord:
    """Ten nodes and two gates, with five retries and a 30-second timeout."""
    research = [_node(f"research-{index}") for index in range(6)]
    return TemplateRecord(name="survey", description="A survey", blueprint=[
        *research,
        _node("draft", depends_on=[node["id"] for node in research], failure_policy="retry",
              max_retries=5, timeout_s=30),
        _node("check", depends_on=["draft"], verifies="draft", max_regenerations=2),
        _node("final", depends_on=["check"]),
        _node("audit", depends_on=["final"], verifies="final", max_regenerations=1),
    ])


def test_developer_limits_admit_a_larger_record(tmp_path):
    record = _large_record()
    catalog = TemplateCatalog(tmp_path)
    for limits, node, field in (
            (None, None, "blueprint"), (TemplateLimits(max_nodes=10), "draft", "max_retries"),
            (TemplateLimits(max_nodes=10, max_retries=5), "draft", "timeout_s"),
            (TemplateLimits(max_nodes=10, max_retries=5, min_timeout_s=30), "audit", "verifies")):
        with pytest.raises(TemplateValidationError) as caught:
            catalog.put(record, limits=limits)
        assert (caught.value.node, caught.value.field) == (node, field)
    assert listing(tmp_path) == {}

    limits = TemplateLimits(max_nodes=10, max_gates=2, max_retries=5, min_timeout_s=30)
    assert validate_template_record(record, limits=limits) is record
    assert catalog.get(catalog.put(record, limits=limits)) == record
    nodes, _ = record.to_template().builder(Task("Survey the market"))
    assert [node.verifies for node in nodes if node.verifies] == ["draft", "final"]


def test_default_limits_are_the_model_plan_limits():
    assert TemplateLimits() == TemplateLimits(
        max_nodes=MODEL_PLAN_MAX_NODES, max_depth=MODEL_PLAN_MAX_DEPTH,
        max_retries=MODEL_PLAN_MAX_RETRIES, max_regenerations=MODEL_PLAN_MAX_REGENERATIONS,
        max_gates=MODEL_PLAN_MAX_GATES, min_timeout_s=MODEL_PLAN_MIN_TIMEOUT_S)


@pytest.mark.parametrize("fields", [
    {"max_nodes": 0}, {"max_depth": 0}, {"max_retries": -1}, {"max_regenerations": True},
    {"max_gates": 1.0}, {"min_timeout_s": -1}, {"min_timeout_s": float("nan")},
    {"min_timeout_s": "60"},
])
def test_invalid_limits_are_refused(fields):
    with pytest.raises(ValueError, match=next(iter(fields))):
        TemplateLimits(**fields)


def test_arguments_of_the_wrong_type_are_refused(tmp_path):
    source = source_record()
    for call in (lambda: validate_template_record([source.to_dict()]),
                 lambda: validate_template_record(source, parent=source.to_dict()),
                 lambda: validate_template_record(source, limits={"max_nodes": 10}),
                 lambda: TemplateCatalog(tmp_path).put("memo")):
        with pytest.raises(TypeError):
            call()
    assert not tmp_path.joinpath("records").exists()


def test_a_validated_gated_record_keeps_its_gate_and_delivers_the_approved_output(tmp_path):
    """The approved memo is delivered, not the judge's verdict."""
    graph, registry = _completed_gated_graph()
    data = json.loads(canonical_bytes(distill_record(graph, name="memo", registry=registry)
                                      .to_dict()))
    record = validate_template_record(data)

    draft, check = record.to_template().builder(Task("Summarize the quarter"))[0]
    assert (check.verifies, check.depends_on, check.max_regenerations) == ("draft", ["draft"], 2)
    assert [(node.failure_policy, node.max_retries, node.timeout_s, node.attach_dep_artifacts,
             node.metadata) for node in (draft, check)] == [
        (FailurePolicy.RETRY, 3, 90.0, False, {"role": "author"}),
        (FailurePolicy.SKIP, 0, None, True, {"role": "adversarial"})]

    catalog = TemplateCatalog(tmp_path)
    stored = catalog.get(catalog.put(data))
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


def test_developer_templates_and_distillation_are_not_validated():
    """Developer code is trusted: only records stored or checked as candidates are."""
    nodes = [Node(id=f"step-{index}", label=f"Step {index}", max_retries=5,
                  failure_policy=FailurePolicy.RETRY, timeout_s=1.0,
                  depends_on=[f"step-{index - 1}"] if index else [])
             for index in range(MODEL_PLAN_MAX_NODES + 1)]
    for node in nodes:
        node.status = NodeStatus.COMPLETED
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)

    template = distill_template(graph, name="long")
    record = distill_record(graph, name="long")

    assert len(template.builder(Task("Anything"))[0]) == MODEL_PLAN_MAX_NODES + 1
    with pytest.raises(TemplateValidationError, match="the limit of 8"):
        validate_template_record(record)
    with pytest.raises(ValueError):
        build_graph_from_model_output({"nodes": [{"id": "a", "label": "x", "max_retries": 4}]})


def test_public_names_are_exported():
    for name in ("TemplateLimits", "TemplateValidationError", "validate_template_record"):
        assert name in smythe.__all__ and getattr(smythe, name) is getattr(smythe.templates, name)
        assert name in smythe.templates.__all__


def test_public_types_resolve_their_type_hints():
    typing.get_type_hints(TemplateLimits)
    typing.get_type_hints(validate_template_record)
    typing.get_type_hints(TemplateCatalog.put)
