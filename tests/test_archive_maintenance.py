"""Archive maintenance: a selective repair against a full recomputation, offline.

The archive is the one in ``examples/18_archive_maintenance.py``: four durable
research runs read bundled sources from a library on the loopback address,
two of those sources shared between runs, extract a finding from each read
with a model step and publish a brief written by local code under a strict
policy with lineage. Then one shared source gets a new version, one published
brief is damaged and one final validator's version is bumped. A repair plan is
applied, and the same four questions are computed again in full in a second
journal. Kept results and changed conclusions are checked separately.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from smythe import SQLiteWorkflowStore
from smythe.lineage import load_lineage
from smythe.workflow import current_decision, decision_history
from smythe.workflow_artifact import current_artifact
from test_examples_smoke import EXAMPLE_SCRIPTS

EXAMPLE = Path(__file__).parents[1] / "examples" / "18_archive_maintenance.py"
REVISED_NODES = {"read-renewals-2026", "extract-renewals-2026", "synthesis"}


@pytest.fixture(scope="module")
def example():
    """The example's module; it imports the examples' source library beside it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(EXAMPLE.parent))
        spec = importlib.util.spec_from_file_location("archive_maintenance_example", EXAMPLE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


def run_work(store, run_id) -> dict[str, int]:
    """One run's work, from its journal: its calls by kind and its operations."""
    phases = [call["key"]["phase"] for call in store.inspect_run(run_id)["calls"]]
    return {"reads": phases.count("tool"), "model steps": len(phases) - phases.count("tool"),
            **{unit: len(store.list_operations(run_id, kind)) for unit, kind in (
                ("local steps", "synthesis"), ("publications", "publication"),
                ("acceptance evaluations", "acceptance"), ("kept results", "reuse"))}}


def total(works) -> dict[str, int]:
    works = list(works)
    return {unit: sum(work[unit] for work in works) for unit in works[0]}


def operation_rows(store, run_id) -> dict[str, tuple]:
    """Each journal operation of the run by key: its kind, state and digests."""
    rows = store._db.execute("SELECT operation_key, kind, state, inputs_sha, result_sha "
                             "FROM workflow_operations WHERE run_id=?", (run_id,))
    return {row[0]: tuple(row[1:]) for row in rows}


def run_rows(store, run_id) -> list[str]:
    """The run's rows in the journal's SQL dump: its run, calls and their evidence,
    checkpoints, operations and events."""
    calls = [row[0] for row in store._db.execute(
        "SELECT call_id FROM workflow_calls WHERE run_id=?", (run_id,))]
    values = [f"'{value}'" for value in (run_id, *calls)]
    return [line for line in store._db.iterdump()
            if line.startswith("INSERT") and any(value in line for value in values)]


def journal_calls(store, run_id) -> list[tuple[str, str]]:
    return [(call["key"]["phase"], call["key"]["scope_id"])
            for call in store.inspect_run(run_id)["calls"]]


def delivered(store, run_id) -> tuple[str, dict]:
    artifact = current_artifact(store, run_id)
    assert artifact.eligible
    return artifact.publication.sha256, current_decision(store, run_id)["record"].to_dict()


def maintain(example, root: Path, *, reuse: bool):
    """Build the archive, change it, plan and apply its repair, and compute it again in full.

    Yields what each step left, with both journals still open.
    """
    library = example.SourceLibrary(example.read_versions(example.VERSIONS))
    with (library, SQLiteWorkflowStore(root / "archive.db") as store,
          SQLiteWorkflowStore(root / "full.db") as full_store):
        runs = example.run_items(example.research_swarm(store, library, root / "artifacts"),
                                 library)
        keys = {run_id: key for key, run_id in runs.items()}
        lineages = {key: load_lineage(store, run_id) for key, run_id in runs.items()}
        outputs = {key: current_artifact(store, run_id).publication.path.read_bytes()
                   for key, run_id in runs.items()}
        earlier = library.current(example.REVISED)
        revised = library.release(example.REVISED)
        damaged = current_artifact(store, runs[example.DAMAGED]).publication
        example.damage(damaged.path)
        damaged_bytes = damaged.path.read_bytes()

        swarm = example.research_swarm(store, library, root / "artifacts", validator="2",
                                       reuse=reuse)
        plan = swarm.plan_repair(sources=library)
        items = {keys[item["run_id"]]: item for item in plan.to_dict()["items"]}
        dumps = {key: run_rows(store, run_id) for key, run_id in runs.items()}
        operations = {key: operation_rows(store, run_id) for key, run_id in runs.items()}
        checkpoints = {key: store.get_checkpoint(run_id)["checkpoint"]["graph"]
                       for key, run_id in runs.items()}
        before = {key: run_work(store, run_id) for key, run_id in runs.items()}
        reads = len(library.reads)
        new_runs = sum(item["work"]["new_runs"] for item in items.values())

        result = swarm.repair(plan, plan_sha256=plan.plan_sha256,
                              max_total_budget_usd=new_runs * example.RUN_BUDGET,
                              reason=example.REASON, sources=library)

        entries = {keys[entry["run_id"]]: entry for entry in result.to_dict()["items"]}
        heads = {key: entries[key].get("successor_run_id", run_id) for key, run_id in runs.items()}
        # What the repair did: work added to each repaired run, and each successor's own.
        selective = {key: {unit: count - (before[key][unit] if heads[key] == runs[key] else 0)
                           for unit, count in run_work(store, heads[key]).items()}
                     for key in runs}
        read_by_repair = len(library.reads) - reads

        full_runs = example.run_items(example.research_swarm(
            full_store, library, root / "full-artifacts", validator="2"), library)
        yield SimpleNamespace(
            store=store, full_store=full_store, library=library, runs=runs, heads=heads,
            full_runs=full_runs, lineages=lineages, outputs=outputs, earlier=earlier,
            revised=revised, damaged=damaged, damaged_bytes=damaged_bytes, plan=plan,
            items=items, result=result, entries=entries, dumps=dumps, operations=operations,
            checkpoints=checkpoints, selective=selective, read_by_repair=read_by_repair,
            full={key: run_work(full_store, run_id) for key, run_id in full_runs.items()},
            policy=example.policy("2"))


@pytest.fixture(scope="module")
def archive(example, tmp_path_factory):
    yield from maintain(example, tmp_path_factory.mktemp("archive"), reuse=True)


@pytest.fixture(scope="module")
def rollback(example, tmp_path_factory):
    yield from maintain(example, tmp_path_factory.mktemp("rollback"), reuse=False)


def without(work: dict[str, int], *units: str) -> dict[str, int]:
    return {unit: count for unit, count in work.items() if unit not in units}


# ---------------------------------------------------------------------------
# The plan and its outcome
# ---------------------------------------------------------------------------


def test_the_plan_calls_for_each_route_and_every_item_is_applied(archive):
    assert {key: (item["action"], item["reasons"]) for key, item in archive.items.items()} == {
        "billing-default": ("regenerate", ["dependency_changed", "policy_changed", "source_changed",
                                           "validator_set_changed"]),
        "renewal-forecast": ("regenerate", ["dependency_changed", "policy_changed",
                                            "source_changed", "validator_set_changed"]),
        "support-staffing": ("rerender", ["artifact_changed", "policy_changed",
                                          "validator_set_changed"]),
        "trial-length": ("recheck", ["policy_changed", "validator_set_changed"]),
    }
    assert {key: (entry["outcome"], entry["decision"])
            for key, entry in archive.entries.items()} == dict.fromkeys(
        archive.runs, ("applied", "accepted"))
    assert archive.result.successor_run_ids == (archive.heads["billing-default"],
                                                archive.heads["renewal-forecast"])
    assert [key for key in archive.runs if archive.heads[key] == archive.runs[key]] == [
        "support-staffing", "trial-length"]


def test_selective_repair_publishes_the_bytes_and_decisions_of_a_full_recomputation(archive):
    identities = {tuple(identity.to_dict().values())
                  for identity in archive.policy.validator_identities}
    for key, run_id in archive.full_runs.items():
        sha256, decision = delivered(archive.store, archive.heads[key])
        full_sha256, full_decision = delivered(archive.full_store, run_id)
        assert sha256 == full_sha256, key
        assert decision == full_decision, key
        assert decision["decision"] == "accepted"
        assert decision["policy_sha256"] == archive.policy.sha256
        assert {tuple(check["validator"].values()) for check in decision["validators"]} == (
            identities)
        assert {check["validator"]["name"]: check["validator"]["version"]
                for check in decision["validators"]}["brief-quotes"] == "2"


def test_selective_repair_does_strictly_less_work_than_full_recomputation(archive):
    selective = total(archive.selective.values())
    full = total(archive.full.values())
    assert without(selective, "kept results") == {
        "reads": 2, "model steps": 2, "local steps": 2, "publications": 3,
        "acceptance evaluations": 4}
    assert without(full, "kept results") == {
        "reads": 9, "model steps": 9, "local steps": 4, "publications": 4,
        "acceptance evaluations": 4}
    assert selective["kept results"] == 6 and full["kept results"] == 0
    new_runs = (len(archive.result.successor_run_ids), len(archive.full_runs))
    assert new_runs == (2, 4)
    units = [unit for unit in selective if unit != "kept results"]
    assert all(selective[unit] <= full[unit] for unit in units)
    assert sum(selective[unit] for unit in units) + new_runs[0] < sum(
        full[unit] for unit in units) + new_runs[1]
    # The library answered exactly the reads the journal records.
    assert archive.read_by_repair == selective["reads"]
    # The repair's own account agrees with the journal.
    assert archive.result.work == {
        "new_runs": 2, "read_calls": 2, "model_calls": 2, "reused_nodes": 6, "publications": 3,
        "acceptance_evaluations": 4}


# ---------------------------------------------------------------------------
# Each route
# ---------------------------------------------------------------------------


def descendants(lineage, node_id: str) -> set[str]:
    """*node_id* and every node of the record that depends on it, directly or not."""
    found = {node_id}
    for node in lineage.nodes:
        if found & set(node["depends_on"]):
            found.add(node["id"])
    return found


def test_items_that_read_the_revised_source_regenerate_only_what_depends_on_it(
        archive, example):
    for key in ("billing-default", "renewal-forecast"):
        lineage, item = archive.lineages[key], archive.items[key]
        regenerated = {node["id"] for node in item["nodes"] if node["action"] == "regenerate"}
        assert regenerated == descendants(lineage, "read-renewals-2026") == REVISED_NODES
        assert {node["id"] for node in item["nodes"] if node["action"] == "reuse"} == {
            node["id"] for node in lineage.nodes} - REVISED_NODES
        # Only the revised source's read and its extraction called anything.
        successor = archive.heads[key]
        assert journal_calls(archive.store, successor) == [
            ("tool", "node/read-renewals-2026"), ("execution", "node/extract-renewals-2026")]
        assert archive.selective[key] == {
            "reads": 1, "model steps": 1, "local steps": 1, "publications": 1,
            "acceptance evaluations": 1, "kept results": len(lineage.nodes) - 3}
    # An item that shares only the unchanged source keeps every node.
    for key in ("support-staffing", "trial-length"):
        assert {node["action"] for node in archive.items[key]["nodes"]} == {
            archive.items[key]["action"]}


def test_the_damaged_brief_is_published_again_without_reads_or_model_steps(archive, example):
    key = example.DAMAGED
    run_id = archive.runs[key]
    assert archive.entries[key]["work"] == {
        "new_runs": 0, "read_calls": 0, "model_calls": 0, "reused_nodes": 0, "publications": 1,
        "acceptance_evaluations": 1}
    assert archive.selective[key] == {
        "reads": 0, "model steps": 0, "local steps": 0, "publications": 1,
        "acceptance evaluations": 1, "kept results": 0}
    current = current_artifact(archive.store, run_id)
    assert (current.publication.name, current.publication.sha256) == (
        "document-p1.md", archive.damaged.sha256)
    assert current.publication.path.read_bytes() == archive.outputs[key]
    # The damaged file keeps what it holds, and is listed as superseded.
    assert archive.damaged.path.read_bytes() == archive.damaged_bytes != archive.outputs[key]
    assert current.superseded == (archive.damaged,)


def test_the_validator_only_item_is_rechecked_without_running_a_node(archive, example):
    store, key = archive.store, "trial-length"
    run_id = archive.runs[key]
    assert archive.entries[key]["work"] == {
        "new_runs": 0, "read_calls": 0, "model_calls": 0, "reused_nodes": 0, "publications": 0,
        "acceptance_evaluations": 1}
    assert archive.selective[key] == {
        "reads": 0, "model steps": 0, "local steps": 0, "publications": 0,
        "acceptance evaluations": 1, "kept results": 0}
    # Every node is as it was: no node ran, at any generation.
    assert store.get_checkpoint(run_id)["checkpoint"]["graph"]["nodes"] == (
        archive.checkpoints[key]["nodes"])
    history = decision_history(store, run_id)
    assert [entry["sequence"] for entry in history] == [0, 1]
    assert history[1]["record"].policy_sha256 == archive.policy.sha256
    assert history[1]["inputs"]["repair"] == {"plan_sha256": archive.plan.plan_sha256,
                                              "reason": example.REASON}
    assert history[0]["record"].policy_sha256 != archive.policy.sha256
    assert current_artifact(store, run_id).publication.path.read_bytes() == archive.outputs[key]


# ---------------------------------------------------------------------------
# Kept results and changed conclusions, inspected separately
# ---------------------------------------------------------------------------


def test_every_kept_result_names_its_reason_and_its_predecessor(archive, example):
    store = archive.store
    for key in ("billing-default", "renewal-forecast"):
        predecessor, successor = archive.runs[key], archive.heads[key]
        lineage = load_lineage(store, successor).to_dict()
        assert lineage["predecessor"] == {"store_id": store.store_id, "run_id": predecessor,
                                          "lineage_sha256": archive.lineages[key].sha256}
        assert lineage["repair"] == {"plan_sha256": archive.plan.plan_sha256,
                                     "reason": example.REASON}
        earlier = {node["id"]: node for node in archive.lineages[key].to_dict()["nodes"]}
        decided = {node["id"]: node for node in archive.items[key]["nodes"]}
        kept = {node["id"]: node for node in lineage["nodes"] if "reused_from" in node}
        assert set(kept) == set(earlier) - REVISED_NODES
        operations = {operation["inputs"]["node_id"]: operation
                      for operation in store.list_operations(successor, "reuse")}
        assert set(operations) == set(kept)
        for node_id, node in kept.items():
            assert node["reused_from"] == {"run_id": predecessor, "generation": 0}
            assert node["result_sha256"] == earlier[node_id]["result_sha256"]
            assert node["inputs"] == earlier[node_id]["inputs"]
            inputs = operations[node_id]["inputs"]
            # The plan's decision, under the plan the repair applied, keeps it.
            assert decided[node_id]["action"] == "reuse"
            assert inputs["reasons"] == decided[node_id]["reasons"]
            assert inputs["plan_sha256"] == archive.plan.plan_sha256
            assert inputs["predecessor"] == lineage["predecessor"]
            assert inputs["result_sha256"] == earlier[node_id]["result_sha256"]
        for node in lineage["nodes"]:
            if node["id"] not in kept:
                assert node["id"] in REVISED_NODES


def test_the_repaired_runs_keep_their_journal_rows(archive):
    store = archive.store
    # A run a successor repaired is untouched: its calls, balances, checkpoints,
    # operations and events.
    for key in ("billing-default", "renewal-forecast"):
        assert run_rows(store, archive.runs[key]) == archive.dumps[key]
    # A run repaired in place only adds records.
    for key in ("support-staffing", "trial-length"):
        after = operation_rows(store, archive.runs[key])
        assert {operation: after[operation] for operation in archive.operations[key]} == (
            archive.operations[key])
        assert set(after) > set(archive.operations[key])
        assert set(archive.dumps[key]) - set(run_rows(store, archive.runs[key])) <= {
            line for line in archive.dumps[key] if line.startswith('INSERT INTO "workflow_runs"')}


def test_changed_conclusions_quote_the_new_version_and_never_the_earlier(archive, example):
    old = example.finding(example.REVISED, archive.earlier.text)
    new = example.finding(example.REVISED, archive.revised.text)
    assert "71%" in old and "64%" in new
    for key in ("billing-default", "renewal-forecast"):
        text = current_artifact(archive.store, archive.heads[key]).publication.path.read_text(
            encoding="utf-8")
        assert f'- {example.REVISED}: "{new}"' in example_section(text)
        assert old not in text
        # The earlier brief quoted the earlier version, and stays as it was.
        assert old in archive.outputs[key].decode("utf-8")
        earlier = current_artifact(archive.store, archive.runs[key]).publication
        assert earlier.path.read_bytes() == archive.outputs[key]
    # Briefs that never read the revised source say what they said.
    for key in ("support-staffing", "trial-length"):
        text = current_artifact(archive.store, archive.heads[key]).publication.path.read_bytes()
        assert text == archive.outputs[key]


def example_section(text: str) -> str:
    return text.split("## Conclusion", 1)[1]


# ---------------------------------------------------------------------------
# The reuse switch
# ---------------------------------------------------------------------------


def test_with_reuse_off_each_regenerated_item_does_the_work_of_a_full_recomputation(
        rollback, archive):
    store = rollback.store
    actions = {key: item["action"] for key, item in rollback.items.items()}
    # Every item that kept results now computes them again; a changed
    # validator alone still calls for a recheck, which computes nothing.
    assert actions == {"billing-default": "regenerate", "renewal-forecast": "regenerate",
                       "support-staffing": "regenerate", "trial-length": "recheck"}
    assert all("reuse_disabled" in rollback.items[key]["reasons"]
               for key, action in actions.items() if action == "regenerate")
    assert {key: entry["outcome"] for key, entry in rollback.entries.items()} == dict.fromkeys(
        rollback.runs, "applied")
    assert rollback.selective["trial-length"] == archive.selective["trial-length"]
    for key, action in actions.items():
        if action != "regenerate":
            continue
        successor = rollback.heads[key]
        assert store.list_operations(successor, "reuse") == []
        # The successor did exactly what a full recomputation of the item does.
        assert rollback.selective[key] == rollback.full[key] == archive.full[key]
        assert delivered(store, successor) == delivered(rollback.full_store,
                                                        rollback.full_runs[key])
        assert delivered(store, successor) == delivered(archive.full_store,
                                                        archive.full_runs[key])
    # In all, it does the work of a full recomputation, except for the item
    # that only a validator change affects, which is decided again.
    full = total(archive.full.values())
    assert total(rollback.selective.values()) == {
        unit: full[unit] - archive.full["trial-length"][unit]
        + rollback.selective["trial-length"][unit] for unit in full}
    # Its decisions and bytes are those of the selective repair.
    for key in rollback.runs:
        assert delivered(store, rollback.heads[key]) == delivered(archive.store,
                                                                  archive.heads[key])
    # Lineage stays readable, and names each run's predecessor.
    for key in ("billing-default", "renewal-forecast", "support-staffing"):
        lineage = load_lineage(store, rollback.heads[key]).to_dict()
        assert lineage["predecessor"]["run_id"] == rollback.runs[key]
        assert not any("reused_from" in node for node in lineage["nodes"])


def test_the_example_runs_in_the_examples_smoke_test():
    assert EXAMPLE in EXAMPLE_SCRIPTS
