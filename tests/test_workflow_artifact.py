"""Durable runs publish their delivered document, and acceptance binds the published bytes.

A run with a renderer publishes the rendered document once into its own
directory, never replacing a file. The document is the output a strict policy
decides, so the decision binds the published SHA-256. A missing or changed
file is not eligible for delivery on inspection or resume, and republishing
writes a new file that becomes the run's current publication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fakes import Crash
from smythe import workflow
from smythe import (
    AcceptanceDecision, AcceptancePolicy, LocalOnly, MarkdownRenderer, OfflineProvider,
    PublishedArtifact, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task, ValidationReport,
    ValidatorIdentity,
)
from smythe.acceptance import text_sha256
from smythe.cli import EXIT_OK, EXIT_RUN_INDETERMINATE, EXIT_RUN_NOT_ACCEPTED, main
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.validation import hard_finding
from smythe.workflow_artifact import PUBLICATION_OPERATION, current_artifact
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import WorkflowConflictError, WorkflowStateError
from test_workflow_acceptance import operations, stop_at

DOCUMENT_HEADING = "# Write the brief\n\n"


class Mentions:
    """A final validator: the text must mention ``needle``. Records every text it checks."""

    checked: list[str] = []

    def __init__(self, needle: str = "offline") -> None:
        self.needle = needle
        self.identity = ValidatorIdentity.for_config("mentions", "1", {"needle": needle})

    def validate(self, subject):
        Mentions.checked.append(subject.text())
        findings = () if self.needle in subject.text() else (
            hard_finding("missing", f"no {self.needle!r}"),)
        return ValidationReport("output", findings, validator=self.identity)


class PlainBrief:
    """A LocalOnly renderer: a plain-text brief with its own file name."""

    filename = "brief.txt"
    media_type = "text/plain; charset=utf-8"

    def render(self, output, task):
        return f"BRIEF: {task.goal}\n{output}\n"


@pytest.fixture(autouse=True)
def fresh_checks():
    Mentions.checked = []


@pytest.fixture
def journal(tmp_path) -> Path:
    return tmp_path / "runs.db"


@pytest.fixture
def store(journal):
    with SQLiteWorkflowStore(journal) as value:
        yield value


def strict(needle: str = "offline") -> AcceptancePolicy:
    return AcceptancePolicy(mode="strict", final_validators=[Mentions(needle)])


def graph() -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft the brief"),
        Node(id="final", label="Finish the brief", depends_on=["draft"]),
    ], task=Task("Write the brief"))


def durable(store, tmp_path, **options) -> Swarm:
    options.setdefault("renderer", MarkdownRenderer())
    options.setdefault("artifact_dir", tmp_path / "artifacts")
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **options)


def run_directory(tmp_path, run_id) -> Path:
    return tmp_path / "artifacts" / run_id


def listing(directory: Path) -> list[str]:
    return sorted(item.name for item in directory.iterdir())


def inspect(capsys, journal, run_id):
    code = main(["runs", "--journal", str(journal), "--json", "inspect", run_id])
    captured = capsys.readouterr()
    assert captured.err == ""
    value = json.loads(captured.out)
    assert value["ok"] is True
    return code, value["inspect"]


def damage(path: Path, how: str) -> None:
    if how == "changed":
        path.write_bytes(b"# Write the brief\n\nedited after publication\n")
    else:
        path.unlink()


# ---------------------------------------------------------------------------
# Publication and the decision on the published bytes
# ---------------------------------------------------------------------------


def test_the_published_document_is_the_output_the_policy_decides(store, tmp_path):
    result = durable(store, tmp_path, acceptance=strict()).execute(graph())
    run_id = result.execution_id
    synthesized = store.load_operation(run_id, "synthesis")["result"]["output"]
    document = DOCUMENT_HEADING + synthesized.rstrip() + "\n"
    path = run_directory(tmp_path, run_id) / "document.md"

    # The run delivers the rendered document, published once in its own directory.
    assert result.output == document and result.accepted
    assert path.read_bytes() == document.encode("utf-8")
    assert listing(path.parent) == [".smythe-run-owner.json", "document.md"]
    # The validator checked the document's bytes, and the decision binds their SHA-256.
    assert Mentions.checked == [document]
    published = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.acceptance.output_sha256 == published == text_sha256(document)

    current = current_artifact(store, run_id)
    assert current.eligible and current.reasons == ()
    assert current.decision == result.acceptance
    assert current.integrity["status"] == "verified"
    assert current.publication == PublishedArtifact(
        sequence=0, directory=str(path.parent.resolve()), name="document.md", sha256=published,
        size_bytes=len(document.encode("utf-8")), media_type="text/markdown; charset=utf-8",
        renderer={"type": "markdown_renderer", "version": 1}, synthesis="synthesis",
        output_sha256=text_sha256(synthesized),
    )
    assert current.publication.path == path.resolve()

    # The recipe records the renderer; the completed checkpoint holds the
    # document and applied the publication with the decision.
    assert store.load_run(run_id)["config"]["renderer"] == {"type": "markdown_renderer", "version": 1}
    assert store.get_checkpoint(run_id)["checkpoint"]["output"] == document
    assert operations(store, run_id)["publication/0"] == (PUBLICATION_OPERATION, "applied")
    assert store.audit(run_id)["ok"]

    # Resuming the completed run returns the same decision, with nothing checked again.
    again = durable(store, tmp_path, acceptance=strict()).resume(run_id)
    assert again.accepted and again.output == document and Mentions.checked == [document]


def test_a_rejected_document_stays_published_but_is_not_eligible(store, tmp_path):
    result = durable(store, tmp_path, acceptance=strict("absent")).execute(graph())

    assert result.acceptance.decision is AcceptanceDecision.REJECTED
    path = run_directory(tmp_path, result.execution_id) / "document.md"
    assert path.read_bytes() == result.rejected_output.encode("utf-8")
    current = current_artifact(store, result.execution_id)
    assert current.decision == result.acceptance
    assert current.reasons == ("not_accepted",) and not current.eligible


def test_a_run_without_a_strict_policy_publishes_without_a_decision(store, tmp_path):
    result = durable(store, tmp_path).execute(graph())

    assert result.acceptance is None and result.output.startswith(DOCUMENT_HEADING)
    current = current_artifact(store, result.execution_id)
    assert current.decision is None and current.integrity["status"] == "verified"
    # Nothing decided the document, so it is never eligible for delivery.
    assert current.reasons == ("no_decision",)


def test_runs_without_a_renderer_keep_their_recipe_and_publish_nothing(store, tmp_path):
    result = durable(store, tmp_path, renderer=None, acceptance=strict()).execute(graph())

    assert "renderer" not in store.load_run(result.execution_id)["config"]
    assert not (tmp_path / "artifacts").exists()
    assert current_artifact(store, result.execution_id) is None
    assert not result.output.startswith("#")
    assert not any(key.startswith("publication/") for key in operations(store, result.execution_id))


def test_a_local_only_renderer_records_its_identity_and_names_its_file(store, tmp_path):
    renderer = LocalOnly(PlainBrief, "plain-brief", "2", role="renderer")
    result = durable(store, tmp_path, renderer=renderer, acceptance=strict()).execute(graph())

    assert store.load_run(result.execution_id)["config"]["renderer"] == {
        "type": "local_only", "role": "renderer", "identity": "plain-brief", "version": "2"}
    path = run_directory(tmp_path, result.execution_id) / "brief.txt"
    assert result.output.startswith("BRIEF: Write the brief\n") and result.accepted
    assert path.read_bytes() == result.output.encode("utf-8")
    current = current_artifact(store, result.execution_id)
    assert current.publication.media_type == "text/plain; charset=utf-8" and current.eligible


# ---------------------------------------------------------------------------
# Damaged files lose delivery eligibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how,status,problem", [
    ("changed", "changed", "the published artifact 'document.md' no longer holds the bytes its "
                           "record bound"),
    ("missing", "missing", "the published artifact 'document.md' is missing"),
])
def test_a_damaged_document_is_not_eligible_on_inspect_or_resume(
    journal, tmp_path, capsys, how, status, problem,
):
    with SQLiteWorkflowStore(journal) as store:
        result = durable(store, tmp_path, acceptance=strict()).execute(graph())
    run_id, document = result.execution_id, result.output
    path = run_directory(tmp_path, run_id) / "document.md"
    damage(path, how)
    on_disk = listing(path.parent)

    with SQLiteWorkflowStore(journal, read_only=True) as store:
        current = current_artifact(store, run_id)
    # The decision still binds the published bytes; the file no longer holds them.
    assert current.decision.accepted and current.integrity["status"] == status
    assert current.reasons == (f"artifact_{status}",) and not current.eligible

    code, value = inspect(capsys, journal, run_id)
    assert code == EXIT_RUN_INDETERMINATE
    assert value["delivery"]["decision"] == "indeterminate"
    assert value["delivery"]["record"]["decision"] == "accepted"
    assert problem in value["delivery"]["reasons"]
    assert value["artifact"]["delivery_eligibility"] == {"eligible": False,
                                                         "reasons": [f"artifact_{status}"]}
    assert value["artifact"]["integrity"]["status"] == status
    assert value["next_action"]["action"] == "republish"
    # Inspection changes no file.
    assert listing(path.parent) == on_disk

    with SQLiteWorkflowStore(journal) as store:
        resumed = durable(store, tmp_path, acceptance=strict()).resume(run_id)
    assert resumed.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert problem in resumed.acceptance.reasons
    assert (resumed.output, resumed.rejected_output) == ("", document)
    assert listing(path.parent) == on_disk


def test_inspection_reports_an_intact_published_document(journal, tmp_path, capsys):
    with SQLiteWorkflowStore(journal) as store:
        result = durable(store, tmp_path, acceptance=strict()).execute(graph())
    before = journal.read_bytes()

    code, value = inspect(capsys, journal, result.execution_id)
    assert code == EXIT_OK and value["delivery"]["decision"] == "accepted"
    artifact = value["artifact"]
    assert artifact["delivery_eligibility"] == {"eligible": True, "reasons": []}
    assert artifact["decision"] == "accepted"
    assert artifact["publication"]["sha256"] == result.acceptance.output_sha256
    assert Path(artifact["path"]) == (run_directory(tmp_path, result.execution_id)
                                      / "document.md").resolve()
    assert value["next_action"]["action"] == "none"
    assert journal.read_bytes() == before

    assert main(["runs", "--journal", str(journal), "inspect", result.execution_id]) == EXIT_OK
    text = capsys.readouterr().out
    assert "(publication 0)" in text and "Integrity: verified" in text
    assert "Delivery eligibility: eligible" in text
    assert result.output not in text


def test_inspection_of_a_rejected_document_keeps_its_exit_code(journal, tmp_path, capsys):
    with SQLiteWorkflowStore(journal) as store:
        result = durable(store, tmp_path, acceptance=strict("absent")).execute(graph())

    code, value = inspect(capsys, journal, result.execution_id)
    assert code == EXIT_RUN_NOT_ACCEPTED and value["delivery"]["decision"] == "rejected"
    assert value["artifact"]["delivery_eligibility"]["reasons"] == ["not_accepted"]
    assert value["next_action"]["action"] == "review_output"


# ---------------------------------------------------------------------------
# Republishing
# ---------------------------------------------------------------------------


def test_republishing_never_clobbers(journal, tmp_path):
    with SQLiteWorkflowStore(journal) as store:
        swarm = durable(store, tmp_path, acceptance=strict())
        result = swarm.execute(graph())
        run_id, document = result.execution_id, result.output
        first = run_directory(tmp_path, run_id) / "document.md"
        damage(first, "changed")
        edited = first.read_bytes()

        record = swarm.republish(run_id)
        # The edited file keeps its bytes; the document goes to the next free name.
        assert first.read_bytes() == edited
        second = first.with_name("document-2.md")
        assert second.read_bytes() == document.encode("utf-8")
        assert (record.sequence, record.name, record.sha256) == (
            1, "document-2.md", result.acceptance.output_sha256)
        assert listing(first.parent) == [".smythe-run-owner.json", "document-2.md", "document.md"]

        # An intact current file is left as it is.
        assert swarm.republish(run_id) == record
        assert store.load_operation(run_id, "publication/2") is None

        # A missing current file is published again; the edited file is never replaced.
        second.unlink()
        third = swarm.republish(run_id)
        assert (third.sequence, third.name) == (2, "document-2.md")
        assert second.read_bytes() == document.encode("utf-8") and first.read_bytes() == edited
        assert {key: value for key, value in operations(store, run_id).items()
                if key.startswith("publication/")} == {
            f"publication/{n}": (PUBLICATION_OPERATION, "applied") for n in range(3)}

        # The run delivers again, under the decision it always had, with nothing validated.
        resumed = durable(store, tmp_path, acceptance=strict()).resume(run_id)
        assert resumed.accepted and resumed.output == document
        assert resumed.acceptance == result.acceptance and Mentions.checked == [document]
        assert store.audit(run_id)["ok"]


def test_the_accessor_returns_the_newest_publication_and_its_decision(store, tmp_path):
    swarm = durable(store, tmp_path, acceptance=strict())
    result = swarm.execute(graph())
    run_id = result.execution_id
    original = current_artifact(store, run_id).publication
    damage(original.path, "missing")
    record = swarm.republish(run_id)

    current = current_artifact(store, run_id)
    assert current.publication == record and record.sequence == original.sequence + 1
    assert current.decision == result.acceptance and current.eligible

    # The newest publication stays current when its file is damaged in turn;
    # the earlier record is never read in its place.
    damage(record.path, "changed")
    current = current_artifact(store, run_id)
    assert current.publication == record
    assert current.reasons == ("artifact_changed",) and not current.eligible


def test_republish_needs_a_completed_run_with_a_renderer(store, tmp_path):
    planned = durable(store, tmp_path, acceptance=strict()).plan(Task("Write the brief"))
    with pytest.raises(WorkflowStateError, match="Only a completed workflow"):
        durable(store, tmp_path, acceptance=strict()).republish(planned.run_ref["run_id"])

    plain = durable(store, tmp_path, renderer=None).execute(graph())
    with pytest.raises(WorkflowStateError, match="no renderer"):
        durable(store, tmp_path, renderer=None).republish(plain.execution_id)
    with pytest.raises(ValueError, match="needs a durable run_store"):
        Swarm(model="offline", provider=OfflineProvider()).republish(plain.execution_id)


# ---------------------------------------------------------------------------
# Interrupted publication
# ---------------------------------------------------------------------------


def test_a_process_lost_after_publishing_reuses_its_file(store, tmp_path, monkeypatch):
    planned = durable(store, tmp_path, acceptance=strict()).plan(Task("Write the brief"))
    run_id = planned.run_ref["run_id"]
    restore = stop_at(monkeypatch, store, "complete_operation",
                      lambda lease, key, *args, **kwargs: key.startswith("publication/"))
    with pytest.raises(Crash):
        durable(store, tmp_path, acceptance=strict()).execute(planned)
    restore()
    directory = run_directory(tmp_path, run_id)
    assert operations(store, run_id)["publication/0"] == (PUBLICATION_OPERATION, "started")
    published = (directory / "document.md").read_bytes()

    result = durable(store, tmp_path, acceptance=strict()).resume(run_id)
    # The file the lost process published is the one recorded; no copy is made.
    assert listing(directory) == [".smythe-run-owner.json", "document.md"]
    assert result.accepted and result.output.encode("utf-8") == published
    assert current_artifact(store, run_id).publication.name == "document.md"
    assert operations(store, run_id)["publication/0"] == (PUBLICATION_OPERATION, "applied")


def test_a_process_lost_after_publication_decides_on_the_published_bytes(store, tmp_path, monkeypatch):
    planned = durable(store, tmp_path, acceptance=strict()).plan(Task("Write the brief"))
    run_id = planned.run_ref["run_id"]
    restore = stop_at(monkeypatch, store, "complete_operation",
                      lambda lease, key, *args, **kwargs: key.startswith("acceptance/"))
    with pytest.raises(Crash):
        durable(store, tmp_path, acceptance=strict()).execute(planned)
    restore()
    assert operations(store, run_id)["publication/0"] == (PUBLICATION_OPERATION, "completed")
    path = run_directory(tmp_path, run_id) / "document.md"
    Mentions.checked = []

    result = durable(store, tmp_path, acceptance=strict()).resume(run_id)
    assert Mentions.checked == [path.read_text(encoding="utf-8")] == [result.output]
    assert result.accepted and listing(path.parent) == [".smythe-run-owner.json", "document.md"]
    assert store.load_operation(run_id, "publication/1") is None


@pytest.mark.parametrize("how,status,problem", [
    ("changed", "changed", "the published artifact 'document.md' no longer holds the bytes its "
                           "record bound"),
    ("missing", "missing", "the published artifact 'document.md' is missing"),
])
@pytest.mark.parametrize("lost", ["before_the_decision", "before_the_completed_checkpoint"])
def test_a_run_resumed_after_publication_never_delivers_a_damaged_document(
    store, tmp_path, monkeypatch, how, status, problem, lost,
):
    planned = durable(store, tmp_path, acceptance=strict()).plan(Task("Write the brief"))
    run_id = planned.run_ref["run_id"]
    if lost == "before_the_decision":
        restore = stop_at(monkeypatch, store, "complete_operation",
                          lambda lease, key, *args, **kwargs: key.startswith("acceptance/"))
    else:
        restore = stop_at(monkeypatch, store, "save_checkpoint",
                          lambda lease, revision, state, **kwargs: state["status"] == "completed")
    with pytest.raises(Crash):
        durable(store, tmp_path, acceptance=strict()).execute(planned)
    restore()
    assert operations(store, run_id)["publication/0"] == (PUBLICATION_OPERATION, "completed")
    path = run_directory(tmp_path, run_id) / "document.md"
    document = path.read_text(encoding="utf-8")
    damage(path, how)
    on_disk = listing(path.parent)

    # The run completes under its decision, but the file no longer holds the
    # decided bytes, so nothing is delivered: the result agrees with the accessor.
    resumed = durable(store, tmp_path, acceptance=strict()).resume(run_id)
    assert resumed.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert problem in resumed.acceptance.reasons
    assert (resumed.output, resumed.rejected_output) == ("", document)
    current = current_artifact(store, run_id)
    assert current.decision.accepted and current.reasons == (f"artifact_{status}",)
    assert store.get_checkpoint(run_id)["checkpoint"]["status"] == "completed"
    # Resuming again gives the same answer, and no file is written or replaced.
    again = durable(store, tmp_path, acceptance=strict()).resume(run_id)
    assert again.acceptance == resumed.acceptance and again.output == ""
    assert listing(path.parent) == on_disk

    # Republishing restores delivery under the decision the run made.
    swarm = durable(store, tmp_path, acceptance=strict())
    assert swarm.republish(run_id).sha256 == current.decision.output_sha256
    delivered = swarm.resume(run_id)
    assert delivered.accepted and delivered.output == document
    assert current_artifact(store, run_id).eligible


def test_checking_the_file_after_completion_never_fails_the_completed_run(
    store, tmp_path, monkeypatch,
):
    def unreadable_journal(*args):
        raise RuntimeError("journal read failed")

    monkeypatch.setattr(workflow, "current_artifact", unreadable_journal)
    with pytest.raises(RuntimeError, match="journal read failed"):
        durable(store, tmp_path, acceptance=strict()).execute(graph())
    monkeypatch.undo()

    (run,) = store.list_runs()
    assert store.get_checkpoint(run["run_id"])["checkpoint"]["status"] == "completed"
    assert durable(store, tmp_path, acceptance=strict()).resume(run["run_id"]).accepted


@pytest.mark.parametrize("file_intact", [True, False])
def test_a_replayed_publication_never_decides_on_other_bytes(
    store, tmp_path, monkeypatch, file_intact,
):
    planned = durable(store, tmp_path, acceptance=strict()).plan(Task("Write the brief"))
    run_id = planned.run_ref["run_id"]
    restore = stop_at(monkeypatch, store, "complete_operation",
                      lambda lease, key, *args, **kwargs: key.startswith("acceptance/"))
    with pytest.raises(Crash):
        durable(store, tmp_path, acceptance=strict()).execute(planned)
    restore()
    path = run_directory(tmp_path, run_id) / "document.md"
    published = path.read_text(encoding="utf-8")
    if not file_intact:
        path.unlink()
    # A renderer that no longer makes the recorded bytes, contrary to its contract.
    monkeypatch.setattr(MarkdownRenderer, "render", lambda self, output, task: "other bytes\n")

    if file_intact:
        result = durable(store, tmp_path, acceptance=strict()).resume(run_id)
        # The decision is made on the published bytes, read back from the file.
        assert result.output == published and Mentions.checked[-1] == published
    else:
        with pytest.raises(WorkflowStateError, match="no longer makes its bytes"):
            durable(store, tmp_path, acceptance=strict()).resume(run_id)
        assert store.get_checkpoint(run_id)["checkpoint"]["status"] == "failed"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AsyncRenderer:
    async def render(self, output, task):
        return output


class BytesRenderer:
    def render(self, output, task):
        return output.encode("utf-8")


class DeviceName:
    filename = "con.md"

    def render(self, output, task):
        return output


def test_renderer_configuration_is_checked(store, tmp_path):
    with pytest.raises(WorkflowBindingError, match="renderer requires a durable run_store"):
        Swarm(model="offline", provider=OfflineProvider(), renderer=MarkdownRenderer())
    with pytest.raises(WorkflowBindingError, match="artifact_dir"):
        durable(store, tmp_path, artifact_dir=None)
    with pytest.raises(WorkflowBindingError, match="add a final validator"):
        durable(store, tmp_path, acceptance=AcceptancePolicy(mode="strict")).execute(graph())
    with pytest.raises(WorkflowBindingError, match="renderer must be MarkdownRenderer"):
        durable(store, tmp_path, renderer=object()).execute(graph())
    with pytest.raises(WorkflowBindingError, match="not a LocalOnly synthesizer"):
        durable(store, tmp_path, renderer=LocalOnly(PlainBrief, "brief", "1", role="synthesizer")
                ).execute(graph())
    for component, message in ((AsyncRenderer, "must be synchronous"),
                               (DeviceName, "simple name with an extension")):
        with pytest.raises(WorkflowBindingError, match=message):
            durable(store, tmp_path, renderer=LocalOnly(component, "custom", "1", role="renderer")
                    ).execute(graph())
    # Every refusal came before a run was created or a call was made.
    assert store.list_runs() == []


def test_a_renderer_that_returns_bytes_fails_the_run_before_publishing(store, tmp_path):
    renderer = LocalOnly(BytesRenderer, "bytes", "1", role="renderer")
    with pytest.raises(WorkflowStateError, match="Renderer must return text, not bytes"):
        durable(store, tmp_path, renderer=renderer).execute(graph())
    (run,) = store.list_runs()
    assert store.get_checkpoint(run["run_id"])["checkpoint"]["status"] == "failed"
    assert listing(run_directory(tmp_path, run["run_id"])) == [".smythe-run-owner.json"]


def test_resume_needs_the_renderer_the_run_started_with(store, tmp_path):
    planned = durable(store, tmp_path).plan(Task("Write the brief"))
    run_id = planned.run_ref["run_id"]
    with pytest.raises(WorkflowConflictError, match="renderer differs"):
        durable(store, tmp_path, renderer=None).resume(run_id)
    other = LocalOnly(PlainBrief, "plain-brief", "2", role="renderer")
    with pytest.raises(WorkflowConflictError, match="renderer differs"):
        durable(store, tmp_path, renderer=other).resume(run_id)
    assert durable(store, tmp_path).resume(run_id).output.startswith(DOCUMENT_HEADING)


def test_a_directory_with_files_the_run_does_not_own_is_refused_before_any_call(store, tmp_path):
    planned = durable(store, tmp_path).plan(Task("Write the brief"))
    run_id = planned.run_ref["run_id"]
    stray = run_directory(tmp_path, run_id) / "notes.txt"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"not the run's")

    with pytest.raises(WorkflowStateError, match="unowned files"):
        durable(store, tmp_path).execute(planned)
    assert store.inspect_run(run_id)["call_count"] == 0
    assert stray.read_bytes() == b"not the run's"
