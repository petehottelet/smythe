"""Read-only HTML publication cannot replace an optimization journal or output."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3

import pytest

from smythe.cli import EXIT_INVALID_INPUT, EXIT_OK, EXIT_OPTIMIZE_STATE, main
from smythe.optimize.ledger import ExperimentLedger, ExperimentLedgerError
from test_optimize_inspection import CAMPAIGN, logical_dump, make_ledger, mutate


def invoke(path, out=None, *, campaign=CAMPAIGN, as_json=True):
    args = ["optimize", "inspect", campaign, "--ledger", str(path)]
    if as_json:
        args.append("--json")
    if out is not None:
        args.extend(("--out", str(out)))
    return main(args)


def link(source, destination, *, hard=False):
    try:
        if hard:
            os.link(source, destination)
        else:
            os.symlink(source, destination)
    except OSError as exc:
        pytest.skip(f"filesystem cannot create the test link: {exc}")


@pytest.mark.parametrize("as_json", [False, True])
def test_export_preserves_exact_existing_stdout_and_all_ledger_evidence(tmp_path, monkeypatch, capsys, as_json):
    from smythe.optimize.engine import OptimizationRunner
    import smythe.optimize.statistics as statistics
    import smythe.cli as cli

    path, out = tmp_path / "evidence.db", tmp_path / "reports" / "report.html"
    make_ledger(path, states=("completed", "completed"), decision=True)
    before = path.read_bytes(), logical_dump(path)

    def forbidden(*_, **__):
        pytest.fail("inspection reran campaign/provider/statistical work")

    monkeypatch.setattr(OptimizationRunner, "run", forbidden)
    monkeypatch.setattr(statistics, "assess_promotion", forbidden)
    monkeypatch.setattr(cli, "simulate_concurrency", forbidden)
    monkeypatch.setattr(cli.ProviderPool, "__init__", forbidden)
    assert invoke(path, as_json=as_json) == EXIT_OK
    original = capsys.readouterr()
    assert invoke(path, out, as_json=as_json) == EXIT_OK
    exported = capsys.readouterr()
    assert exported.out == original.out
    assert exported.err == original.err
    html = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in html.lower() and CAMPAIGN in html
    assert (path.read_bytes(), logical_dump(path)) == before
    assert not list(out.parent.glob(".*.tmp"))


def test_whitespace_campaign_preserves_stdout_while_report_identity_is_canonical(tmp_path, capsys):
    path, out = tmp_path / "evidence.db", tmp_path / "report.html"
    make_ledger(path, states=())
    assert invoke(path, campaign=" " + CAMPAIGN + " ") == EXIT_OK
    original = capsys.readouterr().out
    assert invoke(path, out, campaign=" " + CAMPAIGN + " ") == EXIT_OK
    assert capsys.readouterr().out == original


@pytest.mark.parametrize("kind", ["missing", "foreign", "wal", "shm"])
def test_unreadable_or_active_ledger_creates_no_output_parent(tmp_path, capsys, kind):
    path, out = tmp_path / "source" / "evidence.db", tmp_path / "report-parent" / "report.html"
    if kind == "foreign":
        path.parent.mkdir()
        with closing(sqlite3.connect(path)) as db:
            db.execute("CREATE TABLE foreign_data(value TEXT)")
    elif kind in {"wal", "shm"}:
        make_ledger(path, states=())
        Path(str(path) + "-" + kind).write_bytes(b"retained sidecar")
    before = path.read_bytes() if path.exists() else None
    assert invoke(path, out) == EXIT_OPTIMIZE_STATE
    assert not json.loads(capsys.readouterr().out)["ok"]
    assert not out.parent.exists()
    assert (path.read_bytes() if path.exists() else None) == before


def test_real_live_wal_writer_is_not_checkpointed_or_closed_by_export(tmp_path, capsys):
    path, out = tmp_path / "live.db", tmp_path / "report.html"
    make_ledger(path, states=())
    with ExperimentLedger(path) as writer:
        before = path.read_bytes(), Path(str(path) + "-wal").read_bytes()
        assert invoke(path, out) == EXIT_OPTIMIZE_STATE
        capsys.readouterr()
        assert not writer._closed
        assert (path.read_bytes(), Path(str(path) + "-wal").read_bytes()) == before
    assert not out.exists()


@pytest.mark.parametrize("kind", ["database", "wal", "shm", "journal", "hardlink", "symlink",
                                  "dangling-sidecar-target", "dangling-output-sidecar", "ledger-alias-sidecar"])
def test_protected_paths_and_resolved_sidecar_aliases_are_refused_before_open(tmp_path, monkeypatch, capsys, kind):
    path = tmp_path / "evidence.db"
    make_ledger(path, states=())
    ledger_arg = path
    if kind == "database":
        out = path
    elif kind in {"wal", "shm", "journal"}:
        out = Path(str(path) + "-" + kind)
    elif kind in {"hardlink", "symlink"}:
        out = tmp_path / "alias.html"
        link(path, out, hard=kind == "hardlink")
    elif kind == "dangling-sidecar-target":
        out = tmp_path / "uncreated.html"
        link(out, Path(str(path) + "-wal"))
    elif kind == "dangling-output-sidecar":
        out = tmp_path / "alias.html"
        link(Path(str(path) + "-journal"), out)
    else:
        ledger_arg = tmp_path / "source-alias.db"
        link(path, ledger_arg)
        out = Path(str(ledger_arg) + "-journal")
    before = path.read_bytes()

    def forbidden(*_, **__):
        pytest.fail("protected output opened the ledger")

    monkeypatch.setattr(ExperimentLedger, "__init__", forbidden)
    assert invoke(ledger_arg, out) == EXIT_INVALID_INPUT
    assert not json.loads(capsys.readouterr().out)["ok"]
    assert path.read_bytes() == before
    if kind == "dangling-sidecar-target":
        assert not out.exists()


def test_existing_output_even_identical_is_never_replaced(tmp_path, capsys):
    path, out = tmp_path / "source.db", tmp_path / "report.html"
    make_ledger(path, states=())
    assert invoke(path, out) == EXIT_OK
    capsys.readouterr()
    before = path.read_bytes(), out.read_bytes()
    assert invoke(path, out) == EXIT_INVALID_INPUT
    capsys.readouterr()
    assert (path.read_bytes(), out.read_bytes()) == before


def test_close_time_failure_prevents_rendering_and_all_output_writes(tmp_path, monkeypatch, capsys):
    import smythe.optimize.report as report_module

    path, out = tmp_path / "source.db", tmp_path / "uncreated" / "report.html"
    make_ledger(path, states=())
    before = path.read_bytes()
    original = ExperimentLedger.close

    def fail_close(self):
        original(self)
        raise ExperimentLedgerError("close-time quiescence failed")

    monkeypatch.setattr(ExperimentLedger, "close", fail_close)
    monkeypatch.setattr(report_module, "render_optimization_report", lambda _: pytest.fail("rendered before close"))
    assert invoke(path, out) == EXIT_OPTIMIZE_STATE
    capsys.readouterr()
    assert not out.parent.exists() and path.read_bytes() == before


@pytest.mark.parametrize("kind", ["bad-decision", "bad-contract", "bad-trial-null", "bad-trial-list"])
def test_stored_validation_errors_report_exit_eight_and_preserve_source(tmp_path, capsys, kind):
    path, out = tmp_path / "source.db", tmp_path / "new" / "report.html"
    make_ledger(path, states=("completed", "completed"), decision=True)
    if kind == "bad-decision":
        mutate(path, "UPDATE promotion_decisions SET payload_json='[]'")
    elif kind == "bad-contract":
        mutate(path, "UPDATE campaigns SET contract_json='[]'")
    else:
        payload = "null" if kind == "bad-trial-null" else "[]"
        mutate(path, "UPDATE trial_events SET payload_json=? WHERE event_type='completed'", (payload,))
    before = path.read_bytes()
    assert invoke(path, out) == EXIT_OPTIMIZE_STATE
    error = json.loads(capsys.readouterr().out)
    assert error["error"]["type"] == "ExperimentLedgerError"
    assert path.read_bytes() == before and not out.parent.exists()


def test_concurrent_output_creator_wins_without_replacement(tmp_path, monkeypatch, capsys):
    import smythe.jobs.artifact_io as artifact_io

    path, out = tmp_path / "source.db", tmp_path / "report.html"
    make_ledger(path, states=())
    before = path.read_bytes()
    original = os.link

    def competitor(source, target):
        with open(target, "xb") as stream:
            stream.write(b"another writer won")
        return original(source, target)

    monkeypatch.setattr(artifact_io.os, "link", competitor)
    assert invoke(path, out) == EXIT_OPTIMIZE_STATE
    capsys.readouterr()
    assert out.read_bytes() == b"another writer won"
    assert path.read_bytes() == before and not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("failure", ["file-sync", "directory-sync", "link"])
def test_publication_failure_preserves_evidence_and_cleans_only_owned_temp(tmp_path, monkeypatch, capsys, failure):
    import smythe.jobs.artifact_io as artifact_io

    path, out = tmp_path / "source.db", tmp_path / "report.html"
    make_ledger(path, states=())
    before = path.read_bytes()
    unrelated = tmp_path / ".unrelated.tmp"
    unrelated.write_bytes(b"keep")

    def fail(*_):
        raise OSError("injected publication failure")

    if failure == "file-sync":
        monkeypatch.setattr(artifact_io.os, "fsync", fail)
    elif failure == "directory-sync":
        monkeypatch.setattr(artifact_io, "_fsync_directory", fail)
    else:
        monkeypatch.setattr(artifact_io.os, "link", fail)
    assert invoke(path, out) == EXIT_OPTIMIZE_STATE
    capsys.readouterr()
    assert path.read_bytes() == before
    assert unrelated.read_bytes() == b"keep"
    assert list(tmp_path.glob(".*.tmp")) == [unrelated]
    if failure == "directory-sync":
        assert out.read_bytes().lower().startswith(b"<!doctype html>")
    else:
        assert not out.exists()


def test_oversize_output_is_refused_before_creating_directories(tmp_path, monkeypatch, capsys):
    import smythe.optimize.report as report_module
    from smythe.jobs.artifact_io import MAX_ARTIFACT_BYTES

    path, out = tmp_path / "source.db", tmp_path / "new" / "report.html"
    make_ledger(path, states=())
    before = path.read_bytes()
    monkeypatch.setattr(report_module, "render_optimization_report", lambda _: "x" * (MAX_ARTIFACT_BYTES + 1))
    assert invoke(path, out) == EXIT_OPTIMIZE_STATE
    capsys.readouterr()
    assert not out.parent.exists() and path.read_bytes() == before
