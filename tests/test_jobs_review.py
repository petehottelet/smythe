"""Jobs review packets list only what needs an operator's decision, and change nothing."""

from __future__ import annotations

import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

import smythe.provider as provider_module
from fakes import Crash, crash_when
import smythe.cli as cli
from smythe.cli import (
    EXIT_INTERRUPTED, EXIT_INVALID_INPUT, EXIT_JOB_FAILED, EXIT_JOB_STATE, EXIT_OK, build_parser,
    main,
)
from smythe.jobs import JobManifestV1, delivery, make_approval, operator, preflight_job
from smythe.jobs.inspection import REVIEW_BLOCKS, REVIEW_KINDS, REVIEW_VERSION, review_job
from smythe.jobs.loading import load_manifest
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import SQLiteRunStore
from smythe.provider import Artifact
from smythe.reuse import REASONS
from test_jobs_cli import _jpeg_bytes, _manifest
from test_jobs_lease_fencing import Clock, make_legacy
from test_jobs_operator import fake_launch
from test_jobs_runner import (
    PNG_1X1, _FixedPool, _InvalidCostProvider, _PaletteValidator, _PricedArtifactProvider,
    _priced_manifest, _validated_run,
)

ORGANIZATION = "org-" + "A1b2" * 6  # built here, so no tracked file holds an identifier


def output(capsys):
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    return json.loads(captured.out, parse_constant=lambda value: pytest.fail(f"Nonfinite JSON: {value}"))


def reviewed(capsys, path, run_id, *options):
    assert main(["jobs", "review", run_id, *options, "--store", str(path), "--json"]) == EXIT_OK
    payload = output(capsys)
    assert payload["ok"] is True and payload["command"] == "review"
    return payload["review"]


def commands_parse(packet, *local):
    """Every suggested command is a jobs command line that parses, with no store or local path."""
    for item in packet["items"]:
        for argv in item["resolve_with"]:
            args = build_parser().parse_args(argv)
            assert args.root_command == "jobs" and args.jobs_command == argv[1]
            assert args.run_id == packet["run_id"]
            assert "--store" not in argv
            assert not any(str(path) in token for path in local for token in argv)
    return [argv for item in packet["items"] for argv in item["resolve_with"]]


def holds_no_path(value, *paths):
    """No text anywhere in *value* holds one of *paths*, in either slash form."""
    if isinstance(value, dict):
        return all(holds_no_path(item, *paths) for pair in value.items() for item in pair)
    if isinstance(value, list):
        return all(holds_no_path(item, *paths) for item in value)
    return not isinstance(value, str) or not any(
        form.lower() in value.lower() for path in paths for form in (str(path), path.as_posix()))


def fill(argv, reason="operator decision"):
    return [reason if token == "REASON" else token for token in argv]


def cli_job(tmp_path, capsys, monkeypatch=None, *, rejected=False, attempts=1, count=1):
    """Run the offline manifest through the CLI; a JPEG where it asks for a PNG is rejected."""
    if rejected:
        monkeypatch.setattr(provider_module, "_OFFLINE_PNG", _jpeg_bytes())
    manifest = _manifest(tmp_path, attempts=attempts)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["operations"][0]["count"] = count
    manifest.write_text(json.dumps(data), encoding="utf-8")
    path = tmp_path / "jobs.sqlite3"
    assert main(["jobs", "plan", str(manifest), "--json"]) == EXIT_OK
    token = output(capsys)["plan"]["approval"]["token"]
    code = main(["jobs", "run", str(manifest), "--approve", token, "--store", str(path), "--json"])
    assert code == (EXIT_JOB_FAILED if rejected else EXIT_OK)
    return SimpleNamespace(path=path, run_id=output(capsys)["run"]["run_id"], manifest=manifest)


def kinds(packet):
    return [item["kind"] for item in packet["items"]]


def test_a_clean_completed_run_needs_no_decision(tmp_path, capsys):
    job = cli_job(tmp_path, capsys)
    packet = reviewed(capsys, job.path, job.run_id)

    assert packet["review_version"] == REVIEW_VERSION == 1
    assert packet["run_id"] == job.run_id and packet["status"] == "completed"
    assert packet["items"] == [] and packet["totals"] == dict.fromkeys(REVIEW_KINDS, 0)
    assert packet["page"] == {"limit": 50, "offset": 0, "total": 1, "returned": 1,
                              "has_more": False}
    assert packet["operation_counts"] == {"succeeded": 1}
    assert packet["artifact_integrity"]["counts"] == {"verified": 1}
    # The packet carries no local paths: no manifest root, store or artifact location.
    assert "manifest_root" not in packet
    assert holds_no_path(packet, tmp_path)

    assert main(["jobs", "review", job.run_id, "--store", str(job.path)]) == EXIT_OK
    human = capsys.readouterr()
    assert human.err == "" and "Nothing on this page needs a decision." in human.out
    assert "To run a command" not in human.out


def priced_plan(tmp_path, *, attempts=2):
    """One operation at a $0.10 call ceiling, with room for every attempt."""
    data = _priced_manifest().to_dict()
    data["execution"].update(max_attempts=attempts, max_budget_usd=f"{attempts * 0.10:.2f}")
    return preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def unknown_after_publication(tmp_path, monkeypatch):
    """Stop the process after a priced response is published, before its call settles."""
    store = SQLiteRunStore(tmp_path / "jobs.db")
    plan = priced_plan(tmp_path)
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")])
    runner = JobRunner(store, provider_pool=_FixedPool(provider))
    stop = crash_when(monkeypatch, store, "complete_call")
    with pytest.raises(Crash):
        asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path, run_id="run"))
    stop.restore()
    return store, runner


def test_an_interrupted_worker_and_then_an_unknown_outcome_name_their_decisions(
    tmp_path, monkeypatch,
):
    store, runner = unknown_after_publication(tmp_path, monkeypatch)
    with store, SQLiteRunStore(store.path, read_only=True) as reader:
        # The process stopped after dispatch: the run is still recorded as running.
        interrupted = review_job(reader, "run")
        assert interrupted["status"] == "running"
        [stopped] = interrupted["items"]
        assert (stopped["kind"], stopped["blocks"]) == ("lease_expired", "completion")
        assert stopped["resolve_with"] == [["jobs", "resume", "run"]]
        commands_parse(interrupted, tmp_path)

        resumed = asyncio.run(runner.resume("run"))
        assert resumed["counts"] == {"unknown_outcome": 1}
        packet = review_job(reader, "run")
        [item] = packet["items"]
        [call] = store._call_rows("run")
        [attempt] = resumed["attempts"]
        assert (item["kind"], item["blocks"]) == ("unknown_outcome", "completion")
        assert item["subject"] == {
            "operation_key": "glyph[0]", "operation_id": resumed["operations"][0]["operation_id"],
            "attempt_id": attempt["attempt_id"], "attempt_number": 1, "call_id": call["call_id"]}
        assert item["observed"]["exposure_microusd"] == call["exposure_microusd"] == 100_000
        assert item["observed"]["status"] == "unknown_outcome"
        assert item["observed"]["excerpt"] == "process stopped after dispatch"
        assert item["resolve_with"] == [
            ["jobs", "adopt", "run", "glyph[0]", "--reason", "REASON"],
            ["jobs", "reroll", "run", "glyph[0]", "--acknowledge-unknown", "--reason", "REASON"],
        ]
        assert "Investigate the provider account" in item["decision_needed"]
        assert packet["totals"]["unknown_outcome"] == 1 and sum(packet["totals"].values()) == 1
        commands_parse(packet, tmp_path)
        assert review_job(reader, "run")["items"] == packet["items"]

        # The acknowledged reroll settles it; nothing is left to decide.
        asyncio.run(runner.reroll("run", ["glyph[0]"], reason="provider shows no request",
                                  acknowledge_unknown=True))
        assert review_job(reader, "run")["items"] == []


def test_invalid_provider_accounting_holds_back_admission(tmp_path):
    plan = priced_plan(tmp_path)
    with SQLiteRunStore(tmp_path / "jobs.db") as store:
        runner = JobRunner(store, provider_pool=_FixedPool(_InvalidCostProvider()))
        asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path, run_id="run"))
        [item] = review_job(store, "run")["items"]
    assert (item["kind"], item["blocks"]) == ("invalid_accounting", "admission")
    assert item["observed"]["error_class"] == "invalid_provider_accounting"
    assert item["observed"]["excerpt"].startswith("invalid provider accounting: cost_usd")
    assert item["observed"]["exposure_microusd"] == 100_000
    # Adoption does not settle an invalid charge; only the acknowledged reroll is offered.
    assert item["resolve_with"] == [
        ["jobs", "reroll", "run", "glyph[0]", "--acknowledge-unknown", "--reason", "REASON"]]


def test_rejected_operations_offer_a_reroll_until_no_attempt_is_left(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2)
    first = reviewed(capsys, job.path, job.run_id)
    [item] = first["items"]
    assert (item["kind"], item["blocks"]) == ("rejected", "completion")
    assert item["observed"]["attempts_remaining"] == 1
    assert "expected image/png" in item["observed"]["excerpt"]
    assert item["resolve_with"] == [["jobs", "reroll", job.run_id, "glyph[0]", "--reason", "REASON"]]
    [argv] = commands_parse(first, tmp_path)
    # Item IDs are stable while nothing changes.
    assert reviewed(capsys, job.path, job.run_id)["items"] == first["items"]

    assert main([*fill(argv), "--store", str(job.path), "--json"]) == EXIT_JOB_FAILED
    output(capsys)
    second = reviewed(capsys, job.path, job.run_id)
    [exhausted] = second["items"]
    assert exhausted["kind"] == "attempts_exhausted" and exhausted["resolve_with"] == []
    assert exhausted["item_id"] != item["item_id"]
    assert exhausted["subject"]["attempt_number"] == 2
    assert "new plan and approval" in exhausted["decision_needed"]
    assert second["totals"]["attempts_exhausted"] == 1 and second["totals"]["rejected"] == 0


def test_a_stop_request_asks_for_a_resume(tmp_path, capsys):
    manifest, root = load_manifest(_manifest(tmp_path))
    plan = preflight_job(manifest, manifest_root=root)
    path = tmp_path / "jobs.sqlite3"
    with SQLiteRunStore(path) as store:
        store.create_run(plan, make_approval(plan), manifest_root=root, run_id="idle")
    assert main(["jobs", "stop", "idle", "--timeout-s", "0", "--reason", "check the palette",
                 "--store", str(path), "--json"]) == EXIT_OK
    output(capsys)
    packet = reviewed(capsys, path, "idle")
    [item] = packet["items"]
    assert (item["kind"], item["blocks"]) == ("pause_requested", "admission")
    assert item["observed"]["excerpt"] == "check the palette"
    assert item["observed"]["pause_generation"] == 1
    assert item["resolve_with"] == [["jobs", "resume", "idle"]]
    commands_parse(packet, tmp_path)


def test_an_item_id_changes_with_its_recorded_state_alone(tmp_path):
    manifest, root = load_manifest(_manifest(tmp_path))
    plan = preflight_job(manifest, manifest_root=root)
    with SQLiteRunStore(tmp_path / "jobs.sqlite3") as store:
        store.create_run(plan, make_approval(plan), manifest_root=root, run_id="idle")
        store.request_pause("idle", reason="check the palette")
        [first] = review_job(store, "idle")["items"]
        assert review_job(store, "idle")["items"] == [first]
        # A second stop request, with the same reason: the same kind of item about
        # the same run, whose recorded pause generation has moved on.
        store.request_pause("idle", reason="check the palette")
        [second] = review_job(store, "idle")["items"]
    assert (second["kind"], second["subject"]) == (first["kind"], first["subject"])
    assert second["observed"]["status"] == first["observed"]["status"]
    assert (first["observed"]["pause_generation"], second["observed"]["pause_generation"]) == (1, 2)
    assert second["item_id"] != first["item_id"]


def test_a_budget_overrun_comes_first_and_admits_no_reroll(tmp_path):
    plan = priced_plan(tmp_path)
    # A response without an artifact is rejected at its reported cost, past the call's ceiling.
    provider = _PricedArtifactProvider([], cost_usd=0.25)
    with SQLiteRunStore(tmp_path / "jobs.db") as store:
        runner = JobRunner(store, provider_pool=_FixedPool(provider))
        result = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path,
                                          run_id="run"))
        packet = review_job(store, "run")
    assert result["status"] == "budget_overrun"
    # What holds back admission comes before what holds back completion.
    assert kinds(packet) == ["budget_overrun", "rejected"]
    overrun, rejected = packet["items"]
    assert [REVIEW_BLOCKS.index(item["blocks"]) for item in packet["items"]] == [0, 1]
    assert overrun["blocks"] == "admission" and overrun["resolve_with"] == []
    assert overrun["observed"]["confirmed_microusd"] == 250_000
    assert overrun["observed"]["approved_microusd"] == 200_000
    assert rejected["observed"]["attempts_remaining"] == 1
    assert rejected["resolve_with"] == [] and "budget overran" in rejected["decision_needed"]


@pytest.mark.parametrize("damage", ["deleted", "modified"])
def test_a_damaged_accepted_file_asks_for_a_repair_and_keeps_its_acceptance(
    tmp_path, capsys, damage,
):
    job = cli_job(tmp_path, capsys, attempts=2)
    [published] = (tmp_path / "outputs").rglob("*.png")
    if damage == "deleted":
        published.unlink()
    else:
        published.write_bytes(published.read_bytes() + b"tampered")
    with SQLiteRunStore(job.path, read_only=True) as reader:
        recorded = reader.snapshot(job.run_id)["artifacts"]
    before = job.path.read_bytes()

    packet = reviewed(capsys, job.path, job.run_id)
    [item] = packet["items"]
    status = "missing" if damage == "deleted" else "changed"
    assert (item["kind"], item["blocks"]) == ("artifact_integrity", "delivery")
    assert item["reasons"] == [f"artifact_{status}"] and set(item["reasons"]) <= set(REASONS)
    assert [artifact["status"] for artifact in item["observed"]["artifacts"]] == [status]
    assert item["observed"]["artifacts"][0]["relative_path"] == recorded[0]["relative_path"]
    assert item["resolve_with"] == [
        ["jobs", "reroll", job.run_id, "glyph[0]", "--repair-integrity", "--reason", "REASON"]]
    # The review records nothing: the acceptance and the database bytes are unchanged.
    assert job.path.read_bytes() == before
    with SQLiteRunStore(job.path, read_only=True) as reader:
        assert reader.snapshot(job.run_id)["artifacts"] == recorded
    assert [row["accepted"] for row in recorded] == [1]

    [argv] = commands_parse(packet, tmp_path)
    assert main([*fill(argv), "--store", str(job.path), "--json"]) == EXIT_OK
    output(capsys)
    assert reviewed(capsys, job.path, job.run_id)["items"] == []


def test_an_unsafe_accepted_file_needs_a_person(tmp_path, capsys):
    job = cli_job(tmp_path, capsys, attempts=2)
    [published] = (tmp_path / "outputs").rglob("*.png")
    published.unlink()
    published.mkdir()
    [item] = reviewed(capsys, job.path, job.run_id)["items"]
    assert item["kind"] == "artifact_integrity" and item["reasons"] == ["artifact_unsafe"]
    assert item["resolve_with"] == [] and "Restore the file" in item["decision_needed"]


def test_a_changed_builtin_check_asks_for_a_recheck_the_cli_can_run(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys)
    monkeypatch.setattr(delivery, "ARTIFACT_SPEC_VALIDATOR_VERSION", "2")
    packet = reviewed(capsys, job.path, job.run_id)
    [item] = packet["items"]
    assert (item["kind"], item["blocks"]) == ("delivery_ineligible", "delivery")
    assert item["observed"]["eligibility_reasons"] == ["validator_changed"]
    assert item["reasons"] == ["validator_set_changed"] and set(item["reasons"]) <= set(REASONS)
    [argv] = commands_parse(packet, tmp_path)
    assert argv == ["jobs", "recheck", job.run_id, "glyph[0]", "--reason", "REASON"]

    assert main([*fill(argv), "--store", str(job.path), "--json"]) == EXIT_OK
    output(capsys)
    assert reviewed(capsys, job.path, job.run_id)["items"] == []


def test_a_changed_registered_validator_points_to_the_python_recheck(tmp_path):
    with SQLiteRunStore(tmp_path / "jobs.db") as store:
        _, result, _ = _validated_run(tmp_path, store, _PaletteValidator())
        assert review_job(store, result["run_id"], validators=[_PaletteValidator()])["items"] == []
        packet = review_job(store, result["run_id"], validators=[_PaletteValidator(version="2")])
    [item] = packet["items"]
    assert item["kind"] == "delivery_ineligible" and item["reasons"] == ["validator_set_changed"]
    assert item["resolve_with"] == [] and "JobRunner" in item["decision_needed"]


def test_a_failed_detached_worker_is_reported_without_local_paths(tmp_path, capsys, monkeypatch):
    manifest, root = load_manifest(_manifest(tmp_path))
    plan = preflight_job(manifest, manifest_root=root)
    path = tmp_path / "store" / "jobs.db"
    with SQLiteRunStore(path) as store:
        store.create_run(plan, make_approval(plan), manifest_root=root, run_id="run")
        fake_launch(monkeypatch, store, behavior="failure")
        with pytest.raises(operator.WorkerStartupError):
            operator.launch_worker(store.path, "run")
        worker = operator.read_worker_state(store.path, "run")
        directory = Path(worker["receipt_path"])
        request = operator._read_json(directory / "request.json")
        # A worker's receipt keeps its error unredacted; the review redacts it.
        operator._atomic_json(directory / "finished.json", operator._record(
            request, state="failed", pid=456, exit_code=7, error_type="ProviderConfigurationError",
            error=f"quota exceeded for {ORGANIZATION}; see {directory / 'worker.log'}",
        ))
    packet = reviewed(capsys, path, "run")
    [item] = packet["items"]
    assert (item["kind"], item["blocks"]) == ("worker_failed", "completion")
    assert item["subject"] == {"launch_id": worker["launch_id"]}
    assert item["observed"]["error_class"] == "ProviderConfigurationError"
    excerpt = item["observed"]["excerpt"]
    assert ORGANIZATION not in excerpt and "org-[redacted]" in excerpt
    assert excerpt.startswith("quota exceeded for org-[redacted]; see <store directory>")
    assert holds_no_path(packet, tmp_path, path.resolve().parent)
    assert item["resolve_with"] == [["jobs", "resume", "run"]]
    commands_parse(packet, tmp_path)


def test_stored_errors_are_redacted_and_capped(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2)
    # A record written before errors were redacted can still hold an identifier.
    with closing(sqlite3.connect(job.path)) as db, db:
        db.execute("UPDATE operations SET error=?", (f"billing for {ORGANIZATION} " + "x" * 1000,))
    [item] = reviewed(capsys, job.path, job.run_id)["items"]
    excerpt = item["observed"]["excerpt"]
    assert excerpt.startswith("billing for org-[redacted] ") and ORGANIZATION not in excerpt
    assert len(excerpt) == 300 and excerpt.endswith("…")


def test_a_key_that_looks_like_an_option_still_parses(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2)
    with closing(sqlite3.connect(job.path)) as db, db:
        db.execute("UPDATE operations SET operation_key='-glyph'")
    packet = reviewed(capsys, job.path, job.run_id)
    [argv] = commands_parse(packet, tmp_path)
    assert argv == ["jobs", "reroll", job.run_id, "--reason", "REASON", "--", "-glyph"]
    assert build_parser().parse_args(argv).operation_keys == ["-glyph"]


def test_operations_are_paged_and_items_are_ordered(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2, count=3)
    first = reviewed(capsys, job.path, job.run_id, "--limit", "2")
    assert [item["subject"]["operation_key"] for item in first["items"]] == ["glyph[0]", "glyph[1]"]
    assert first["page"] == {"limit": 2, "offset": 0, "total": 3, "returned": 2, "has_more": True}
    assert first["operation_counts"] == {"rejected": 3}
    rest = reviewed(capsys, job.path, job.run_id, "--limit", "2", "--offset", "2")
    assert [item["subject"]["operation_key"] for item in rest["items"]] == ["glyph[2]"]
    every = reviewed(capsys, job.path, job.run_id)
    assert every["items"] == first["items"] + rest["items"]
    assert len({item["item_id"] for item in every["items"]}) == 3

    assert main(["jobs", "review", job.run_id, "--limit", "2", "--store", str(job.path)]) == EXIT_OK
    assert "More operations: use --offset 2" in capsys.readouterr().out


def no_providers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Review initialized a provider or runner")

    for name in ("smythe.cli.ProviderPool", "smythe.jobs.providers.ProviderPool",
                 "smythe.jobs.runner.ProviderPool", "smythe.cli.JobRunner"):
        monkeypatch.setattr(name, forbidden)


def dump(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        return list(db.iterdump())


def test_review_reads_without_writing_migrating_or_constructing_providers(
    tmp_path, capsys, monkeypatch,
):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2)
    rows = dump(job.path)
    before, modified = job.path.read_bytes(), job.path.stat().st_mtime_ns
    no_providers(monkeypatch)

    assert reviewed(capsys, job.path, job.run_id)["items"]
    assert main(["jobs", "review", job.run_id, "--store", str(job.path)]) == EXIT_OK
    capsys.readouterr()

    assert job.path.read_bytes() == before and job.path.stat().st_mtime_ns == modified
    # Readers of a WAL database share its index files; nothing is written into the log.
    wal = Path(str(job.path) + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    assert dump(job.path) == rows


def test_review_never_enables_wal(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2)
    with closing(sqlite3.connect(job.path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    sidecars = [Path(str(job.path) + suffix) for suffix in ("-wal", "-shm", "-journal")]
    assert not any(path.exists() for path in sidecars)
    before, rows = job.path.read_bytes(), dump(job.path)

    assert reviewed(capsys, job.path, job.run_id)["items"]

    assert not any(path.exists() for path in sidecars)
    assert job.path.read_bytes() == before and dump(job.path) == rows
    with closing(sqlite3.connect(job.path.as_uri() + "?mode=ro", uri=True)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("delete",)


@pytest.mark.parametrize("version", [2, 3])
def test_legacy_stores_are_reviewed_without_migration(tmp_path, capsys, version):
    path = tmp_path / "legacy.db"
    make_legacy(path, tmp_path, Clock(), leased=True)
    if version == 3:
        # Rebuild the version-3 shape: lease fencing without pause control or namespaces.
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("ALTER TABLE runs ADD COLUMN lease_epoch INTEGER NOT NULL DEFAULT 1")
            db.execute("ALTER TABLE attempts ADD COLUMN lease_owner_id TEXT")
            db.execute("ALTER TABLE attempts ADD COLUMN lease_epoch INTEGER NOT NULL DEFAULT 0")
            db.execute("ALTER TABLE run_leases ADD COLUMN epoch INTEGER NOT NULL DEFAULT 1")
            db.execute("PRAGMA user_version=3")
    before, rows = path.read_bytes(), dump(path)

    packet = reviewed(capsys, path, "run")
    with SQLiteRunStore(path, read_only=True) as reader:
        assert review_job(reader, "run")["items"] == packet["items"] == []

    assert packet["schema_version"] == version
    assert path.read_bytes() == before and dump(path) == rows


def test_invalid_options_and_unknown_runs_have_stable_exit_codes(tmp_path, capsys):
    job = cli_job(tmp_path, capsys)
    for options in (["--limit", "0"], ["--limit", "501"], ["--offset", "-1"]):
        assert main(["jobs", "review", job.run_id, *options, "--store", str(job.path),
                     "--json"]) == EXIT_INVALID_INPUT
        assert output(capsys)["error"]["type"] == "ValueError"
    assert main(["jobs", "review", "missing-run", "--store", str(job.path), "--json"]) == EXIT_JOB_STATE
    assert output(capsys)["error"]["type"] == "JobNotFoundError"


def test_json_is_one_compact_document(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2)
    assert main(["jobs", "review", job.run_id, "--store", str(job.path), "--json"]) == EXIT_OK
    text = capsys.readouterr().out
    assert text.count("\n") == 1
    assert json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":")) == text.strip()


def test_an_interrupted_json_review_writes_no_partial_packet(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2)
    dumps = json.dumps

    def interrupted(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("ok") and value.get("command") == "review":
            raise KeyboardInterrupt
        return dumps(value, *args, **kwargs)

    monkeypatch.setattr("smythe.cli.json.dumps", interrupted)
    assert main(["jobs", "review", job.run_id, "--store", str(job.path), "--json"]) == EXIT_INTERRUPTED
    payload = output(capsys)
    assert payload == {"ok": False, "command": "review",
                       "error": {"type": "KeyboardInterrupt", "message": "review interrupted"}}


def test_an_interrupted_human_review_prints_nothing_partial(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2)
    terminal_text, calls = cli._terminal_text, []

    def interrupted(value, **kwargs):
        calls.append(value)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return terminal_text(value, **kwargs)

    monkeypatch.setattr(cli, "_terminal_text", interrupted)
    assert main(["jobs", "review", job.run_id, "--store", str(job.path)]) == EXIT_INTERRUPTED
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "error: review interrupted\n"


def test_human_review_shows_each_decision_and_command(tmp_path, capsys, monkeypatch):
    job = cli_job(tmp_path, capsys, monkeypatch, rejected=True, attempts=2)
    assert main(["jobs", "review", job.run_id, "--store", str(job.path)]) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert lines[0] == f"Run: {job.run_id}"
    assert "Needs a decision: 1 (operation 1 of 1, limit 50)" in lines
    assert any(line.startswith("[completion] rejected: glyph[0] (item ") for line in lines)
    assert "  Observed: rejected" in lines
    # Quoted for this host's shell; both renderings are checked below.
    command = cli._command_line(["jobs", "reroll", job.run_id, "glyph[0]", "--reason", "REASON"])
    assert f"  Resolve with: smythe {command}" in lines
    assert lines[-1] == ("To run a command, add the --store option used here and replace "
                         "REASON with your own reason.")


def human_packet(*, items=(), returned=1, total=1, offset=0, limit=50, not_checked=0):
    """A review packet as the human output reads it."""
    return {
        "run_id": "safe-run", "name": "glyphs", "status": "failed",
        "operation_counts": {"rejected": total},
        "page": {"limit": limit, "offset": offset, "total": total, "returned": returned,
                 "has_more": offset + returned < total},
        "artifact_integrity": {"counts": {"not_checked": not_checked} if not_checked else {}},
        "items": [{"item_id": "0" * 64, "kind": "rejected", "blocks": "completion",
                   "subject": {"operation_key": "glyph[0]"},
                   "observed": {"status": "rejected", "error_class": None, "excerpt": None},
                   "decision_needed": "Reroll it with a reason, or leave it rejected.",
                   "resolve_with": [argv]} for argv in items],
    }


def emitted(capsys, value, **options):
    cli._emit_job_review(value, **options)
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out.splitlines()


def test_human_review_counts_the_operations_a_page_holds(capsys):
    reroll = ["jobs", "reroll", "safe-run", "glyph[0]", "--reason", "REASON"]
    header = [line for line in emitted(capsys, human_packet(items=[reroll]), store_given=False)
              if line.startswith("Needs a decision")]
    assert header == ["Needs a decision: 1 (operation 1 of 1, limit 50)"]
    lines = emitted(capsys, human_packet(items=[reroll, reroll], returned=2, total=3, limit=2),
                    store_given=False)
    assert "Needs a decision: 2 (operations 1-2 of 3, limit 2)" in lines
    assert "More operations: use --offset 2" in lines
    lines = emitted(capsys, human_packet(returned=1, total=3, offset=2, limit=2), store_given=False)
    assert "Needs a decision: 0 (operation 3 of 3, limit 2)" in lines
    lines = emitted(capsys, human_packet(returned=0, total=3, offset=5), store_given=False)
    assert "Needs a decision: 0 (none of 3 operations at offset 5, limit 50)" in lines


def test_human_review_reminds_of_the_store_whenever_it_prints_a_command(capsys):
    # Only the hint to hash every accepted file names a command.
    lines = emitted(capsys, human_packet(not_checked=2), store_given=True)
    assert lines[-2] == ("Accepted files not hashed within the 64 MiB limit: 2; "
                         "smythe jobs reroll safe-run --dry-run hashes them all.")
    assert lines[-1] == "To run a command, add the --store option used here."
    # No reminder without a --store option, or without a command.
    lines = emitted(capsys, human_packet(not_checked=2), store_given=False)
    assert not any(line.startswith("To run a command") for line in lines)
    lines = emitted(capsys, human_packet(), store_given=True)
    assert lines[-1] == "Nothing on this page needs a decision."


def test_suggested_commands_are_quoted_for_the_local_shell(capsys):
    reroll = ["jobs", "reroll", "safe-run", "glyph[0]", "--reason", "REASON"]
    adopt = ["jobs", "adopt", "safe-run", "--reason", "REASON", "--", "-odd key"]
    value = human_packet(items=[reroll, adopt], returned=2, total=2)
    # cmd.exe keeps single quotes as part of an argument, so Windows gets its own quoting.
    lines = emitted(capsys, value, store_given=False, windows=True)
    assert "  Resolve with: smythe jobs reroll safe-run glyph[0] --reason REASON" in lines
    assert '  Resolve with: smythe jobs adopt safe-run --reason REASON -- "-odd key"' in lines
    lines = emitted(capsys, value, store_given=False, windows=False)
    assert "  Resolve with: smythe jobs reroll safe-run 'glyph[0]' --reason REASON" in lines
    assert "  Resolve with: smythe jobs adopt safe-run --reason REASON -- '-odd key'" in lines
    assert cli._command_line(['say "hi"', "a b"], windows=True) == r'"say \"hi\"" "a b"'
    assert cli._command_line(['say "hi"', "a b"], windows=False) == "'say \"hi\"' 'a b'"


def test_human_review_escapes_stored_terminal_controls(capsys):
    cli._emit_job_review({
        "run_id": "safe-run", "name": "unsafe\x1b[2J\nname", "status": "failed",
        "operation_counts": {"rejected": 1},
        "page": {"limit": 50, "offset": 0, "total": 1, "returned": 1, "has_more": False},
        "artifact_integrity": {"counts": {}},
        "items": [{
            "item_id": "0" * 64, "kind": "rejected", "blocks": "completion",
            "subject": {"operation_key": "key\x1b]0;title\x07"},
            "observed": {"status": "rejected", "error_class": None, "excerpt": "bad\x1b[31mred"},
            "decision_needed": "Reroll it\x1b[0m.", "resolve_with": [],
        }],
    }, store_given=False)
    human = capsys.readouterr().out
    assert "\x1b" not in human and "\x07" not in human
    assert "unsafe\\u001b[2J\\u000aname" in human and "bad\\u001b[31mred" in human
    assert "key\\u001b]0;title\\u0007" in human
