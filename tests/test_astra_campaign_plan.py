"""Offline protocol validation, counterbalancing and portable provenance."""

from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from benchmarks.astra_campaign import CampaignPlanError, build_schedule, prepare_campaign
from benchmarks.astra_campaign._json import canonical, digest
from benchmarks.astra_campaign.schedule import ARMS, WILLIAMS_ROWS
from benchmarks.astra_campaign.tasks import DATA_DIR


def test_preparation_contains_all_trials_and_keeps_paid_execution_blocked():
    receipt = prepare_campaign()
    assert len(receipt["schedules"]["pilot"]) == 12
    assert len(receipt["schedules"]["main"]) == 200
    assert receipt["api_calls"] == 0
    assert receipt["total_api_budget_nanousd"] is None
    assert receipt["paid_execution_allowed"] is receipt["claimable"] is False
    assert len(receipt["blockers"]) == 4
    assert receipt["independent_main_tasks"] == 10
    assert "not a paid campaign runtime freeze" in receipt["source_hash_scope"]
    assert "results" not in receipt and "quality" not in receipt
    saved_hash = receipt.pop("preparation_sha256")
    assert saved_hash == digest(canonical(receipt).encode())


@pytest.mark.parametrize("seed", [0, 1, 14173, 2**63 - 1])
@pytest.mark.parametrize("stage,blocks", [("pilot", 3), ("main", 50)])
def test_schedule_is_reproducible_complete_and_balances_order(seed, stage, blocks):
    rows = build_schedule(stage, seed)
    assert rows == build_schedule(stage, seed)
    assert [row["ordinal"] for row in rows] == list(range(4 * blocks))
    assert len({row["trial_id"] for row in rows}) == len(rows)
    assert len({(row["task_id"], row["repetition"], row["arm_id"]) for row in rows}) == len(rows)
    expected_arms = {arm[0] for arm in ARMS}
    position_counts = Counter()
    predecessors = Counter()
    arm_map = {arm[0]: arm[1:] for arm in ARMS}
    for index in range(blocks):
        block = rows[4 * index:4 * index + 4]
        assert {row["arm_id"] for row in block} == expected_arms
        assert len({(row["task_id"], row["repetition"]) for row in block}) == 1
        assert len({(row["task_sha256"], row["source_sha256"]) for row in block}) == 1
        for row in block:
            assert row["block"] == index
            assert (row["model"], row["strategy"]) == arm_map[row["arm_id"]]
            position_counts[row["arm_id"], row["position"]] += 1
        predecessors.update((left["arm_id"], right["arm_id"]) for left, right in zip(block, block[1:]))
    for arm in expected_arms:
        counts = [position_counts[arm, position] for position in range(4)]
        assert max(counts) - min(counts) <= 1
    if stage == "main":
        assert len(predecessors) == 12
        assert max(predecessors.values()) - min(predecessors.values()) <= 1
        assert set(Counter(row["task_id"] for row in rows).values()) == {20}
        assert set(Counter(row["arm_id"] for row in rows).values()) == {50}


def test_williams_design_balances_every_directed_predecessor_in_four_blocks():
    counts = Counter((left, right) for row in WILLIAMS_ROWS for left, right in zip(row, row[1:]))
    assert counts == {(left, right): 1 for left in range(4) for right in range(4) if left != right}
    assert all(sorted(row) == [0, 1, 2, 3] for row in WILLIAMS_ROWS)


def test_different_seed_changes_order_without_changing_membership():
    first, second = build_schedule("main", 1), build_schedule("main", 2)
    assert [row["trial_id"] for row in first] != [row["trial_id"] for row in second]
    assert {row["trial_id"] for row in first} == {row["trial_id"] for row in second}
    assert {row["task_id"] for row in first}.isdisjoint(
        row["task_id"] for row in build_schedule("pilot", 1)
    )


@pytest.mark.parametrize("seed", [True, False, -1, 2**63, 1.0, "1", None])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(CampaignPlanError):
        build_schedule("main", seed)


@pytest.mark.parametrize("stage", [None, "held-out", "", 1, []])
def test_invalid_stage_is_rejected(stage):
    with pytest.raises(CampaignPlanError):
        build_schedule(stage, 0)


@pytest.mark.parametrize("field,value", [
    ("version", True), ("total_api_budget_nanousd", 0), ("total_api_budget_nanousd", 1_000_000),
    ("pack_manifest_sha256", "0" * 64), ("models", ["gpt-6-astra"]),
    ("status", "authorized"),
])
def test_protocol_cannot_silently_change_or_authorize_spend(tmp_path, field, value):
    root = tmp_path / "data"
    shutil.copytree(DATA_DIR, root)
    path = root / "protocol.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CampaignPlanError):
        prepare_campaign(directory=root)


@pytest.mark.parametrize("field,key,value", [
    ("request_policy", "max_output_tokens", True),
    ("request_policy", "sdk_retries", 2),
    ("execution_policy", "graph_policy_persisted", True),
    ("evaluation_policy", "quality_noninferiority_margin", .5),
    ("price_snapshot", "long_context_threshold", 1),
])
def test_nested_policy_changes_require_a_new_protocol(tmp_path, field, key, value):
    root = tmp_path / "data"
    shutil.copytree(DATA_DIR, root)
    path = root / "protocol.json"
    data = json.loads(path.read_text())
    data[field][key] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CampaignPlanError):
        prepare_campaign(directory=root)


def test_whole_pack_and_preparation_sources_have_lf_crlf_hash_parity(tmp_path, monkeypatch):
    import benchmarks.astra_campaign.protocol as module

    original = prepare_campaign()
    root = tmp_path / "package"
    shutil.copytree(DATA_DIR.parent, root)
    for path in root.rglob("*"):
        if path.suffix in {".json", ".py"}:
            content = path.read_bytes().replace(b"\r\n", b"\n")
            path.write_bytes(content.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(module, "__file__", str(root / "protocol.py"))
    assert prepare_campaign(directory=root / "data") == original
    # Whitespace other than line endings is still provenance-significant.
    path = root / "tasks.py"
    path.write_bytes(path.read_bytes() + b"# changed preparation source\r\n")
    assert prepare_campaign(directory=root / "data")["preparation_sha256"] != original["preparation_sha256"]


def test_preparation_does_not_import_runtime_or_sdks_or_connect(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    script = """
import socket, sys
sys.path.insert(0, sys.argv[1])
def forbidden(*a, **kw): raise AssertionError('network attempted')
socket.socket.connect = forbidden
from benchmarks.astra_campaign import prepare_campaign
value = prepare_campaign()
assert value['api_calls'] == 0
assert not {'smythe', 'openai', 'anthropic', 'google.genai'} & sys.modules.keys()
print(value['status'])
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(repository)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline-preparation-only"
