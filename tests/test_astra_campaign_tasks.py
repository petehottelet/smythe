"""Frozen sources, independent answer-key arithmetic and evaluator separation."""

from collections import Counter
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
import json
import re
import shutil

import pytest

from benchmarks.astra_campaign import CampaignPlanError, check_output, load_task_pack, provider_task
from benchmarks.astra_campaign._json import MAX_FILE_BYTES, digest
from benchmarks.astra_campaign.tasks import DATA_DIR, MAX_OUTPUT_BYTES, SHAPES


@pytest.fixture(scope="module")
def cases():
    return {case.task_id: case for case in load_task_pack().tasks}


def documents(case):
    return {doc["id"]: doc["content"] for doc in provider_task(case)["context"]["source_pack"]["documents"]}


def passing(case, output):
    receipt = check_output(case, json.dumps(output))
    assert receipt == {"deterministic_passed": True, "failed_checks": [],
                       "quality_evaluated": False, "accepted": None}


def test_pack_counts_and_rubric_material_are_separate_from_provider_inputs(cases):
    assert Counter(case.stage for case in cases.values()) == {"main": 10, "pilot": 3}
    assert Counter(case.shape for case in cases.values() if case.stage == "main") == dict.fromkeys(SHAPES, 2)
    for case in cases.values():
        task = provider_task(case)
        assert set(task) == {"goal", "constraints", "done_when", "context"}
        assert set(task["context"]) == {"source_pack"}
        assert task["done_when"] == []  # No hidden judge rubric injected into runtime requirements.
        assert not {"rubric", "checks", "expected", "anchors", "stage", "shape"} & task.keys()
        assert case.rubric and case.checks
        encoded = json.dumps(task)
        assert not any(criterion["criterion"] in encoded for criterion in case.rubric)
        assert not any("criterion-" in text for text in task["constraints"])
        assert case.stage in {"pilot", "main"}


def test_provider_and_evaluator_views_are_detached(cases):
    case = cases["main-calendar"]
    task = provider_task(case)
    task["context"]["source_pack"]["documents"][0]["content"] = "mutated"
    task["constraints"].clear()
    rubric = case.rubric
    rubric[0]["anchors"]["4"] = "mutated"
    case.checks.clear()
    assert provider_task(case)["constraints"]
    assert "mutated" not in json.dumps(provider_task(case))
    assert case.rubric[0]["anchors"]["4"] != "mutated" and case.checks


@pytest.mark.parametrize("identity,formats", [
    ("pilot-calendar", ["%d %B %Y", "%m/%d/%Y", "%d %b %Y", "%Y-%m-%d"]),
    ("main-calendar", ["%d %b %Y", "%d/%m/%Y", "%B %d, %Y", "%Y-%m-%d", "%d %B %Y", "%d/%m/%Y"]),
])
def test_calendar_keys_derive_from_unambiguous_source_rules(cases, identity, formats):
    case = cases[identity]
    source = json.loads(documents(case)["calendar"])
    correct = [datetime.strptime(text, fmt).date().isoformat() for text, fmt in zip(source["dates"], formats, strict=True)]
    passing(case, correct)
    assert not check_output(case, json.dumps(correct[::-1]))["deterministic_passed"]


def test_identifier_keys_preserve_source_numbers_and_zero(cases):
    case = cases["main-identifiers"]
    source = json.loads(documents(case)["identifiers"])
    output = []
    for value in source["input"]:
        letters, digits = re.fullmatch(r"([A-Za-z]{2})[ _-]*(\d+)", value.strip()).groups()
        output.append(f"{letters.upper()}-{int(digits):04d}")
    passing(case, output)
    output[-2] = "ZX-0001"
    assert not check_output(case, json.dumps(output))["deterministic_passed"]


@pytest.mark.parametrize("identity", ["pilot-relays", "main-field-links", "main-tunnel-links"])
def test_parallel_candidate_keys_use_each_source_and_both_limits(cases, identity):
    case = cases[identity]
    source = {key: json.loads(value) for key, value in documents(case).items()}
    requirements = source.pop("requirements")
    profiles = {}
    for name, values in source.items():
        profiles[name] = {**values, "feasible": values["energy_mj"] <= requirements["maximum_energy_mj_per_message"]
                          and values["range_m"] >= requirements["minimum_range_m"]}
    ranking = sorted((key for key, value in profiles.items() if value["feasible"]),
                     key=lambda key: (profiles[key]["energy_mj"], -profiles[key]["range_m"]))
    output = {"profiles": profiles, "ranking": ranking, "reasoning": "Numeric comparisons are checked separately from explanation quality."}
    passing(case, output)
    for values in output["profiles"].values():
        values["energy_mj"] = float(values["energy_mj"])
        values["range_m"] = float(values["range_m"])
    passing(case, output)  # Equivalent JSON numeric forms are equally correct.
    key = ranking[0]
    output["profiles"][key]["feasible"] = 1
    assert not check_output(case, json.dumps(output))["deterministic_passed"]  # bool != int


def test_production_chain_keeps_loose_stock_out_of_shipped_quantity(cases):
    case = cases["main-production-chain"]
    source = json.loads(documents(case)["production"])
    after_test = int(source["initial_units"] * Decimal(".90"))
    after_coating = int(after_test * Decimal(".95"))
    allocated = after_coating - source["reserve_units_after_coating"]
    cartons = allocated // source["units_per_carton"]
    backlog = source["order_units"] - cartons * source["units_per_carton"]
    output = {"steps": {"after_test": after_test, "after_coating": after_coating,
                         "allocated": allocated, "explanation": "Sequence uses prior-stage quantities."},
              "shipments": cartons, "backlog": backlog, "decision": "expedite" if backlog > 24 else "normal"}
    assert (cartons, backlog) == (41, 36)
    passing(case, output)
    output["backlog"] = source["order_units"] - allocated
    assert not check_output(case, json.dumps(output))["deterministic_passed"]


def test_capacity_chain_floors_quality_units_before_full_boxes(cases):
    case = cases["main-capacity-chain"]
    source = json.loads(documents(case)["capacity"])
    gross = source["machines"] * source["minutes_per_machine"]
    net = int(gross * Decimal(".85"))
    started = net // source["minutes_per_unit"]
    survived = int((started * Decimal(".92")).to_integral_value(rounding=ROUND_FLOOR))
    delivered = survived // source["units_per_box"] * source["units_per_box"]
    output = {"steps": {"gross_minutes": gross, "net_minutes": net, "started_units": started,
                        "passing_units": survived, "explanation": "Floor the yield before packing."},
              "deliverable_units": delivered, "shortfall": source["order_units"] - delivered,
              "decision": "overtime" if delivered < source["order_units"] else "regular"}
    assert survived == 281  # 281.52 must not round to 282.
    passing(case, output)
    output["steps"]["passing_units"] = 282
    assert not check_output(case, json.dumps(output))["deterministic_passed"]


@pytest.mark.parametrize("identity", ["pilot-membership", "main-studio-copy", "main-delivery-copy"])
def test_audit_numeric_keys_follow_authoritative_policy_without_leaking_defect_answers(cases, identity):
    case = cases[identity]
    policy = json.loads(documents(case)["policy"])
    if "monthly_credits" in policy:
        monthly, annual = policy["monthly_credits"], policy["annual_credits"]
        percent = float((Decimal(1) - Decimal(annual) / (12 * monthly)) * 100)
        facts = {"monthly": monthly, "annual": annual, "savings_percent": round(percent, 2)}
        facts.update({"guest_passes": policy["guest_passes_per_month"]} if "guest_passes_per_month" in policy
                     else {"storage_gb": policy["storage_gb"]})
    else:
        unit = policy["pack_credits"] / policy["labels_per_pack"]
        facts = {"labels_per_pack": policy["labels_per_pack"], "pack_credits": policy["pack_credits"],
                 "unit_credits": unit, "savings_percent": round((1 - unit / policy["single_label_credits"]) * 100, 2)}
    output = {"facts": facts, "corrected_copy": "Placeholder deliberately does not receive a quality or acceptance verdict.",
              "defects": {kind: {"a quoted phrase": "an explanation"} for kind in ("spelling", "arithmetic", "policy")}}
    passing(case, output)
    assert not any("exact phrase keys" in line for line in provider_task(case)["constraints"])
    output["facts"]["savings_percent"] = 99
    assert not check_output(case, json.dumps(output))["deterministic_passed"]


@pytest.mark.parametrize("identity,survivors,unsupported", [
    ("main-library-hours", ["bounded-pilot", "representative-survey"], ["funded-rollout", "universal-demand"]),
    ("main-warehouse-routing", ["staffed-trial", "broader-testing"], ["ready-this-month", "universal-speedup"]),
])
def test_adversarial_checks_do_not_claim_to_judge_argument_quality(cases, identity, survivors, unsupported):
    case = cases[identity]
    source = documents(case)
    assert len(json.loads(source["claims"])) == 4  # Includes supported as well as unsupported claims.
    output = {"case": "Example case", "critique": "Example critique",
              "survivors": dict.fromkeys(survivors, "A bounded action"),
              "unsupported_claims": unsupported, "source_ids": list(source)}
    passing(case, output)
    output["source_ids"].append("fabricated-document")
    assert not check_output(case, json.dumps(output))["deterministic_passed"]


@pytest.mark.parametrize("text", ["", "not JSON", '```json\n[]\n```', '{"x":1,"x":2}',
                                  '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}',
                                  "x" * (MAX_OUTPUT_BYTES + 1), None],
                         ids=["empty", "text", "fence", "duplicate", "nan", "infinity",
                              "overflow", "oversize", "none"])
def test_invalid_output_is_failed_without_quality_claims(cases, text):
    assert check_output(cases["main-calendar"], text) == {
        "deterministic_passed": False, "failed_checks": ["json-output"],
        "quality_evaluated": False, "accepted": None,
    }


@pytest.fixture
def copy_pack(tmp_path):
    root = tmp_path / "pack"
    shutil.copytree(DATA_DIR, root)
    return root


def rewrite(root, name, update):
    path = root / name
    data = json.loads(path.read_text())
    update(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest_path = root / "pack-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"][name] = digest(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_changed_or_missing_source_fails_the_hash_boundary(copy_pack):
    path = copy_pack / "sources/main-calendar.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(CampaignPlanError, match="hash mismatch"):
        load_task_pack(copy_pack)
    path.unlink()
    with pytest.raises(CampaignPlanError, match="Cannot read"):
        load_task_pack(copy_pack)


def test_unmanifested_task_is_rejected(copy_pack):
    (copy_pack / "tasks/extra.json").write_text("{}")
    with pytest.raises(CampaignPlanError, match="unmanifested"):
        load_task_pack(copy_pack)


@pytest.mark.parametrize("field,value", [("version", True), ("stage", "pilot"), ("shape", "single-step"),
                                        ("rubric", []), ("checks", []), ("source_file", "../outside.json")])
def test_rehashed_but_invalid_task_still_fails_schema(copy_pack, field, value):
    rewrite(copy_pack, "tasks/main-capacity-chain.json", lambda task: task.update({field: value}))
    with pytest.raises(CampaignPlanError):
        load_task_pack(copy_pack)


@pytest.mark.parametrize("change", ["kind", "path", "number", "duplicate", "citation"])
def test_check_dsl_rejects_invalid_semantics(copy_pack, change):
    def mutate(task):
        if change == "kind":
            task["checks"][0]["kind"] = "eval-python"
        if change == "path":
            task["checks"][0]["path"] = [True]
        if change == "number":
            task["checks"].append({"id": "huge-number", "kind": "number", "path": [],
                                   "expected": 10**400, "tolerance": 0})
        if change == "duplicate":
            task["checks"].append(task["checks"][0])
        if change == "citation":
            task["checks"].append({"id": "bad-citation", "kind": "citations", "path": [],
                                   "allowed": ["missing"], "minimum": 1})
    rewrite(copy_pack, "tasks/main-calendar.json", mutate)
    with pytest.raises(CampaignPlanError):
        load_task_pack(copy_pack)


@pytest.mark.parametrize("content", [b'{"version":1,"version":1}', b'{"value":NaN}', b'{"value":1e999}', b'\xff'])
def test_corrupt_json_is_rejected_before_hash_validation(copy_pack, content):
    (copy_pack / "sources/main-calendar.json").write_bytes(content)
    with pytest.raises(CampaignPlanError):
        load_task_pack(copy_pack)


def test_raw_file_bound_precedes_line_ending_normalization(copy_pack):
    (copy_pack / "sources/main-calendar.json").write_bytes(b"\r\n" * (MAX_FILE_BYTES // 2 + 1))
    with pytest.raises(CampaignPlanError, match="exceeds"):
        load_task_pack(copy_pack)


@pytest.mark.parametrize("name", ["../outside.json", "sources/../../outside.json", "C:/outside.json", "sources\\x.json"])
def test_manifest_path_escape_is_rejected(copy_pack, name):
    path = copy_pack / "pack-manifest.json"
    data = json.loads(path.read_text())
    old = next(iter(data["file_sha256"]))
    data["file_sha256"][name] = data["file_sha256"].pop(old)
    path.write_text(json.dumps(data))
    with pytest.raises(CampaignPlanError):
        load_task_pack(copy_pack)
