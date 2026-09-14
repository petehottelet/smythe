"""Offline checks for blind rubric evidence and bounded judge accounting."""

from copy import deepcopy
import json

import pytest

from benchmarks import astra_evaluation as evaluation
from benchmarks.astra_campaign import load_task_pack


@pytest.fixture
def case():
    return next(c for c in load_task_pack().tasks if c.task_id == "pilot-relays")


def response(case):
    scores = {"criteria": [{"id": r["id"], "score": 3, "reason": "Supported by the supplied numeric facts."}
                           for r in case.rubric], "material_defects": []}
    return {"modelVersion": evaluation.MODEL,
            "usageMetadata": {"promptTokenCount": 100, "totalTokenCount": 130,
                              "candidatesTokenCount": 20, "thoughtsTokenCount": 10,
                              "cachedContentTokenCount": 0, "serviceTier": "standard"},
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(scores)}]}}]}


def test_text_contract_exposes_type_requirement_without_expected_answers(case):
    contract = evaluation.text_contract(case)
    assert "/reasoning" in contract and "nonempty string" in contract
    assert "600" not in contract and "1800" not in contract
    assert "expected" not in contract and "rubric" not in contract
    assert evaluation.text_contract(next(c for c in load_task_pack().tasks if c.task_id == "pilot-calendar")) is None


def test_judge_receives_sources_and_rubric_but_no_arm_or_execution_receipts(case):
    request = evaluation.judge_request(case, "saved answer")
    material = json.loads(request["contents"][0]["parts"][0]["text"])
    assert set(material) == {"task", "sources", "rubric", "deliverable"}
    assert material["rubric"] == case.rubric
    assert request["serviceTier"] == "standard"
    assert not {"tools", "cachedContent"} & request.keys()


def test_reasoning_is_charged_exactly_once_and_cache_is_distinct(case):
    raw = response(case)
    raw["usageMetadata"]["cachedContentTokenCount"] = 40
    price = evaluation.price_judge_response(raw)
    assert price["billed_output_tokens"] == 30
    assert price["cost_lower_nanousd"] == price["cost_upper_nanousd"] == 60 * 2000 + 40 * 200 + 30 * 12000
    assert price["cost_exact"] is True


def test_unreported_cache_is_an_interval_not_an_invented_zero(case):
    raw = response(case)
    del raw["usageMetadata"]["cachedContentTokenCount"]
    price = evaluation.price_judge_response(raw)
    assert price["reported_cached_tokens"] is None and price["cost_exact"] is False
    assert price["cost_lower_nanousd"] == 100 * 200 + 30 * 12000
    assert price["cost_upper_nanousd"] == 100 * 2000 + 30 * 12000


@pytest.mark.parametrize("field,value", [
    ("promptTokenCount", None), ("promptTokenCount", True), ("promptTokenCount", 200001),
    ("totalTokenCount", 99), ("totalTokenCount", 131), ("candidatesTokenCount", -1),
    ("thoughtsTokenCount", 31), ("cachedContentTokenCount", 101),
    ("serviceTier", "priority"), ("serviceTier", None), ("toolUsePromptTokenCount", 1),
    ("promptTokensDetails", [{"modality": "IMAGE", "tokenCount": 1}]),
])
def test_incomplete_or_out_of_scope_accounting_is_rejected(case, field, value):
    raw = response(case)
    raw["usageMetadata"][field] = value
    with pytest.raises(ValueError):
        evaluation.price_judge_response(raw)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "bool", "range", "empty", "truncated"])
def test_judge_cannot_forge_complete_rubric(case, kind):
    raw = response(case)
    scores = json.loads(raw["candidates"][0]["content"]["parts"][0]["text"])
    if kind == "missing":
        scores["criteria"].pop()
    elif kind == "duplicate":
        scores["criteria"].append(deepcopy(scores["criteria"][0]))
    elif kind in {"bool", "range"}:
        scores["criteria"][0]["score"] = True if kind == "bool" else 5
    elif kind == "empty":
        scores["criteria"][0]["reason"] = ""
    else:
        raw["candidates"][0]["finishReason"] = "MAX_TOKENS"
    raw["candidates"][0]["content"]["parts"][0]["text"] = json.dumps(scores)
    with pytest.raises(ValueError):
        evaluation.decode_scores(raw, [c["id"] for c in case.rubric])


@pytest.fixture
def transport(monkeypatch, case):
    state = {"calls": [], "raw": response(case), "interrupt": False}

    def post(method, payload, key):
        state["calls"].append(method)
        if method == "countTokens":
            return 200, b'{"totalTokens":100}'
        if state["interrupt"]:
            raise OSError("offline interrupted transport")
        return 200, json.dumps(state["raw"]).encode()

    monkeypatch.setattr(evaluation, "_post", post)
    return state


def run(case, path, output="answer", cap=1_000_000_000):
    return evaluation.run_judgment(case, output, directory=path, allowance_nanousd=cap,
                                   api_key="offline-not-a-key", model_version=evaluation.MODEL)


def test_replay_verifies_raw_scores_and_never_rebuys(case, tmp_path, transport):
    first = run(case, tmp_path)
    assert first == run(case, tmp_path)
    assert first["human_calibrated"] is first["claimable"] is False
    assert transport["calls"] == ["countTokens", "generateContent"]
    raw_path = next(tmp_path.glob("*.raw.json"))
    raw_path.write_bytes(b'{}')
    with pytest.raises(ValueError, match="Raw judge receipt"):
        run(case, tmp_path)
    assert len(transport["calls"]) == 2


def test_unknown_attempt_blocks_new_spending(case, tmp_path, transport):
    transport["interrupt"] = True
    with pytest.raises(OSError):
        run(case, tmp_path)
    with pytest.raises(ValueError, match="unresolved"):
        run(case, tmp_path, output="different answer")
    assert len(transport["calls"]) == 2


def test_budget_denial_precedes_generation(case, tmp_path, transport):
    with pytest.raises(ValueError, match="cannot admit"):
        run(case, tmp_path, cap=1)
    assert transport["calls"] == ["countTokens"]
    assert not list(tmp_path.glob("*.started.json"))


def test_raw_bytes_survive_decode_failure_and_are_not_rebought(case, tmp_path, transport):
    transport["raw"]["candidates"][0]["finishReason"] = "MAX_TOKENS"
    with pytest.raises(ValueError, match="complete candidate"):
        run(case, tmp_path)
    assert len(list(tmp_path.glob("*.raw.json"))) == 1
    with pytest.raises(ValueError, match="unresolved"):
        run(case, tmp_path)
    assert len(transport["calls"]) == 2
