"""Benchmark accounting includes all returned provider usage without paid calls."""

from __future__ import annotations

import asyncio
import json

import pytest

from benchmarks.harness import BenchmarkTask, make_swarm, offline_provider
from benchmarks.provider_usage import BLENDED_USD_PER_TOKEN, UsageRecordingProvider
from benchmarks.run_shape_suite import run_one
from smythe.budget import Sentinel
from smythe.provider import CompletionResult, OfflineProvider
from smythe.tools import ChatMessage


def test_provider_usage_includes_dynamic_planning_and_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = UsageRecordingProvider(offline_provider("smythe_dynamic"))
    swarm = make_swarm("smythe_dynamic", provider, "offline")
    result = swarm.execute(BenchmarkTask("sample", "Write an analysis").to_task())
    usage = provider.snapshot()

    assert len(result.graph.nodes) == 3
    assert usage["call_count"] == 4
    assert usage["total_tokens"] == 950  # 650 planning + 3 x 100 execution
    assert usage["prompt_tokens"] == 370
    assert usage["completion_tokens"] == 580
    assert usage["scope"] == "planning_execution_synthesis"


def test_shape_benchmark_uses_full_workflow_cost_estimate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = BenchmarkTask("sample", "Write an analysis")
    dynamic = run_one(task, "smythe_dynamic", live=False)
    fixed = run_one(task, "fixed_pipeline", live=False)

    assert dynamic["usage"]["total_tokens"] == 950
    assert dynamic["cost_usd"] == pytest.approx(950 * BLENDED_USD_PER_TOKEN)
    assert fixed["cost_usd"] == pytest.approx(300 * BLENDED_USD_PER_TOKEN)
    assert dynamic["cost_scope"] == "planning_execution_synthesis"
    assert dynamic["cost_is_estimate"] is True
    assert dynamic["wall_s"] is None


def test_framework_benchmark_uses_direct_usage_including_planning(tmp_path, monkeypatch):
    from benchmarks import run_framework_h2h

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_framework_h2h,
        "OpenAIProvider",
        lambda: offline_provider("smythe_dynamic"),
    )
    record = run_framework_h2h.run_smythe(
        "smythe_dynamic", BenchmarkTask("sample", "Write an analysis"),
    )
    assert record["tokens"] == 950
    assert record["tokens_source"] == "provider_response_usage"
    assert record["usage"]["call_count"] == 4


def test_legacy_fixed_token_derivation_recovers_response_token_counts():
    budget = Sentinel()
    counts = (1132, 3987, 6718)
    for index, tokens in enumerate(counts):
        budget.record(str(index), CompletionResult("ok", prompt_tokens=tokens))
    assert round(budget.total_cost_usd / BLENDED_USD_PER_TOKEN) == sum(counts)


def test_chat_delegation_records_once_and_preserves_budget_hooks():
    class BudgetedProvider(OfflineProvider):
        def budget_estimate_usd(self, model):
            return 0.1

        def requires_explicit_budget_estimate(self, model):
            return True

    provider = UsageRecordingProvider(BudgetedProvider())
    result = asyncio.run(provider.chat("system", [ChatMessage("user", "prompt")], "offline"))
    assert result.total_tokens == provider.total_tokens == 100
    assert provider.snapshot()["call_count"] == 1
    assert provider.budget_estimate_usd("offline") == 0.1
    assert provider.requires_explicit_budget_estimate("offline") is True


@pytest.mark.parametrize("score", [0, 11, True, 1.5, "8", None])
@pytest.mark.parametrize("module_name", ["run_framework_h2h", "run_shape_suite"])
def test_benchmark_judge_rejects_invalid_rubric_scores(monkeypatch, module_name, score):
    from importlib import import_module

    module = import_module(f"benchmarks.{module_name}")
    monkeypatch.setattr(
        module, "GeminiProvider",
        lambda: OfflineProvider(responses=[json.dumps({"overall": score})]),
    )
    assert module.judge("A deliverable", ["A criterion"]) is None
