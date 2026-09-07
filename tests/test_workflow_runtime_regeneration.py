"""Billed descendants survive batched checkpoint and regeneration crashes."""

import asyncio
from collections import Counter
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from smythe import Swarm
from smythe.async_executor import AsyncExecutor
from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.verifier import node_generation
from smythe.workflow_store import SQLiteWorkflowStore


class ProcessLoss(BaseException):
    pass


@pytest.mark.parametrize("crash_after_commit", [False, True])
def test_cancelled_billed_descendant_is_consumed_across_regeneration_resume(
    tmp_path, monkeypatch, crash_after_commit,
):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    counts = Counter()
    old_consumer_ids = []
    cancelled = []
    crashed = []
    finalized = []

    def wire(body):
        return SimpleNamespace(content=json.dumps(body).encode(), status_code=200,
                               headers={"x-request-id": "req_test"})

    async def count(**payload):
        return wire({"input_tokens": 100})

    async def generate(**payload):
        prompt = payload["input"][-1]["content"]
        name = next(name for name in ("Draft", "Consumer", "Judge")
                    if prompt == name or f"Your step: {name}" in prompt or prompt.startswith(name + "\n"))
        ordinal = counts[name]
        counts[name] += 1
        if name == "Judge" and ordinal == 0:
            assert await asyncio.to_thread(entered.wait, 30), "consumer never entered its paid finalizer"
            text = "FAIL: regenerate the draft"
        else:
            text = "PASS" if name == "Judge" else f"{name} generation {ordinal}"
        return wire({
            "id": f"resp_{name}_{ordinal}", "model": payload["model"], "status": "completed",
            "service_tier": "default", "usage": {"input_tokens": 100, "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}},
            "output": [{"type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}],
        })

    def client(self):
        if self._client is not None:
            return self._client
        return SimpleNamespace(base_url="https://api.openai.com/v1/", max_retries=0, close=AsyncMock(),
            responses=SimpleNamespace(with_raw_response=SimpleNamespace(create=AsyncMock(side_effect=generate)),
                input_tokens=SimpleNamespace(with_raw_response=SimpleNamespace(count=AsyncMock(side_effect=count)))))

    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", client)
    original_finalize = ExecutorBase.finalize_node_result

    def finalize(self, node, result):
        generation = node_generation(node)
        if node.id == "consumer" and generation == 0:
            entered.set()
            try:
                assert release.wait(30), "paid finalizer was never released"
                return original_finalize(self, node, result)
            finally:
                finished.set()
                finalized.append((node.id, generation))
        finalized.append((node.id, generation))
        return original_finalize(self, node, result)

    monkeypatch.setattr(ExecutorBase, "finalize_node_result", finalize)
    source = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="consumer", label="Consumer", depends_on=["draft"]),
        Node(id="judge", label="Judge", depends_on=["draft"], verifies="draft", max_regenerations=1),
    ])

    with SQLiteWorkflowStore(tmp_path / "regeneration.db") as store:
        def runtime():
            return Swarm(model="gpt-6-astra", provider=OpenAIResponsesProvider(api_key="no-network", max_output_tokens=100),
                         run_store=store, parallel=True, max_concurrency=2,
                         checkpoint_every_n_nodes=1000, max_budget_usd=1)

        original_cancel = AsyncExecutor._cancel_and_settle

        async def cancel(self, active, affected_ids=None):
            if affected_ids and "consumer" in affected_ids and not cancelled:
                matching = [task for task, node in active.items() if node.id == "consumer"]
                assert matching and entered.is_set() and not finished.is_set()
                run_id = store.list_runs()[0]["run_id"]
                old = next(call for call in store.inspect_run(run_id)["calls"]
                           if call["key"]["scope_id"] == "node/consumer" and call["key"]["generation"] == 0)
                assert old["billing_state"] == "known" and old["result_state"] == "accepted"
                old_consumer_ids.append(old["call_id"])
                # Release only after cancellation has really been requested.
                # The normal path cancels again and must still drain the writer.
                for task in matching:
                    task.cancel()
                    assert task.cancelling()
                cancelled.append(True)
                release.set()
            return await original_cancel(self, active, affected_ids)

        monkeypatch.setattr(AsyncExecutor, "_cancel_and_settle", cancel)
        original_save = store.save_checkpoint

        def save(lease, revision, state, **kwargs):
            reset = any(node["id"] == "consumer" and node["metadata"].get("execution_generation") == 1
                        for node in state["graph"]["nodes"])
            if crashed:
                raise ProcessLoss("process stopped")
            if reset:
                assert finished.is_set(), "reset happened before the paid finalizer settled"
                crashed.append(True)
                if crash_after_commit:
                    original_save(lease, revision, state, **kwargs)
                raise ProcessLoss("lost process at generation checkpoint")
            return original_save(lease, revision, state, **kwargs)

        monkeypatch.setattr(store, "save_checkpoint", save)

        async def first():
            try:
                with pytest.raises(ProcessLoss):
                    await asyncio.wait_for(runtime().execute_async(source), timeout=60)
            finally:
                release.set()

        asyncio.run(first())
        assert crashed and cancelled and finished.is_set()
        assert counts == {"Draft": 1, "Consumer": 1, "Judge": 1}
        run_id = store.list_runs()[0]["run_id"]
        assert store.inspect_run(run_id)["confirmed_nanousd"] == 4_500_000
        assert store.load_replay(old_consumer_ids[0])["result_state"] == ("applied" if crash_after_commit else "accepted")
        monkeypatch.setattr(store, "save_checkpoint", original_save)
        result = runtime().resume(run_id)
        assert counts == {"Draft": 2, "Consumer": 2, "Judge": 2}
        assert len(set(finalized)) == 6
        assert all(node_generation(node) == 1 for node in result.graph.nodes)
        assert result.cost_is_complete and result.total_cost_usd == .009
        accounting = store.inspect_run(run_id)
        assert accounting["confirmed_nanousd"] == 9_000_000
        assert accounting["status"] == "completed"
        assert all(call["result_state"] == "applied" for call in accounting["calls"])
        assert store.load_replay(old_consumer_ids[0])["cost_nanousd"] == 1_500_000
