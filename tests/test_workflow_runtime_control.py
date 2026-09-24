"""Recovery consumes verifier controls before any new supervisor call."""

from collections import Counter
import asyncio
import json

import pytest

from smythe import LocalOnly, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task
from smythe.graph import ExecutionGraph, Node, NodeStatus, Revision, Topology
from smythe.provider import CompletionResult
from smythe.supervisor import LLMSupervisor, SUPERVISOR_SYSTEM_PROMPT
from smythe.workflow_store import WorkflowLeaseError


class ProcessLost(BaseException):
    pass


@pytest.mark.parametrize("boundary", ["pending_verdict", "regeneration_intent", "pending_replayed_verdict"])
def test_pending_rejection_is_consumed_before_resumed_supervision(tmp_path, monkeypatch, boundary):
    dispatched = []
    judges = []

    async def complete(self, system, prompt, model):
        dispatched.append((system, prompt))
        if system == SUPERVISOR_SYSTEM_PROMPT:
            return CompletionResult(json.dumps({"change": False, "reason": "No change"}), cost_usd=0)
        if prompt.startswith("Judge"):
            judges.append(prompt)
            return CompletionResult("FAIL: revise" if len(judges) == 1 else "PASS", cost_usd=0)
        return CompletionResult("Draft accepted for review", cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    with SQLiteWorkflowStore(tmp_path / "verification.db") as store:
        provider = OfflineProvider()
        swarm = Swarm(
            model="offline", provider=provider, architect=SimpleArchitect(), run_store=store,
            supervisor=LLMSupervisor(provider, model="offline", only_terminal=False), max_revisions=1,
        )
        graph = ExecutionGraph([Topology.SERIAL], [
            Node(id="draft", label="Draft"),
            Node(id="judge", label="Judge", depends_on=["draft"], verifies="draft", max_regenerations=1),
        ])
        original = store.save_checkpoint
        crashed = []

        def save(lease, revision, state, **kwargs):
            if crashed:
                raise ProcessLost("Process already stopped")
            judge = next(node for node in state["graph"]["nodes"] if node["id"] == "judge")
            metadata = judge["metadata"]
            ready = (metadata.get("verification_receipt", {}).get("state") == "pending"
                     if boundary != "regeneration_intent" else "regeneration_intent" in metadata)
            if ready and boundary == "pending_replayed_verdict":
                # A successful journal replay still carries the earlier
                # failure until the ordinary node-update callback clears it.
                # The verifier's forced control checkpoint precedes that hook.
                metadata.update(
                    response_error={"type": "ProviderAccountingError", "message": "Earlier interruption"},
                    accounting_invalid=True, accounting_error="Earlier interruption", cost_usd_unknown=True,
                )
            saved = original(lease, revision, state, **kwargs)
            if ready:
                crashed.append(lease.run_id)
                raise ProcessLost("Stopped after durable verification control")
            return saved

        monkeypatch.setattr(store, "save_checkpoint", save)
        with pytest.raises(ProcessLost):
            swarm.execute(graph)
        run_id = crashed[0]
        saved = store.get_checkpoint(run_id)
        old_calls = store.inspect_run(run_id)["calls"]
        assert len(old_calls) == len(dispatched) == 3
        assert all(call["result_state"] == "applied" for call in old_calls)
        assert saved["checkpoint"]["graph"]["nodes"][1]["result"].startswith("FAIL")

        monkeypatch.setattr(store, "save_checkpoint", original)
        restored = swarm.resume(run_id)
        calls = store.inspect_run(run_id)["calls"]
        assert len(dispatched) == len(calls) == 7
        assert len(judges) == 2
        assert Counter(call["key"]["generation"] for call in calls
                       if call["key"]["phase"] == "supervision") == {0: 1, 1: 2}
        assert all(call["result_state"] == "applied" for call in calls)
        assert {call["call_id"] for call in old_calls}.issubset({call["call_id"] for call in calls})
        assert all(node.status is NodeStatus.COMPLETED for node in restored.graph.nodes)
        assert all(node.metadata["execution_generation"] == 1 for node in restored.graph.nodes)
        assert restored.graph.nodes[1].metadata["regenerations_used"] == 1
        if boundary == "pending_replayed_verdict":
            assert saved["checkpoint"]["graph"]["nodes"][1]["metadata"]["accounting_invalid"]
            assert not {"response_error", "accounting_invalid", "accounting_error", "cost_usd_unknown"}.intersection(
                restored.graph.nodes[1].metadata,
            )
        assert store.audit(run_id)["ok"] and restored.cost_is_complete


@pytest.mark.parametrize("mutation", ["drop", "rewire"])
def test_saved_proposal_cannot_change_a_journal_accepted_worker(tmp_path, monkeypatch, mutation):
    """Checkpoint lag cannot turn already dispatched work back into editable work."""
    dispatched = []

    async def scenario():
        entered, release, accepted = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def complete(self, system, prompt, model):
            dispatched.append((system, prompt))
            if system == SUPERVISOR_SYSTEM_PROMPT:
                if "was 'a'." in prompt:
                    release.set()
                    await asyncio.wait_for(accepted.wait(), 15)
                    proposal = {"change": True, "reason": "Saved proposal", "add": [],
                                "drop": ["b"] if mutation == "drop" else [],
                                "rewire": {"b": ["a"]} if mutation == "rewire" else {}}
                    return CompletionResult(json.dumps(proposal), cost_usd=0)
                return CompletionResult('{"change":false}', cost_usd=0)
            if prompt == "A":
                await asyncio.wait_for(entered.wait(), 15)
            elif prompt == "B":
                entered.set()
                await release.wait()
            return CompletionResult(f"Accepted {prompt}", cost_usd=0)

        monkeypatch.setattr(OfflineProvider, "complete", complete)
        with SQLiteWorkflowStore(tmp_path / f"proposal-{mutation}.db") as store:
            provider = OfflineProvider()
            swarm = Swarm(
                model="offline", provider=provider, architect=SimpleArchitect(), run_store=store,
                supervisor=LLMSupervisor(provider, model="offline", only_terminal=False), max_revisions=1,
                parallel=True, max_concurrency=2, checkpoint_every_n_nodes=100,
            )
            save, accept = store.save_checkpoint, store.accept_result
            crashed = []

            def accept_result(*args, **kwargs):
                record = accept(*args, **kwargs)
                if record["key"]["scope_id"] == "node/b":
                    accepted.set()
                return record

            def save_checkpoint(lease, revision, state, **kwargs):
                operation = store.load_operation(lease.run_id, "supervision/a/0")
                if crashed or operation is not None and operation["state"] == "completed":
                    crashed.append(lease.run_id)
                    raise ProcessLost("Parsed proposal saved, graph disposition not saved")
                return save(lease, revision, state, **kwargs)

            monkeypatch.setattr(store, "accept_result", accept_result)
            monkeypatch.setattr(store, "save_checkpoint", save_checkpoint)
            try:
                with pytest.raises(ProcessLost):
                    await swarm.execute_async(ExecutionGraph([Topology.FORK_JOIN], [
                        Node(id="a", label="A"), Node(id="b", label="B"),
                    ]))
            finally:
                release.set()
            run_id = crashed[0]
            checkpoint = store.get_checkpoint(run_id)["checkpoint"]
            assert {node["id"]: node["status"] for node in checkpoint["graph"]["nodes"]} == {
                "a": "completed", "b": "running",
            }
            before = store.inspect_run(run_id)["calls"]
            b_call = next(call for call in before if call["key"]["scope_id"] == "node/b")
            assert b_call["result_state"] == "accepted"
            operation = store.load_operation(run_id, "supervision/a/0")
            proposal = operation["result"]["revision"]
            assert proposal["drop_node_ids"] == (["b"] if mutation == "drop" else [])
            assert proposal["rewire"] == ({"b": ["a"]} if mutation == "rewire" else {})

            monkeypatch.setattr(store, "save_checkpoint", save)
            resumed = await swarm.aresume(run_id)
            nodes = {node.id: node for node in resumed.graph.nodes}
            assert set(nodes) == {"a", "b"}
            assert nodes["b"].depends_on == [] and nodes["b"].result == "Accepted B"
            assert all(node.status is NodeStatus.COMPLETED for node in nodes.values())
            assert nodes["a"].metadata["workflow_supervision"]["revision_applied"] is False
            after = store.inspect_run(run_id)["calls"]
            assert len(dispatched) == len(after) == 4
            assert all(call["result_state"] == "applied" for call in after)
            assert next(call for call in after if call["call_id"] == b_call["call_id"])["result_state"] == "applied"
            saved_operation = store.load_operation(run_id, "supervision/a/0")
            assert saved_operation["result"] == operation["result"]  # retain the original proposal
            assert saved_operation["state"] == "applied" and store.audit(run_id)["ok"]

    asyncio.run(scenario())


@pytest.mark.parametrize("request_matches", [False, True])
def test_revision_cannot_add_a_node_under_an_id_with_journaled_calls(
    tmp_path, monkeypatch, request_matches,
):
    """Calls are keyed by node id and generation. "A" runs, the gate resets it,
    and a revision drops it; a new generation-0 node under its id then
    conflicted with its saved call, or replayed its result without running."""
    judges, steps = [], []

    async def complete(self, system, prompt, model):
        if prompt.startswith("Judge"):
            judges.append(prompt)
            return CompletionResult("FAIL: revise" if len(judges) == 1 else "PASS", cost_usd=0)
        if prompt.startswith("Step"):
            steps.append(prompt)
        return CompletionResult("Accepted " + prompt.splitlines()[0], cost_usd=0)

    class Review:
        async def review(self, graph, node, **kwargs):
            ids = {item.id for item in graph.nodes}
            if node.id == "draft" and node.metadata.get("execution_generation", 0) == 0:
                return Revision(add_nodes=(Node(id="A", label="Step A", depends_on=["draft"]),),
                                rewire={"judge": ("draft", "A")})
            if node.id == "draft" and "A" in ids:
                return Revision(drop_node_ids=("A",), rewire={"judge": ("draft",)})
            if node.id == "judge" and "A" not in ids:
                if not request_matches:
                    revised = Node(id="A", label="Step A, revised", depends_on=["draft"])
                    return Revision(add_nodes=(revised,))
                # A dependent keeps the new "A" non-terminal, so its prompt matches the old one.
                return Revision(add_nodes=(Node(id="A", label="Step A", depends_on=["draft"]),
                                           Node(id="B", label="Step B", depends_on=["A"])))
            return None

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    with SQLiteWorkflowStore(tmp_path / "readd.db") as store:
        swarm = Swarm(
            model="offline", provider=OfflineProvider(), architect=SimpleArchitect(), run_store=store,
            supervisor=LocalOnly(Review, "readd-review", "1", role="supervisor"), max_revisions=3,
        )
        result = swarm.execute(ExecutionGraph([Topology.SERIAL], [
            Node(id="draft", label="Draft"),
            Node(id="judge", label="Judge", depends_on=["draft"], verifies="draft", max_regenerations=1),
        ]))
        assert [(node.id, node.status) for node in result.graph.nodes] == [
            ("draft", NodeStatus.COMPLETED), ("judge", NodeStatus.COMPLETED),
        ]
        assert result.output == "Accepted Draft"
        assert len(judges) == 2 and len(steps) == 1
        rejected = [span for span in result.trace if span["status"] == "revision_rejected"]
        assert len(rejected) == 1
        assert "already has calls in the durable journal" in rejected[0]["error"]
        assert store.get_checkpoint(result.execution_id)["checkpoint"]["revisions_used"] == 2
        keys = [call["key"] for call in store.inspect_run(result.execution_id)["calls"]]
        assert [key["generation"] for key in keys if key["scope_id"] == "node/A"] == [0]
        assert swarm.resume(result.execution_id).output == result.output
        assert store.audit(result.execution_id)["ok"]


def test_grouped_supervisor_journal_error_is_not_committed_as_nochange(tmp_path):
    error = WorkflowLeaseError("Fenced supervisor operation")

    class GroupedReview:
        async def review(self, *args, **kwargs):
            raise ExceptionGroup("context exit", [ValueError("ordinary"), error])

    with SQLiteWorkflowStore(tmp_path / "group.db") as store:
        swarm = Swarm(
            model="offline", provider=OfflineProvider(), architect=SimpleArchitect(), run_store=store,
            supervisor=LocalOnly(GroupedReview, "grouped-review", "1", role="supervisor"), max_revisions=1,
        )
        planned = swarm.plan(Task("One worker"))
        with pytest.raises(WorkflowLeaseError) as caught:
            swarm.execute(planned)
        assert caught.value is error
        run_id, node_id = planned.run_ref["run_id"], planned.nodes[0].id
        operation = store.load_operation(run_id, f"supervision/{node_id}/0")
        assert operation["state"] == "started" and operation["result"] is None
        assert store.inspect_run(run_id)["call_count"] == 1
