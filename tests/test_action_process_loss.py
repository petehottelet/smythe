"""A worker killed while its external write is in flight leaves an unknown outcome, never a resend.

A real child process claims the dispatch of an approved action and sends it to
the fake external service over loopback HTTP. The service commits the write
and holds its answer at a gate, and the test kills the child with
TerminateProcess or SIGKILL, so no cleanup handler runs and its lease is never
released. A successor then recovers the run past the lost lease.
"""

from __future__ import annotations

import time

import pytest

from fakes import Gate, ManualClock, kill_and_reap, launch_child
from fixture_service import FixtureService, serve_http
from smythe import SQLiteWorkflowStore
from smythe.actions import ActionStateError, check_authority
from smythe.workflow_actions import begin_attempt, recover_actions
from test_action_approvals import Host, approved, completed, prepare
from test_workflow_acceptance import replies  # noqa: F401

# Child code: dispatch the action once and send its write, which the service holds.
CHILD = r"""
import sys

from fixture_connector import HttpFixtureConnector
from fixture_service import Endpoint
from smythe import SQLiteWorkflowStore
from smythe.actions import load_action
from smythe.workflow_actions import begin_attempt, claim_dispatch, recover_actions

path, port, action_id, approval_id = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
with SQLiteWorkflowStore(path) as store:
    action = load_action(store, action_id)
    lease = store.acquire_lease(action.run_id, "dispatching-worker", ttl_s=60)
    recover_actions(store, lease)
    attempt = begin_attempt(store, lease, action_id, approval_id)
    claim_dispatch(store, lease, attempt["attempt_id"])
    document = store.get_checkpoint(action.run_id)["checkpoint"]["output"].encode("utf-8")
    connector = HttpFixtureConnector(Endpoint("127.0.0.1", port), action.profile)
    connector.submit(action.action_key, action.intent, document, 600)
sys.exit("The held write was answered")
"""


def test_a_killed_dispatch_recovers_as_an_unknown_outcome_without_a_resend(tmp_path, replies):  # noqa: F811
    path, clock, host = tmp_path / "runs.db", ManualClock(time.time_ns()), Host()
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        action = prepare(store, completed(store, tmp_path, replies, renderer=None))
        approval = approved(store, host, action, clock)
    write = action.profile.write_ceiling_nanousd
    # A fixed service clock keeps the service's snapshots comparable.
    service, gate = FixtureService(clock_ns=ManualClock()), Gate()
    service.hold(action.action_key, gate)  # Commit the write, then hold its answer.

    with serve_http(service) as endpoint:
        process = launch_child(CHILD, path, endpoint.port, action.action_id, approval.approval_id)
        try:
            try:
                assert gate.entered.wait(60), kill_and_reap(process)
                held = service.snapshot()
                reaped = kill_and_reap(process)
            finally:
                kill_and_reap(process)
            assert reaped.killed and reaped.returncode != 0
            assert held["requests"][action.action_key] == {
                "received": 1, "committed": 1, "deduplicated": 0, "refused": 0, "acknowledged": 0,
                "ack_dropped": 0, "object_ids": ["obj_1"]}

            # The child's lease is never released; a successor recovers once it lapses.
            with SQLiteWorkflowStore(path, clock_ns=ManualClock.after(120)) as store:
                (stopped,) = store.inspect_run(action.run_id)["actions"]["items"]
                (attempt,) = stopped["attempts"]
                assert (stopped["state"], attempt["state"], attempt["reserved_nanousd"]) == (
                    "dispatched", "dispatched", write)
                lease = store.acquire_lease(action.run_id, "successor", ttl_s=60)
                assert lease.epoch == attempt["dispatch_epoch"] + 1

                assert recover_actions(store, lease) == {
                    "released": [], "outcome_unknown": [attempt["attempt_id"]]}

                (recovered,) = store.inspect_run(action.run_id)["actions"]["items"]
                assert recovered["state"] == "outcome_unknown"
                assert (recovered["confirmed_nanousd"], recovered["reserved_nanousd"],
                        recovered["unknown_nanousd"]) == (0, 0, write)
                assert recovered["attempts"] == [{**attempt, "state": "outcome_unknown",
                                                  "reserved_nanousd": 0, "unknown_nanousd": write}]
                # The approval was not consumed, but nothing resends a write that may have
                # taken effect, and the exposure stays.
                assert check_authority(store, action.action_id, host).approval_id == approval.approval_id
                with pytest.raises(ActionStateError, match="is outcome_unknown"):
                    begin_attempt(store, lease, action.action_id, approval.approval_id)
                assert recover_actions(store, lease) == {"released": [], "outcome_unknown": []}
                assert store.audit(action.run_id)["ok"]
            # Recovery sent nothing to the service.
            assert service.snapshot() == held
        finally:
            gate.release()
