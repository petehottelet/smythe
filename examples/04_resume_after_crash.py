"""Durable execution: crash mid-run, then resume from the checkpoint.

    python examples/04_resume_after_crash.py

A three-node pipeline is executed with a provider that dies on the
second node. The Swarm checkpoints after every node, so the failed run
leaves a resumable state on disk. A second Swarm — standing in for a
fresh process — resumes it: the first node's result is restored from
the checkpoint (not re-executed) and only the remaining work runs.

Runs fully offline. Checkpoints are written to a temp directory.
"""

import tempfile

from _providers import DemoProvider, pick_provider

from smythe import FileCheckpointStore, Swarm
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider import CompletionResult


class CrashyProvider(DemoProvider):
    """DemoProvider that fails exactly once on the 'analyze' node."""

    def __init__(self) -> None:
        super().__init__()
        self.crashed = False

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        if prompt.startswith("Analyze") and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated crash: provider died mid-run")
        return await super().complete(system, prompt, model)


def make_graph() -> ExecutionGraph:
    gather = Node(id="gather", label="Gather raw notes on the topic")
    analyze = Node(id="analyze", label="Analyze the notes", depends_on=["gather"])
    report = Node(id="report", label="Write the final report", depends_on=["analyze"])
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[gather, analyze, report])


provider, model = pick_provider()
if not isinstance(provider, DemoProvider):
    print("(API key detected, but this example always uses the offline provider\n"
          " so the crash can be simulated deterministically.)\n")
provider = CrashyProvider()

checkpoint_dir = tempfile.mkdtemp(prefix="smythe-checkpoints-")
store = FileCheckpointStore(checkpoint_dir)

print("=== First run (will crash on node 2 of 3) ===")
swarm = Swarm(provider=provider, model=model, checkpoint_store=store, parallel=True)
try:
    swarm.execute(make_graph())
except RuntimeError as exc:
    print(f"Execution failed: {exc}")

[execution_id] = store.list_ids()
state = store.load(execution_id)
print(f"\nCheckpoint on disk: {checkpoint_dir}\\{execution_id}.json")
for node in state["graph"]["nodes"]:
    print(f"  {node['id']:<8} {node['status']}")

print("\n=== Resume in a 'new process' ===")
fresh_swarm = Swarm(provider=provider, model=model, checkpoint_store=store)
result = fresh_swarm.resume(execution_id)

print(result.output)
print(f"\nResumed execution {result.execution_id} completed.")
print("Node 'gather' was restored from the checkpoint, not re-executed.")
