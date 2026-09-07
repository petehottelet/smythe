"""Plan, execute, inspect, and resume one text workflow without an API key."""

from pathlib import Path
from tempfile import TemporaryDirectory

from smythe import OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task


def main():
    with TemporaryDirectory(prefix="smythe-workflow-") as directory:
        with SQLiteWorkflowStore(Path(directory) / "runs.db") as store:
            swarm = Swarm(
                model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                run_store=store, max_budget_usd=1.00, max_concurrency=4,
            )
            graph = swarm.plan(Task("Write a short explanation of a directed acyclic graph."))
            result = swarm.execute(graph)
            resumed = swarm.resume(result.execution_id)
            assert resumed.output == result.output
            assert store.audit(result.execution_id)["ok"]
            print(result.output)
            print(result.workflow_accounting)


if __name__ == "__main__":
    main()
