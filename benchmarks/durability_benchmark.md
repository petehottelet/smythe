# Durability benchmark: dispatch exposure after kill and resume

The question under test is: **after an abrupt process termination during a
wide fan-out, how many operation IDs are dispatched more than once after
resume?** A repeated dispatch is conservative duplicate-spend exposure. It is
not proof of an invoice charge because a real provider may reject, cancel, or
deduplicate a request.

Harness: [run_durability_benchmark.py](run_durability_benchmark.py). Current
raw record: [results/durability_kill_resume_v2.json](results/durability_kill_resume_v2.json).
The workload is entirely offline and uses no API keys.

## Evidence model

Each simulated attempt appends two JSONL records:

1. `dispatched`, written and `fsync`'d before simulated remote latency; and
2. `completed`, written after the simulated response returns.

Both records carry a stable operation ID and a fresh attempt ID. The
orchestrator kills the worker when the durable dispatch count reaches the
configured threshold. After resume, the harness validates the exact expected
operation-ID inventory and reports:

- dispatches and completions observed at the kill point;
- attempts still in flight at the kill point;
- total dispatches after resume;
- repeated dispatches, computed per operation ID; and
- the exact replayed operation IDs.

Recording before the simulated request is deliberately conservative: a killed
process may record exposure immediately before work that never reaches a real
provider. Recording after completion, however, would be unsafe because it
would omit requests that reached a provider but outlived their client.

## Current v2 result

Run on 2026-07-16 using Windows, Python 3.11.7, N=64, concurrency 8,
100 ms simulated latency, hard kill at 32 durable dispatches, and
`FileCheckpointStore(checkpoint_every_n_nodes=1)`:

| System | Repeated dispatches | In flight at kill | Resume completed |
|---|---:|---:|---:|
| smythe, rep 1 | 8 | 8 | 64/64 |
| smythe, rep 2 | 4 | 1 | 64/64 |
| smythe, rep 3 | 8 | 4 | 64/64 |

Repeated dispatches averaged 6.7 and ranged from 4 to 8. Two runs replayed
`n0024` through `n0031`; one replayed `n0028` through `n0031`. The observed
maximum remained one concurrency wave. Some attempts completed before
termination but had not yet reached a durable node checkpoint, which is why
the replay set can be larger than the instantaneous in-flight count.

LangGraph was not installed in the environment used for this v2 run, so the
previous framework comparison has **not** been re-established under the new
accounting. The historical
`results/durability_kill_resume.json` record used completion-time log lines and
must not be cited as billed-call or dispatch-exposure evidence.

## Protocol

- **Cell A — fan-out scheduler overhead.** Independent broadcast nodes, 25 ms
  simulated calls, concurrency 16, and no checkpoint persistence. Both
  dispatch and completion events are recorded without per-event `fsync`.
- **Cell B — kill and resume.** N=64, 100 ms calls, concurrency 8, durable
  event logging, and a hard process kill at 32 dispatches. Smythe uses a file
  checkpoint after every completed node. The optional comparison lane uses
  LangGraph `AsyncSqliteSaver` with `durability="sync"`.

The star graph intentionally represents a worst-case wide fan-out. Deep serial
chains can have materially different replay behavior.

## Boundaries

- This is a process-kill test, not a host power-loss or disk-failure test.
  `FileCheckpointStore` uses atomic replacement but does not currently `fsync`
  the checkpoint file and containing directory.
- This lane exercises legacy `Swarm`/`FileCheckpointStore`, not Jobs. Jobs
  records an interrupted dispatched call as `unknown_outcome` and does not
  automatically replay it. A dedicated Jobs kill lane remains required.
- The simulation excludes provider retries, rate limits, server-side request
  persistence, and invoice reconciliation.
- The call-event file is the conservative benchmark recorder. It is not a
  provider-side or independently hosted audit log.
- Resume wall time includes interpreter, JSON event `fsync`, and checkpoint
  startup overhead; repeated dispatch count is the primary Cell B metric.

## Reproducing

```bash
python -m pip install -e ".[dev]"
python benchmarks/run_durability_benchmark.py --cell b

# Optional framework comparison
python -m pip install langgraph langgraph-checkpoint-sqlite
python benchmarks/run_durability_benchmark.py --cell b
```

Use `--quick` for one repetition and `--out PATH` to preserve a machine-readable
record.
