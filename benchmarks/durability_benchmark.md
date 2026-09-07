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

## Current v2 result (2026-08-05, framework comparison re-established)

Run on Windows, Python 3.12.7, N=64, concurrency 8, 100 ms simulated
latency, hard kill at 32 durable dispatches; Smythe uses
`FileCheckpointStore(checkpoint_every_n_nodes=1)`, LangGraph uses
`AsyncSqliteSaver` with `durability="sync"` (strongest persistence both
sides), 3 reps each:

| System | Repeated dispatches | Resume wall | Resume completed |
|---|---:|---:|---:|
| smythe, rep 1 | 8 | 5.6 s | 64/64 |
| smythe, rep 2 | 8 | 6.0 s | 64/64 |
| smythe, rep 3 | 8 | 4.6 s | 64/64 |
| langgraph, rep 1 | 32 | 15.9 s | 64/64 |
| langgraph, rep 2 | 32 | 17.0 s | 64/64 |
| langgraph, rep 3 | 32 | 19.3 s | 64/64 |

Smythe's repeated dispatches were exactly one concurrency wave (8) in every
rep; LangGraph re-dispatched all 32 previously dispatched operations in every rep, because
Pregel checkpoints at superstep boundaries and a wide broadcast is one
superstep. That is **75% fewer repeated dispatches** in this 64-node kill
profile. Provider billing and wider jobs were not measured by this record.

At the kill point, Smythe had completed 24 operations with 8 attempts still in
flight in each rep. LangGraph had completed 24, 24, and 25 operations, with 8,
8, and 7 attempts still in flight. The repeated-dispatch count includes both
completed and interrupted attempts when they are dispatched again.

Cell A in the same record shows per-node fan-out overhead near parity
(smythe 0.99–1.19 ms/node vs LangGraph 0.89–1.49 ms/node at N ≤ 1024) with
graph construction at ~2 ms vs ~396 ms for N=1024. Published as measured:
LangGraph's raw scheduler is not slower than Smythe's; the differences that
matter here are replay exposure and resume time.

The historical `results/durability_kill_resume.json` record used
completion-time log lines and must not be cited as billed-call or
dispatch-exposure evidence.

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
