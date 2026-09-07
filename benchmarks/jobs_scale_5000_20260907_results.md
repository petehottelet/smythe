# Jobs scale and recovery: 5,000 operations

**Status: reviewed, claimable offline correctness observation; no known
measurement defects within that scope.**
One offline campaign on 7 September 2026 completed 5,000 artifact operations
after a hard process kill, safe resume, and eight explicit rerolls. The
[original result](results/jobs_scale_5000_20260907_f1/result.json) is preserved
unchanged in the [evidence bundle](results/jobs_scale_5000_20260907_f1/README.md).

This is a local correctness observation. Every output is the same 70-byte,
1×1 PNG fixture. It provides no comparative speed, model-quality, distinct-glyph,
or production-image capacity claim.

## Observed recovery

| Stage | Accepted operations | Pending operations | Interrupted or unknown operations |
|---|---:|---:|---:|
| After the hard kill | 2,492 | 2,500 | 8 still recorded as dispatched/running |
| After safe resume | 4,992 | 0 | 8 retained as `unknown_outcome` |
| After explicit rerolls | 5,000 | 0 | 0 operations awaiting an accepted result |

Safe resume issued no calls for previously accepted operations and did not
repeat the eight interrupted calls. Explicitly acknowledging a reroll for
those eight operations created new attempts linked to their unknown parents.
The final ledger retains **5,008 calls: 5,000 succeeded and eight unknown**.
Those historical unknown call records remain after replacement outputs are
accepted. Resuming the completed run made **zero new provider entries**.

All 5,000 accepted files total 350,000 bytes. Their identical content hash is
expected; operation IDs, attempt IDs, accepted pointers, and durable provider
entry records establish whether work was repeated. This workload is separate
from the original SVG generation and native screensaver catalogs.

## Frozen workload and host

The [predeclared protocol](jobs_scale_benchmark.md) used concurrency eight,
two attempts per operation, SQLite WAL mode with `synchronous=FULL`, a
30-second lease TTL, and one-second renewal. A deterministic barrier held
eight calls after the 2,500th provider entry and after 2,492 earlier outputs
had committed. The parent terminated its owned Windows worker with
`TerminateProcess`, then waited for actual lease expiry without editing the
database. The recorded lease wait was 27.25 seconds.

Producer source was frozen at
[`4bb7c0295cec658c7118d7bf9305764dae9b1c57`](https://github.com/petehottelet/smythe/commit/4bb7c0295cec658c7118d7bf9305764dae9b1c57),
with 59 source-file hashes checked before and after execution. This is the
**Jobs schema-v3 implementation** at that revision. Later schema-v4 namespace,
pause, and detached-operator changes are outside this campaign's coverage.

The host reported Windows kernel/build `10.0.26200`, AMD64, 32 logical CPUs,
Python 3.11.9, SQLite 3.45.1, and Pillow 11.1.0. Other local development and
test work occurred during the campaign; an idle host was not a performance
gate.

## Recorded timing and accounting

| Measurement | Seconds |
|---|---:|
| Initial worker launch through hard-kill exit | 2,858.47 |
| Safe-resume worker launch through exit | 4,410.64 |
| Explicit-reroll worker launch through exit | 9.27 |
| Completed-resume worker launch through exit | 2.22 |
| Real lease-expiry wait | 27.25 |
| Crash, resume, and final validation combined | 6.63 |
| Remaining parent setup/coordination, calculated by subtraction | 1.74 |
| **Complete harness runtime** | **7,316.22** |

The total is about two hours and two minutes. Timings include local SQLite
persistence, per-entry/return JSONL `fsync`, subprocess imports, artifact I/O,
and benchmark observation. They describe this single run and are not a
throughput comparison. Independently logged fixture-provider concurrency
peaked at eight. The runner's active-operation metric also includes artifact
finalization; its first-phase value is unavailable because the process was
killed before it could publish final metrics.

There were **zero remote API calls and zero API charges**. Approved,
confirmed, reserved, and exposure balances stayed at zero. The eight
interrupted call records retain incomplete/estimated cost flags, and the
run's estimate flag remains set after recovery. This explicit zero-cost
fixture does not establish live-provider billing reconciliation.

## Independent archive review

The [review passed](results/jobs_scale_5000_20260907_f1/review.json): 5,000
operations, 5,008 calls, 20,051 durable events, all accepted files, and all
eight explicit reroll lineages reconcile. The record remains
`comparative_claimable: false`.

The producer exited and its owned workers were confirmed closed before
archiving. The [archive tool](archive_jobs_scale.py) preserves every original
campaign file, the original result, frozen producer source, and separate
review source. It verifies each ZIP member against its recorded raw size and
SHA-256 before opening SQLite from an extracted temporary copy.

Review checks reconcile operation/call identities, event chronology, actual
kill and lease expiry, all accepted files, preserved pointers, reroll lineage,
and zero-cost balances. Historical accepted sets are reconstructed from the
final ledger, provider logs, timestamps, and retained receipt-subset hashes.
The bundle contains the final database, not historical database snapshots.

The [bundle guide](results/jobs_scale_5000_20260907_f1/README.md) documents the
15,088 archive members, exact hashes, and reproduction commands. The
[25-operation development pilots](results/jobs_scale_preflight_20260907/README.md)
retain their failed and completed outcomes separately. The existing
[matched framework recovery study](durability_benchmark.md) remains separate
comparative evidence.
