# Jobs scale harness pilots — September 7, 2026

These are retained **25-operation harness-development pilots**, both offline
and at concurrency eight. They contain no 5,000-operation result and support
no comparative performance, production-image capacity, or speedup claim.

| Pilot | Status | Lease TTL / heartbeat | Observation |
| --- | --- | --- | --- |
| [Aggressive lease](ttl5_failed/diagnostic.json) | Failed; unbound diagnostic | 5 s / 0.25 s | The start barrier reached 12 provider entries and four accepted operations. The owned worker was hard-killed. Resume then failed with `RunLeaseError`. |
| [Default lease](ttl30_passed/result.json) | Completed; pre-freeze pilot | 30 s / 1 s | 25 artifacts accepted; 33 provider entries, including eight explicitly acknowledged rerolls; zero automatic redispatches of unknown or already accepted operations. |

The failed pilot's [original resume log](ttl5_failed/resume-worker.log) is
retained. Its uncommitted harness did not serialize provenance before failure;
neither an exact source revision nor missing source hashes are reconstructed.
This pilot used a shorter lease than Jobs' production default. The recorded
failure does not establish a failure at the default lease setting.

The successful pilot ran on a dirty working tree based on
`aec976aab1b9bab005b51b6ff2f4fbba0864d776`. Its original record retains source
hashes and the dirty-file list, so that base revision alone does not identify
the executed source. Those source hashes use the then-current raw-byte policy,
before the final harness adopted LF-normalized source hashes. The harness also
gained stricter call-ledger and interrupted-campaign checks after this pilot.
Do not present this pilot as a run of the subsequently frozen source commit.

All artifacts are the same deterministic 70-byte, 1×1 PNG; identical content
hashes are expected. Operation, attempt, and call identities establish which
work ran again. The successful pilot's raw SQLite database, phase results,
provider-entry logs, and 25 accepted PNGs are included for inspection. Its
recorded 57.16-second elapsed time includes 29.27 seconds waiting for real
lease expiry and the harness' durable evidence logging. This is one local
observation, with no statistical or comparative interpretation.

[archive.json](archive.json) records every file under the two pilot folders
with its size and raw-byte SHA-256. The original result
and diagnostic files are preserved byte-for-byte. Their original evidence
inventories were checked against the packaged files, and the successful
pilot's accepted receipts were checked against its database and PNG bytes.
Local Git attributes preserve original evidence bytes across checkouts.

The current [campaign protocol](../../jobs_scale_benchmark.md) defines the
separate 5,000-operation run. These pilots are diagnostics and must not be
promoted as that campaign's result.
