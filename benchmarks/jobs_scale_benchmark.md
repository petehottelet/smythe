# Jobs scale and recovery check

**Status: protocol in preparation; no 5,000-operation result published yet.**
This is an offline correctness and recovery campaign. It does not measure
model quality, glyph generation, provider latency, or comparative throughput.

## Workload

Execute 5,000 independent artifact operations through `JobRunner`, using
SQLite WAL mode with `synchronous=FULL`. Each operation returns the same tiny
1×1 PNG fixture. Identical image hashes are expected; operation and attempt
identities determine whether work was repeated. Provider API charges are zero.

The manifest freezes operation count, concurrency, attempt allowance, output
contract, and budget. Each operation has two allowed attempts. The parent
process owns the worker subprocess and the fresh campaign directory. The
campaign never kills an unrelated process or reuses a previous result folder.

The default configuration uses eight concurrent operations, the production
30-second lease TTL, and a one-second renewal interval. Run from a checkout
with the Jobs dependencies installed:

```bash
python benchmarks/jobs_scale_benchmark.py \
  --workdir smythe/tmp/jobs-scale-evidence \
  --output smythe/tmp/jobs-scale-result.json
```

Both paths must be new. The result file must sit outside the evidence folder.
Worker logs, the SQLite journal, provider-entry logs, and accepted files remain
on disk after completion or failure. A failure result is diagnostic evidence;
it does not count as a completed campaign.

## Interruption and recovery

1. Start the approved job in a worker subprocess. Log provider entry with its
   operation, attempt, and call identity before waiting at the crash barrier.
2. At the midpoint of provider entries, hold the final concurrent calls before
   they return. Wait until earlier accepted artifacts have committed, then
   hard-kill the owned worker. This fixes the interruption state without
   relying on a guessed sleep duration.
3. Wait for the actual run lease to expire. Do not edit lease timestamps.
4. Resume pending work through the public runner. Preserve every previously
   accepted pointer and artifact hash. Interrupted dispatched operations must
   become `unknown_outcome` and must not be automatically reissued.
5. Explicitly reroll only those unknown operations with
   `acknowledge_unknown=True`, retaining their previous attempts.
6. Verify all 5,000 accepted files, recorded sizes, SHA-256 hashes, MIME types,
   and dimensions. Resume the completed job again and verify zero new provider
   entries.

The explicit reroll is part of the protocol. Jobs does not promise automatic
deduplication of a request whose remote outcome is unknown. The fixture makes
that ambiguity safe to exercise without a paid provider.

## Evidence

The result must retain phase timing, operation counts, actual provider-entry
counts, interrupted identities, accepted-pointer preservation, attempt lineage,
artifact validation, and the completed-resume check. Timing includes the
runner's local persistence work; validation and lease-wait time are reported
separately. Every required check must pass before the record is marked complete.

Measure provider concurrency independently. The runner's existing
`peak_active_calls` metric counts active operation scopes, including artifact
finalization; it is not a measurement of simultaneous provider execution.
The killed worker cannot return its final runner metrics, so unavailable
values remain unavailable.

Record platform, Python, SQLite, Pillow, revision, harness and runtime source
hashes, fixture byte size, artifact totals, and database disk size. Keep output
references relative. Retain failed or incomplete campaigns with their status;
do not select a faster rerun as the reported result.
Python source hashes use LF-normalized tracked files. Retained evidence hashes
cover the original bytes, including provider logs and the SQLite database.

This is a single-host engineering check. It supports only the observed
completion and recovery assertions. It provides no speedup estimate,
production-image capacity claim, or statistical latency comparison.

## Graph-depth coverage

Jobs are flat artifact fan-out. Their scale result does not establish deep-DAG
execution. Separate graph regressions cover iterative validation, cycle
detection, depth, dependency order, revisions, and serial execution beyond
Python's default recursion limit. Those checks must preserve existing
dependency ordering and reject invalid revisions before mutating the graph.
