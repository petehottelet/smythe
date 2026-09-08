# Concurrent SQLite initialization

Status: implemented and reviewed; full qualification in progress. Unreleased
after 0.7.0.
This work preserves the existing journal formats, evidence, and public commands.

## Observed failures

The frozen `cf93435` offline suite passed 3,335 tests with 10 skipped and one
failure. Two fresh Autotune constructors collided while enabling WAL mode.
The failed run and its unchanged source manifest are retained. This was a
fresh-initialization failure; the regression also covers v3 migration.

A separate bounded probe used actual Jobs and workflow constructors with
independent SQLite connections on Windows, Python 3.11.9, SQLite 3.45.1.
It stopped each case at the first observed failure:

| Store | Starting database | Observed failure |
|---|---|---|
| Jobs | Fresh | Second paired trial: `SQLITE_BUSY` while enabling WAL |
| Jobs | Valid, DELETE journal mode | Second paired trial: `SQLITE_BUSY` while enabling WAL |
| Workflow | Fresh | First paired trial: one opener observed the other's partially created schema |
| Workflow | Valid, DELETE journal mode | Fourth paired trial: `SQLITE_BUSY` while enabling WAL |

All nine retained probe databases passed `integrity_check`; their foreign-key
checks were empty. These are constructor correctness diagnostics, not a
failure-rate estimate, throughput benchmark, or evidence of corrupted data.
No provider calls were made. The original sources, errors, and probe outcomes
remain preserved separately from later verification runs.

## Shared WAL setup

Use one internal helper for Autotune, Jobs, and workflow journals. Keep each
store's schema preflight before journal changes. Recheck that preflight on a
WAL retry, and validate the current schema again after acquiring the write lock.

SQLite can return `SQLITE_BUSY` without invoking its configured busy handler
when waiting could deadlock. A larger timeout alone does not fix this path.
[SQLite busy-handler contract](https://www.sqlite.org/c3ref/busy_handler.html).

The helper retries only verified SQLite BUSY errors from WAL setup. It uses a
bounded monotonic retry budget, closes each cursor, and restores the caller's
busy timeout on every exit. Schema validation keeps its original timeout and
does not acquire a new size-dependent time limit. Other errors propagate.
Require the returned journal mode to be `wal`; SQLite can otherwise leave the
previous mode in place. [WAL activation](https://www.sqlite.org/wal.html#activating_and_configuring_wal_mode).

## Atomic schema creation

After WAL setup, acquire `BEGIN IMMEDIATE` and re-read the schema under that
write lock. A second initializer must use the first initializer's completed
schema and persistent identity. It must not trust a stale preflight decision
that the database was empty.

Execute the existing fixed DDL statements inside that transaction. Avoid
`executescript` inside a caller-owned transaction because its implicit commit
would split initialization. Create the workflow store UUID once, in the same
transaction as its tables. Reopen and migration preserve existing identities,
records, costs, leases, events, and schema versions.

On failure, roll back and close the connection. Reject foreign or unsupported
schemas before changing them. Read-only constructors perform no migration,
WAL setup, or write-lock acquisition.

## Qualification gates

Focused qualification passes all 229 Autotune/helper cases and all 260 Jobs
and workflow cases across their complete selected test files. Ruff and
independent reviews pass. Nine deterministic regression cases fail against
the retained `cf93435` source and pass after the repair. Old and new
`sqlite_master` definitions compare identically for both stores.

The regressions are committed in
[shared WAL tests](../tests/test_sqlite_initialization.py),
[Autotune constructor tests](../tests/test_optimize_lease.py), and
[Jobs/workflow initialization tests](../tests/test_jobs_workflow_initialization.py).
The full-suite result remains pending; focused checks do not replace it.

1. A real reader lock forces WAL contention; releasing it permits construction.
   Retain the old-source failure and verify the bounded retry and error paths.
2. Barrier-controlled concurrent initializers observe a complete schema and
   one persistent workflow identity. Keep the actual paired-opener diagnostics.
3. An injected DDL failure rolls back the entire schema creation. A new opener
   can then initialize the empty database normally.
4. Existing evidence, read-only bytes, unsupported-schema rejection, migration
   rules, lease fencing, and budget accounting retain their behavior.
5. Run each changed test file, relevant storage and Autotune checks, independent
   review, Ruff, the full frozen offline suite, and platform CI before closing
   the item in the [completion record](coming-soon-2026-09-07.md).

The related Jobs fixture cleanup gives 45 previously unclosed test-owned
connections explicit teardown. It preserves runner and thread lifetimes and
does not change production ownership of caller-provided stores.
