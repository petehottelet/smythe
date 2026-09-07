# Complete text-workflow accounting

This implements the next item in the [completion campaign](coming-soon-2026-09-07.md)
and supplies the accounting foundation for the [Astra protocol](../benchmarks/astra_benchmark_plan.md).

## Public contract

Opt in with `Swarm(run_store=SQLiteWorkflowStore(path))`. Create the durable
run before routing or planning. A graph returned by `plan()` carries its store,
run, and recipe identities; `execute(graph)` continues that run. The existing
unmanaged Swarm API remains available.

The first version supports text workflows with native Astra/Sol Responses and
stateless offline planning/echo fixtures. It binds every reachable built-in
component before the first provider call. Explicit `LocalOnly` factories can
declare custom deterministic components with no provider calls or external
side effects. This declaration is a caller contract, not a Python sandbox.

Reject configured unsupported providers, inherited custom implementations,
external skill hydration, live memory, scripted/artifact offline providers,
and tool runtimes before the first provider call. Validate generated nodes and
reject attachments before worker admission. Durable tool side effects and
idempotent memory projections require their own journals.

## Durable boundaries

1. Save the complete Task, static registry, component recipe, provider settings,
   price version, and budget before routing or planning.
2. Bind each call to phase, component/node, generation, durable invocation,
   repair/retry attempt, and turn. Reusing an identity with changed request,
   neutral tool-name map, adapter, or price bindings fails.
3. Preserve input-count evidence and reserve the exact conservative quote in
   one transactional ledger. Only the first successful dispatch claim permits
   a generation request.
4. Commit raw response bytes before pricing or decoding. Known bills replace
   their reservations once; unknown outcomes retain separate exposure.
5. Save native accepted output separately from application acceptance. An
   invalid planner response remains a paid attempt even when a repair succeeds.
6. Save parsed graph and registry IDs before execution. Freeze supervisor input
   before its call and persist its decision, including no-change outcomes.
7. Atomically save graph/control state with consumed call and operation IDs.
   Preserve every incurred charge across retries, regeneration, and resume.
8. On recovery, finish local pricing, decoding, and graph application from
   saved evidence. A dispatched call without saved response evidence is unknown
   and cannot be sent again automatically.

## Storage and isolation

`SQLiteWorkflowStore` has its own database identity and schema. It uses WAL,
FULL synchronization, and immediate write transactions. Monetary values are
canonical integer nanoUSD text; Python performs exact arithmetic. Transactional
run balances avoid repeated scans of all calls. Recovery audits recompute them.

Lease epochs fence admission, dispatch, checkpoints, and control transitions.
A stale worker may append immutable evidence for its exact dispatch token;
it cannot admit work or advance the graph. Lease loss, unknown exposure, and
overruns close admission. A resumed run keeps its original budget and counters.

Each run receives fresh component and registry snapshots. A binding tree
shares one snapshot/session for a shared source provider. Sessions close in
their creating event loop; a later `execute(graph)` reopens connections while
keeping the same durable run identity.

Routine inspection contains safe identifiers, phases, usage, prices, statuses,
and evidence hashes. Raw requests, outputs, and encrypted reasoning require
explicit evidence access. Result totals come from the journal; the legacy
execution budget view must not charge the same completion again.

## Required checks

- Exact-fit concurrent admission, duplicate dispatch claims, known overruns,
  unknown exposure followed by a separate known charge, and balance audits.
- Crashes between preparation, quote, reservation, dispatch, raw persistence,
  settlement, native acceptance, application acceptance, and checkpoint commit.
- Stale lease fencing, immutable evidence conflicts, malformed persisted data,
  and inspection without raw content leakage.
- Routing and every planner repair, worker, verifier, supervisor, and synthesis
  call present in the same ledger; deterministic phases record zero API cost.
- Plan/execute handoff, failed planning recovery, frozen supervisor replay,
  graph revisions, verification generations, and completed synthesis replay.
- All unsupported component paths rejected before any paid call; concurrent
  runs remain isolated; the complete offline suite and GitHub CI pass.
