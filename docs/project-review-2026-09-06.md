# Project review — 6 September 2026

**Rating: 7/10. Keep the architecture and harden the runtime. A complete rewrite
would discard useful work without addressing the main problems.**

Smythe has a clear product: generate an inspectable execution graph for a goal,
then run it within a durable execution envelope. The repository implements
both ideas, with working provider adapters, tools, checkpoints, artifact jobs,
verification, supervision, and bounded optimization. The next improvement is
consistency: the older Swarm runtime should meet the stronger validation and
accounting standards already present in Jobs and Autotune.

This review examined the checkout at
`63941dc72ff7e94bb445c19d3e0932097ff7a6de`. It covered the public API, graph and
execution machinery, providers and MCP, persistence, assets, Jobs, Autotune,
tests, packaging, workflows, examples, and documentation. The tracked Python
runtime has 48 modules and 18,847 lines; 60 test files contain 839 test functions
before parameterization. These counts describe the reviewed commit, not a
coverage percentage.

The runtime findings below were reproduced with deterministic local providers.
No paid provider calls were made for this code review. Benchmark measurements
and the screensaver presentation are reviewed in their own
[protocol](../benchmarks/glyph_screensaver_benchmark.md) and
[source guide](../screensaver/README.md).

## Assessment

| Area | Rating | Assessment |
|---|---:|---|
| Product and architecture | 8/10 | The generated graph and durable envelope are useful, distinct abstractions. Planning, assignment, execution, and synthesis have explicit boundaries. |
| Core correctness | 6/10 | Budget validation, concurrent verification, task propagation, and serial failure behavior have reproducible gaps. |
| Jobs and Autotune | 8/10 | Strict contracts, integer monetary accounting, durable dispatch records, run leases, unknown outcomes, and held-out evaluation provide a strong base. |
| Testing and delivery | 8/10 | Broad offline tests, Python 3.11–3.13 CI, Windows coverage, package installation checks, and benchmark smoke tests. The missing cases are interactions between otherwise tested features. |
| API and maintainability | 7/10 | Small dependency footprint and useful public types. Several large modules and duplicated execution paths make behavior drift easier. |
| Evidence and positioning | 6/10 | Committed records and reproducible charts are strengths. Lifecycle accounting and scope must be corrected before promoting some efficiency claims. |

The overall 7/10 is an engineering judgment for a pre-1.0 framework. It is not
an arithmetic benchmark score or a claim of production readiness for every
workload.

## Priority fixes

### 1. Validate every monetary input before changing budget state

**Priority: high.**

[`Sentinel.__init__`](../smythe/budget.py#L98),
[`reserve`](../smythe/budget.py#L150), and
[`_cost_of`](../smythe/budget.py#L175) reject some negative numbers but do not
consistently reject NaN, infinity, booleans, or negative token counts.
`CompletionResult` has no corresponding validation boundary.

Observed locally:

```python
budget = Sentinel(1.0)
budget.reserve("a", float("nan"))
budget.add_cost("a", CompletionResult("paid", cost_usd=50.0))
budget.check("b")  # returns without raising
```

The breakdown records `$50`, while `total_cost_usd` becomes NaN. Comparisons
against the `$1` ceiling then stop working. A result with `prompt_tokens=-1000000`
also reduces the recorded spend by `$3` at the default rate. This is an input
integrity defect within the documented trusted-operator model; a faulty custom
provider or configuration is enough to trigger it.

**Specification:** validate finite, nonnegative monetary values and integer,
nonnegative token counts before admission, reconciliation, or checkpoint
restore. Reject invalid values without mutating balances. Reuse the strict
money conventions in [Jobs models](../smythe/jobs/models.py#L98).

**Acceptance:** table-driven tests cover NaN, positive/negative infinity,
booleans, strings, negative usage, and malformed restored costs. Serial and
parallel execution both stop before another call and preserve valid accrued
spend. Exact-fit budgets continue to pass.

### 2. Invalidate active descendants when a verifier rejects their input

**Priority: high.**

[`_reset_for_regeneration`](../smythe/executor_base.py#L218) resets completed
and failed descendants, but leaves running descendants untouched at
[line 242](../smythe/executor_base.py#L242).
[`AsyncExecutor.run`](../smythe/async_executor.py#L189) then rebuilds scheduling
state while those tasks remain active.

An offline reproduction used a draft with two parallel children: a judge and
a consumer. The consumer started from draft v1. The judge rejected v1, the
draft regenerated to v2, and the judge passed v2. The completed graph was:

```text
draft    completed    draft-v2
judge    completed    PASS
consumer completed    used-v1
```

The final graph contains an accepted replacement and a result derived from
its rejected predecessor. The existing downstream regeneration test covers
finished descendants, not this in-flight timing.

**Specification:** identify the full invalidated subtree, cancel and await
active descendants, settle any charges already incurred, then reset and
reschedule them. Alternatively, bind every attempt to dependency generation
identifiers and discard outputs whose inputs are no longer current. Persist
that dependency identity for recovery.

**Acceptance:** use event-controlled providers to force rejection while a
descendant is running. No stale descendant result may become the delivered
output. Verify cancellation, already-billed completions, overlapping judges,
and crash/resume during regeneration.

### 3. Preserve the complete Task through planning, execution, and resume

**Priority: high.**

[`Task.context`](../smythe/task.py#L16) is documented as context forwarded to
agents, and the [planner prompt](../smythe/prompts.py#L186) receives it. The
execution stamp in [`Swarm._stamp_task_context`](../smythe/swarm.py#L209) includes
the goal, constraints, and acceptance criteria, but omits `Task.context`.
The condition at [line 223](../smythe/swarm.py#L223) skips the whole stamp when
the node label equals the goal, also dropping constraints and acceptance
criteria for that common single-node case.

Observed locally: `Task("Review invoice", constraints=["Do not omit currency"],
done_when=["Return all line items"], context={"invoice": "Invoice ABC 123"})`
produces the execution prompt `Review invoice` when the node has the same
label. Changing its label includes the constraints and criteria, but still
omits the invoice.

The inspect-then-execute path has another boundary to address.
`plan(task)` stamps node metadata, so that metadata survives `execute(graph)`.
However, [`execute_async`](../smythe/swarm.py#L264) and
[`_execute_sync`](../smythe/swarm.py#L333) set the run's Task to `None` for graph
input. Structured Task data is then absent from the checkpoint and supervisor
argument, and planner memory does not record the run.

**Specification:** carry the Task with a planned execution object or accept an
explicit Task alongside an inspected graph. Serialize task payloads and data
with clear boundaries so document content remains data. Only omit duplicated
goal text; never omit constraints, acceptance criteria, or context.

**Acceptance:** the same fake provider sees the same effective request through
`execute(task)`, `plan(task)` followed by `execute(graph)`, and resume. Cover
single-node and multi-node graphs, custom architects, Task context, constraints,
acceptance criteria, supervisor inputs, and memory recording.

### 4. Make serial HALT stop new provider work

**Priority: high for paid workflows.**

[`Executor.run`](../smythe/executor.py#L44) catches ordinary node errors,
stores the first error, and continues traversing the graph. An offline graph
with an initial failing node and an independent sibling made both provider
calls, then raised the first error:

```text
calls:    ["fail", "paid sibling"]
statuses: [("a", "failed"), ("b", "completed")]
```

The [parallel executor](../smythe/async_executor.py#L199) already treats HALT
as a global stop and cancels active siblings. A change from serial to parallel
therefore changes the failure contract as well as concurrency.

**Specification:** centralize the stop decision and prohibit new provider
dispatch after a terminal HALT or exhausted RETRY failure. Preserve the chosen
downstream status semantics and completed work. If continuing independent
branches is useful, expose it as an explicit separate policy.

**Acceptance:** one shared failure-policy scenario matrix runs against both
executors. Cover independent siblings, dependent SKIP nodes, retry exhaustion,
budget errors, and post-billing artifact failures. No queued paid call starts
after the stop is observed.

### 5. Account for the complete paid lifecycle

**Priority: high for benchmark claims and budget expectations.**

The [LLM planner](../smythe/planner.py#L117) and
[LLM supervisor](../smythe/supervisor.py#L159) call providers without the run
Sentinel. Swarm reports execution and synthesis charges. The architecture
guide currently documents this narrower scope, but a generated-versus-fixed
workflow benchmark needs the planner's cost and time too.

An offline provider charged `$4` for planning and `$1` for execution under
`max_budget_usd=1.10`. The run succeeded with:

```text
provider call total:  $5.00
SwarmResult total:   $1.00
cost_is_complete:    True
```

**Specification:** introduce a run-level call recorder shared by routing,
planning, supervision, execution, tools, and synthesis. Report totals and
phase subtotals. State whether USD values are measured, estimated, or unknown.
For budgets, reserve before every paid call and preserve unknown exposure
through failure and resume. Apply the same scope to every benchmark lane.

**Acceptance:** fake providers charge distinct amounts in each phase. The
total reconciles with all observed calls, including retries and failed plans.
A planner cannot consume spend outside a lifecycle ceiling. Benchmark records
store phase costs, call counts, actual workflow configuration, and wall-clock
boundaries; earlier incomplete campaigns remain diagnostic evidence.

## Improvements by subsystem

| Subsystem | Keep | Improve next |
|---|---|---|
| Planning and graph | Three planning tiers, inspectable DAGs, validated revisions, capability inventory | Preserve Task identity; require a declared deliverable contract; validate node fields consistently across Python, YAML, planner JSON, and checkpoint loading. |
| Execution | Bounded active-task admission, explicit retries, timeouts, cancellation, per-call recording | Resolve the four correctness findings above; share failure-policy logic between executors; make verification generations explicit. |
| Durability | Atomic artifact replacement, versioned checkpoints, terminal checkpoints | Use unique checkpoint temporary files and durable flushing; define ownership for concurrent resume. The file checkpoint store uses a fixed `.json.tmp` path and an instance-local lock at [checkpoint.py:228](../smythe/checkpoint.py#L228), while Jobs already has stronger storage primitives. |
| Jobs | Approval bound to exact plans, SQLite WAL/FULL durability, dispatch journal, leases, explicit unknown outcomes | Reuse these primitives in the main Swarm runtime; add operator inspection and integrate production asset policies into manifests. |
| Autotune | Immutable contracts, allowlisted mutation, paired comparisons, confirmation and held-out evaluation | Add campaign ownership and evaluator process isolation before live generalization. Keep the offline simulator distinct from measured provider performance. |
| Providers and MCP | Lazy optional SDK imports, vendor-neutral messages/tools, allowlists, environment-variable names in persisted configuration | Unify adapter configuration and costing; test supported minimum/current SDKs. Offer strict capability assignment: [Registry.assign](../smythe/registry.py#L112) currently creates a generalist when no agent satisfies required capabilities. |
| Assets | Separate production/concept policies, deterministic finishing, validation and hash receipts | Keep exact production constraints separate from model judgment. Promote Jobs artifact validation without implying the simpler Swarm artifact records have all the same guarantees. |
| Observability and memory | Structured lifecycle events, explicit revision/regeneration traces, reviewable history | Record durable phase totals and request identifiers. Distinguish wall time from summed node duration; memory currently sums span durations, which grows with parallel work. |
| Packaging and CI | Minimal core install, provider extras, wheel smoke tests, offline tests, trusted PyPI publishing | Test README installation commands from clean environments; use dependency constraints for reproducible benchmark campaigns; establish type checks and a measured coverage floor. |
| Documentation and examples | Concise README plus subsystem guides, architecture diagram, offline acquisition example | Keep exact claims prominent and protocol scope adjacent. Put planned capabilities in Coming soon while retaining limitations beside the behavior they constrain. |
| Screensaver | Shared glyph catalog across web, Windows, macOS, and Linux | Preserve catalog parity across renderers and native artifact checks; add notarized Mac distribution and native Wayland integration. Tune rendering independently of benchmark timing. |

The security policy correctly describes a single-tenant, trusted-operator
runtime. MCP tools inherit operator permissions; multi-tenant isolation and
tool sandboxing are separate projects, not guarantees a README can add. The
core uses safe YAML loading and parameterized database operations, and the
Jobs boundary includes path confinement, attachment fingerprints, bounded
image decoding, and conservative unknown outcomes. The most valuable near-term
security work is consistent input validation and preservation of authority
boundaries as data passes through planning and tools.

## Refactor, do not rewrite

Retain `Task`, `ExecutionGraph`, provider/tool contracts, inspectable planning,
the async scheduler, and the durable Jobs/Autotune storage model. Fix their
invariants before introducing another abstraction layer.

Then split modules along existing boundaries. `provider.py` has 972 lines;
`jobs/store.py` has 1,512; `optimize/engine.py` has 1,356; and
`optimize/ledger.py` has 1,764. Extract vendor adapters, shared validators,
storage schema/migrations, and transaction operations behind existing public
interfaces. Keep compatibility tests around the extraction. A language change
would not solve task loss, stale dependencies, incomplete accounting, or
measurement design.

For large graphs, replace recursive depth, cycle, and topological walks with
iterative algorithms. A valid reverse-ordered chain of 1,500 nodes raised
`RecursionError` in [`ExecutionGraph.validate`](../smythe/graph.py#L484), while
the same dependency structure in forward order passed. The acceptance case
should cover deep and wide DAGs in arbitrary input order; throughput tests on
wide graphs alone do not cover depth.

## Delivery sequence

1. **Runtime correctness:** fix monetary validation, active-subtree
   invalidation, complete Task propagation, and consistent HALT behavior.
   Add deterministic regression tests for each reproduced case.
2. **Complete evidence:** price and time the paid lifecycle, rerun a frozen
   matched protocol, and promote only results that satisfy its quality gates.
   Preserve historical and diagnostic records.
3. **Operational consolidation:** bring the main Swarm runtime onto shared
   durable accounting and ownership primitives; add an operator inspection
   surface and clean-environment quickstart checks.
4. **Maintainability:** split the largest modules, add gradual type checking,
   enforce a measured coverage floor, and test graph order/depth and provider
   compatibility as explicit support boundaries.

The release quickstart should use the provider extra matching its selected
model: the core `pip install smythe` installs PyYAML only. The default `Swarm`
does use [`LLMArchitect`](../smythe/swarm.py#L157), so its inspection example
demonstrates generated topology. `SimpleArchitect` is a deterministic
single-node alternative and should be labeled accordingly.

## Verification record

The code-review reproductions ran on local Python 3.11 and used custom
in-memory providers. They confirmed the monetary, verification, Task
propagation, serial HALT, lifecycle-cost, and deep-graph behaviors reported
above. They did not measure live-provider quality or speed.

The configured CI includes Ubuntu tests on Python 3.11, 3.12, and 3.13, a
Windows suite, package installation smoke tests, and offline benchmark smoke
tests. In [CI run 31119124734](https://github.com/petehottelet/smythe/actions/runs/31119124734)
at the reviewed commit, Python 3.12 and 3.13 tests succeeded. The Python 3.11,
Windows, lint, package, and benchmark jobs failed at **Set up job**, before
repository commands. In [screensaver run 31119124768](https://github.com/petehottelet/smythe/actions/runs/31119124768),
the macOS build succeeded and Windows failed at the same setup boundary.
These workflow failures are infrastructure/setup failures, not evidence of
failed assertions or a broken screensaver build.

The final local offline suite passed **1,022 tests with 4 skips** on Python
3.11; `ruff check .` passed. The source distribution and wheel built successfully,
and a clean virtual environment imported the installed wheel and ran the Jobs
schema CLI. [CI run 34093505237](https://github.com/petehottelet/smythe/actions/runs/34093505237)
passed lint, all three Python versions, Windows, package installation, and
offline benchmark smoke checks for the review changes. This review does not
infer live test status from a static badge.

[Native run 34093505361](https://github.com/petehottelet/smythe/actions/runs/34093505361)
passed Windows, Apple Silicon, Intel Mac, and Ubuntu 22.04/24.04 execution checks.
The Mac jobs tested the same universal bundle; the Linux jobs tested the same
ELF executable. The downloaded Windows binary also passed the local native
smoke test. Published packages and receipts are identified in
[build provenance](../screensaver/dist/BUILD_INFO.json). These checks cover
rendering, motion, resizing, and native lifecycle behavior; Mac notarization,
Windows multi-monitor dispatch, and native Wayland integration remain outside
the verified scope.
