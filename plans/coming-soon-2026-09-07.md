# Coming soon completion campaign

Requested sequence: complete an item, review implementation and evidence,
run its regression tests and the offline suite, push, inspect the published
result, and continue to the next feasible item. Historical benchmark records
remain unchanged; new measurements receive new records and explicit scope.

Version [0.7.0](https://github.com/petehottelet/smythe/releases/tag/v0.7.0)
is published from `729fda3` on GitHub and
[PyPI](https://pypi.org/project/smythe/0.7.0/).
All nine [final CI jobs](https://github.com/petehottelet/smythe/actions/runs/34164206893)
passed: each Linux Python 3.11–3.13 suite passed 3,273 tests with 24 skipped;
Windows passed 3,282 with 15 skipped. macOS operator/checkpoint checks passed
177 with 12 skipped, and the installed OpenAI 3.8.0 contract lane passed 502.
Python 3.13's pytest summary reports 139 warnings, including SQLite resource
warnings; the raw output remains recorded for follow-up.
The [release checklist](https://github.com/petehottelet/smythe/issues/15)
retains artifact identities and publication verification.
The final local suite passed 3,287 tests with 10 skipped in 1,633.63 seconds
on the same frozen source. The [release report](../docs/release-0.7.0.md)
records the fresh PyPI installation and downloaded distribution hashes.

## Work queue

| Item | Status | Completion evidence |
|---|---|---|
| Numeric cost and usage validation | Completed | 1,666 offline tests passed, 4 skipped; Ruff and independent review passed; strict provider, ledger, admission, Jobs, and checkpoint boundaries |
| Cross-platform evidence checks | Completed | 1,681 offline tests passed, 4 skipped; 66 targeted Linux checks passed; [all CI jobs](https://github.com/petehottelet/smythe/actions/runs/34128905987) and [native verification](https://github.com/petehottelet/smythe/actions/runs/34128905943) passed for `5ab8578`; measurements and artwork unchanged |
| Serial halt | Completed | 1,705 offline tests passed, 4 skipped; 24 shared serial/parallel failure-policy cases; independent review passed |
| Active-descendant verification | Completed | 1,765 offline tests passed, 4 skipped; 60 new cancellation, settled-charge, generation-identity, and crash/resume cases; independent review and [all CI jobs](https://github.com/petehottelet/smythe/actions/runs/34132523693) passed for `ec632bb` |
| Complete Task propagation | Completed | 1,877 offline tests passed, 4 skipped; 112 new snapshot, consumer, handoff, mutation, and recovery cases; independent review passed; Jobs heartbeat regression now uses an event barrier and rejects injected inline finalization |
| Astra Responses and native usage | Completed | 2,253 offline tests passed, 7 skipped; all 379 new native tests passed with isolated OpenAI SDK 3.8.0; independent review and [all eight CI jobs](https://github.com/petehottelet/smythe/actions/runs/34137894365) passed for `ee3bcde` |
| Complete workflow accounting | Completed | 2,523 offline tests passed, 8 skipped; independent review, Ruff, 165 local documentation targets, 23 native provenance bindings, and [all eight CI jobs](https://github.com/petehottelet/smythe/actions/runs/34142534009) passed for `bd0ad62`. [Workflow guide](../docs/workflow-accounting.md): exact phase costs, request quotes, fenced dispatch, saved responses, frozen control decisions, and atomic recovery |
| Astra pilot and matched campaign | Pilot runtime qualified offline; awaiting API-spend ceiling | 79 pilot runtime checks pass on frozen source with 61 unchanged source hashes; independent review and Ruff pass. Source-bound allowances, retained native evidence, immutable trial outcomes, and recovery use the prepared 13 task/source packs and 12-trial schedule. Paid pilot calibration, 200-workflow main execution, and judge accounting remain pending. |
| Saved workflow graph limits and Astra preparation | Completed | 2,883 offline tests passed, 8 skipped on frozen source `6245578`; Ruff, 190 local documentation targets, 23 native provenance bindings, independent policy/documentation review, 119 focused workflow tests, 82 preparation tests, and [all eight CI jobs](https://github.com/petehottelet/smythe/actions/runs/34151055127) passed. The preparation receipt reproduces exactly under its portable text-hash policy. |
| Jobs list and inspection | Completed | 2,682 offline tests passed, 8 skipped; independent code and visual review, 140 local documentation targets, 23 native provenance bindings, and [all eight CI jobs](https://github.com/petehottelet/smythe/actions/runs/34146003750) passed for `aec976a`. Read-only run lists, bounded inspection, escaped HTML reports, exact Jobs costs, attempt lineage, and timestamped artifact integrity |
| Deep graph execution and scale harness | Completed | 2,722 offline tests passed, 8 skipped on frozen source `003e157`; Ruff, 180 local documentation targets, 23 native provenance bindings, independent graph/harness reviews, and [all eight CI jobs](https://github.com/petehottelet/smythe/actions/runs/34149130653) passed. Iterative traversal executes a reverse-ordered 5,000-node chain. Sixteen scale-harness checks include a real subprocess kill, lease expiry, resume, and explicit unknown rerolls. [Retained 25-operation pilots](../benchmarks/results/jobs_scale_preflight_20260907/README.md) are separate diagnostics. |
| Jobs ownership fencing | Completed | 2,925 offline tests passed, 8 skipped on frozen source `4bb7c02`; all 151 focused Jobs tests and independent runtime review pass. Ruff, [all eight main CI jobs](https://github.com/petehottelet/smythe/actions/runs/34154426388), and [all native builds](https://github.com/petehottelet/smythe/actions/runs/34154426442) pass on published `5c1939b`. The WAL snapshot regression rejects an intentionally broken autocommit reader. Schema v3 checks live epochs and attempt provenance; lease time is sampled after lock acquisition. |
| Jobs 5,000-operation campaign | Completed and independently reconciled | Frozen `4bb7c02`, schema v3: 5,000 accepted identical 1×1 PNG fixtures after a real hard kill, actual lease expiry, safe resume, and eight explicit rerolls. Zero accepted work reissued; zero calls on completed resume; zero API charges. All 15,088 archive members and 20,051 ledger events are retained in the [reviewed evidence](../benchmarks/jobs_scale_5000_20260907_results.md). One Windows campaign, concurrency eight; no comparative timing or model-quality claim. |
| Detachable jobs and durable pauses | Released in 0.7.0; final platform CI passed | Earlier frozen suites passed 3,126 and then 3,274 tests. Remote Windows CI exposed denied process breakaway. The final repair refuses denied startup before worker creation or authorization. Its [frozen qualification](../benchmarks/results/windows_operator_20260907/README.md) passed all 122 operator/CLI cases, then all 21 CLI cases after a direct-call assertion. Positive local survival and three controlled zero-call refusals have separate receipts; rejected fallback diagnostics remain retained. Final Windows, Linux, and macOS checks pass on `729fda3`. |
| Artifact directory ownership | Released in 0.7.0; final platform CI passed | Full-suite qualification at `f1bffa5`: 3,126 passed, 8 skipped; Ruff passed. All 13 namespace tests pass. The isolated old-writer negative control reproduces an actual overwrite; the fixed path preserves the sentinel and records an unknown outcome. Persistent namespaces, pre-dispatch claims, and exclusive publication retain legacy receipt paths. Independent reviews and final `729fda3` platform checks pass. |
| Atomic file checkpoints | Released in 0.7.0; final platform CI passed | Frozen `0a7fabc`: 3,274 passed, 10 skipped in 1,943.75 seconds; Ruff passed. All 34 focused checks pass, with one POSIX-only check skipped on Windows. Concurrent writers complete independently; failed writes preserve the old snapshot. An old-writer negative control reproduces the shared-temp collision. Independent review and final `729fda3` Windows/Linux/macOS checks pass. |
| PyPI README and release artifacts | Published 0.7.0; registry and fresh-install checks passed | Final Windows candidate distributions from `729fda3` pass strict Twine validation, complete member inspection, and an identical sdist-to-wheel rebuild. All 22 build-hook checks pass. Downloaded PyPI files match [trusted publication](https://github.com/petehottelet/smythe/actions/runs/34165199495) byte for byte. A fresh PyPI installation passes the exact README through real SDK 3.8.0 with synthetic responses and all 12 offline Jobs commands, with no paid provider calls. All six original release attachments match their uploaded bytes. |
| Scale evidence archive and chart gate | Released in 0.7.0; final CI passed | All 37 archive/reconciliation tests passed. The completed archive passed independent reconciliation and a second check of every member hash. All 64 chart/gate tests pass after integration; the new monochrome figure passed visual review and all 20 text bounds fit. Eight earlier charts/diagrams remain byte-identical. Full platform suites pass on `729fda3`. |
| Autotune campaign ownership | Implementation and independent review underway | A two-connection reproduction showed that a foreign ledger writer could complete another runner's dispatched trial. The proposed fix adds explicit campaign leases, owner epochs, dispatch provenance, transactional migration, heartbeat renewal, and conservative crash recovery. This is unreleased work after 0.7.0. |
| Renderer performance and native exploration | Measurement campaign complete; target missed | [Six primary sessions](../benchmarks/renderer_performance_20260907_results.md) measured 56.21–56.24 draws/s and 18.1 ms P95 callback intervals. Three separate blank-page controls showed similar cadence; 57 travel/resize cycles passed over 600.95 seconds under declared competing load. Stronger arithmetic/source review, 42 Python checks, and seven JavaScript checks pass. Current browser review passes 31 checks. Every failed attempt and superseded review is retained. Native explorer parity and physical presentation remain separate. |
| Native signing and Wayland | Prerequisite review pending | Developer ID/notarization access and native display integration tests |

## Astra preflight

On 7 September 2026, read-only model retrieval returned HTTP 200 for the exact
`gpt-6-astra` and `gpt-5.6-sol` IDs using the configured account. This confirms
model access, not generation quality or billing. No paid campaign calls have
been made at this stage. Official model, pricing, and prompt-cache documentation
was rechecked. The initial runtime fixes prevent malformed accounting and
failure handling from invalidating the campaign.
