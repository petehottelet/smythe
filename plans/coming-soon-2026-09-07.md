# Coming soon completion campaign

Requested sequence: complete an item, review implementation and evidence,
run its regression tests and the offline suite, push, inspect the published
result, and continue to the next feasible item. Historical benchmark records
remain unchanged; new measurements receive new records and explicit scope.

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
| Astra pilot and matched campaign | Offline preparation complete; awaiting API-spend ceiling | 13 original task/source packs, 12 pilot and 200 main workflow schedules, portable preparation receipt, and factual checks separate from quality judgments. Paid runtime integration, pilot calibration, and judge accounting remain pending. |
| Saved workflow graph limits and Astra preparation | Completed | 2,883 offline tests passed, 8 skipped on frozen source `6245578`; Ruff, 190 local documentation targets, 23 native provenance bindings, independent policy/documentation review, 119 focused workflow tests, 82 preparation tests, and [all eight CI jobs](https://github.com/petehottelet/smythe/actions/runs/34151055127) passed. The preparation receipt reproduces exactly under its portable text-hash policy. |
| Jobs list and inspection | Completed | 2,682 offline tests passed, 8 skipped; independent code and visual review, 140 local documentation targets, 23 native provenance bindings, and [all eight CI jobs](https://github.com/petehottelet/smythe/actions/runs/34146003750) passed for `aec976a`. Read-only run lists, bounded inspection, escaped HTML reports, exact Jobs costs, attempt lineage, and timestamped artifact integrity |
| Deep graph execution and scale harness | Completed | 2,722 offline tests passed, 8 skipped on frozen source `003e157`; Ruff, 180 local documentation targets, 23 native provenance bindings, independent graph/harness reviews, and [all eight CI jobs](https://github.com/petehottelet/smythe/actions/runs/34149130653) passed. Iterative traversal executes a reverse-ordered 5,000-node chain. Sixteen scale-harness checks include a real subprocess kill, lease expiry, resume, and explicit unknown rerolls. [Retained 25-operation pilots](../benchmarks/results/jobs_scale_preflight_20260907/README.md) are separate diagnostics. |
| Jobs ownership fencing | Completed; remote verification next | 2,925 offline tests passed, 8 skipped on frozen source `4bb7c02`; all 151 focused Jobs tests and independent runtime review pass. Ruff passes. The WAL snapshot regression rejects an intentionally broken autocommit reader. Schema v3 checks live epochs and attempt provenance; lease time is sampled after lock acquisition. |
| Jobs 5,000-operation campaign | Awaiting fenced-source publication | Run the reviewed harness against the frozen, qualified ownership implementation. Keep actual kill/expiry/recovery evidence and every failed attempt. |
| Detachable jobs and durable pauses | Implementation and focused verification | 19 pause tests and 114 operator tests pass; real process and final compatibility checks remain. Private launch handshake, atomic lease/status observations, durable admission control, and generation-bound stop/resume. |
| Artifact directory ownership | Confirmed defect; fix in progress | Custom run IDs that alias on a case-insensitive filesystem can share an artifact directory. Add persistent directory identity while preserving legacy receipt/recovery behavior before the operator release. |
| Renderer performance and native exploration | Browser review passed; follow-up timing campaign frozen | Current review passes 31 checks and refreshes the screenshot. The first timing attempt stopped before samples; its receipt and both metadata preflights are retained. [Follow-up plan](../benchmarks/results/glyph_rain_regl_20260907_followup1_plan.json) binds explicit browser launch and the active graphics context. Native explorer parity remains separate. |
| Native signing and Wayland | Prerequisite review pending | Developer ID/notarization access and native display integration tests |

## Astra preflight

On 7 September 2026, read-only model retrieval returned HTTP 200 for the exact
`gpt-6-astra` and `gpt-5.6-sol` IDs using the configured account. This confirms
model access, not generation quality or billing. No paid campaign calls have
been made at this stage. Official model, pricing, and prompt-cache documentation
was rechecked. The initial runtime fixes prevent malformed accounting and
failure handling from invalidating the campaign.
