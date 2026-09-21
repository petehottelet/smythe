# What 200 Astra and Sol workflows tell us about generated agent plans

Smythe turns a goal into an inspectable task graph and runs it with budgets,
bounded concurrency, traces, and recovery. We tested when generating that graph
helps by comparing it with a fixed research → analysis → writing pipeline.

The study recorded **200 workflows**: ten project-authored, supplied-source
tasks, five repetitions, two models, and two planning strategies. Every
scheduled outcome remains in the results, including one failed workflow.

The result is mixed. Generated plans can reduce the number of execution nodes,
but planning and longer-running graphs add overhead. On these tasks, generated
graphs increased mean wall time for both models and mean workflow cost for Astra.

| Arm | Automatic acceptance | Median time | P95 time | Workflow charges |
|---|---:|---:|---:|---:|
| Astra / fixed | 50/50 | 23.64 s | 47.85 s | $3.5782200 |
| Astra / generated | 49/50 | 20.95 s | 76.42 s | $5.3826700 |
| Sol / fixed | 47/50 | 19.26 s | 37.23 s | $1.2788070 |
| Sol / generated | 45/50 | 29.68 s | 65.72 s | $2.3679962–$2.5376412 |

Astra's generated arm had a lower median but a longer tail. Its paired mean
time difference was **+8.92 seconds**, with a descriptive 95% task-clustered
interval of **+0.22 to +17.98 seconds**. Its mean cost difference was
**+$0.0361 per workflow**, with an interval of **+$0.0033 to +$0.0719**.
Sol's generated arm increased mean time by **9.97 seconds**; its interval was
**+3.79 to +16.12 seconds**.

The frozen acceptance rule accepted **191/200** workflows. A human reviewed all
eight disputed available answers and accepted each at 4/4. Those judgments are
reported separately: the primary automatic classifications remain unchanged,
and the workflow with no output remains failed. This supports a transparent
account of the disagreement, not a quality-superiority claim.

One failed Sol call returned no usage receipt. Its full **$0.169645** reservation
remains in the accounting bounds. Exact cost comparisons involving that arm are
withheld. Known usage is valued at the campaign's frozen list prices; it is not
an invoice reconciliation.

The engineering lesson is to make graph selection explicit. Use a known graph
when the workflow is stable; use generated plans when task structure needs to
vary, then measure whether that flexibility earns its cost. Smythe supports
both within the same execution controls. The proposed routing improvements
still need separate tests; this study does not prove their benefit.

The results include planning and execution, native ledgers, saved outputs,
blind judge reasoning, human ratings, and reproducible charts. These were ten
reused synthetic tasks in five related families, not an external holdout or a
framework comparison. A separate experiment is needed to isolate scheduler
parallelism or compare modern framework adapters.

Read the [full study, methods, and evidence](results/astra_20260913_main/README.md).


[Offline reproduction and hashes](results/astra_20260913_main/README.md#publication-reproduction).
