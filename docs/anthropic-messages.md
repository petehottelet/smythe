# Native Claude Messages

Available since **Smythe 0.8.0**. Install `pip install "smythe[anthropic]==0.9.0"`
and set `ANTHROPIC_API_KEY` locally. `AnthropicMessagesProvider` supports
`claude-fable-5-1` text workflows through Anthropic's global Messages endpoint.

```python
from smythe import AnthropicMessagesProvider, SQLiteWorkflowStore, Swarm, Task

with SQLiteWorkflowStore("fable-runs.db") as store:
    swarm = Swarm(
        model="claude-fable-5-1",
        provider=AnthropicMessagesProvider(
            reasoning_effort="medium", max_output_tokens=8192,
        ),
        run_store=store,
        max_budget_usd=5.00,
        parallel=True,
        max_concurrency=8,
    )
    result = swarm.execute(Task("Summarize the supplied text.", context={"text": "Your source text"}))
    print(result.output)
    print(result.workflow_accounting)
```

This example makes paid API calls. The adapter selects Standard service,
disables SDK retries, and uses Fable's adaptive thinking with explicit effort.
It accepts text only. Tools, attachments, assistant prefill, alternative
endpoints, and provider fallbacks are outside this adapter's scope.

The journal saves request and response bytes before interpreting the output.
Prices use integer nanoUSD. Ordinary input, cache reads, five-minute writes,
one-hour writes, and output are separate categories. Claude's ordinary input
count excludes cache tokens; thinking is included in billed output once.
Missing usage, inconsistent cache totals, unknown models or service tiers,
and interrupted dispatches keep their reserved exposure and close admission.
Refusals and truncated output retain their known charge. A 401, 403, 404, 413
or 429 response, such as a 429 `rate_limit_error`, settles at zero cost and
follows the node's failure policy when two conditions hold. Its body must be
Anthropic's plain error object, `{"type": "error", "error": {"type": "...",
"message": "..."}, "request_id": "..."}`, with no usage or other field. The
SDK must have reported it as that status's error, such as `RateLimitError`,
not as a transport failure. Every other 4xx response keeps its reserved
exposure. That includes a 400 such as "Output blocked by content filtering
policy", which can follow billed generation. 5xx responses, including 529
`overloaded_error`, also keep their reserved exposure, because the journal
cannot show they were not billed. See
[workflow accounting](workflow-accounting.md#admission-and-cost) for the exact
conditions.

Input counts are estimates. The quote reserves the entire output cap and
input headroom at the highest supported input rate. A measured overrun remains
charged and blocks further admission. Accepted responses replay locally with
no token-count or generation request. See [workflow accounting](workflow-accounting.md).

The [Fable protocol](../benchmarks/fable_51_benchmark_plan.md) defines the matched
benchmark and the separately measured Claude Code Ultracode comparison.
Ultracode uses Code's Workflow tool; it is not a Messages API effort value.
The [native pilot](../benchmarks/results/fable_20260914_pilot/README.md) and
[ten-task Code study](../benchmarks/results/fable_code_20260914/README.md) have
complete execution and billing records. Actual human pilot ratings gate the
100-run native main comparison.
