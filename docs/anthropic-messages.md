# Native Claude Messages

**Unreleased.** Install the current checkout with `pip install -e ".[anthropic]"`
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
Refusals and truncated output retain their known charge.

Input counts are estimates. The quote reserves the entire output cap and
input headroom at the highest supported input rate. A measured overrun remains
charged and blocks further admission. Accepted responses replay locally with
no token-count or generation request. See [workflow accounting](workflow-accounting.md).

The [Fable protocol](../benchmarks/fable_51_benchmark_plan.md) defines the matched
benchmark and the separately measured Claude Code Ultracode comparison.
Ultracode uses Code's Workflow tool; it is not a Messages API effort value.
