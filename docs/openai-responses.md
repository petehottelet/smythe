# Native OpenAI Responses

`OpenAIResponsesProvider` runs text and function-tool requests with
`gpt-6-astra` and `gpt-5.6-sol`. Each response carries native usage and a
dated, model-specific token price. Smythe 0.7.0 includes this explicit adapter:

```bash
pip install "smythe[openai]==0.7.0"
```

Set `OPENAI_API_KEY`. Select the provider explicitly; automatic provider
selection remains unchanged. OpenAI SDK 3.8.0 or later is required and loaded
when opening a session or making the first request.

## Count and price a request before generation

`prepare()` validates and freezes a request locally. `quote()` calls the
native input-token counter and prices the full output cap plus the most
expensive applicable input category. The quote is bound to the complete
request's SHA-256 hash.

```python
import asyncio
from decimal import Decimal

from smythe import OpenAIResponsesProvider
from smythe.tools import ChatMessage


async def main():
    provider = OpenAIResponsesProvider(
        reasoning_effort="medium",
        max_output_tokens=8192,
    )
    async with provider.session() as pooled:
        request = pooled.prepare(
            system="Answer in one concise paragraph.",
            messages=[ChatMessage(role="user", content="Explain a directed acyclic graph.")],
            model="gpt-6-astra",
        )
        quote = await pooled.quote(request)
        if quote.ceiling_usd > Decimal("1.00"):
            raise RuntimeError("The request exceeds this example's generation allowance")

        envelope = await pooled.dispatch(request)
        # An application can persist envelope.body before pricing or decoding it.
        result = pooled.decode(envelope)
    print(result.text)
    print(result.native_receipt["cost_usd"])


asyncio.run(main())
```

This example performs a token-count request and, if the quote fits, one paid
generation attempt. The count request is separate from generation. A quote
does not reserve a shared workflow budget. `complete()` and `chat()` each
make one generation attempt and never count inputs automatically.

`session()` reuses HTTP connections within one event loop and closes them on
exit. Use its bound provider only inside that context. Calls without a session
create and close their own client, including calls made through separate
synchronous planning and execution loops.

The request uses Standard service at the global OpenAI endpoint, `store=False`,
disabled truncation, and no SDK retries. The default reasoning effort is
`medium`; the default output cap is 8,192 tokens, including reasoning.
Unsupported sampling parameters, hosted tools, images, audio, regional
endpoints, and compatible third-party endpoints are outside this adapter's
scope.

## Usage receipts

The price table is versioned as
`openai-native-standard-global-2026-09-07-v1`. It records ordinary input,
cache reads, cache writes, and output separately, using integer billionths of
a dollar. Reasoning tokens are part of billed output and are counted once.
Long-context rates apply to the entire request when input exceeds 272,000
tokens. These are dated published list prices, not an account-specific invoice.
[Price schedule and formula](../benchmarks/astra_benchmark_plan.md#published-prices-and-cost-formula).

Pricing uses the returned model and effective service tier. Missing cache
counters, invalid numeric types, unsupported model identities or tiers, and
unknown billable modalities produce an unknown-cost receipt. The requested
model never fills a gap in returned billing evidence.

`CompletionResult.native_receipt` contains JSON-safe usage, exact cost as a
decimal string and integer nanoUSD, price version, request/response hashes,
and identifiers. Required billing-category completeness and optional usage-detail
completeness are reported separately. Raw response bytes remain in `response_envelope` for explicit
application handling. Encrypted reasoning and native continuation items are
excluded from ordinary traces and checkpoints.

## Function tools and failed responses

The existing tool loop carries the complete native output sequence into the
next request, preserving reasoning items, phases, and function-call IDs.
Function definitions retain optional schema fields. A tool result must match
one outstanding native call, and continuation belongs to its originating
provider and model.

`ProviderResponseError` retains a known charge when generation was billed but
its output is incomplete, malformed, or unusable. `ProviderAccountingError`
retains unknown exposure when billing evidence cannot be reconciled. Execution
records known charges once, stops further dispatch, and preserves unresolved
in-memory reservations. These failures cannot trigger an automatic node retry or be
converted into tool feedback that causes another paid turn. Saved error
markers block ordinary resume until the response or accounting is reconciled.

## Swarm budget scope

Pass this provider to `Swarm(provider=provider, model="gpt-6-astra")` to use
native pricing for execution. A capped Swarm also requires an explicit,
inclusive `max_cost_per_call_usd` on the provider. It must cover every request
the application permits, including growing tool history. The runtime reserves
that ceiling before each native tool turn and reconciles the returned charge.
Supplying a ceiling does not change the API's token cap.

The ordinary Swarm ledger covers execution and synthesis. Add
[`run_store=SQLiteWorkflowStore(...)`](workflow-accounting.md) to bind every
text-workflow phase and attempt to one durable ledger, including routing,
planning, and supervision. That path counts and reserves each exact native
request, persists raw evidence, and recovers saved responses locally.
Its request quote replaces the legacy per-call estimate for admission.
Existing Chat Completions calls continue to use the blended estimate when
their provider supplies no explicit cost.

[Cost guardrails](budgets.md) · [Task handoffs](tasks.md) ·
[Astra campaign plan](../benchmarks/astra_benchmark_plan.md).
