# Glyph Screensaver Fan-Out Benchmark

This workload measures Smythe on a visually inspectable, highly parallel artifact job:
64 independent fictional cyber glyphs are generated as one 64-node
`BROADCAST_REDUCE` graph, validated objectively, and assembled into a digital-rain
screensaver package.

The aesthetic is reference-inspired--not copied. It uses the general vocabulary of green
digital rain: a black field, luminous descending columns, varied trails, and bright leading
glyphs. The offline marks are generated from original deterministic 7x9 procedural patterns;
they are not extracted from a font, logo, screenshot, film frame, or other source image.

## Run it

The default run is local, deterministic, and costs $0:

```bash
python benchmarks/run_glyph_screensaver.py
```

It sweeps concurrency 1, 4, 8, and 16 with a 250 ms simulated provider latency by default:

```bash
python benchmarks/run_glyph_screensaver.py --latency-s 0.25
```

The 250 ms delay models an asynchronous remote request while keeping a full sweep practical.
It is deliberately longer than local PNG journaling, so scheduler scaling is not buried under
fixed machine overhead. Use `--latency-s 0` as a separate low-latency smoke profile; that mode
is useful for measuring overhead, not for estimating remote-call speedup.

For a quick mechanics smoke test, reduce the graph width. The atlas, preview, GIF, and HTML
are assembled only for the full 64-glyph suite:

```bash
python benchmarks/run_glyph_screensaver.py --glyphs 8 --concurrencies 1,4
```

## Protocol

- One `ExecutionGraph` containing one independent `Node` task per glyph. Here a node is a
  schedulable Smythe graph task, not a claim that each glyph owns a persistent autonomous
  persona.
- Every label comes from `glyph_prompt()` and produces exactly one provider call.
- Offline calls use `ProceduralGlyphProvider`; the default 250 ms latency represents an
  asynchronous remote request while deterministic tile rendering makes results repeatable.
- Every provider artifact is normalized to PNG at exactly 128x128.
- The run passes only when every node completes, every tile is a valid 128x128 PNG, and all
  tile SHA-256 hashes are unique.
- Generation wall time excludes deterministic validation and assembly so the concurrency
  comparison isolates graph execution. End-to-end and validation wall times are also recorded.
- Throughput is completed glyphs divided by generation wall time.
- Speedup is concurrency-1 wall time divided by the candidate wall time.
- Parallel efficiency is speedup divided by concurrency.
- The fastest fully valid 64-glyph run supplies the tiles for a 1024x1024 atlas, 1920x1080
  still preview, 640x360 looping GIF, and self-contained 1920x1080 HTML canvas screensaver.

The JSON evidence record includes protocol and environment snapshots, each run's timing,
throughput, speedup, efficiency, cost completeness flags, errors, and SHA-256-bound output
receipts. Offline results default to `benchmarks/results/glyph_screensaver_offline.json`.

## Optional GPT Image lane

The live lane uses the Image API and current `gpt-image-2` default with low-quality
1024x1024 PNG output. OpenAI's official guide describes `gpt-image-2` as the latest GPT Image
model, documents `low` as the fastest draft setting, and notes that square outputs are
typically fastest: [OpenAI image-generation guide](https://developers.openai.com/api/docs/guides/image-generation).

Live execution is deliberately one chosen concurrency, not a paid sweep:

```bash
python benchmarks/run_glyph_screensaver.py --live --concurrency 8 \
  --max-cost-per-call-usd 0.01 --max-budget-usd 0.64
```

Guardrails are fail-closed:

- `OPENAI_API_KEY` must already be present in the environment.
- Both budget flags must be explicit and positive.
- The whole-job budget must cover `glyph count x inclusive per-call ceiling` before any call.
- Missing credentials or ceilings are errors; the script never silently falls back to offline.
- Nodes do not retry, so the declared 64-call ceiling is the whole generation envelope.
- The recorded live cost is Smythe's conservative configured ceiling, not a claimed invoice.
- Current pricing is intentionally not hard-coded. Verify the official guide/calculator and
  choose an inclusive input-plus-output ceiling immediately before a paid run.

`gpt-image-2` does not currently support transparent output backgrounds. The live artifacts
are therefore normalized into 128x128 RGBA PNG containers, but the benchmark does not claim
that their pixels have transparent backgrounds.

## Interpretation boundaries

The offline sweep is a controlled executor benchmark, not evidence about a particular image
API's production latency or rate limits. Its procedural provider intentionally holds prompt,
artifact size, call count, and per-call delay constant so the measured variable is Smythe's
bounded scheduling concurrency. Results at very low simulated latency are expected to expose
fixed rendering and journaling overhead rather than linear speedup. The live lane is ecological
evidence for one account, model, region, prompt set, and moment in time; repeat it before making
capacity or purchasing claims.
