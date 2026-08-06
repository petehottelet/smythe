# Glyph Screensaver Fan-Out Benchmark

This workload measures Smythe on a visually inspectable, highly parallel artifact job:
192 independent fictional cyber glyphs are generated as one 192-node
`BROADCAST_REDUCE` graph, validated objectively, and assembled into a digital-rain
screensaver package.

The aesthetic is reference-inspired--not copied. It uses the general vocabulary of green
digital rain: a black field, luminous descending columns, varied trails, and bright leading
glyphs. The offline marks are generated from an original deterministic stroke grammar --
horizontal bars, vertical stems, hooks, enclosures, press diagonals, bowls, tail sweeps,
and diacritic dots composed on an ideograph grid with occasional serif nubs. The
vocabulary is reminiscent of hand-drawn ideographs and romanesque letterforms without
reproducing any real character; nothing is extracted from a font, logo, screenshot, film
frame, or other source image.

## Run it

The default run is local, deterministic, and costs $0:

```bash
python benchmarks/run_glyph_screensaver.py
```

It sweeps concurrency 1, 4, 8, and 16 with a 250 ms simulated provider latency by default:

```bash
python benchmarks/run_glyph_screensaver.py --latency-s 0.25
```

The 250 ms delay models a fast asynchronous remote request while keeping a full sweep
practical. At that latency, fixed local work -- deterministic tile rendering plus
durable (fsync'd) artifact journaling at roughly 70 ms per tile on the reference
Windows/NTFS machine -- is a significant share of the envelope, so the default sweep
understates scheduler scaling and its plateau partly measures local disk throughput.
Use `--latency-s 0` as a separate fixed-overhead smoke profile.

### Realistic-latency profile

The published realistic profile sets the simulated latency to **5.8 s per call**,
the measured mean serial per-image latency of the live image lane in
[image_benchmarks.md](image_benchmarks.md) (46.3 s for 8 images). At realistic image-API
latency, per-tile journaling is noise and the sweep isolates what the executor
contributes to a real wide artifact job:

```bash
python benchmarks/run_glyph_screensaver.py --latency-s 5.8 \
  --concurrencies 1,4,8,16,32,64 \
  --results benchmarks/results/glyph_screensaver_offline_realistic.json \
  --out smythe_artifacts/glyph_screensaver/offline_realistic
```

For a quick mechanics smoke test, reduce the graph width. The atlas, preview, GIF, and
HTML are assembled only for the full 192-glyph suite:

```bash
python benchmarks/run_glyph_screensaver.py --glyphs 8 --concurrencies 1,4
```

## Protocol

- One `ExecutionGraph` containing one independent `Node` task per glyph. Here a node is a
  schedulable Smythe graph task, not a claim that each glyph owns a persistent autonomous
  persona.
- Every label comes from `glyph_prompt()` and produces exactly one provider call.
- Offline calls use `ProceduralGlyphProvider`; the simulated latency represents an
  asynchronous remote request while deterministic stroke rendering makes results
  repeatable.
- Every provider artifact is normalized to PNG at exactly 128x128.
- The run passes only when every node completes, every tile is a valid 128x128 PNG, and all
  tile SHA-256 hashes are unique.
- Generation wall time excludes deterministic validation and assembly so the concurrency
  comparison isolates graph execution. End-to-end and validation wall times are also recorded.
- Throughput is completed glyphs divided by generation wall time.
- Speedup is concurrency-1 wall time divided by the candidate wall time.
- Parallel efficiency is speedup divided by concurrency.
- The fastest fully valid 192-glyph run supplies the tiles for a 2048x1536 contact-sheet
  atlas (16x12), a 1920x1080 still preview, a 640x360 looping GIF, and a self-contained
  1920x1080 HTML canvas screensaver.

The JSON evidence record includes protocol and environment snapshots, each run's timing,
throughput, speedup, efficiency, cost completeness flags, errors, and SHA-256-bound output
receipts. Offline results default to `benchmarks/results/glyph_screensaver_offline.json`.

## Optional live image lane

The live lane supports two providers via `--live-provider`:

- `openai` (default): the Image API's current `gpt-image-2` default with
  low-quality 1024x1024 PNG output. OpenAI's official guide describes
  `gpt-image-2` as the latest GPT Image model, documents `low` as the fastest
  draft setting, and notes that square outputs are typically fastest:
  [OpenAI image-generation guide](https://developers.openai.com/api/docs/guides/image-generation).
- `gemini`: `gemini-2.5-flash-image` through `GeminiProvider` with a 1:1
  aspect configuration; requires `GOOGLE_API_KEY`. The recorded per-image
  estimate is $0.039 (the asset suite's convention) while budget enforcement
  reserves the explicit `--max-cost-per-call-usd` ceiling.

Live execution is deliberately one chosen concurrency, not a paid sweep:

```bash
python benchmarks/run_glyph_screensaver.py --live --concurrency 8 \
  --max-cost-per-call-usd 0.01 --max-budget-usd 1.92

python benchmarks/run_glyph_screensaver.py --live --live-provider gemini \
  --concurrency 8 --max-cost-per-call-usd 0.06 --max-budget-usd 12.00
```

Guardrails are fail-closed:

- `OPENAI_API_KEY` must already be present in the environment.
- Both budget flags must be explicit and positive.
- The whole-job budget must cover `glyph count x inclusive per-call ceiling` before any call.
- Missing credentials or ceilings are errors; the script never silently falls back to offline.
- Nodes do not retry, so the declared 192-call ceiling is the whole generation envelope.
- The recorded live cost is Smythe's conservative configured ceiling, not a claimed invoice.
- Current pricing is intentionally not hard-coded. Verify the official guide/calculator and
  choose an inclusive input-plus-output ceiling immediately before a paid run.

`gpt-image-2` does not currently support transparent output backgrounds. The live artifacts
are therefore normalized into 128x128 RGBA PNG containers, but the benchmark does not claim
that their pixels have transparent backgrounds.

## Live lane results (2026-08-05)

Four attempts were made; every record — including the failures — is
committed, because each failure exercised a guardrail or exposed a real
defect:

| Attempt | Record | Outcome |
|---|---|---|
| 1 — OpenAI | [glyph_screensaver_live_openai_credit_exhausted.json](results/glyph_screensaver_live_openai_credit_exhausted.json) | The account had no credits. The lane failed closed on the first 429s; no fallback, no retry storm. |
| 2 — Gemini | [glyph_screensaver_live_gemini_tokencount_crash.json](results/glyph_screensaver_live_gemini_tokencount_crash.json) | **Framework bug found:** the SDK returned `usage_metadata` with `None` token counts on an image-only response, crashing cost recording mid-fan-out (`int + NoneType`). Fixed in `smythe/provider.py` with a regression test. |
| 3 — Gemini | [glyph_screensaver_live_gemini_plaque_reject.json](results/glyph_screensaver_live_gemini_plaque_reject.json) | All 192 calls completed in 141 s, but the model drew the "luminous tile" prompt literally — framed glass plaques — and the objective normalization gate refused 97 of them as non-separable. The prompt was tightened to demand a flat 2D mark on plain black. |
| 4 — Gemini, final | [glyph_screensaver_live.json](results/glyph_screensaver_live.json) | **184 of 192 tiles generated and objectively validated in 121.0 s at concurrency 8.** The run then halted itself at the exact-fit budget boundary: after 191 ceiling reservations of $0.06 against the $11.52 limit, accumulated floating-point error rejected the 192nd with $0.06 nominally remaining. **Second framework bug found and fixed** (nano-dollar admission tolerance in `smythe/budget.py`, with a regression test). The framework chose to stop rather than overspend. |

Interpretation notes:

- Recorded run costs use the conservative $0.06 per-call ceiling, not invoice
  data; the provider's own billing is authoritative. Attempt 2's crash means
  its record understates the calls it actually dispatched before failing.
- The 121 s / 184-image figure is ecological evidence for one account, model,
  region, prompt set, and moment; it is not a rate-limit or capacity claim.
- A clean 192/192 run after both fixes is one command away (use a budget
  with headroom, for example `--max-budget-usd 12.00`); it was not re-run in
  this session to avoid a further paid attempt.

## Interpretation boundaries

The offline sweep is a controlled executor benchmark, not evidence about a particular image
API's production latency or rate limits. Its procedural provider intentionally holds prompt,
artifact size, call count, and per-call delay constant so the measured variable is Smythe's
bounded scheduling concurrency. Results at very low simulated latency are expected to expose
fixed rendering and journaling overhead rather than linear speedup; the realistic-latency
profile exists because the 250 ms default demonstrably does not bury that overhead on the
reference machine. The 5.8 s figure is one measured account/model/moment, not a universal
image-API constant. The live lane is ecological evidence for one account, model, region,
prompt set, and moment in time; repeat it before making capacity or purchasing claims.
