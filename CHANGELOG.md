# Changelog

All notable changes to **smythe** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with the pre-1.0 stability note below.

## Versioning policy (pre-1.0)

While the project is on a `0.x` line, the public API is **not yet stable**:

- `0.x` minor bumps (e.g. `0.1.0` -> `0.2.0`) MAY include backward-incompatible changes.
  Each minor release will document its breaking changes in this file.
- `0.x.y` patch bumps (e.g. `0.1.0` -> `0.1.1`) are **non-breaking** and contain only
  bug fixes, documentation, or internal improvements.
- The first stable release will be `1.0.0`. Until then, pin to a specific minor
  version (`smythe~=0.1.0`) if API stability matters to you.

Releases 0.1.0 through 0.8.1 are recorded, unchanged, in the
[changelog archive](CHANGELOG-ARCHIVE.md).

---

## [Unreleased]

### Added

- `OpenAIImageProvider(background=...)` accepts `"auto"` (default),
  `"opaque"` or `"transparent"`. It is sent only when not `"auto"`, so
  existing requests are unchanged; a transparent background requires PNG or
  WebP output.
- Noumenon benchmark lanes for transparent PNGs and SVG conversion.
  `--background transparent` keeps the provider's alpha, asks for a
  transparent background in the prompt, gates every tile on an objective
  transparency check, and refuses models without transparent output before
  any call. `--vectorize` traces every accepted tile into an even-odd SVG in
  pure Python and Pillow and accepts it only when it rasterizes back to the
  source mask at an IoU of at least 0.98.
- `tools/evidence_archive.py` builds a deterministic archive of retired
  benchmark evidence from git history and verifies it against a committed
  pointer, every member's SHA-256 and, optionally, the recorded commit.
  [`benchmarks/archive/`](benchmarks/archive/README.md) pins the first archive:
  the first glyph fan-out campaign, the browser renderer timing study,
  superseded glyph partitions, review records and the withdrawn native package
  receipts, read from `v0.8.2`.

### Changed

- **The glyph fan-out example is now Noumenon.** The harness is
  `benchmarks/run_noumenon.py` (assets in `benchmarks/noumenon_assets.py`), its
  records are `benchmarks/results/noumenon_*.json`, its report is
  [`benchmarks/noumenon_benchmark.md`](benchmarks/noumenon_benchmark.md), and its
  charts are `assets/benchmarks/noumenon_scaling.svg` and `assets/noumenon/`.
- **The Noumenon benchmark was re-run.** The four-width realistic-latency sweep
  and the default 250 ms profile ran again from a clean checkout under the
  unchanged protocol. At 192 nodes, concurrency 64 now measures 52.32×
  (21.7 s against 1,133.3 s serially); the superseded August records,
  which measured 56.21×, are in the evidence archive. Live GPT Image lanes
  recorded 192 transparent PNGs and their SVG conversions.
- **The screensaver apps moved to the
  [Noumenon repository](https://github.com/petehottelet/noumenon)**: the web
  explorer, the Windows, macOS and Linux ports, the catalog exporters, their
  tests and the screensaver workflow. The glyph generator, its contours and the
  192-glyph catalog stay in Smythe under `benchmarks/noumenon/`, where the SVG
  workflow benchmark imports them.
- A live Noumenon run that halts, for example on a provider rate limit,
  records what its completed calls charged and marks the total incomplete,
  instead of recording no cost.
- Benchmark environment snapshots mark a checkout `dirty` only when tracked
  files differ from the recorded revision, and count untracked files
  separately in `untracked_files`.
- Released changelog sections 0.1.0 through 0.8.1 moved, unchanged, to
  [`CHANGELOG-ARCHIVE.md`](CHANGELOG-ARCHIVE.md).

### Removed

- **Repo Doctor moved to its own repository,
  [repodoctor](https://github.com/petehottelet/repodoctor).** The skill, its
  archive builder `tools/skill_archive.py`, its tests and its distribution
  workflow left Smythe, and Smythe releases no longer attach the skill ZIP.

## [0.8.2] - 2026-09-25

This correctness release closes gaps in the execution envelope: rejected
durable plans that stopped a run for good, verification gates a supervisor
could revise away, refused provider replies counted as output, and image
frames decoded before they were bounded. Some fixes tighten rules that 0.8.1
accepted; each is listed under **Changed**. The
[upgrade guide](docs/release-0.8.2.md) lists what to check before upgrading.

### Fixed

- **A durable run repairs a generated plan it cannot execute instead of
  stopping for good.** A schema-valid plan that set `attach_dep_artifacts`, or
  broke the run's `WorkflowGraphPolicy`, was rejected only after its planning
  call was journaled, so every resume replayed it and failed. `LLMArchitect`
  and `ConstrainedArchitect` now check each plan against the run's graph rules
  inside their retry loop and send the reason back to the model; each repair is
  a new journaled call. Resuming a run that 0.8.1 stopped this way asks for a
  repair.
- A durable plan with a node id the journal cannot key (over 256 characters, or
  with a control character) is repaired before planning is saved, instead of
  failing at the node's first call on every resume.
- In a durable run, a revision that adds a node under the id of a node with
  journaled calls is rejected and traced. It raised `WorkflowConflictError` on
  every resume, or replayed the old node's result when the request matched.
- **A supervisor revision can no longer drop a verification gate or cut it off
  from the node it judges.** A revision that dropped a verifier let the output
  it judged through unverified, and one that rewired the verifier off its target
  let a parallel judge finish first, so its FAIL verdict was discarded.
  `ExecutionGraph.apply_revision` now rejects a revision that drops a verifier
  or the node it judges, or that leaves the verifier no longer depending on
  that node, directly or through the nodes between them. The rule applies to
  every supervisor; a rejected revision is traced and the run continues. A
  revision can still add steps after a gated node, and the gate does not judge
  their output.
- **Generated plans may contain at most one gating node, which must list the
  node it verifies in `depends_on`.** Every node could be a gate: an 8-node plan
  with seven gates on one node made 78 node calls. A gate without that edge
  could run before its target and have its verdict discarded. Such plans are
  retried with the problem named.
- A gate inside a `ConstrainedArchitect` template judges its own renamed
  target. Composition prefixed node ids but not `verifies`, so the gate's
  verdict was always discarded.
- `LLMSupervisor` limits how many revision-added nodes a run's graph may hold
  with `max_total_added_nodes` (default 8, the generated-plan node limit).
  `max_added_nodes` bounded only one revision, so five revisions could append 15
  nodes. Nodes a revision adds carry `"added_by_revision": true` in their
  metadata, so resuming does not reset the count. An added node's label may be
  at most 500 characters.
- A node can no longer use the id `__synthesis__`, the key under which
  `LLM_MERGE` synthesis books its charge; a node with that id shared the
  synthesis entry in the budget breakdown.
- **Jobs artifact inspection bounds every animation frame, not only the
  first.** A GIF frame that extends past the logical screen grew the canvas
  during decoding: a 74-byte GIF was accepted at 10×10 after a 718 MB decode at
  9001×9001. The canvas every GIF frame needs is now checked from the bytes
  before Pillow parses them, and each frame, including each picture of a
  multi-picture JPEG, against the per-image and aggregate pixel limits before
  it is decoded. Pillow reads on past two GIF block terminators where the
  format ends a block, so a frame hidden after one escaped the check; a GIF
  with bytes after such a terminator is refused as undecodable. The GIF check
  also covers `validate_image`, `finish_image` and the design checks, where
  Pillow allocated an oversized first frame while opening the file.
- WebP canvases are checked before libwebp allocates them. A 40-byte lossy
  WebP, or a 26-byte lossless one, declaring 16383×16383 made libwebp allocate
  about 2 GB. This covers Jobs inspection, `finish_image`, `validate_image` and
  the design checks, which now open a path once, so the bytes checked are the
  bytes decoded.
- Jobs artifact inspection no longer changes the process's warning filters.
  Escalating `DecompressionBombWarning` inside `warnings.catch_warnings()` is
  not thread-safe, so overlapping inspections could leave the escalation
  installed for the whole process. Every undecodable image, including a GIF on
  which Pillow raises `EOFError`, `IndexError` or `struct.error`, is reported as
  `ArtifactInspectionError`.
- Distilled templates ignore a model-supplied param named `task` or `params`
  instead of raising `TypeError`, which cost a paid planner retry.
- **Refused and filtered replies no longer count as output.** With the
  Anthropic, OpenAI and Gemini SDK providers, an Anthropic `refusal`, an OpenAI
  `content_filter` finish, or any Gemini finish except `STOP`, `MAX_TOKENS` or an
  unspecified one (such as `SAFETY`, `RECITATION` or `MALFORMED_FUNCTION_CALL`)
  completed its node with any partial text, and a filtered OpenAI turn ran its
  tool calls. So did a Gemini response to a blocked prompt (no candidates and a
  `prompt_feedback.block_reason`) and an OpenAI message with `refusal` set.
  The node now fails with `OutputRefusedError` after its cost is recorded, no
  tool call from that turn runs, and its failure policy applies. A refused plan
  is retried with the stop reason named, a refused supervisor review applies no
  revision, and a refused `LLM_MERGE` synthesis fails the run.
- Twelve exceptions whose constructors take more than a message could not be
  unpickled or copied intact. Most, including `SentinelAlert`,
  `BudgetReconciliationError`, `ProviderRequestRejectedError` and
  `ResponseQuoteError`, raised `TypeError`; `WorkerStartupInterrupted` raised
  `ValueError`; `OutputTruncatedError` came back with its message wrapped twice
  and `AssetPreflightError` with its message split into characters. They now
  survive `pickle`, `copy.copy` and `copy.deepcopy` with their message and
  fields, including across process boundaries.

### Changed

- A generated node `timeout_s` must be at least 60 seconds
  (`smythe.loader.MODEL_PLAN_MIN_TIMEOUT_S`); a shorter one is retried. A timeout
  cancels calls already sent, so their spend bought nothing, and a durable run
  blocked on unknown billing. Developer-written Python and YAML graphs are
  unchanged.
- `ConstrainedArchitect` limits the composed graph to `max_nodes` (default 64),
  checked after each builder call; a selection over the limit is retried. A
  66-byte reply could compose 200,001 nodes. Builders receive model-chosen
  params and must bound them.
- In a durable run, a generated plan that the graph policy or the plain-text
  rule rejects now costs up to `max_retries` repair calls, and planning that
  still fails raises `ArchitectError` instead of `WorkflowBindingError`.
- In a durable run, a node id must be at most 256 characters with no control
  characters (`smythe.workflow.MAX_NODE_ID_CHARS`). A caller-built graph, or a
  resumed saved graph, with another id raises `WorkflowBindingError` before any
  node runs.
- Any supervisor revision, including one from a custom `Supervisor`, that drops
  a verification gate or its target, cuts the gate off from its target, or adds
  a node with the id `__synthesis__` is rejected and traced as
  `revision_rejected`.
- `ExecutionGraph.validate()` raises `ValueError` for the node id
  `__synthesis__` in any graph, including when resuming a checkpoint or durable
  run that contains one.
- Jobs reports an animated GIF whose frames extend past its logical screen at
  the size of its enlarged canvas, and counts every frame at that size toward
  the aggregate pixel limit.
- A GIF with bytes after a block terminator that Pillow would read past is
  refused by Jobs inspection, `validate_image`, `finish_image` and the design
  checks.
- `OpenAIProvider` and `GeminiProvider` report the stop reasons `refusal`
  (OpenAI only), `content_filter` and (Gemini only) `incomplete` where they
  reported `end_turn` or `tool_use`; see the [execution guide](docs/execution.md). Stop
  reasons outside `TRUNCATED_STOP_REASONS` and `REFUSED_STOP_REASONS`,
  including any a custom provider returns, still count as complete.
  `OutputTruncatedError` now subclasses `IncompleteOutputError`.

### Added

- `IncompleteOutputError` and `OutputRefusedError`, exported from `smythe`, and
  `smythe.provider.REFUSED_STOP_REASONS`.
- `LLMSupervisor(max_total_added_nodes=...)` and
  `ConstrainedArchitect(max_nodes=...)`.
- `smythe.loader.MODEL_PLAN_MAX_GATES` and `MODEL_PLAN_MIN_TIMEOUT_S`;
  `smythe.graph.SYNTHESIS_NODE_ID` and `REVISION_ADDED_KEY`;
  `smythe.workflow.MAX_NODE_ID_CHARS`.

[Unreleased]: https://github.com/petehottelet/smythe/compare/v0.8.2...HEAD
[0.8.2]: https://github.com/petehottelet/smythe/compare/v0.8.1...v0.8.2
