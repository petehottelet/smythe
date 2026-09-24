# Smythe 0.8.2

[GitHub release](https://github.com/petehottelet/smythe/releases/tag/v0.8.2) ·
[PyPI package](https://pypi.org/project/smythe/0.8.2/)

Smythe 0.8.2 is a correctness release. It closes gaps in the execution
envelope: a durable run repairs a generated plan it cannot execute instead of
stopping for good, a supervisor revision can no longer drop a verification gate
or cut it off from its target, refused or filtered provider replies stop
counting as output, and Jobs bounds every image frame before decoding it.

```bash
pip install "smythe[openai]==0.8.2"
# For native Claude Messages:
pip install "smythe[anthropic]==0.8.2"
```

## What changes

- **Durable planning repairs rejected plans.** `LLMArchitect` and
  `ConstrainedArchitect` check each plan against the run's graph policy and
  plain-text node rules inside their retry loop, so a plan the run cannot
  execute goes back to the model with the reason instead of wedging the run.
- **Verification gates are protected.** No supervisor revision can drop a
  gate or the node it judges, or leave the gate no longer depending on that
  node, directly or through the nodes between them. A revision can still add
  steps after a gated node, and the gate does not judge their output. A
  generated plan may contain one gate, which must depend on its target, and a
  gate inside a `ConstrainedArchitect` template now judges its own renamed
  target.
- **Supervised growth is bounded.** `LLMSupervisor` keeps at most 8
  revision-added nodes in a run's graph by default (`max_total_added_nodes`),
  and an added node's label is at most 500 characters.
- **Refused and filtered replies fail their node.** With the Anthropic,
  OpenAI and Gemini SDK providers, a refusal, a content filter or an
  unfinished Gemini response raises `OutputRefusedError` after its cost is
  recorded, and no tool call from that turn runs.
- **Jobs bounds every image frame.** GIF frames that grow the canvas, and
  WebP canvases, are checked from the bytes before Pillow or libwebp allocates
  memory, and inspection no longer changes the process's warning filters.
- **Exceptions survive pickle and copy,** including across process
  boundaries.

Full behavior and fixes are in the [changelog](../CHANGELOG.md#082---2026-09-23).

## Upgrade from 0.8.1

Smythe is pre-1.0. This patch release tightens rules that 0.8.1 accepted where
that was needed to close a correctness gap. Check these before upgrading:

- **Generated plans** may contain at most one gating node, which must list the
  node it verifies in `depends_on`, and a node `timeout_s` must be at least 60
  seconds. A plan that breaks these rules is retried, which can cost an extra
  planning call.
- **Durable planning.** A generated plan that the run's graph policy or
  plain-text rule rejects now costs up to `max_retries` repair calls, and
  planning that still fails raises `ArchitectError` instead of
  `WorkflowBindingError`. A 0.8.1 run stopped by such a rejection resumes into
  a repair.
- **Supervisors.** A revision from any supervisor, including your own, that
  drops a verification gate or its target, or cuts the gate off from its
  target, is rejected and traced. `LLMSupervisor`
  refuses proposals past `max_total_added_nodes` (default 8) or with an added
  label over 500 characters.
- **ConstrainedArchitect** composes at most 64 nodes by default
  (`ConstrainedArchitect(max_nodes=...)`). Template builders receive
  model-chosen params and must bound them.
- **SDK providers.** An Anthropic `refusal`, an OpenAI `content_filter` finish
  or any Gemini finish except `STOP`, `MAX_TOKENS` or an unspecified one fails
  its node under the node's failure policy; `RETRY` makes another billed call.
  Durable runs are unaffected, because they accept only the native providers,
  which already reject unfinished responses.
- **Reserved id.** No graph may use the node id `__synthesis__`, so 0.8.2
  cannot resume a checkpoint or durable run that contains a node with that id.
- **Jobs** reports an animated GIF whose frames extend past its logical screen
  at the size of its enlarged canvas.
- **Rollback.** Checkpoints stay at version 4. A durable run whose plan was
  repaired under 0.8.2, or that uses a non-default `max_total_added_nodes` or
  `ConstrainedArchitect(max_nodes=...)`, cannot be resumed by 0.8.1; finish it
  on 0.8.2 or start a new run.

## Verify a checkout or package

Install `.[dev,openai,anthropic]` from the tagged checkout and run:

```bash
python -m ruff check .
python -m mypy
python -m pytest tests/ -q
python examples/14_durable_text_workflow.py
```

The example uses fixture responses and makes no provider calls. The
[distribution guide](distribution.md) covers archive inventories, source-to-wheel
reproduction, clean installed-package checks and the source test profile.
Package README links resolve against `v0.8.2`. The release workflow retains
the actual published distributions and SHA-256 hashes; verify downloaded PyPI
bytes against that receipt as described in [Releasing](../RELEASING.md).
