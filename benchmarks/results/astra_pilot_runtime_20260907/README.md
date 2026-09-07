# Astra pilot runtime: offline qualification

**79 tests passed in 684.20 seconds** on 7 September 2026. This qualifies
offline runtime contracts on the source identified below. No paid campaign
has run, no live allowance is authorized by this record, and it makes no
model-quality, API-cost, or performance comparison claim.

The complete [test log](pytest.log) and [machine-readable qualification](qualification.json)
are retained here. The elapsed time is test-suite duration, not pilot latency.

## Tested source

The test ran in an isolated worktree at
[`f1bffa53d5a060acc400dd85f4ceb9c408faa334`](https://github.com/petehottelet/smythe/commit/f1bffa53d5a060acc400dd85f4ceb9c408faa334)
with these two explicit additions:

| File | SHA-256 after CRLF-to-LF normalization |
|---|---|
| [benchmarks/astra_runtime.py](../../astra_runtime.py) | `c1f5f6e7fe9fd953f546245be3852a197beb7176ac55fbb9a221ed81b6b419fa` |
| [tests/test_astra_runtime.py](../../../tests/test_astra_runtime.py) | `abb781f66f6741451b1ecd56d178a9a13fc5a4040143cddc1714454354fbebff` |

The [original before manifest](sources-before.json) binds all 61 files:
tracked Smythe Python sources, `pyproject.toml`, and the two additions. The
test wrapper verified that this manifest was unchanged after the run.
The source links above aid navigation; the hashes define the tested versions.
Later checkpoint, version, packaging, and documentation edits are outside
this particular focused qualification and require their release checks.

## What was exercised

- Strict spending caps, source and destination binding, approval refusal,
  campaign writer exclusion, and closed main/judge commands.
- All 12 scheduled pilot arms through the durable workflow ledger, with
  fixed three-stage and generated graphs, exact model/graph limits, and
  parallel execution of independent generated nodes.
- Failed and completed outcomes, unknown billing, same-identity recovery,
  immutable accounting/output receipts, and rejection of altered checkpoint,
  response, event, or unaccepted-quote evidence.
- Linked/reparse paths, receipt collisions, ignored scratch exclusion, and
  preservation of the original campaign preparation and task data.

The tests replace `OpenAIResponsesProvider._get_client` with an in-process
fake Responses client and use fixture dependency labels, including
`offline-contract-sdk`. They exercise the native adapter and ledger through
mocked transport. They do **not** qualify installed SDK serialization, live
endpoint or account access, model quality, or real spending. Their synthetic
allowance and usage values do not authorize a campaign or measure API charges.
Real-SDK release checks, when available, have separate receipts.

## Receipt provenance and reproduction

`pytest.log` and `sources-before.json` are byte-for-byte copies of the original
qualification artifacts. Historical import, Ruff, and end-of-run source checks
were observed in tool output but were not saved as standalone logs. The
[revalidation receipt](revalidation.json) and [after manifest](sources-after.json)
were captured later from the unchanged frozen worktree; their capture time and
scope are explicit. Revalidation confirmed the import root, scoped Ruff, all
61 source hashes, and all 34 tracked preparation files, including the unchanged
[historical receipt](../astra_preparation_20260907.json).

To reproduce, create an isolated checkout of the base commit, add the exact two
files above, and verify every entry in `sources-before.json` using its stated
LF normalization. Install that checkout's development dependencies, select it
as the import root, and run from that checkout:

```bash
python -m pytest tests/test_astra_runtime.py -q
python -m ruff check benchmarks/astra_runtime.py tests/test_astra_runtime.py
```

API credentials are unnecessary. Compare the source manifest again afterward.
The [pilot guide](../../astra_runtime.md) describes the separate live execution
requirements; this record leaves paid execution, main, judging, and quality
acceptance closed.
