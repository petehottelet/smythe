# Jobs 5,000-operation evidence bundle

**Independent review passed; claimable for the stated offline correctness
observation.** This is one campaign using identical 1×1 PNG fixtures. It makes no comparative
speed or model-quality claim. See the
[result report](../../jobs_scale_5000_20260907_results.md) and
[frozen protocol](../../jobs_scale_benchmark.md).

| File | Contents |
|---|---|
| [result.json](result.json) | Original terminal producer result, unchanged |
| [evidence.zip](evidence.zip) | Complete campaign tree, original result, frozen producer source, and separate review source |
| [inventory.json](inventory.json) | Every archive member's raw SHA-256, size, kind, and original modification time; archive/result hashes |
| [review.json](review.json) | Successful independent reconciliation, required checks, exact source/runtime identities, and no known measurement defects |

The 7,875,780-byte archive contains 15,088 members: 5,085 files and 10,003
directories. The original result's SHA-256 is
`96df796d8dfe452e8ef7167f4fd7648b59ea0410a95cd5be228497e9e4626695`;
the archive's SHA-256 is
`aad4eec40c613c1199812c25b692cc6f87f324fa2e8394151ec43acb8fd54751`.

The archive contains `campaign/` with the final SQLite database and sidecars,
worker logs, configuration, provenance, provider-entry/return logs, recovery
receipts, and all 5,000 accepted PNGs. `source/` contains the frozen producer
Python files. `review-source/` contains the archiver, chart validator, drawing
helpers, and archive regression tests. Archive members use byte-exact hashes;
producer source identity additionally uses LF-normalized hashes matched to
the original Git blobs.

Producer revision:
`4bb7c0295cec658c7118d7bf9305764dae9b1c57` (Jobs schema v3).
The producer and all owned workers exited before archiving. SQLite review
uses only an extracted temporary copy. Historical state is reconstructed
from final ledger events, provider logs, and retained receipt-subset hashes;
this archive does not contain historical database snapshots.

To verify the archive's member hashes from the repository root:

```python
import json
from pathlib import Path
from benchmarks.archive_jobs_scale import verify_archive

bundle = Path("benchmarks/results/jobs_scale_5000_20260907_f1")
inventory = json.loads((bundle / "inventory.json").read_bytes())
verify_archive(bundle / "evidence.zip", inventory["members"])
```

To repeat reconciliation, extract `evidence.zip` to a new directory and use a
checkout at the exact producer revision. Run the retained review code with
Python and Pillow installed; choose a new output directory:

```bash
python EXTRACTED/review-source/benchmarks/archive_jobs_scale.py \
  --campaign EXTRACTED/campaign \
  --result EXTRACTED/result.json \
  --source-root FROZEN_4BB7C02_CHECKOUT \
  --out NEW_REVIEW_DIRECTORY
```

No provider SDK or API key is needed. Keep the original bundle unchanged.
