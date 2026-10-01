# Jobs 5,000-operation evidence bundle

**Independent review passed; claimable for the stated offline correctness
observation.** This is one campaign using identical 1×1 PNG fixtures. It makes no comparative
speed or model-quality claim. See the
[result report](../../jobs_scale_5000_20260907_results.md) and
[frozen protocol](../../jobs_scale_benchmark.md).

| File | Contents |
|---|---|
| [result.json](result.json) | Original terminal producer result; only its database digest was recomputed on September 30, 2026 |
| [evidence.zip](evidence.zip) | Complete campaign tree, original result, frozen producer source, and separate review source |
| [inventory.json](inventory.json) | Every archive member's raw SHA-256, size, kind, and original modification time; archive/result hashes |
| [review.json](review.json) | Successful independent reconciliation, required checks, exact source/runtime identities, and no known measurement defects |

The 7,847,919-byte archive contains 15,088 members: 5,085 files and 10,003
directories. The result's SHA-256 is
`c80c52bb377a82468f4fa957eb086f2bdaf5a9f0a9de8fa4485cd102ad0b7067`;
the archive's SHA-256 is
`2c841d8d059f2ada3dcd72d8c15a97f0a3c953173d043a3418329fcdc438bf6d`.

On September 30, 2026, a local path in the database's `runs.manifest_root`
value was replaced with the placeholder `<checkout>`. No other database row
changed. The database, result, inventory, review and archive digests and sizes
were recomputed, and the retained reconciliation still passes with no known
measurement defects.
[path-redaction-20260930.json](path-redaction-20260930.json) lists each changed
file's original and current SHA-256.

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
