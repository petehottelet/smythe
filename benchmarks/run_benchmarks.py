"""Run the Smythe benchmark suite.

    python benchmarks/run_benchmarks.py                      # offline mechanics run
    python benchmarks/run_benchmarks.py --task research-memo --baseline smythe_dynamic
    python benchmarks/run_benchmarks.py --judge              # real mode: score quality

Offline (no API keys): every task runs through all three baselines with
deterministic providers — verifies the harness end to end at zero cost,
produces no quality numbers. With a provider key set, the same command
runs real models, and --judge adds blind rubric scoring.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from harness import BASELINES, load_tasks, offline_provider, run_one
from judge import score_output

# Rendering uses characters legacy cp1252 Windows consoles can't encode.
for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

BENCH_DIR = Path(__file__).parent


def real_provider():
    """Return (provider, model) for the first configured API key, else None."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from smythe.provider import AnthropicProvider
        return AnthropicProvider(), "claude-opus-4-8"
    if os.environ.get("OPENAI_API_KEY"):
        from smythe.provider import OpenAIProvider
        return OpenAIProvider(), "gpt-5.2"
    if os.environ.get("GOOGLE_API_KEY"):
        from smythe.provider import GeminiProvider
        return GeminiProvider(), "gemini-3-flash"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="run only this task name")
    parser.add_argument("--baseline", default=None, choices=BASELINES,
                        help="run only this baseline")
    parser.add_argument("--offline", action="store_true",
                        help="force offline mode even if an API key is set")
    parser.add_argument("--judge", action="store_true",
                        help="score outputs against the rubric (real mode only)")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="write results JSON to FILE")
    args = parser.parse_args()

    real = None if args.offline else real_provider()
    offline = real is None
    if offline:
        print("Running offline: mechanics only, no quality scores, no cost.")
        print("Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY for real runs.\n")
        if args.judge:
            print("--judge ignored offline: scoring canned text is meaningless.\n")

    tasks = load_tasks(BENCH_DIR / "tasks")
    if args.task:
        tasks = [t for t in tasks if t.name == args.task]
        if not tasks:
            print(f"No task named {args.task!r}", file=sys.stderr)
            return 2
    baselines = [args.baseline] if args.baseline else list(BASELINES)

    records = []
    for bench_task in tasks:
        for baseline in baselines:
            if offline:
                provider, model = offline_provider(baseline), "offline-model"
            else:
                provider, model = real
            record = run_one(bench_task, baseline, provider, model, offline=offline)
            if args.judge and not offline:
                record["quality"] = score_output(
                    provider, model, bench_task.goal, bench_task.rubric,
                    record["output"],
                )
            records.append(record)
            print(f"  done: {bench_task.name} / {baseline}"
                  f"  nodes={record['nodes']}  topology={record['topology']}")

    print(f"\n{'task':<24} {'baseline':<16} {'nodes':>5} {'depth':>5} "
          f"{'cost_usd':>9} {'quality':>8}")
    for r in records:
        quality = r["quality"]["overall"] if r["quality"] else "-"
        print(f"{r['task']:<24} {r['baseline']:<16} {r['nodes']:>5} {r['depth']:>5} "
              f"{r['cost_usd']:>9.4f} {quality:>8}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = BENCH_DIR / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
