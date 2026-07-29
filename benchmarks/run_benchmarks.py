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
    """Return (provider, executor model, judge model) for the first
    configured API key, else None.

    The judge deliberately runs on a different model than the executor
    to reduce self-preference bias in scoring.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        from smythe.provider import AnthropicProvider
        return AnthropicProvider(), "claude-opus-4-8", "claude-sonnet-5"
    if os.environ.get("OPENAI_API_KEY"):
        from smythe.provider import OpenAIProvider
        return OpenAIProvider(), "gpt-5.2", "gpt-5.2"
    if os.environ.get("GOOGLE_API_KEY"):
        from smythe.provider import GeminiProvider
        return GeminiProvider(), "gemini-3-flash", "gemini-3-flash"
    return None


_JUDGE_PROVIDERS: dict[str, object] = {}


def judge_provider_for(judge_model: str, executor_provider):
    """Provider for the judge, chosen from the judge model's name.

    The judge should be able to run on a different vendor than the
    executor — that is the point of using a different model — so it
    cannot simply reuse the executor's provider. Falls back to the
    executor's provider when the model is not recognized.
    """
    if judge_model in _JUDGE_PROVIDERS:
        return _JUDGE_PROVIDERS[judge_model]
    lower = (judge_model or "").lower()
    provider = executor_provider
    try:
        if lower.startswith("gemini") or lower.startswith("nano-banana"):
            from smythe.provider import GeminiProvider
            provider = GeminiProvider()
        elif lower.startswith("claude"):
            from smythe.provider import AnthropicProvider
            provider = AnthropicProvider()
        elif lower.startswith(("gpt", "o1", "o3", "o4")):
            from smythe.provider import OpenAIProvider
            provider = OpenAIProvider()
    except Exception:
        provider = executor_provider
    _JUDGE_PROVIDERS[judge_model] = provider
    return provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="run only this task name")
    parser.add_argument("--baseline", default=None, choices=BASELINES,
                        help="run only this baseline")
    parser.add_argument("--offline", action="store_true",
                        help="force offline mode even if an API key is set")
    parser.add_argument("--judge", action="store_true",
                        help="score outputs against the rubric (real mode only)")
    parser.add_argument("--judge-model", default=None, metavar="MODEL",
                        help="override the judge model (default: a different "
                             "model than the executor, to reduce self-preference)")
    parser.add_argument("--runs", type=int, default=1, metavar="N",
                        help="repeat each (task, baseline) N times")
    parser.add_argument("--ablate-terminal-note", action="store_true",
                        help="ablation: blank the terminal deliverable note so "
                             "final nodes are not told their output is the "
                             "deliverable (for A/B comparisons)")
    parser.add_argument("--ablate-task-context", action="store_true",
                        help="ablation: root nodes see only their planned "
                             "label, not the original task payload "
                             "(for A/B comparisons)")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="write results JSON to FILE")
    args = parser.parse_args()

    if args.ablate_terminal_note:
        import smythe.executor_base as _eb
        _eb.TERMINAL_DELIVERABLE_NOTE = ""
        print("ABLATION: terminal deliverable note disabled for this run.\n")
    if args.ablate_task_context:
        import smythe.executor_base as _eb2
        _eb2.INCLUDE_TASK_CONTEXT = False
        print("ABLATION: task context disabled for this run.\n")

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
            for run_index in range(args.runs):
                if offline:
                    provider, model = offline_provider(baseline), "offline-model"
                    judge_model = None
                else:
                    provider, model, judge_model = real
                    judge_model = args.judge_model or judge_model
                try:
                    record = run_one(bench_task, baseline, provider, model,
                                     offline=offline)
                except Exception as exc:
                    # Provider timeouts and 5xx are facts of life across a
                    # 45-cell campaign. Losing one cell is a gap in the
                    # table; losing the campaign is an hour and real money.
                    print(f"  FAILED {bench_task.name}/{baseline} "
                          f"run={run_index}: {type(exc).__name__}: {exc}")
                    records.append({
                        "task": bench_task.name, "baseline": baseline,
                        "model": model, "offline": offline, "run": run_index,
                        "quality": None, "cost_usd": None, "nodes": None,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    })
                    continue
                record["error"] = None
                record["run"] = run_index
                record["judge_model"] = None
                record["ablate_terminal_note"] = args.ablate_terminal_note
                record["ablate_task_context"] = args.ablate_task_context
                if args.judge and not offline:
                    try:
                        record["quality"] = score_output(
                            judge_provider_for(judge_model, provider),
                            judge_model, bench_task.goal,
                            bench_task.rubric, record["output"],
                        )
                    except Exception as exc:
                        # An unscored cell is a gap in the table; a lost
                        # campaign is hours and dollars of real work.
                        record["quality"] = None
                        record["judge_error"] = str(exc)[:200]
                        print(f"  judge failed on {bench_task.name}/{baseline}: {exc}")
                    record["judge_model"] = judge_model
                records.append(record)
                print(f"  done: {bench_task.name} / {baseline} run={run_index}"
                      f"  nodes={record['nodes']}  topology={record['topology']}")

    print(f"\n{'task':<24} {'baseline':<16} {'runs':>4} {'quality':>14} "
          f"{'mean_cost':>10}")
    for bench_task in tasks:
        for baseline in baselines:
            cell = [r for r in records
                    if r["task"] == bench_task.name and r["baseline"] == baseline]
            if not cell:
                continue
            costs = [r["cost_usd"] for r in cell]
            scores = [r["quality"]["overall"] for r in cell if r["quality"]]
            if scores:
                quality = (f"{sum(scores) / len(scores):.1f} "
                           f"[{min(scores)}-{max(scores)}]")
            else:
                quality = "-"
            print(f"{bench_task.name:<24} {baseline:<16} {len(cell):>4} "
                  f"{quality:>14} {sum(costs) / len(costs):>10.4f}")

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
