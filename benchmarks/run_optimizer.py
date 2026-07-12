"""Autoresearch-style optimizer for the brand asset suite.

    python benchmarks/run_optimizer.py --iterations 2 --budget-usd 1.50
    python benchmarks/run_optimizer.py --iterations 20 --budget-usd 12  # overnight

The loop (the karpathy/autoresearch pattern, applied to prompts instead
of train.py): an LLM proposes ONE targeted change to the brand config's
prompt policies, the full suite runs live, a vision judge scores brand
consistency, and the change is kept only if it clears the noise floor.

Grounded in measured judge variance (results/judge_variance.json:
stdev 0.49 on identical pixels): each candidate is scored as the MEAN
OF 3 INDEPENDENT JUDGINGS (sd of the mean ~0.28) and must beat the
incumbent by more than KEEP_MARGIN = 0.5 (~2 sigma). Hard gates that
cannot be sweet-talked: all 8 assets must pass format compliance, or
the candidate is rejected regardless of score.

Every experiment is journaled to results/optimizer_journal.jsonl —
kept or reverted, with the proposal, rationale, scores, and cost.
Iteration cost ~= $0.45; the loop stops before exceeding --budget-usd.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from run_asset_suite import judge_assets, load_brand, main_async as run_suite  # noqa: E402

from smythe.provider import GeminiProvider  # noqa: E402

KEEP_MARGIN = 0.5
JUDGINGS_PER_CANDIDATE = 3
EST_ITERATION_COST = 0.45
PROPOSER_MODEL = "gemini-flash-lite-latest"

TUNABLE_FIELDS = (
    "logo_lock_instruction",
    "no_text_instruction",
    "logo_prompt",
)

PROPOSER_PROMPT = """You are optimizing prompt policies for a parallel \
brand-asset image pipeline. Propose ONE targeted change to ONE field to \
improve the vision judge's brand-consistency score.

Tunable fields: {fields} (or "scenes.<asset_id>" for one scene).

Current values:
{current}

Latest judge feedback (score {score}, per-asset defects):
{defects}

Experiment history (most recent last; do not repeat failed ideas):
{history}

Respond with STRICT JSON only, no prose, no code fences:
{{"field": "<field name>", "value": "<complete replacement text>", \
"rationale": "<one sentence>"}}"""


def apply_proposal(brand: dict, proposal: dict) -> dict:
    candidate = copy.deepcopy(brand)
    field, value = proposal["field"], proposal["value"]
    if field.startswith("scenes."):
        asset_id = field.split(".", 1)[1]
        if asset_id not in candidate["scenes"]:
            raise ValueError(f"unknown scene {asset_id!r}")
        candidate["scenes"][asset_id] = value
    elif field in TUNABLE_FIELDS:
        candidate[field] = value
    else:
        raise ValueError(f"field {field!r} is not tunable")
    return candidate


async def propose(brand: dict, last_run: dict, history: list[dict]) -> dict:
    current = {f: brand.get(f, "<default>") for f in TUNABLE_FIELDS}
    defects = {
        e["asset"]: e.get("defects", [])
        for e in last_run.get("assets", [])
        if e.get("defects")
    }
    hist_lines = [
        f"- {h['proposal']['field']}: {h['proposal']['rationale']} -> "
        f"score {h['score']} ({'KEPT' if h['kept'] else 'reverted'})"
        for h in history[-8:]
    ] or ["(none yet)"]
    prompt = PROPOSER_PROMPT.format(
        fields=", ".join(TUNABLE_FIELDS),
        current=json.dumps(current, indent=1),
        score=last_run.get("brand_judge", {}).get("overall"),
        defects=json.dumps(defects, indent=1),
        history="\n".join(hist_lines),
    )
    result = await GeminiProvider().complete(
        "You optimize prompts. Strict JSON only.", prompt, PROPOSER_MODEL,
    )
    text = result.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    return json.loads(text)


async def score_candidate(
    brand: dict, out: Path, *, run_index: int,
) -> tuple[dict, float | None, float]:
    """Run the suite once, then extra judgings for a noise-resistant mean."""
    run = await run_suite(brand, True, out / f"iter_{run_index}", None, True)
    cost = run["cost_usd"]
    scores = [run["brand_judge"]["overall"]] if run.get("brand_judge") else []
    finals = {
        e["asset"]: e["final_path"] for e in run["assets"] if e.get("final_path")
    }
    for _ in range(JUDGINGS_PER_CANDIDATE - 1):
        verdict = await judge_assets(
            brand=brand, live=True, out=out / f"iter_{run_index}",
            logo_path=run["brand_logo"]["path"], finals=finals,
        )
        cost += verdict["cost_usd"]
        if verdict["overall"] is not None:
            scores.append(verdict["overall"])
    mean_score = round(mean(scores), 2) if scores else None
    return run, mean_score, cost


async def main_async(args) -> None:
    brand = load_brand(args.brand)
    out = Path(args.out)
    journal_path = Path("benchmarks/results/optimizer_journal.jsonl")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    spent = 0.0

    print("[baseline] scoring incumbent config...")
    base_run, best_score, cost = await score_candidate(brand, out, run_index=0)
    spent += cost
    print(f"[baseline] mean of {JUDGINGS_PER_CANDIDATE} judgings: "
          f"{best_score} (${spent:.2f} spent)")
    last_run = base_run

    for i in range(1, args.iterations + 1):
        if spent + EST_ITERATION_COST > args.budget_usd:
            print(f"[stop] budget: ${spent:.2f} spent, "
                  f"next iteration would exceed ${args.budget_usd:.2f}")
            break
        proposal = await propose(brand, last_run, history)
        try:
            candidate = apply_proposal(brand, proposal)
        except (ValueError, KeyError) as exc:
            print(f"[iter {i}] invalid proposal rejected: {exc}")
            continue
        print(f"[iter {i}] {proposal['field']}: {proposal['rationale']}")

        run, score, cost = await score_candidate(candidate, out, run_index=i)
        spent += cost
        gates_ok = run["assets_pass"] == "8/8"
        kept = bool(
            gates_ok and score is not None and best_score is not None
            and score > best_score + KEEP_MARGIN
        )
        if kept:
            brand, best_score, last_run = candidate, score, run
        entry = {
            "iteration": i,
            "proposal": proposal,
            "score": score,
            "best_score": best_score,
            "format_gates_ok": gates_ok,
            "kept": kept,
            "spent_usd": round(spent, 4),
        }
        history.append(entry)
        with journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[iter {i}] score={score} best={best_score} "
              f"{'KEPT' if kept else 'reverted'} (${spent:.2f} spent)")

    final_config = Path("benchmarks/results/optimizer_best_config.json")
    final_config.write_text(json.dumps(brand, indent=2), encoding="utf-8")
    print(f"\nBest score {best_score}; config -> {final_config}; "
          f"journal -> {journal_path}; total ${spent:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand", default="benchmarks/brands/metacortex.json")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--budget-usd", type=float, default=1.50)
    parser.add_argument("--out", default="smythe_artifacts/optimizer")
    args = parser.parse_args()
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY required (the optimizer is live-only)")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
