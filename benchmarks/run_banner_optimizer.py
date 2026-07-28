"""Autoresearch optimizer for banner-ad creative direction.

    python benchmarks/run_banner_optimizer.py --offline --iterations 2   # mechanics, free
    python benchmarks/run_banner_optimizer.py \
        --logo 00_project_files/osiris_light.png \
        --logo-dark 00_project_files/osiris_dark.png \
        --iterations 4 --search-k 2 \
        --budget-usd 8 --iteration-cost-ceiling-usd 1.4 \
        --max-cost-per-call-usd 0.06

The karpathy/autoresearch loop (the same pattern as run_optimizer.py for
the asset suite, here applied to the *banner* pipeline): an LLM proposes
ONE targeted change to the creative direction, the full banner suite
runs live, the ad-craft vision judge scores every finished banner, and
the change is KEPT only if it beats the incumbent by more than the
noise-margin.

Search space (the three text blocks that steer the live creative
director in run_banner_suite.py):
  - creative_worldview : the campaign's aesthetic worldview
  - brief              : the product/palette/style brief
  - craft_canon        : the brand-agnostic ad-craft rules

Objective: the mean judge score across EVERY candidate of EVERY
placement (not just the winners). Averaging over all candidates x all
placements both reduces judge noise and rewards a direction that
reliably produces good candidates rather than one lucky hero. The
winner-mean (what actually ships) is recorded alongside as a secondary
metric.

HONEST LIMITATION, stated up front: after the manual design fixes the
banner judge already scores most placements 8-9 (ceiling compression,
the same effect documented in the framework head-to-head). The optimizer
may therefore measure a null result on a well-tuned config - that is a
legitimate scientific outcome, reported as-is, not hidden. The judge is
also same-vendor (Gemini) as the generator; cross-vendor judging is a
documented hardening path, not yet wired here.

Budget: the caller must supply an inclusive per-image ceiling, a
conservative whole-iteration ceiling, and a total budget. The baseline
is not scored unless one full iteration fits inside the budget, and the
loop stops before any iteration that would exceed it - identical
guardrails to run_optimizer.py.
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

from benchmarks.run_banner_suite import (  # noqa: E402
    load_brand,
    main_async as run_banner_suite,
)
from smythe.provider import GeminiProvider  # noqa: E402

KEEP_MARGIN = 0.4  # heuristic noise guard on a 1-10 scale, averaged over
# ~14-21 candidates; NOT a confidence bound (see run_optimizer.py's note).
PROPOSER_MODEL = "gemini-flash-latest"
PROPOSER_COST_PER_TOKEN_USD = 0.000003
CRITERIA = ("focal_clarity", "photo_quality", "tonal_harmony", "cta_craft")

TUNABLE_FIELDS = ("creative_worldview", "brief", "craft_canon")

PROPOSER_PROMPT = """You are optimizing the creative direction for a \
parallel banner-ad image pipeline. An LLM art director turns three text \
blocks into photographic concepts; a vision judge then scores the \
finished banners on focal clarity, photo quality, tonal harmony, and CTA \
craft. Propose ONE targeted change to ONE of the three blocks to raise \
the judge's mean score.

Tunable fields: {fields}.

Current values:
{current}

Latest judge feedback (mean score {score}; per-placement weak criteria \
and defects):
{feedback}

Experiment history (most recent last; do not repeat failed ideas):
{history}

Improve the direction with a concrete, specific edit - sharper hero \
framing, better light, cleaner reserved zones, stronger tonal contrast. \
Return the COMPLETE replacement text for the field (not a diff). Respond \
with STRICT JSON only, no prose, no code fences:
{{"field": "<field name>", "value": "<complete replacement text>", \
"rationale": "<one sentence>"}}"""


def objective_from_record(record: dict) -> tuple[float | None, float | None]:
    """(all-candidate mean, winner mean) across placements, 1-10 scale.

    The all-candidate mean is the optimization objective (more signal,
    less gameable); the winner mean is the shipped-quality secondary.
    """
    all_means: list[float] = []
    winner_means: list[float] = []
    for p in record.get("placements", []):
        sm = p.get("score_means", {})
        if sm:
            all_means.extend(sm.values())
            w = p.get("winner")
            if w in sm:
                winner_means.append(sm[w])
            continue
        # Fallback when score_means is absent (k=1 or unjudged): average
        # the winner's raw criteria.
        jw = p.get("judge", {}).get(p.get("winner"), {})
        nums = [jw.get(c) for c in CRITERIA if isinstance(jw.get(c), (int, float))]
        if nums:
            m = sum(nums) / len(nums)
            all_means.append(m)
            winner_means.append(m)
    obj = round(mean(all_means), 3) if all_means else None
    win = round(mean(winner_means), 3) if winner_means else None
    return obj, win


def _feedback(record: dict) -> str:
    """Compact per-placement judge feedback for the proposer."""
    lines = []
    for p in record.get("placements", []):
        jw = p.get("judge", {}).get(p.get("winner"), {})
        weak = [c for c in CRITERIA
                if isinstance(jw.get(c), (int, float)) and jw[c] <= 8]
        defects = [d for d in jw.get("defects", [])
                   if d and d.lower() not in ("none", "no defects")]
        if weak or defects:
            bits = []
            if weak:
                bits.append("weak: " + ", ".join(
                    f"{c}={jw[c]}" for c in weak))
            if defects:
                bits.append("defects: " + "; ".join(defects[:2]))
            lines.append(f"- {p['placement']}: {' | '.join(bits)}")
    return "\n".join(lines) or "(no weak criteria - near ceiling)"


def _offline_proposal(brand: dict, i: int) -> dict:
    """Deterministic proposal for offline mechanics runs (no LLM call)."""
    return {
        "field": "brief",
        "value": brand["brief"] + f" (offline variant {i})",
        "rationale": f"offline mechanics variant {i}",
    }


def apply_proposal(brand: dict, proposal: dict) -> dict:
    candidate = copy.deepcopy(brand)
    field, value = proposal["field"], proposal["value"]
    if field not in TUNABLE_FIELDS:
        raise ValueError(f"field {field!r} is not tunable")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"empty replacement value for {field!r}")
    candidate[field] = value
    return candidate


async def propose(
    brand: dict, last_record: dict, history: list[dict],
) -> tuple[dict, float]:
    current = {f: brand.get(f, "<default>") for f in TUNABLE_FIELDS}
    hist_lines = [
        f"- {h['proposal']['field']}: {h['proposal']['rationale']} -> "
        f"score {h['score']} ({'KEPT' if h['kept'] else 'reverted'})"
        for h in history[-8:]
    ] or ["(none yet)"]
    prompt = PROPOSER_PROMPT.format(
        fields=", ".join(TUNABLE_FIELDS),
        current=json.dumps(current, indent=1)[:4000],
        score=last_record.get("_objective"),
        feedback=_feedback(last_record),
        history="\n".join(hist_lines),
    )
    result = await GeminiProvider(max_tokens=8192).complete(
        "You optimize advertising creative direction. Strict JSON only.",
        prompt, PROPOSER_MODEL,
    )
    text = result.text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"proposer returned no JSON object: {text[:200]!r}")
    proposal = json.loads(text[start:end + 1])
    est_cost = (
        result.cost_usd if result.cost_usd is not None
        else result.total_tokens * PROPOSER_COST_PER_TOKEN_USD
    )
    return proposal, est_cost


async def score_config(
    brand: dict, *, live: bool, out: Path, run_index: int, k: int,
    logo: str | None, dark_logo: str | None, max_cost_per_call_usd: float,
) -> tuple[dict, float | None, float | None, float]:
    """Run the banner suite once; return (record, objective, winner, cost)."""
    record = await run_banner_suite(
        brand, live, out / f"iter_{run_index}", logo, True, k,
        max_cost_per_call_usd, publish_dir=None, dark_logo_arg=dark_logo,
    )
    obj, win = objective_from_record(record)
    record["_objective"] = obj
    return record, obj, win, record.get("cost_usd", 0.0)


async def main_async(args: argparse.Namespace) -> None:
    brand = load_brand(args.brand)
    out = Path(args.out)
    journal_path = Path(args.journal)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    spent = 0.0
    live = not args.offline

    print("[baseline] scoring incumbent creative direction...")
    base_record, best_obj, best_win, cost = await score_config(
        brand, live=live, out=out, run_index=0, k=args.search_k,
        logo=args.logo, dark_logo=args.logo_dark,
        max_cost_per_call_usd=args.max_cost_per_call_usd,
    )
    spent += cost
    last_record = base_record
    print(f"[baseline] objective(all-cand)={best_obj} winner-mean={best_win} "
          f"(${spent:.2f} spent)")

    for i in range(1, args.iterations + 1):
        if live and spent + args.iteration_cost_ceiling_usd > args.budget_usd:
            print(f"[stop] budget: ${spent:.2f} spent, next iteration would "
                  f"exceed ${args.budget_usd:.2f}")
            break
        try:
            if live:
                proposal, prop_cost = await propose(brand, last_record, history)
            else:
                proposal, prop_cost = _offline_proposal(brand, i), 0.0
            spent += prop_cost
            candidate = apply_proposal(brand, proposal)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[iter {i}] invalid proposal rejected: {exc}")
            continue
        print(f"[iter {i}] {proposal['field']}: {proposal['rationale']}")

        record, obj, win, cost = await score_config(
            candidate, live=live, out=out, run_index=i, k=args.search_k,
            logo=args.logo, dark_logo=args.logo_dark,
            max_cost_per_call_usd=args.max_cost_per_call_usd,
        )
        spent += cost
        gates_ok = record["placements_pass"] == f"{len(record['placements'])}/" \
            f"{len(record['placements'])}"
        kept = bool(
            gates_ok and obj is not None and best_obj is not None
            and obj > best_obj + KEEP_MARGIN
        )
        if kept:
            brand, best_obj, best_win, last_record = (
                candidate, obj, win, record)
        entry = {
            "iteration": i,
            "proposal": proposal,
            "score": obj,
            "winner_mean": win,
            "best_score": best_obj,
            "format_gates_ok": gates_ok,
            "kept": kept,
            "iteration_cost_usd": round(prop_cost + cost, 4),
            "spent_usd": round(spent, 4),
        }
        history.append(entry)
        with journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[iter {i}] objective={obj} best={best_obj} "
              f"{'KEPT' if kept else 'reverted'} (${spent:.2f} spent)")

    best_config = Path(args.best_config)
    best_config.parent.mkdir(parents=True, exist_ok=True)
    best_config.write_text(json.dumps(brand, indent=2), encoding="utf-8")
    print(f"\nBest objective {best_obj} (winner-mean {best_win}); "
          f"config -> {best_config}; journal -> {journal_path}; "
          f"total ${spent:.2f}")

    # Held-out generality: does the tuned craft_canon transfer to a
    # different brand? Run the held-out brand with its own baseline
    # craft, then with the tuned craft, and report the delta.
    if args.heldout_brand:
        await _heldout_transfer(args, brand, spent)


async def _heldout_transfer(
    args: argparse.Namespace, tuned_brand: dict, spent: float,
) -> None:
    held = load_brand(args.heldout_brand)
    out = Path(args.out) / "heldout"
    live = not args.offline
    print(f"\n[held-out] transfer test on {held['name']} "
          f"(does the tuned craft_canon generalize?)")

    base_held, base_obj, base_win, c1 = await score_config(
        held, live=live, out=out, run_index=0, k=args.search_k,
        logo=args.heldout_logo, dark_logo=args.heldout_logo_dark,
        max_cost_per_call_usd=args.max_cost_per_call_usd,
    )
    tuned_held = copy.deepcopy(held)
    tuned_held["craft_canon"] = tuned_brand.get("craft_canon", "")
    _, tuned_obj, tuned_win, c2 = await score_config(
        tuned_held, live=live, out=out, run_index=1, k=args.search_k,
        logo=args.heldout_logo, dark_logo=args.heldout_logo_dark,
        max_cost_per_call_usd=args.max_cost_per_call_usd,
    )
    delta = (
        round(tuned_obj - base_obj, 3)
        if tuned_obj is not None and base_obj is not None else None
    )
    result = {
        "heldout_brand": held["name"],
        "baseline_craft_objective": base_obj,
        "tuned_craft_objective": tuned_obj,
        "delta": delta,
        "transfers": bool(delta is not None and delta > KEEP_MARGIN),
        "cost_usd": round(c1 + c2, 4),
    }
    Path(args.heldout_out).write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"[held-out] {held['name']}: baseline craft {base_obj} -> "
          f"tuned craft {tuned_obj} (delta {delta}); "
          f"transfers={result['transfers']}; record -> {args.heldout_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand", default="benchmarks/brands/osiris_banners.json")
    parser.add_argument("--offline", action="store_true",
                        help="mechanics only, zero cost (no proposer, uses fallback concepts)")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--search-k", type=int, default=2,
                        help="candidates per placement during the search")
    parser.add_argument("--logo", default=None)
    parser.add_argument("--logo-dark", default=None)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--iteration-cost-ceiling-usd", type=float, default=None)
    parser.add_argument("--max-cost-per-call-usd", type=float, default=0.0)
    parser.add_argument("--out", default="smythe_artifacts/banner_optimizer")
    parser.add_argument("--journal",
                        default="benchmarks/results/banner_optimizer_journal.jsonl")
    parser.add_argument("--best-config",
                        default="benchmarks/results/banner_optimizer_best.json")
    parser.add_argument("--heldout-brand", default=None,
                        help="second banner brand config for the transfer test")
    parser.add_argument("--heldout-logo", default=None)
    parser.add_argument("--heldout-logo-dark", default=None)
    parser.add_argument("--heldout-out",
                        default="benchmarks/results/banner_optimizer_heldout.json")
    args = parser.parse_args()

    if not args.offline:
        if not os.environ.get("GOOGLE_API_KEY"):
            raise SystemExit("GOOGLE_API_KEY required for live optimization "
                             "(use --offline for mechanics)")
        for flag in ("budget_usd", "iteration_cost_ceiling_usd"):
            if getattr(args, flag) is None or getattr(args, flag) <= 0:
                raise SystemExit(f"--{flag.replace('_', '-')} must be positive in live mode")
        if args.max_cost_per_call_usd <= 0:
            raise SystemExit("--max-cost-per-call-usd must be positive in live mode")
        if args.iteration_cost_ceiling_usd > args.budget_usd:
            raise SystemExit(
                "Refusing to start: --budget-usd must cover at least one "
                "--iteration-cost-ceiling-usd.")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
