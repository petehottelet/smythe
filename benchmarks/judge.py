"""LLM judge — scores a baseline's output against the task rubric.

Only meaningful with a real provider; offline runs skip judging. The
judge sees the task, the rubric, and the output — never which baseline
produced it, so scores can't favor a system by name.
"""

from __future__ import annotations

import asyncio
import json
import re

from smythe.provider import Provider

JUDGE_SYSTEM = """\
You are a strict evaluator of written deliverables. You will receive a
task, a rubric, and one candidate output. Score the output against each
rubric criterion from 1 (fails) to 10 (excellent), then give an overall
score. Be harsh: 7 means genuinely good. Respond with JSON only:
{"criteria": [{"criterion": "...", "score": N}, ...], "overall": N,
"weaknesses": "one sentence"}"""


def build_judge_prompt(goal: str, rubric: list[str], output: str) -> str:
    criteria = "\n".join(f"- {c}" for c in rubric)
    return (
        f"TASK:\n{goal}\n\n"
        f"RUBRIC:\n{criteria}\n\n"
        f"CANDIDATE OUTPUT:\n{output}\n\n"
        "Score it. JSON only."
    )


def parse_judge_response(text: str) -> dict:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    data = json.loads(stripped)
    if "overall" not in data:
        raise ValueError("Judge response missing 'overall'")
    return data


def score_output(
    provider: Provider,
    model: str,
    goal: str,
    rubric: list[str],
    output: str,
    *,
    max_retries: int = 2,
) -> dict:
    """Return the judge's verdict for one output. Raises after retries."""

    async def _score() -> dict:
        last_error: Exception | None = None
        for _ in range(1 + max_retries):
            result = await provider.complete(
                JUDGE_SYSTEM, build_judge_prompt(goal, rubric, output), model,
            )
            try:
                return parse_judge_response(result.text)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"Judge produced unparseable output: {last_error}")

    return asyncio.run(_score())
