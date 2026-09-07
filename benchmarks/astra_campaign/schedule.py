"""Seeded four-arm Williams schedules; no execution or provider construction."""

from __future__ import annotations

from ._json import CampaignPlanError, canonical, digest
from .tasks import TaskPack, load_task_pack

ARMS = (
    ("astra-fixed", "gpt-6-astra", "fixed_pipeline"),
    ("astra-dynamic", "gpt-6-astra", "smythe_dynamic"),
    ("sol-fixed", "gpt-5.6-sol", "fixed_pipeline"),
    ("sol-dynamic", "gpt-5.6-sol", "smythe_dynamic"),
)
# Every arm occurs once in each position and follows each other arm once
# across four rows. Fifty main blocks use each row 12 or 13 times.
WILLIAMS_ROWS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))


def validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise CampaignPlanError("Seed must be an integer between 0 and 2**63 - 1")


def build_schedule(stage: str, seed: int, pack: TaskPack | None = None) -> list[dict]:
    """Order task/repetition blocks and counterbalance arms inside each block."""
    validate_seed(seed)
    pack = load_task_pack() if pack is None else pack
    tasks = pack.for_stage(stage)
    repetitions = 1 if stage == "pilot" else 5

    def key(value):
        return digest(canonical(["astra-order-v1", seed, stage, value]).encode())

    arms = sorted(ARMS, key=lambda arm: key(arm[0]))
    blocks = sorted(((task, rep) for task in tasks for rep in range(1, repetitions + 1)),
                    key=lambda pair: key([pair[0].task_id, pair[1]]))
    shift = int(key("row-offset")[:8], 16) % 4
    schedule = []
    for block_index, (task, rep) in enumerate(blocks):
        row = WILLIAMS_ROWS[(block_index + shift) % 4]
        for position, arm_index in enumerate(row):
            arm_id, model, strategy = arms[arm_index]
            schedule.append({
                "ordinal": len(schedule), "block": block_index, "position": position,
                "trial_id": f"{stage}--{task.task_id}--r{rep}--{arm_id}",
                "stage": stage, "task_id": task.task_id, "repetition": rep,
                "arm_id": arm_id, "model": model, "strategy": strategy,
                "task_sha256": task.task_sha256, "source_sha256": task.source_sha256,
            })
    return schedule
