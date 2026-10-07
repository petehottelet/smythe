"""Run each Autotune trial as one durable text workflow, with its exact cost.

:func:`workflow_trial_evaluator` adapts durable :class:`~smythe.swarm.Swarm`
runs to the :data:`~smythe.optimize.engine.TrialEvaluator` contract. Each
trial runs exactly one workflow, whose budget is the trial's ceiling and
whose run id the trial key names (:func:`trial_run_id`), and reports the
journal's confirmed nanoUSD as micro-USD rounded up, so a charge is never
under-reported.

A trial whose workflow cannot be reported raises :class:`WorkflowTrialError`
with one of a few fixed texts: a configuration refused before the run
exists; a run that already exists for the trial; a workflow that leaves any
reserved or unknown spend, or any unknown call; one that fails; and a failed
measurement. ``OptimizationRunner`` then records the
trial as unknown, at its full ceiling, and stops the campaign. The runner
keeps that error's text in its ledger, so the text is fixed and never a
provider's. The original error stays chained in the process for diagnosis.

Nothing here retries. The runner owns trials and never dispatches one twice;
the workflow journal owns calls. A trial whose run id already exists in the
journal is refused, never resumed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from typing import Any

from smythe._money import CEILING, FLOOR, MICROUSD, NANOUSD, convert, from_units, to_units
from smythe.optimize.engine import (
    OptimizationError,
    TrialContext,
    TrialEvaluator,
    TrialOutcome,
)

_REFUSED = "trial workflow was refused before it started"
_EXISTS = "trial already has a workflow, which is never run again"
_FAILED = "trial workflow failed with all of its spend confirmed"
_UNKNOWN = "trial workflow left reserved or unknown spend"
_MEASURE = "trial measurement failed"


class WorkflowTrialError(OptimizationError):
    """A trial's workflow gave no outcome that can be reported.

    The message is always one of this module's fixed texts.
    """


def trial_run_id(trial_key: str) -> str:
    """The durable run id of the workflow that evaluates trial *trial_key*.

    A run id holds at most 64 letters, digits, ``_`` and ``-``, so it names
    the trial by the first 192 bits of its key's SHA-256.
    """
    if type(trial_key) is not str or not trial_key:
        raise ValueError("trial_key must be a non-empty string")
    return "autotune-" + hashlib.sha256(trial_key.encode("utf-8")).hexdigest()[:48]


def workflow_trial_evaluator(
    runtime_factory: Callable[[TrialContext, float], Any],
    task_for: Callable[[TrialContext], Any],
    measure: Callable[[TrialContext, Any], tuple[Mapping[str, int | float], Mapping[str, bool]]],
) -> TrialEvaluator:
    """Return a trial evaluator that runs one durable workflow per trial.

    ``runtime_factory(context, max_budget_usd)`` returns the
    :class:`~smythe.swarm.Swarm`, with a durable ``run_store``, that runs the
    trial's candidate; its ``max_budget_usd`` must be the one given, which is
    exactly ``context.ceiling_microusd`` x 1,000 nanoUSD.
    ``task_for(context)`` returns the trial's :class:`~smythe.task.Task`.
    The run is created under :func:`trial_run_id` and runs once. Only a run
    that completes with all of its spend confirmed is measured:
    ``measure(context, result)`` returns the trial's metrics and gates as a
    pair of mappings from its :class:`~smythe.swarm.SwarmResult`. The
    outcome's cost is the run's confirmed nanoUSD rounded up to micro-USD.

    Every other case raises :class:`WorkflowTrialError` (see the module).
    Measurement must be local: whatever it spends is not counted.
    """
    for name, part in (("runtime_factory", runtime_factory), ("task_for", task_for),
                       ("measure", measure)):
        if not callable(part):
            raise TypeError(f"{name} must be callable")

    async def evaluate(context: TrialContext) -> TrialOutcome:
        from smythe.swarm import Swarm
        from smythe.task import Task, task_to_dict
        from smythe.workflow_store import SQLiteWorkflowStore, WorkflowNotFoundError

        run_id = trial_run_id(context.trial_key)
        try:
            budget_nanousd = convert(context.ceiling_microusd, MICROUSD, NANOUSD)
            # A Swarm budget is a float, which must hold the ceiling exactly.
            budget_usd = float(from_units(context.ceiling_microusd, MICROUSD))
            if to_units(budget_usd, NANOUSD, FLOOR) != budget_nanousd:
                raise ValueError("The trial ceiling has no exact float USD budget")
            swarm = runtime_factory(context, budget_usd)
            if not isinstance(swarm, Swarm) or type(swarm._run_store) is not SQLiteWorkflowStore:
                raise TypeError("runtime_factory must return a Swarm with a durable run_store")
            if (swarm.max_budget_usd is None
                    or to_units(swarm.max_budget_usd, NANOUSD, FLOOR) != budget_nanousd):
                raise ValueError("The Swarm's max_budget_usd is not the trial's budget")
            task = task_for(context)
            if not isinstance(task, Task):
                raise TypeError("task_for must return a Task")
            store = swarm._run_store
            # Building the recipe checks the configuration before the run exists.
            recipe = swarm._workflow_runtime().recipe
            try:
                store.load_run(run_id)
                exists = True
            except WorkflowNotFoundError:
                exists = False
            if not exists:
                store.create_run(task_to_dict(task), recipe, budget_nanousd, run_id=run_id)
        except Exception as error:
            raise WorkflowTrialError(_REFUSED) from error
        if exists:
            raise WorkflowTrialError(_EXISTS)

        result = failure = None
        try:
            # A run without a checkpoint plans its task, as execute() would.
            result = await swarm.aresume(run_id)
        except Exception as error:
            failure = error
        try:
            state = store.inspect_run(run_id)
        except Exception as error:
            raise WorkflowTrialError(_UNKNOWN) from error
        if state["reserved_nanousd"] or state["unknown_nanousd"] or state["unknown_calls"]:
            raise WorkflowTrialError(_UNKNOWN) from failure
        if failure is not None:
            raise WorkflowTrialError(_FAILED) from failure
        if state["status"] != "completed" or getattr(result, "execution_id", None) != run_id:
            raise WorkflowTrialError(_FAILED)
        cost = convert(state["confirmed_nanousd"], NANOUSD, MICROUSD, CEILING)
        try:
            metrics, gates = measure(context, result)
            return TrialOutcome(metrics=metrics, gates=gates, actual_cost_microusd=cost)
        except Exception as error:
            raise WorkflowTrialError(_MEASURE) from error

    return evaluate


__all__ = ["WorkflowTrialError", "trial_run_id", "workflow_trial_evaluator"]
