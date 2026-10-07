"""End-to-end offline execution tests for Jobs v1."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import threading
from contextlib import ExitStack

import pytest
import smythe.jobs.runner as runner_module

from fakes import Crash, crash_when
from smythe.jobs import ApprovalError, JobManifestV1, PreflightError, make_approval, preflight_job
from smythe.jobs.models import (
    DEFAULT_CALL_TIMEOUT_S,
    DEFAULT_MAX_WALL_SECONDS,
    MAX_CALL_TIMEOUT_S,
)
from smythe.jobs.providers import ProviderPool
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import (
    InvalidTransitionError, JobNotFoundError, OperationStatus, RunLeaseError, RunStatus,
    SQLiteRunStore,
)
from smythe.provider import Artifact, CompletionResult, Provider

Image = pytest.importorskip("PIL.Image")


_png_buffer = io.BytesIO()
Image.new("RGBA", (1, 1), (255, 0, 0, 255)).save(_png_buffer, format="PNG")
PNG_1X1 = _png_buffer.getvalue()


@pytest.fixture
def store_factory():
    """Close test-owned stores after all assertions and workers finish."""
    with ExitStack() as stack:
        def create(path):
            return stack.enter_context(SQLiteRunStore(path))

        yield create


def _manifest(*, count=4, attempts=1):
    return JobManifestV1.from_dict(
        {
            "version": 1,
            "name": "runner-test",
            "profiles": [
                {
                    "name": "default",
                    "provider": "offline",
                    "model": "offline-image",
                    "max_cost_per_call_usd": "0",
                    "options": {"artifacts_per_call": 1},
                }
            ],
            "operations": [
                {
                    "key": "glyph",
                    "count": count,
                    "prompt": "Generate one glyph",
                    "profile": "default",
                    "artifact": {
                        "mime_type": "image/png",
                        "width": 1,
                        "height": 1,
                    },
                }
            ],
            "execution": {
                "max_concurrency": 2,
                "max_attempts": attempts,
                "max_budget_usd": "0",
                "output_directory": "outputs",
            },
        }
    )


def test_runner_executes_bounded_offline_job_and_persists_artifacts(tmp_path, store_factory):
    plan = preflight_job(_manifest(), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store)

    result = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))

    assert result["status"] == RunStatus.COMPLETED.value
    assert result["counts"] == {OperationStatus.SUCCEEDED.value: 4}
    assert result["execution_metrics"]["peak_active_calls"] <= 2
    assert result["cost"]["confirmed_microusd"] == 0
    assert len(result["artifacts"]) == 4
    for artifact in result["artifacts"]:
        assert (tmp_path / "outputs" / result["artifact_directory"] / artifact["relative_path"]).is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace semantics")
def test_runner_accepts_equivalent_windows_extended_path_namespace(tmp_path, store_factory, monkeypatch):
    """A concurrent mkdir may make resolve() add the extended path prefix."""

    original_resolve = runner_module.Path.resolve

    def resolve_with_namespace(path, *args, **kwargs):
        resolved = original_resolve(path, *args, **kwargs)
        value = str(resolved)
        if path.name.startswith("run-") and not value.startswith("\\\\?\\"):
            return runner_module.Path("\\\\?\\" + value)
        return resolved

    monkeypatch.setattr(runner_module.Path, "resolve", resolve_with_namespace)
    plan = preflight_job(_manifest(count=2), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")

    result = asyncio.run(
        JobRunner(store).start(
            plan,
            make_approval(plan),
            manifest_root=tmp_path,
            run_id="namespace-run",
        )
    )

    assert result["status"] == RunStatus.COMPLETED.value
    assert result["counts"] == {OperationStatus.SUCCEEDED.value: 2}


def test_internal_provider_transport_timeout_covers_approved_contract_max(tmp_path, store_factory):
    runner = JobRunner(store_factory(tmp_path / "jobs.db"))

    assert runner.providers._request_timeout_s == MAX_CALL_TIMEOUT_S
    assert runner.call_timeout_s is None
    assert runner.max_wall_seconds is None


def test_operator_deadlines_can_only_tighten_the_approved_plan(tmp_path, store_factory):
    data = _manifest(count=1).to_dict()
    data["execution"]["call_timeout_s"] = 900
    data["execution"]["max_wall_seconds"] = 7200
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)

    default_runner = JobRunner(store_factory(tmp_path / "default.db"))
    strict_runner = JobRunner(
        store_factory(tmp_path / "strict.db"),
        call_timeout_s=120,
        max_wall_seconds=600,
    )
    loose_runner = JobRunner(
        store_factory(tmp_path / "loose.db"),
        call_timeout_s=1200,
        max_wall_seconds=9000,
    )

    assert runner_module._plan_timeout(
        plan,
        ("call_timeout_s",),
        operator_ceiling=default_runner.call_timeout_s,
        fallback=DEFAULT_CALL_TIMEOUT_S,
    ) == 900
    assert runner_module._plan_timeout(
        plan,
        ("call_timeout_s",),
        operator_ceiling=strict_runner.call_timeout_s,
        fallback=DEFAULT_CALL_TIMEOUT_S,
    ) == 120
    assert runner_module._plan_timeout(
        plan,
        ("max_wall_seconds",),
        operator_ceiling=loose_runner.max_wall_seconds,
        fallback=DEFAULT_MAX_WALL_SECONDS,
    ) == 7200


class _SequenceProvider(Provider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.calls += 1
        artifacts = [] if self.calls == 1 else [Artifact(PNG_1X1, "image/png")]
        return CompletionResult(text=f"attempt {self.calls}", artifacts=artifacts)


class _FixedPool(ProviderPool):
    def __init__(self, provider: Provider) -> None:
        super().__init__()
        self.provider = provider

    def get(self, operation):
        return self.provider

    @staticmethod
    def validate_operation(operation):
        return None

    @staticmethod
    def preflight(operation, **_kwargs):
        return None


def test_selective_reroll_replaces_only_rejected_operation(tmp_path, store_factory):
    manifest = _manifest(count=1, attempts=2)
    plan = preflight_job(manifest, manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store, provider_pool=_FixedPool(_SequenceProvider()))

    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))
    operation_key = first["operations"][0]["operation_key"]
    assert first["status"] == RunStatus.FAILED.value
    assert first["counts"] == {OperationStatus.REJECTED.value: 1}

    second = asyncio.run(runner.reroll(first["run_id"], [operation_key], reason="missing glyph"))

    assert second["status"] == RunStatus.COMPLETED.value
    assert second["counts"] == {OperationStatus.SUCCEEDED.value: 1}
    assert second["operations"][0]["attempt_count"] == 2
    assert len(second["attempts"]) == 2


def test_start_refuses_a_used_approval_before_any_provider_call(tmp_path, store_factory):
    plan = preflight_job(_manifest(count=1, attempts=2), manifest_root=tmp_path)
    approval = make_approval(plan)
    store = store_factory(tmp_path / "jobs.db")
    provider = _SequenceProvider()
    runner = JobRunner(store, provider_pool=_FixedPool(provider))
    first = asyncio.run(runner.start(plan, approval, manifest_root=tmp_path, run_id="first"))
    calls, before = provider.calls, list(store._connection.iterdump())

    with pytest.raises(ApprovalError, match="run 'first'.*smythe jobs resume first"):
        asyncio.run(runner.start(plan, approval, manifest_root=tmp_path))
    with pytest.raises(TypeError, match="allow_approval_reuse must be a boolean"):
        asyncio.run(runner.start(plan, approval, manifest_root=tmp_path, allow_approval_reuse=1))
    assert provider.calls == calls
    assert list(store._connection.iterdump()) == before

    second = asyncio.run(runner.start(
        plan, approval, manifest_root=tmp_path, run_id="second", allow_approval_reuse=True,
    ))
    assert first["run_id"] == "first" and second["run_id"] == "second"
    assert provider.calls > calls
    created = store.snapshot("second", include_events=True)["events"][0]
    assert created["event_type"] == "run_created"
    assert created["payload"]["approval_reuse"] == {"prior_run_count": 1, "prior_run_ids": ["first"]}


def test_reroll_records_its_reason_on_the_new_attempt(tmp_path, store_factory):
    plan = preflight_job(_manifest(count=1, attempts=2), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store, provider_pool=_FixedPool(_SequenceProvider()))
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))
    operation_key = first["operations"][0]["operation_key"]

    second = asyncio.run(runner.reroll(first["run_id"], [operation_key], reason="missing glyph"))

    assert [attempt["reason"] for attempt in second["attempts"]] == [None, "missing glyph"]
    assert second["attempts"][1]["parent_attempt_id"] == second["attempts"][0]["attempt_id"]


class _AmbiguousProvider(Provider):
    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        raise ConnectionError("response lost")


def test_provider_exception_after_dispatch_becomes_unknown_outcome(tmp_path, store_factory):
    plan = preflight_job(_manifest(count=1, attempts=2), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store, provider_pool=_FixedPool(_AmbiguousProvider()))

    result = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))

    assert result["status"] == RunStatus.NEEDS_ATTENTION.value
    assert result["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}
    assert store.pending_operations(result["run_id"]) == []


def test_runner_rejects_attachment_changed_after_approval(tmp_path, store_factory):
    attachment = tmp_path / "input.png"
    attachment.write_bytes(b"approved-input")
    data = _manifest(count=1).to_dict()
    data["operations"][0]["attachments"] = ["input.png"]
    manifest = JobManifestV1.from_dict(data)
    plan = preflight_job(manifest, manifest_root=tmp_path)
    approval = make_approval(plan)
    attachment.write_bytes(b"different-input")
    store = store_factory(tmp_path / "jobs.db")

    with pytest.raises(ValueError, match="changed after preflight"):
        asyncio.run(JobRunner(store).start(plan, approval, manifest_root=tmp_path))


@pytest.mark.parametrize(
    "run_id",
    ["../outside", "..\\outside", ".", "", "name/child", "name\\child"],
)
def test_runner_rejects_unsafe_custom_run_id(tmp_path, store_factory, run_id):
    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")

    with pytest.raises(ValueError, match="run_id"):
        asyncio.run(
            JobRunner(store).start(
                plan,
                make_approval(plan),
                manifest_root=tmp_path,
                run_id=run_id,
            )
        )


class _BlockingProvider(Provider):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.entered.set()
        await self.release.wait()
        return CompletionResult(
            text="completed after release",
            artifacts=[Artifact(PNG_1X1, "image/png")],
        )


def test_runner_lease_heartbeat_blocks_concurrent_resume(tmp_path):
    async def scenario() -> None:
        plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
        first_store = SQLiteRunStore(tmp_path / "jobs.db")
        second_store = SQLiteRunStore(tmp_path / "jobs.db")
        provider = _BlockingProvider()
        first_runner = JobRunner(
            first_store,
            provider_pool=_FixedPool(provider),
            lease_ttl_s=30,
            lease_heartbeat_s=0.05,
        )
        task = asyncio.create_task(
            first_runner.start(
                plan,
                make_approval(plan),
                manifest_root=tmp_path,
                run_id="leased-run",
            )
        )
        try:
            await asyncio.wait_for(provider.entered.wait(), timeout=30)
            acquired = first_store.get_run_lease("leased-run")
            assert acquired is not None
            await asyncio.sleep(0.12)
            renewed = first_store.get_run_lease("leased-run")
            assert renewed is not None
            assert renewed.heartbeat_at_ns > acquired.heartbeat_at_ns
            with pytest.raises(RunLeaseError, match="leased"):
                await JobRunner(second_store).resume("leased-run")
            provider.release.set()
            result = await asyncio.wait_for(task, timeout=30)
            assert result["status"] == RunStatus.COMPLETED.value
            assert first_store.get_run_lease("leased-run") is None
        finally:
            provider.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            first_store.close()
            second_store.close()

    asyncio.run(scenario())


def _priced_manifest(*, count=1):
    data = _manifest(count=count).to_dict()
    data["profiles"][0].update(
        {
            "provider": "openai_image",
            "model": "gpt-image-2",
            "max_cost_per_call_usd": "0.10",
            "options": {},
        }
    )
    data["execution"]["max_budget_usd"] = f"{count * 0.10:.2f}"
    return JobManifestV1.from_dict(data)


class _NeverReturningProvider(Provider):
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _InvalidCostProvider(Provider):
    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return CompletionResult(
            text="invalid billing metadata",
            artifacts=[Artifact(PNG_1X1, "image/png")],
            cost_usd=float("nan"),
        )


class _EstimatedCostProvider(Provider):
    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return CompletionResult(
            text="estimated billing metadata",
            artifacts=[Artifact(PNG_1X1, "image/png")],
            cost_usd=0.08,
            cost_usd_is_estimate=True,
        )


class _ImmediateArtifactProvider(Provider):
    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return CompletionResult(
            text="ready",
            artifacts=[Artifact(PNG_1X1, "image/png")],
        )


def test_never_returning_provider_hits_call_deadline_and_becomes_unknown(tmp_path, store_factory):
    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(
        store,
        provider_pool=_FixedPool(_NeverReturningProvider()),
        call_timeout_s=0.05,
    )

    result = asyncio.run(
        runner.start(plan, make_approval(plan), manifest_root=tmp_path)
    )

    assert result["status"] == RunStatus.NEEDS_ATTENTION.value
    assert result["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}
    assert "deadline after dispatch" in result["operations"][0]["error"]


def test_whole_run_deadline_cancels_dispatched_call_conservatively(tmp_path, store_factory, monkeypatch):
    class EnteredProvider(_NeverReturningProvider):
        def __init__(self):
            self.entered = asyncio.Event()

        async def complete(self, system, prompt, model):
            self.entered.set()
            return await super().complete(system, prompt, model)

    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = EnteredProvider()
    runner = JobRunner(
        store,
        provider_pool=_FixedPool(provider),
        call_timeout_s=10,
        max_wall_seconds=0.05,
    )
    original_wait_for = asyncio.wait_for

    async def deadline_after_dispatch(awaitable, timeout):
        if timeout != 0.05:
            return await original_wait_for(awaitable, timeout)
        # This regression targets expiry after durable dispatch. Arm its short
        # timer at provider entry so filesystem preparation cannot win the race.
        execution = asyncio.ensure_future(awaitable)
        try:
            await original_wait_for(provider.entered.wait(), 30)
            return await original_wait_for(execution, timeout)
        finally:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)

    monkeypatch.setattr(asyncio, "wait_for", deadline_after_dispatch)

    with pytest.raises(TimeoutError, match="wall deadline"):
        asyncio.run(
            runner.start(
                plan,
                make_approval(plan),
                manifest_root=tmp_path,
                run_id="wall-deadline",
            )
        )

    assert provider.entered.is_set()
    snapshot = store.snapshot("wall-deadline")
    assert snapshot["status"] == RunStatus.NEEDS_ATTENTION.value
    assert snapshot["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}


def test_whole_run_deadline_before_dispatch_leaves_work_pending(tmp_path, store_factory, monkeypatch):
    class CountingProvider(_ImmediateArtifactProvider):
        calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            return await super().complete(system, prompt, model)

    release = threading.Event()
    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = CountingProvider()
    runner = JobRunner(store, provider_pool=_FixedPool(provider), max_wall_seconds=0.05)

    def delayed_preparation(_snapshot, _root):
        assert release.wait(30), "preparation barrier was never released"

    monkeypatch.setattr(runner, "_claim_artifact_directory", delayed_preparation)

    async def scenario():
        try:
            return await runner.start(plan, make_approval(plan), manifest_root=tmp_path,
                                      run_id="pre-dispatch-deadline")
        finally:
            release.set()

    with pytest.raises(TimeoutError, match="wall deadline"):
        asyncio.run(scenario())
    snapshot = store.snapshot("pre-dispatch-deadline")
    assert provider.calls == 0
    assert snapshot["counts"] == {OperationStatus.PENDING.value: 1}
    assert store._call_rows("pre-dispatch-deadline") == []
    assert snapshot["cost"]["reserved_microusd"] == 0
    assert store.get_run_lease("pre-dispatch-deadline") is None


def test_invalid_provider_cost_is_immediately_journaled_unknown(tmp_path, store_factory):
    plan = preflight_job(_priced_manifest(), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store, provider_pool=_FixedPool(_InvalidCostProvider()))

    result = asyncio.run(
        runner.start(plan, make_approval(plan), manifest_root=tmp_path)
    )

    assert result["status"] == RunStatus.NEEDS_ATTENTION.value
    assert result["cost"]["confirmed_microusd"] == 0
    assert result["cost"]["exposure_microusd"] == 100_000
    assert "invalid provider accounting: cost_usd" in result["operations"][0]["error"]
    assert "finite non-negative" in result["operations"][0]["error"]


@pytest.mark.parametrize("mutated", [False, True], ids=["constructor", "mutated"])
@pytest.mark.parametrize(
    "field,value", [("cost_usd", float("nan")), ("prompt_tokens", True), ("cost_usd", 1e308)],
    ids=["nan-cost", "boolean-usage", "durable-cost-overflow"],
)
def test_invalid_accounting_stops_queued_calls_and_fresh_runner_resume(
    tmp_path, store_factory, mutated, field, value,
):
    class InvalidAccountingProvider(Provider):
        def __init__(self):
            self.calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            if mutated:
                result = CompletionResult(text="invalid", cost_usd=0.08)
                setattr(result, field, value)
                return result
            return CompletionResult(text="invalid", **{field: value})

    data = _priced_manifest(count=3).to_dict()
    data["execution"].update(max_concurrency=1, max_attempts=2, max_budget_usd="0.60")
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = InvalidAccountingProvider()
    pool = _FixedPool(provider)
    result = asyncio.run(
        JobRunner(store, provider_pool=pool).start(
            plan, make_approval(plan), manifest_root=tmp_path,
        )
    )
    assert provider.calls == 1
    assert result["status"] == RunStatus.NEEDS_ATTENTION.value
    assert result["counts"] == {"unknown_outcome": 1, "pending": 2}
    assert len(result["attempts"]) == 1
    assert result["cost"]["confirmed_microusd"] == 0
    assert result["cost"]["exposure_microusd"] == 100_000
    assert result["cost"]["reserved_microusd"] == 0
    assert not result["artifacts"]
    events = store.snapshot(result["run_id"], include_events=True)["events"]
    assert sum(event["event_type"] == "call_dispatched" for event in events) == 1
    assert sum(event["event_type"] == "unknown_outcome" for event in events) == 1

    fresh_runner = JobRunner(store, provider_pool=pool)
    resumed = asyncio.run(fresh_runner.resume(result["run_id"]))
    assert provider.calls == 1
    assert resumed["counts"] == result["counts"]
    assert resumed["cost"] == result["cost"]
    assert resumed["execution_metrics"]["operations_started"] == 0


def test_invalid_accounting_requires_explicit_reroll_acknowledgement(tmp_path, store_factory):
    class CorrectedProvider(Provider):
        def __init__(self):
            self.calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            return CompletionResult(
                text="response", cost_usd=float("nan") if self.calls == 1 else 0.08,
                artifacts=[Artifact(PNG_1X1, "image/png")],
            )

    data = _priced_manifest().to_dict()
    data["execution"].update(max_attempts=2, max_budget_usd="0.20")
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = CorrectedProvider()
    runner = JobRunner(store, provider_pool=_FixedPool(provider))
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))
    key = first["operations"][0]["operation_key"]
    with pytest.raises(InvalidTransitionError):
        asyncio.run(runner.reroll(first["run_id"], [key], reason="adapter corrected"))
    assert provider.calls == 1
    result = asyncio.run(
        runner.reroll(
            first["run_id"], [key], reason="adapter corrected; prior charge remains unknown",
            acknowledge_unknown=True,
        )
    )
    assert provider.calls == 2
    assert result["counts"] == {"succeeded": 1}
    assert result["cost"]["confirmed_microusd"] == 80_000
    assert result["cost"]["exposure_microusd"] == 100_000
    assert not result["cost"]["cost_is_complete"]


def test_invalid_accounting_preserves_already_dispatched_sibling_cost(tmp_path, store_factory, monkeypatch):
    classified = asyncio.Event()

    class ConcurrentProvider(Provider):
        def __init__(self):
            self.calls = 0
            self.both_dispatched = asyncio.Event()

        async def complete(self, system, prompt, model):
            self.calls += 1
            index = self.calls
            if index == 1:
                await self.both_dispatched.wait()
                return CompletionResult(text="invalid", cost_usd=float("inf"))
            self.both_dispatched.set()
            # Wait until the first call has been durably classified.
            await asyncio.wait_for(classified.wait(), timeout=10)
            return CompletionResult(
                text="already dispatched", cost_usd=0.08,
                artifacts=[Artifact(PNG_1X1, "image/png")],
            )

    plan = preflight_job(_priced_manifest(count=4), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    original_mark_unknown = store.mark_unknown_outcome

    def mark_unknown(call_id, error, **kwargs):
        original_mark_unknown(call_id, error, **kwargs)
        classified.set()

    monkeypatch.setattr(store, "mark_unknown_outcome", mark_unknown)
    provider = ConcurrentProvider()
    result = asyncio.run(
        JobRunner(store, provider_pool=_FixedPool(provider)).start(
            plan, make_approval(plan), manifest_root=tmp_path, run_id="concurrent-invalid",
        )
    )
    assert provider.calls == 2
    assert result["counts"] == {"unknown_outcome": 1, "succeeded": 1, "pending": 2}
    assert result["cost"]["confirmed_microusd"] == 80_000
    assert result["cost"]["exposure_microusd"] == 100_000
    assert result["cost"]["reserved_microusd"] == 0
    assert len(result["artifacts"]) == 1


def test_stored_call_errors_keep_no_account_identifier(tmp_path, store_factory):
    organization = "org-" + "A1b2" * 6  # built here, so no tracked file holds an identifier

    class AccountRefusal(Provider):
        async def complete(self, system, prompt, model):
            raise ConnectionError(f"429 in organization {organization} on images per min")

    class UnconfiguredPool(_FixedPool):
        """Passes the run's start checks, then refuses each dispatch's own lookup."""

        def __init__(self, provider):
            super().__init__(provider)
            self.lookups = 0

        def get(self, operation):
            self.lookups += 1
            if self.lookups > 2:
                raise RuntimeError(f"project for organization {organization} is not enabled")
            return self.provider

    data = _priced_manifest(count=2).to_dict()
    data["execution"]["max_concurrency"] = 1
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    for name, pool, status, error in (
        ("after", _FixedPool(AccountRefusal()), "unknown_outcome",
         "provider failure after dispatch: 429 in organization org-[redacted] on images per min"),
        ("before", UnconfiguredPool(AccountRefusal()), "failed",
         "project for organization org-[redacted] is not enabled"),
    ):
        store = store_factory(tmp_path / f"{name}.db")
        result = asyncio.run(JobRunner(store, provider_pool=pool).start(
            plan, make_approval(plan), manifest_root=tmp_path))
        assert result["counts"] == {status: 2}
        assert [item["error"] for item in result["operations"]] == [error, error]
        assert organization not in repr(store.snapshot(result["run_id"], include_events=True))


def test_arbitrary_provider_message_cannot_impersonate_accounting_classification(tmp_path, store_factory):
    class MisleadingFailure(Provider):
        def __init__(self):
            self.calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            raise ConnectionError("invalid provider accounting: arbitrary remote message")

    data = _priced_manifest(count=2).to_dict()
    data["execution"]["max_concurrency"] = 1
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = MisleadingFailure()
    result = asyncio.run(
        JobRunner(store, provider_pool=_FixedPool(provider)).start(
            plan, make_approval(plan), manifest_root=tmp_path,
        )
    )
    assert provider.calls == 2
    assert result["counts"] == {"unknown_outcome": 2}
    assert all(
        item["error"].startswith("provider failure after dispatch: ")
        for item in result["operations"]
    )


def test_estimated_provider_cost_remains_exposure_not_confirmed(tmp_path, store_factory):
    plan = preflight_job(_priced_manifest(), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store, provider_pool=_FixedPool(_EstimatedCostProvider()))

    result = asyncio.run(
        runner.start(plan, make_approval(plan), manifest_root=tmp_path)
    )

    assert result["status"] == RunStatus.COMPLETED.value
    assert result["cost"]["confirmed_microusd"] == 0
    assert result["cost"]["exposure_microusd"] == 100_000
    assert not result["cost"]["cost_is_complete"]
    assert result["cost"]["cost_contains_estimates"]


_UNDECODABLE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16  # a PNG header with no image data


def _png(size):
    buffer = io.BytesIO()
    Image.new("RGB", size, (0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class _PricedArtifactProvider(Provider):
    """Return one queued artifact list per call, each with a known cost."""

    def __init__(self, *responses, cost_usd=0.04):
        self.responses = list(responses)
        self.cost_usd = cost_usd
        self.calls = 0

    async def complete(self, system, prompt, model):
        self.calls += 1
        artifacts = self.responses[min(self.calls, len(self.responses)) - 1]
        return CompletionResult(text=f"response {self.calls}", artifacts=list(artifacts),
                                cost_usd=self.cost_usd)


def _events(store, run_id, event_type):
    events = store.snapshot(run_id, include_events=True)["events"]
    return [event for event in events if event["event_type"] == event_type]


def _findings(record):
    return [finding["code"] for result in record["validators"] for report in result["reports"]
            for finding in report["findings"]]


def test_accepted_call_records_the_builtin_artifact_check(tmp_path, store_factory):
    from smythe.validation import ValidatorIdentity

    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    result = asyncio.run(JobRunner(store).start(plan, make_approval(plan), manifest_root=tmp_path))

    [artifact] = result["artifacts"]
    [completed] = _events(store, result["run_id"], "call_completed")
    record = completed["payload"]["delivery_acceptance"]
    assert (record["version"], record["scope"], record["decision"]) == (1, "artifacts", "accepted")
    assert record["reasons"] == []
    assert record["attempt_id"] == result["attempts"][0]["attempt_id"]
    assert record["subjects"] == [{"path": artifact["relative_path"], "sha256": artifact["sha256"],
                                   "size_bytes": artifact["size_bytes"], "published": True}]
    identity = ValidatorIdentity.for_config(
        "smythe.jobs.artifact-spec", "1", {"mime_type": "image/png", "width": 1, "height": 1},
    ).to_dict()
    assert record["validators"] == [{
        "validator": identity, "outcome": "passed",
        "reports": [{"path": artifact["relative_path"], "outcome": "passed", "findings": [],
                     "validator": identity}],
    }]
    assert store.schema_version == 4


def test_rejected_call_records_findings_with_expected_and_observed_values(tmp_path, store_factory):
    plan = preflight_job(_manifest(count=1, attempts=2), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = _PricedArtifactProvider([], [Artifact(_png((2, 3)), "image/png")], cost_usd=None)
    runner = JobRunner(store, provider_pool=_FixedPool(provider))
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))
    key = first["operations"][0]["operation_key"]
    second = asyncio.run(runner.reroll(first["run_id"], [key], reason="wrong size"))

    empty, wrong_size = (event["payload"]["delivery_acceptance"]
                         for event in _events(store, first["run_id"], "artifact_rejected"))
    assert empty["decision"] == "rejected" and empty["subjects"] == []
    assert empty["reasons"] == ["provider returned no artifacts"]
    assert empty["validators"][0]["reports"][0]["path"] == "response"
    assert _findings(empty) == ["no_artifacts"]
    [artifact] = second["artifacts"]
    assert artifact["accepted"] == 0  # Rejected bytes stay published for review.
    [report] = wrong_size["validators"][0]["reports"]
    assert report["path"] == artifact["relative_path"] and report["outcome"] == "rejected"
    [finding] = report["findings"]
    assert (finding["code"], finding["expected"], finding["observed"]) == (
        "dimensions_mismatch", [1, 1], [2, 3])
    assert wrong_size["reasons"] == [second["attempts"][1]["error"]]
    assert "expected 1x1, observed 2x3" in second["attempts"][1]["error"]


def test_undecodable_bytes_with_a_known_cost_are_a_settled_rejection(tmp_path, store_factory):
    data = _priced_manifest().to_dict()
    data["execution"].update(max_attempts=2, max_budget_usd="0.20")
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = _PricedArtifactProvider([Artifact(_UNDECODABLE_PNG, "image/png")],
                                       [Artifact(PNG_1X1, "image/png")])
    runner = JobRunner(store, provider_pool=_FixedPool(provider))

    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))

    assert first["status"] == RunStatus.FAILED.value
    assert first["counts"] == {OperationStatus.REJECTED.value: 1}
    assert first["cost"]["confirmed_microusd"] == 40_000
    assert first["cost"]["exposure_microusd"] == 0 and first["cost"]["cost_is_complete"]
    assert first["artifacts"] == []
    assert "could not be fully decoded" in first["operations"][0]["error"]
    [call] = store._call_rows(first["run_id"])
    assert (call["status"], call["confirmed_microusd"], call["exposure_microusd"]) == (
        "rejected", 40_000, 0)
    [rejected] = _events(store, first["run_id"], "artifact_rejected")
    assert _findings(rejected["payload"]["delivery_acceptance"]) == ["artifact_undecodable"]
    assert not list((tmp_path / "outputs").rglob("*.png"))

    # A settled rejection is rerolled without a duplicate-spend acknowledgement.
    key = first["operations"][0]["operation_key"]
    second = asyncio.run(runner.reroll(first["run_id"], [key], reason="undecodable image"))
    assert provider.calls == 2
    assert second["counts"] == {OperationStatus.SUCCEEDED.value: 1}
    assert second["cost"]["confirmed_microusd"] == 80_000


@pytest.mark.parametrize("limit,code", [
    ("MAX_ARTIFACT_BYTES", "artifact_too_large"),
    ("MAX_TOTAL_ARTIFACT_BYTES_PER_CALL", "artifact_bytes_limit"),
    ("MAX_ARTIFACTS_PER_CALL", "artifact_count_limit"),
])
def test_over_limit_artifacts_are_a_settled_rejection(tmp_path, store_factory, monkeypatch, limit, code):
    monkeypatch.setattr(runner_module, limit, 1)
    plan = preflight_job(_priced_manifest(), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")] * 2)

    result = asyncio.run(JobRunner(store, provider_pool=_FixedPool(provider)).start(
        plan, make_approval(plan), manifest_root=tmp_path,
    ))

    assert result["counts"] == {OperationStatus.REJECTED.value: 1}
    assert result["cost"]["confirmed_microusd"] == 40_000 and result["cost"]["exposure_microusd"] == 0
    assert result["artifacts"] == []
    [rejected] = _events(store, result["run_id"], "artifact_rejected")
    assert set(_findings(rejected["payload"]["delivery_acceptance"])) == {code}
    assert not list((tmp_path / "outputs").rglob("*.png"))


def test_invalid_accounting_keeps_undecodable_bytes_an_unknown_outcome(tmp_path, store_factory):
    class MutatedCostProvider(Provider):
        async def complete(self, system, prompt, model):
            result = CompletionResult(text="invalid", cost_usd=0.04,
                                      artifacts=[Artifact(_UNDECODABLE_PNG, "image/png")])
            result.cost_usd = float("nan")
            return result

    plan = preflight_job(_priced_manifest(), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    result = asyncio.run(JobRunner(store, provider_pool=_FixedPool(MutatedCostProvider())).start(
        plan, make_approval(plan), manifest_root=tmp_path,
    ))

    assert result["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}
    assert result["operations"][0]["error"].startswith("invalid provider accounting: ")
    assert result["cost"]["exposure_microusd"] == 100_000
    assert _events(store, result["run_id"], "artifact_rejected") == []


class _PaletteValidator:
    """A registered validator whose verdict a test supplies."""

    def __init__(self, verdict=None, *, version="1", colors=("#ff0000",)):
        from smythe.validation import ValidatorIdentity

        self.identity = ValidatorIdentity.for_config("brand-palette", version,
                                                     {"colors": list(colors)})
        self.verdict = verdict
        self.subjects = []

    def validate(self, subject):
        from smythe.validation import ValidationReport

        self.subjects.append(subject)
        if self.verdict is not None:
            return self.verdict(self, subject)
        return ValidationReport("any path", validator=self.identity)


def _validated_manifest(*, count=1, attempts=2):
    data = _manifest(count=count, attempts=attempts).to_dict()
    data["operations"][0]["artifact"]["validators"] = [{"name": "brand-palette", "version": "1"}]
    return JobManifestV1.from_dict(data)


def _validated_run(tmp_path, store, validator, provider=None):
    plan = preflight_job(_validated_manifest(), manifest_root=tmp_path, validators=[validator])
    pool = _FixedPool(provider or _ImmediateArtifactProvider())
    runner = JobRunner(store, provider_pool=pool, validators=[validator])
    result = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))
    [event] = (_events(store, result["run_id"], "call_completed")
               + _events(store, result["run_id"], "artifact_rejected"))
    return runner, result, event["payload"]["delivery_acceptance"]


def test_registered_validator_checks_the_published_bytes_and_is_recorded(tmp_path, store_factory):
    validator = _PaletteValidator()
    _, result, record = _validated_run(tmp_path, store_factory(tmp_path / "jobs.db"), validator)

    assert result["counts"] == {OperationStatus.SUCCEEDED.value: 1}
    [artifact] = result["artifacts"]
    [subject] = validator.subjects
    assert subject.data == PNG_1X1 and subject.sha256 == artifact["sha256"]
    assert record["decision"] == "accepted"
    builtin, palette = record["validators"]
    assert builtin["validator"]["name"] == "smythe.jobs.artifact-spec"
    assert palette == {"validator": validator.identity.to_dict(), "outcome": "passed", "reports": [
        {"path": artifact["relative_path"], "outcome": "passed", "findings": [],
         "validator": validator.identity.to_dict()}]}


def test_registered_validator_rejection_rejects_the_attempt(tmp_path, store_factory):
    from smythe.validation import ValidationReport, advisory_finding, hard_finding

    def off_palette(validator, subject):
        findings = [advisory_finding(f"note-{index}", "close to the edge") for index in range(40)]
        findings.append(hard_finding("off_palette", "uses #00ff00", expected=["#ff0000"],
                                     observed=["#00ff00"]))
        return ValidationReport("tile", tuple(findings), validator=validator.identity)

    validator = _PaletteValidator(off_palette)
    _, result, record = _validated_run(tmp_path, store_factory(tmp_path / "jobs.db"), validator)

    assert result["counts"] == {OperationStatus.REJECTED.value: 1}
    assert record["decision"] == "rejected"
    [artifact] = result["artifacts"]
    assert artifact["accepted"] == 0
    assert "artifact-01.png: brand-palette: uses #00ff00" in result["operations"][0]["error"]
    [report] = record["validators"][1]["reports"]
    assert report["outcome"] == "rejected" and report["path"] == artifact["relative_path"]
    # Hard findings are kept first, and at most 32 findings are recorded.
    assert len(report["findings"]) == 32
    assert report["findings"][0]["code"] == "off_palette"


class _Unserializable:
    pass


@pytest.mark.parametrize("verdict,problem", [
    (lambda validator, subject: (_ for _ in ()).throw(RuntimeError("secret detail")),
     "raised RuntimeError"),
    (lambda validator, subject: True, "returned bool, not a ValidationReport"),
    (lambda validator, subject: _coroutine(), "returned coroutine, not a ValidationReport"),
    (lambda validator, subject: _report(_PaletteValidator(version="9").identity),
     "returned a report for another validator"),
    (lambda validator, subject: _report(validator.identity, observed=_Unserializable()),
     "returned findings that are not JSON data"),
    (lambda validator, subject: _report(validator.identity, outcome="error"), "could not decide"),
], ids=["raises", "bool", "coroutine", "other-identity", "not-json", "error-report"])
def test_validator_errors_are_indeterminate_and_never_unknown_outcomes(
    tmp_path, store_factory, verdict, problem,
):
    validator = _PaletteValidator(verdict)
    store = store_factory(tmp_path / "jobs.db")
    runner, result, record = _validated_run(tmp_path, store, validator)

    assert result["counts"] == {OperationStatus.REJECTED.value: 1}
    assert result["cost"]["exposure_microusd"] == 0 and result["cost"]["cost_is_complete"]
    assert record["decision"] == "indeterminate"
    assert [item["outcome"] for item in record["validators"]] == ["passed", "error"]
    error = result["operations"][0]["error"]
    assert f"validator 'brand-palette' {problem}" in error and "secret detail" not in error
    # No duplicate-spend acknowledgement is needed to try again.
    key = result["operations"][0]["operation_key"]
    validator.verdict = None
    second = asyncio.run(runner.reroll(result["run_id"], [key], reason="validator fixed"))
    assert second["counts"] == {OperationStatus.SUCCEEDED.value: 1}


async def _coroutine():
    return None


def _report(identity, *, observed=None, outcome=None):
    from smythe.validation import ValidationReport, advisory_finding

    findings = () if observed is None else (advisory_finding("note", "a note", observed=observed),)
    return ValidationReport("tile", findings, outcome, identity)


def test_runs_refuse_validators_the_runner_has_not_registered(tmp_path, store_factory):
    validator = _PaletteValidator()
    plan = preflight_job(_validated_manifest(), manifest_root=tmp_path, validators=[validator])
    store = store_factory(tmp_path / "jobs.db")
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")], cost_usd=None)
    pool = _FixedPool(provider)

    for registered in ([], [_PaletteValidator(colors=("#0000ff",))]):
        with pytest.raises(PreflightError, match="brand-palette"):
            asyncio.run(JobRunner(store, provider_pool=pool, validators=registered).start(
                plan, make_approval(plan), manifest_root=tmp_path))
    assert store.list_runs() == []

    run_id = store.create_run(plan, make_approval(plan), manifest_root=tmp_path)
    with pytest.raises(PreflightError, match="'brand-palette'.*not registered"):
        asyncio.run(JobRunner(store, provider_pool=pool).resume(run_id))
    with pytest.raises(ApprovalError, match="plan hash"):
        asyncio.run(JobRunner(store, provider_pool=pool, validators=[
            _PaletteValidator(colors=("#0000ff",))]).resume(run_id))
    assert provider.calls == 0
    assert store.snapshot(run_id)["counts"] == {OperationStatus.PENDING.value: 1}
    resumed = asyncio.run(JobRunner(store, provider_pool=pool, validators=[validator]).resume(run_id))
    assert resumed["counts"] == {OperationStatus.SUCCEEDED.value: 1}


def test_resume_rechecks_accepted_bytes_and_reports_a_corrupted_file(tmp_path, store_factory):
    plan = preflight_job(_manifest(count=2), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    first = asyncio.run(JobRunner(store).start(plan, make_approval(plan), manifest_root=tmp_path))
    corrupted, intact = first["artifacts"]
    path = tmp_path / "outputs" / first["artifact_directory"] / corrupted["relative_path"]
    path.write_bytes(b"\x00" * corrupted["size_bytes"])
    before = store.snapshot(first["run_id"], include_events=True)
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")], cost_usd=None)

    resumed = asyncio.run(JobRunner(store, provider_pool=_FixedPool(provider)).resume(first["run_id"]))

    assert provider.calls == 0
    assert resumed["status"] == RunStatus.COMPLETED.value
    assert resumed["counts"] == {OperationStatus.SUCCEEDED.value: 2}
    by_attempt = {op["accepted_attempt_id"]: op for op in resumed["operations"]}
    assert by_attempt[corrupted["attempt_id"]]["delivery_eligibility"] == {
        "eligible": False, "reasons": ["artifact_changed"]}
    assert by_attempt[intact["attempt_id"]]["delivery_eligibility"] == {
        "eligible": True, "reasons": []}
    assert resumed["artifact_integrity"]["counts"] == {"changed": 1, "verified": 1}
    integrity = {item["artifact_id"]: item["integrity"]["status"] for item in resumed["artifacts"]}
    assert integrity == {corrupted["artifact_id"]: "changed", intact["artifact_id"]: "verified"}
    # The finding is reported, never written back: recorded acceptance is unchanged.
    after = store.snapshot(first["run_id"], include_events=True)
    assert [item["accepted"] for item in after["artifacts"]] == [1, 1]
    assert after["operations"] == before["operations"] and after["artifacts"] == before["artifacts"]
    assert not {"call_completed", "artifact_rejected"} & {
        event["event_type"] for event in after["events"][len(before["events"]):]}


def _jpeg_1x1():
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 255)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _unknown_after_publication(tmp_path, store, monkeypatch, provider, *, manifest=None,
                               validators=()):
    """Stop the process after a response's bytes are published, before its call settles.

    Resuming then records the dispatched call as an unknown outcome.
    """
    validators = list(validators)
    plan = preflight_job(manifest or _priced_manifest(), manifest_root=tmp_path,
                         validators=validators)
    runner = JobRunner(store, provider_pool=_FixedPool(provider), validators=validators)
    stop = crash_when(monkeypatch, store, "complete_call")
    with pytest.raises(Crash):
        asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path, run_id="adopt"))
    stop.restore()
    resumed = asyncio.run(runner.resume("adopt"))
    assert resumed["status"] == RunStatus.NEEDS_ATTENTION.value
    assert resumed["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}
    return runner, resumed


def test_adoption_accepts_published_bytes_without_a_call_and_keeps_the_exposure(
    tmp_path, store_factory, monkeypatch,
):
    store = store_factory(tmp_path / "jobs.db")
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")])
    runner, unknown = _unknown_after_publication(tmp_path, store, monkeypatch, provider)
    assert unknown["cost"]["exposure_microusd"] == 100_000 and unknown["artifacts"] == []
    [published] = (tmp_path / "outputs").rglob("*.png")
    key = unknown["operations"][0]["operation_key"]

    adopted = asyncio.run(runner.adopt("adopt", key, reason="the provider billed one image"))

    assert provider.calls == 1
    assert adopted["status"] == RunStatus.COMPLETED.value
    assert adopted["counts"] == {OperationStatus.SUCCEEDED.value: 1}
    [operation], [attempt], [artifact] = (adopted[name] for name in ("operations", "attempts", "artifacts"))
    assert operation["accepted_attempt_id"] == attempt["attempt_id"]
    assert operation["attempt_count"] == 1 and operation["error"] is None
    assert artifact["attempt_id"] == attempt["attempt_id"] and artifact["accepted"] == 1
    assert artifact["sha256"] == hashlib.sha256(published.read_bytes()).hexdigest()
    assert operation["delivery_acceptance"]["decision"] == "accepted"
    assert operation["delivery_eligibility"] == {"eligible": True, "reasons": []}
    # The attempt and its call keep their unknown outcome: no charge is settled.
    assert attempt["status"] == OperationStatus.UNKNOWN_OUTCOME.value
    [call] = store._call_rows("adopt")
    assert (call["status"], call["confirmed_microusd"], call["exposure_microusd"]) == (
        "unknown_outcome", 0, 100_000)
    assert adopted["cost"]["confirmed_microusd"] == 0
    assert adopted["cost"]["exposure_microusd"] == 100_000
    assert adopted["cost"]["cost_is_complete"] is False
    assert _events(store, "adopt", "call_completed") == []
    [event] = _events(store, "adopt", "unknown_outcome_adopted")
    assert event["payload"]["reason"] == "the provider billed one image"
    assert (event["payload"]["call_id"], event["payload"]["attempt_id"]) == (
        call["call_id"], attempt["attempt_id"])
    assert event["payload"]["exposure_microusd"] == 100_000
    assert event["payload"]["artifacts"] == [
        {"relative_path": artifact["relative_path"], "sha256": artifact["sha256"]}]
    assert event["payload"]["delivery_acceptance"]["decision"] == "accepted"

    # Adopted work is not adopted twice, rerolled or dispatched again.
    with pytest.raises(InvalidTransitionError, match="no unknown outcome"):
        asyncio.run(runner.adopt("adopt", key, reason="again"))
    with pytest.raises(InvalidTransitionError, match="cannot be rerolled"):
        asyncio.run(runner.reroll("adopt", [key], reason="again", acknowledge_unknown=True))
    assert asyncio.run(runner.resume("adopt"))["execution_metrics"]["operations_started"] == 0
    assert provider.calls == 1


def test_adoption_refuses_bytes_that_are_not_the_complete_accepted_response(
    tmp_path, store_factory, monkeypatch,
):
    store = store_factory(tmp_path / "jobs.db")
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")])
    runner, unknown = _unknown_after_publication(tmp_path, store, monkeypatch, provider)
    key = unknown["operations"][0]["operation_key"]
    [published] = (tmp_path / "outputs").rglob("*.png")
    extra = published.parent / "artifact-02.png"
    before = store.snapshot("adopt", include_events=True)

    for change, problem in (
        (lambda: published.write_bytes(_UNDECODABLE_PNG), "could not be fully decoded"),
        (published.unlink, "artifact-01.png is missing"),
        (lambda: published.write_bytes(_jpeg_1x1()), "expected image/png, observed image/jpeg"),
        (lambda: (published.write_bytes(PNG_1X1), extra.write_bytes(PNG_1X1)),
         "artifact-02.png is not part of the response"),
    ):
        change()
        with pytest.raises(InvalidTransitionError, match="cannot be adopted") as refused:
            asyncio.run(runner.adopt("adopt", key, reason="bytes look complete"))
        assert problem in str(refused.value)
        after = store.snapshot("adopt", include_events=True)
        assert after["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}
        assert after["operations"] == before["operations"] and after["artifacts"] == []
        assert after["cost"] == before["cost"]
        assert not {"unknown_outcome_adopted", "call_completed"} & {
            event["event_type"] for event in after["events"]}

    # The complete response is still adoptable after the refusals.
    extra.unlink()
    adopted = asyncio.run(runner.adopt("adopt", key, reason="bytes restored"))
    assert adopted["counts"] == {OperationStatus.SUCCEEDED.value: 1}
    assert provider.calls == 1


def test_adoption_runs_the_registered_validators_on_the_published_bytes(
    tmp_path, store_factory, monkeypatch,
):
    from smythe.validation import ValidationReport, hard_finding

    def off_palette(validator, subject):
        return ValidationReport("tile", (hard_finding("off_palette", "uses #00ff00"),),
                                validator=validator.identity)

    validator = _PaletteValidator(off_palette)
    store = store_factory(tmp_path / "jobs.db")
    runner, unknown = _unknown_after_publication(
        tmp_path, store, monkeypatch, _ImmediateArtifactProvider(),
        manifest=_validated_manifest(), validators=[validator])
    key = unknown["operations"][0]["operation_key"]
    published = validator.subjects[0].data

    with pytest.raises(InvalidTransitionError, match="brand-palette: uses #00ff00"):
        asyncio.run(runner.adopt("adopt", key, reason="looks right"))
    assert store.snapshot("adopt")["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}

    validator.verdict = None
    adopted = asyncio.run(runner.adopt("adopt", key, reason="palette confirmed"))

    assert validator.subjects[-1].data == published
    assert adopted["counts"] == {OperationStatus.SUCCEEDED.value: 1}
    [event] = _events(store, "adopt", "unknown_outcome_adopted")
    assert [result["outcome"] for result in event["payload"]["delivery_acceptance"]["validators"]] == [
        "passed", "passed"]
    assert adopted["operations"][0]["delivery_eligibility"] == {"eligible": True, "reasons": []}


def test_adoption_is_refused_in_any_other_state_and_while_another_owner_holds_the_lease(
    tmp_path, store_factory, monkeypatch,
):
    store = store_factory(tmp_path / "jobs.db")
    runner, unknown = _unknown_after_publication(
        tmp_path, store, monkeypatch, _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")]))
    key = unknown["operations"][0]["operation_key"]

    lease = store.acquire_run_lease("adopt", "other-worker", ttl_s=30)
    with pytest.raises(RunLeaseError, match="other-worker"):
        asyncio.run(runner.adopt("adopt", key, reason="bytes verified"))
    store.release_run_lease("adopt", "other-worker", lease=lease)
    with pytest.raises(ValueError, match="reason"):
        asyncio.run(runner.adopt("adopt", key, reason=" "))
    with pytest.raises(JobNotFoundError):
        asyncio.run(runner.adopt("adopt", "missing[0]", reason="bytes verified"))
    assert store.snapshot("adopt")["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}

    # Completed, rejected and pending operations have no unknown outcome to adopt.
    plan = preflight_job(_manifest(count=3, attempts=1), manifest_root=tmp_path)
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")], [],
                                       [Artifact(PNG_1X1, "image/png")], cost_usd=None)
    other = JobRunner(store, provider_pool=_FixedPool(provider))
    run_id = store.create_run(plan, make_approval(plan), manifest_root=tmp_path, run_id="settled")
    asyncio.run(other.resume(run_id))
    snapshot = store.snapshot(run_id)
    assert snapshot["counts"] == {"succeeded": 2, "rejected": 1}
    for operation in snapshot["operations"]:
        with pytest.raises(InvalidTransitionError, match="no unknown outcome"):
            asyncio.run(other.adopt(run_id, operation["operation_key"], reason="bytes verified"))
    pending = store.create_run(plan, make_approval(plan), manifest_root=tmp_path, run_id="pending",
                               allow_approval_reuse=True)
    with pytest.raises(InvalidTransitionError, match="state pending"):
        asyncio.run(other.adopt(pending, "glyph[0]", reason="bytes verified"))


def test_an_unknown_outcome_that_published_nothing_cannot_be_adopted(tmp_path, store_factory):
    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store, provider_pool=_FixedPool(_AmbiguousProvider()))
    result = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))
    key = result["operations"][0]["operation_key"]

    with pytest.raises(InvalidTransitionError, match="artifact-01.png is missing"):
        asyncio.run(runner.adopt(result["run_id"], key, reason="provider shows a response"))
    assert store.snapshot(result["run_id"])["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}


def test_slow_artifact_finalizer_does_not_starve_lease_heartbeat(
    tmp_path, monkeypatch
):
    import smythe.jobs.runner as runner_module

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_write = runner_module.atomic_publish_bytes

    def slow_write(path, data):
        entered.set()
        try:
            assert release.wait(timeout=15), "artifact writer was never released"
            original_write(path, data)
        finally:
            finished.set()

    monkeypatch.setattr(runner_module, "atomic_publish_bytes", slow_write)

    async def scenario() -> None:
        plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
        store = SQLiteRunStore(tmp_path / "jobs.db")
        runner = JobRunner(
            store,
            provider_pool=_FixedPool(_ImmediateArtifactProvider()),
            lease_ttl_s=30,
            lease_heartbeat_s=0.05,
        )
        task = asyncio.create_task(
            runner.start(
                plan,
                make_approval(plan),
                manifest_root=tmp_path,
                run_id="slow-finalizer",
            )
        )
        try:
            loop = asyncio.get_running_loop()
            setup_deadline = loop.time() + 15
            while not entered.is_set() and not task.done() and loop.time() < setup_deadline:
                await asyncio.sleep(0.01)
            assert entered.is_set(), "artifact writer did not start within the setup deadline"
            assert not finished.is_set(), "artifact writer stopped before the heartbeat check"
            before = store.get_run_lease("slow-finalizer")
            assert before is not None
            heartbeat_deadline = loop.time() + 2
            after = before
            while after.heartbeat_at_ns <= before.heartbeat_at_ns and loop.time() < heartbeat_deadline:
                await asyncio.sleep(0.01)
                after = store.get_run_lease("slow-finalizer")
                assert after is not None
            assert after.heartbeat_at_ns > before.heartbeat_at_ns
            # The lease must advance while the writer remains blocked, not
            # after a fixed delay happened to outlast the filesystem work.
            assert not release.is_set() and not finished.is_set()
            release.set()
            result = await asyncio.wait_for(asyncio.shield(task), timeout=15)
            assert result["status"] == RunStatus.COMPLETED.value
        finally:
            release.set()
            try:
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=15)
            finally:
                store.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Integrity repair of corrupted accepted artifacts
# ---------------------------------------------------------------------------


def _damage(path, change):
    if change == "flip":
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
    elif change == "truncate":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        path.unlink()


def _artifact_dump(store):
    return [line for line in store._connection.iterdump() if line.startswith('INSERT INTO "artifacts"')]


def _accepted_file(tmp_path, result, artifact):
    return tmp_path / "outputs" / result["artifact_directory"] / artifact["relative_path"]


def _owner(result, artifact):
    return next(item for item in result["operations"]
                if item["accepted_attempt_id"] == artifact["attempt_id"])


@pytest.mark.parametrize("change,status", [("flip", "changed"), ("truncate", "changed"),
                                           ("delete", "missing")])
def test_integrity_repair_regenerates_a_corrupted_accepted_png(tmp_path, store_factory, change, status):
    from smythe.jobs.inspection import inspect_job

    plan = preflight_job(_manifest(count=2, attempts=2), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store)
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path, run_id="repair"))
    corrupted, intact = first["artifacts"]
    key = _owner(first, corrupted)["operation_key"]
    _damage(_accepted_file(tmp_path, first, corrupted), change)

    with SQLiteRunStore(tmp_path / "jobs.db", read_only=True) as reader:
        inspected = {item["operation_key"]: item for item in inspect_job(reader, "repair")["operations"]}
    assert inspected[key]["delivery_eligibility"] == {"eligible": False, "reasons": [f"artifact_{status}"]}
    assert inspected[_owner(first, intact)["operation_key"]]["delivery_eligibility"]["eligible"]
    dump = _artifact_dump(store)

    repaired = asyncio.run(runner.reroll("repair", [key], reason="accepted PNG damaged",
                                         repair_integrity=True))

    assert repaired["status"] == RunStatus.COMPLETED.value
    assert repaired["counts"] == {OperationStatus.SUCCEEDED.value: 2}
    assert repaired["execution_metrics"]["operations_started"] == 1
    operation = next(item for item in repaired["operations"] if item["operation_key"] == key)
    attempts = [item for item in repaired["attempts"] if item["operation_id"] == operation["operation_id"]]
    # The new attempt alone records the reason, linked to the attempt it replaces.
    assert [(item["attempt_number"], item["reason"]) for item in attempts] == [
        (1, None), (2, "accepted PNG damaged")]
    assert attempts[1]["parent_attempt_id"] == corrupted["attempt_id"]
    assert operation["accepted_attempt_id"] == attempts[1]["attempt_id"]
    assert operation["attempt_count"] == 2
    assert operation["delivery_eligibility"] == {"eligible": True, "reasons": []}
    assert [item["reason"] for item in repaired["attempts"]
            if item["operation_id"] != operation["operation_id"]] == [None]
    # New artifact rows are added; the earlier rows stay byte-identical.
    after = _artifact_dump(store)
    assert set(dump) < set(after) and len(after) == len(dump) + 1
    [new] = [item for item in repaired["artifacts"] if item["attempt_id"] == attempts[1]["attempt_id"]]
    assert new["accepted"] == 1 and "/attempt-0002/" in new["relative_path"]
    assert hashlib.sha256(_accepted_file(tmp_path, repaired, new).read_bytes()).hexdigest() == new["sha256"]
    # Totals stay consistent: one more settled call and nothing reserved or exposed.
    calls = store._call_rows("repair")
    assert len(calls) == 3 and {call["status"] for call in calls} == {"succeeded"}
    assert repaired["cost"]["confirmed_microusd"] == sum(call["confirmed_microusd"] for call in calls)
    assert (repaired["cost"]["reserved_microusd"], repaired["cost"]["exposure_microusd"]) == (0, 0)
    # The returned recheck covers each operation's accepted attempt.
    assert repaired["artifact_integrity"]["counts"] == {"verified": 2}
    [event] = _events(store, "repair", "integrity_repair_queued")
    [finding] = event["payload"]["findings"]
    assert (event["payload"]["version"], event["payload"]["reason"]) == (1, "accepted PNG damaged")
    assert {name: finding[name] for name in ("artifact_id", "relative_path", "sha256", "status")} == {
        "artifact_id": corrupted["artifact_id"], "relative_path": corrupted["relative_path"],
        "sha256": corrupted["sha256"], "status": status}
    with SQLiteRunStore(tmp_path / "jobs.db", read_only=True) as reader:
        report = inspect_job(reader, "repair")
    assert all(item["delivery_eligibility"]["eligible"] for item in report["operations"])


def test_integrity_repair_uses_the_original_approval_and_attempt_allowance(tmp_path, store_factory):
    data = _priced_manifest().to_dict()
    data["execution"].update(max_attempts=2, max_budget_usd="0.20")
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")])
    runner = JobRunner(store, provider_pool=_FixedPool(provider))
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))
    [artifact] = first["artifacts"]
    key = first["operations"][0]["operation_key"]
    _accepted_file(tmp_path, first, artifact).unlink()

    repaired = asyncio.run(runner.reroll(first["run_id"], [key], reason="file deleted",
                                         repair_integrity=True))

    assert provider.calls == 2
    assert repaired["cost"]["approved_microusd"] == first["cost"]["approved_microusd"] == 200_000
    assert repaired["cost"]["confirmed_microusd"] == 80_000
    assert sum(call["confirmed_microusd"] for call in store._call_rows(first["run_id"])) == 80_000
    # A second repair needs an attempt the plan no longer allows.
    [_old, new] = repaired["artifacts"]
    _accepted_file(tmp_path, repaired, new).unlink()
    with pytest.raises(InvalidTransitionError, match="exhausted"):
        asyncio.run(runner.reroll(first["run_id"], [key], reason="deleted again",
                                  repair_integrity=True))
    assert provider.calls == 2
    assert store.snapshot(first["run_id"])["counts"] == {OperationStatus.SUCCEEDED.value: 1}


def test_integrity_repair_refuses_verified_or_unsafe_bytes_and_other_owners(tmp_path, store_factory):
    plan = preflight_job(_manifest(count=1, attempts=2), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store)
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path, run_id="kept"))
    [artifact] = first["artifacts"]
    key = first["operations"][0]["operation_key"]
    before = store.snapshot("kept")

    def unchanged():
        after = store.snapshot("kept", include_events=True)
        assert [after[name] for name in ("operations", "attempts", "artifacts")] == [
            before[name] for name in ("operations", "attempts", "artifacts")]
        assert not _events(store, "kept", "integrity_repair_queued")

    with pytest.raises(InvalidTransitionError, match="no missing or changed accepted file"):
        asyncio.run(runner.reroll("kept", [key], reason="looks off", repair_integrity=True))
    # A casual reroll of a verified success stays refused.
    with pytest.raises(InvalidTransitionError, match="cannot be rerolled"):
        asyncio.run(runner.reroll("kept", [key], reason="looks off"))
    with pytest.raises(ValueError, match="unknown outcome"):
        asyncio.run(runner.reroll("kept", [key], reason="r", repair_integrity=True,
                                  acknowledge_unknown=True))
    with pytest.raises(TypeError, match="repair_integrity"):
        asyncio.run(runner.reroll("kept", [key], reason="r", repair_integrity=1))
    unchanged()

    path = _accepted_file(tmp_path, first, artifact)
    path.unlink()
    path.mkdir()  # Not a regular file: the bytes cannot be known.
    with pytest.raises(InvalidTransitionError, match="is unsafe"):
        asyncio.run(runner.reroll("kept", [key], reason="replaced", repair_integrity=True))
    unchanged()

    path.rmdir()
    lease = store.acquire_run_lease("kept", "other-worker", ttl_s=30)
    with pytest.raises(RunLeaseError, match="other-worker"):
        asyncio.run(runner.reroll("kept", [key], reason="file deleted", repair_integrity=True))
    store.release_run_lease("kept", "other-worker", lease=lease)
    unchanged()


def test_integrity_repair_refuses_a_lease_lost_while_hashing(tmp_path, monkeypatch):
    from fakes import ManualClock

    clock = ManualClock.after(0)
    plan = preflight_job(_manifest(count=1, attempts=2), manifest_root=tmp_path)
    with SQLiteRunStore(tmp_path / "jobs.db", clock_ns=clock) as store, \
            SQLiteRunStore(tmp_path / "jobs.db", clock_ns=clock) as intruder:
        runner = JobRunner(store)
        first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path, run_id="lost"))
        _accepted_file(tmp_path, first, first["artifacts"][0]).unlink()
        check = runner_module.check_artifact_integrity

        def expire_during_check(*args, **kwargs):
            # The files are hashed while this runner holds the run's lease.
            assert store.get_run_lease("lost").owner_id.startswith("runner-")
            summary = check(*args, **kwargs)
            clock.expire_lease(runner.lease_ttl_s)
            intruder.acquire_run_lease("lost", "intruder")
            return summary

        monkeypatch.setattr(runner_module, "check_artifact_integrity", expire_during_check)
        with pytest.raises(RunLeaseError, match="intruder"):
            asyncio.run(runner.reroll("lost", [first["operations"][0]["operation_key"]],
                                      reason="file deleted", repair_integrity=True))
        assert store.snapshot("lost")["counts"] == {OperationStatus.SUCCEEDED.value: 1}
        assert not _events(store, "lost", "integrity_repair_queued")


def test_resume_reports_a_corrupted_file_a_rejected_repair_left_accepted(tmp_path, store_factory):
    plan = preflight_job(_manifest(count=1, attempts=3), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")], [], cost_usd=None)
    runner = JobRunner(store, provider_pool=_FixedPool(provider))
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path, run_id="kept"))
    [artifact] = first["artifacts"]
    key = first["operations"][0]["operation_key"]
    _damage(_accepted_file(tmp_path, first, artifact), "flip")

    rejected = asyncio.run(runner.reroll("kept", [key], reason="flipped byte", repair_integrity=True))

    assert rejected["status"] == RunStatus.FAILED.value
    [operation] = rejected["operations"]
    assert operation["status"] == OperationStatus.REJECTED.value
    # The accepted attempt stays named until a new attempt is accepted.
    assert operation["accepted_attempt_id"] == artifact["attempt_id"]
    assert operation["delivery_eligibility"] == {"eligible": False, "reasons": ["artifact_changed"]}
    before = store.snapshot("kept", include_events=True)

    resumed = asyncio.run(runner.resume("kept"))

    assert resumed["status"] == RunStatus.FAILED.value
    assert resumed["execution_metrics"]["operations_started"] == 0
    assert resumed["artifact_integrity"]["counts"] == {"changed": 1}
    assert resumed["operations"][0]["delivery_eligibility"]["reasons"] == ["artifact_changed"]
    after = store.snapshot("kept", include_events=True)
    assert after["operations"] == before["operations"] and after["artifacts"] == before["artifacts"]
    assert provider.calls == 2


# ---------------------------------------------------------------------------
# Rechecks of accepted outputs under changed validators
# ---------------------------------------------------------------------------


def _off_palette(validator, subject):
    from smythe.validation import ValidationReport, hard_finding

    return ValidationReport("tile", (hard_finding("off_palette", "uses #00ff00"),),
                            validator=validator.identity)


def _eligibility(database, run_id, validators=None):
    from smythe.jobs.inspection import inspect_job

    with SQLiteRunStore(database, read_only=True) as reader:
        return {item["operation_key"]: item["delivery_eligibility"]
                for item in inspect_job(reader, run_id, validators=validators)["operations"]}


def test_a_validator_version_bump_needs_a_recheck_which_records_a_new_decision(
    tmp_path, store_factory,
):
    store = store_factory(tmp_path / "jobs.db")
    original = _PaletteValidator()
    _, first, record = _validated_run(tmp_path, store, original)
    run_id, [operation] = first["run_id"], first["operations"]
    key = operation["operation_key"]
    bumped = _PaletteValidator(version="2")
    provider = _PricedArtifactProvider([Artifact(PNG_1X1, "image/png")], cost_usd=None)
    rechecker = JobRunner(store, provider_pool=_FixedPool(provider), validators=[bumped])

    # The call's decision used version 1, so version 2 requires a recheck.
    assert _eligibility(tmp_path / "jobs.db", run_id, [bumped]) == {
        key: {"eligible": False, "reasons": ["validator_changed"]}}
    assert _eligibility(tmp_path / "jobs.db", run_id) == _eligibility(
        tmp_path / "jobs.db", run_id, [original]) == {key: {"eligible": True, "reasons": []}}
    # Resuming under the changed validator is still refused.
    with pytest.raises(PreflightError, match="version '1'"):
        asyncio.run(rechecker.resume(run_id))

    rechecked = asyncio.run(rechecker.recheck(run_id, [key], reason="palette version 2"))

    assert provider.calls == 0 and rechecked["execution_metrics"]["operations_started"] == 0
    assert [subject.data for subject in bumped.subjects] == [PNG_1X1]
    [item] = rechecked["operations"]
    assert item["status"] == OperationStatus.SUCCEEDED.value
    assert item["accepted_attempt_id"] == operation["accepted_attempt_id"]
    assert item["delivery_eligibility"] == {"eligible": True, "reasons": []}
    acceptance = item["delivery_acceptance"]
    assert (acceptance["source"], acceptance["decision"], acceptance["accepted"]) == (
        "reevaluated", "accepted", True)
    assert [result["validator"]["version"] for result in acceptance["validators"]] == ["1", "2"]
    assert rechecked["status"] == RunStatus.COMPLETED.value
    assert _eligibility(tmp_path / "jobs.db", run_id, [bumped]) == {
        key: {"eligible": True, "reasons": []}}
    [event] = _events(store, run_id, "delivery_acceptance_reevaluated")
    [artifact] = first["artifacts"]
    assert event["payload"]["version"] == 1 and event["payload"]["reason"] == "palette version 2"
    assert event["payload"]["attempt_id"] == operation["accepted_attempt_id"]
    assert event["payload"]["outcome"] == "accepted" and event["payload"]["findings"] == []
    assert event["payload"]["artifacts"] == [
        {"relative_path": artifact["relative_path"], "sha256": artifact["sha256"]}]
    assert event["payload"]["validators"][1]["validator"] == bumped.identity.to_dict()
    # The call's own decision is kept, and the run's lease was released.
    [completed] = _events(store, run_id, "call_completed")
    assert completed["payload"]["delivery_acceptance"] == record
    assert store.get_run_lease(run_id) is None


def test_a_newer_built_in_check_needs_a_recheck(tmp_path, store_factory, monkeypatch):
    import smythe.jobs.delivery as delivery

    plan = preflight_job(_manifest(count=2), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store)
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path,
                                     run_id="built-in"))
    rechecked_key, other_key = (item["operation_key"] for item in first["operations"])
    monkeypatch.setattr(delivery, "ARTIFACT_SPEC_VALIDATOR_VERSION", "2")

    resumed = asyncio.run(runner.resume("built-in"))
    assert {item["operation_key"]: item["delivery_eligibility"]["reasons"]
            for item in resumed["operations"]} == {
        rechecked_key: ["validator_changed"], other_key: ["validator_changed"]}

    rechecked = asyncio.run(runner.recheck("built-in", [rechecked_key], reason="check version 2"))

    assert {item["operation_key"]: item["delivery_eligibility"]["reasons"]
            for item in rechecked["operations"]} == {rechecked_key: [], other_key: ["validator_changed"]}
    [event] = _events(store, "built-in", "delivery_acceptance_reevaluated")
    [builtin] = event["payload"]["validators"]
    assert builtin["validator"]["name"] == "smythe.jobs.artifact-spec"
    assert builtin["validator"]["version"] == "2" and builtin["outcome"] == "passed"


def test_a_recheck_that_rejects_leaves_the_operation_succeeded_but_not_eligible(
    tmp_path, store_factory,
):
    store = store_factory(tmp_path / "jobs.db")
    runner, first, _ = _validated_run(tmp_path, store, _PaletteValidator())
    run_id, [operation] = first["run_id"], first["operations"]
    key = operation["operation_key"]
    before = store.snapshot(run_id)
    stricter = _PaletteValidator(_off_palette, version="2")

    rechecked = asyncio.run(JobRunner(store, validators=[stricter]).recheck(
        run_id, [operation["operation_id"]], reason="stricter palette"))

    [item] = rechecked["operations"]
    assert rechecked["status"] == RunStatus.COMPLETED.value
    assert item["status"] == OperationStatus.SUCCEEDED.value
    assert item["accepted_attempt_id"] == operation["accepted_attempt_id"]
    assert item["delivery_eligibility"] == {"eligible": False, "reasons": ["not_accepted"]}
    assert item["delivery_acceptance"]["decision"] == "rejected"
    assert item["delivery_acceptance"]["reasons"] == ["artifact-01.png: brand-palette: uses #00ff00"]
    after = store.snapshot(run_id)
    for name in ("operations", "attempts", "artifacts", "cost", "counts"):
        assert after[name] == before[name]
    assert [artifact["accepted"] for artifact in after["artifacts"]] == [1]
    assert _eligibility(tmp_path / "jobs.db", run_id, [stricter]) == {
        key: {"eligible": False, "reasons": ["not_accepted"]}}
    # Under the validators its plan names, the call's own decision still holds.
    assert _eligibility(tmp_path / "jobs.db", run_id) == {key: {"eligible": True, "reasons": []}}
    # The accepted bytes verify, so an integrity repair cannot regenerate them.
    with pytest.raises(InvalidTransitionError, match="no missing or changed"):
        asyncio.run(runner.reroll(run_id, [key], reason="rejected by palette 2",
                                  repair_integrity=True))


def test_a_recheck_is_refused_before_any_validator_decides(tmp_path, store_factory):
    store = store_factory(tmp_path / "jobs.db")
    validator = _PaletteValidator()
    plan = preflight_job(_validated_manifest(count=2), manifest_root=tmp_path, validators=[validator])
    runner = JobRunner(store, provider_pool=_FixedPool(_ImmediateArtifactProvider()),
                       validators=[validator])
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path,
                                     run_id="refused"))
    keys = [item["operation_key"] for item in first["operations"]]
    damaged, intact = first["artifacts"]
    decided = len(validator.subjects)

    def unchanged():
        assert not _events(store, "refused", "delivery_acceptance_reevaluated")
        assert len(validator.subjects) == decided
        assert store.get_run_lease("refused") is None

    _damage(_accepted_file(tmp_path, first, damaged), "flip")
    with pytest.raises(InvalidTransitionError, match=f"{damaged['relative_path']} is changed"):
        asyncio.run(runner.recheck("refused", keys, reason="palette check"))
    unchanged()
    _accepted_file(tmp_path, first, damaged).unlink()
    with pytest.raises(InvalidTransitionError, match="is missing"):
        asyncio.run(runner.recheck("refused", keys, reason="palette check"))
    unchanged()

    intact_key = _owner(first, intact)["operation_key"]
    with pytest.raises(PreflightError, match="'brand-palette'.*not registered"):
        asyncio.run(JobRunner(store).recheck("refused", [intact_key], reason="palette check"))
    with pytest.raises(JobNotFoundError):
        asyncio.run(runner.recheck("refused", ["missing[0]"], reason="palette check"))
    for keys_value, error in (([], ValueError), ("glyph[0]", TypeError)):
        with pytest.raises(error, match="operation key"):
            asyncio.run(runner.recheck("refused", keys_value, reason="palette check"))
    for reason in ("", " ", None, "x" * 2049):
        with pytest.raises(ValueError, match="reason"):
            asyncio.run(runner.recheck("refused", [intact_key], reason=reason))
    lease = store.acquire_run_lease("refused", "other-worker", ttl_s=30)
    with pytest.raises(RunLeaseError, match="other-worker"):
        asyncio.run(runner.recheck("refused", [intact_key], reason="palette check"))
    store.release_run_lease("refused", "other-worker", lease=lease)
    unchanged()

    # Only a succeeded operation has accepted output to recheck.
    rejected_plan = preflight_job(_manifest(count=1, attempts=2), manifest_root=tmp_path)
    rejected = asyncio.run(JobRunner(store, provider_pool=_FixedPool(_SequenceProvider())).start(
        rejected_plan, make_approval(rejected_plan), manifest_root=tmp_path, run_id="rejected"))
    with pytest.raises(InvalidTransitionError, match="state rejected has no accepted output"):
        asyncio.run(JobRunner(store).recheck(
            "rejected", [rejected["operations"][0]["operation_key"]], reason="check"))
    assert not _events(store, "rejected", "delivery_acceptance_reevaluated")


# ---------------------------------------------------------------------------
# Repair previews
# ---------------------------------------------------------------------------


def _previewed(runner, run_id, keys=None):
    return {item["operation_key"]: (item["status"], item["action"], item["reasons"],
                                    item["attempts_remaining"], item["new_plan_required"])
            for item in runner.repair_preview(run_id, keys)["operations"]}


def test_a_repair_preview_matches_integrity_repair_eligibility(tmp_path, store_factory):
    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store)
    plan = preflight_job(_manifest(count=5, attempts=2), manifest_root=tmp_path)
    first = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path,
                                     run_id="preview"))
    last = preflight_job(_manifest(count=1, attempts=1), manifest_root=tmp_path)
    spent = asyncio.run(runner.start(last, make_approval(last), manifest_root=tmp_path,
                                     run_id="spent"))
    damage = dict(zip(("intact", "flip", "truncate", "delete", "directory"), first["artifacts"]))
    keys = {change: _owner(first, artifact)["operation_key"] for change, artifact in damage.items()}
    for change, artifact in damage.items():
        path = _accepted_file(tmp_path, first, artifact)
        if change == "directory":
            path.unlink()
            path.mkdir()  # Not a regular file: the bytes cannot be known.
        elif change != "intact":
            _damage(path, change)
    _accepted_file(tmp_path, spent, spent["artifacts"][0]).unlink()

    previews = {run_id: _previewed(runner, run_id) for run_id in ("preview", "spent")}

    assert {change: previews["preview"][key] for change, key in keys.items()} == {
        "intact": ("succeeded", "reuse", [], 1, False),
        "flip": ("succeeded", "regenerate", ["artifact_changed"], 1, False),
        "truncate": ("succeeded", "regenerate", ["artifact_changed"], 1, False),
        "delete": ("succeeded", "regenerate", ["artifact_missing"], 1, False),
        "directory": ("succeeded", "review", ["artifact_unsafe"], 1, False),
    }
    # No attempt is left, so regenerating it takes a new plan and approval.
    assert list(previews["spent"].values()) == [
        ("succeeded", "regenerate", ["artifact_missing"], 0, True)]

    # An integrity repair accepts exactly the operations the preview regenerates in their run.
    for run_id, items in previews.items():
        for key, (_status, action, _reasons, _left, new_plan) in items.items():
            try:
                asyncio.run(runner.reroll(run_id, [key], reason="preview check",
                                          repair_integrity=True))
            except InvalidTransitionError:
                repaired = False
            else:
                repaired = True
            assert repaired == (action == "regenerate" and not new_plan), key


def test_a_repair_preview_reports_states_attempts_and_budget_and_changes_nothing(
    tmp_path, store_factory,
):
    data = _priced_manifest(count=5).to_dict()
    data["execution"].update(max_attempts=2, max_budget_usd="1.00")
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, make_approval(plan), manifest_root=tmp_path, run_id="states")
    unknown, rejected, _pending, running, succeeded = store.pending_operations(run_id)
    lease = store.acquire_run_lease(run_id, "worker-a")

    def dispatched(operation):
        attempt = store.begin_attempt(run_id, operation["operation_id"], lease=lease)
        permit = store.prepare_call(attempt["attempt_id"], 100_000, lease=lease)
        store.mark_call_dispatched(permit.call_id, lease=lease)
        return permit.call_id

    def settle(call_id, artifacts, accepted):
        store.complete_call(call_id, cost_microusd=80_000, cost_is_complete=True,
                            cost_is_estimate=False, artifacts=artifacts, result_text="done",
                            accepted=accepted, lease=lease)

    store.mark_unknown_outcome(dispatched(unknown), "connection lost", lease=lease)
    settle(dispatched(rejected), [], False)
    store.begin_attempt(run_id, running["operation_id"], lease=lease)
    settle(dispatched(succeeded), [{
        "relative_path": "artifacts/glyph_4_/attempt-0001/artifact-01.png", "mime_type": "image/png",
        "sha256": "a" * 64, "size_bytes": 68, "width": 1, "height": 1}], True)
    before = list(store._connection.iterdump())

    # The worker still holds the run's lease: a preview takes none and writes nothing.
    preview = JobRunner(store).repair_preview(run_id)

    assert list(store._connection.iterdump()) == before
    assert store.get_run_lease(run_id).owner_id == "worker-a"
    assert (preview["version"], preview["run_id"]) == (1, "states")
    assert preview["status"] == store.get_run(run_id)["status"]
    assert {item["operation_key"]: (item["status"], item["action"], item["reasons"],
                                    item["attempts_remaining"], item["new_plan_required"])
            for item in preview["operations"]} == {
        "glyph[0]": ("unknown_outcome", "review", ["outcome_unknown"], 1, False),
        "glyph[1]": ("rejected", "regenerate", ["node_incomplete"], 1, False),
        "glyph[2]": ("pending", "regenerate", ["node_incomplete"], 2, False),
        "glyph[3]": ("running", "review", ["run_in_progress"], 1, False),
        # A legacy decision has no recorded validator, and its file is gone.
        "glyph[4]": ("succeeded", "regenerate", ["artifact_missing", "validator_set_changed"], 1,
                     False),
    }
    assert preview["counts"] == {"reuse": 0, "recheck": 0, "regenerate": 3, "review": 2}
    assert preview["budget"] == {
        "approved_microusd": 1_000_000, "confirmed_microusd": 160_000,
        "exposure_microusd": 100_000, "reserved_microusd": 0, "remaining_microusd": 740_000,
        "overrun": False, "unit": "microUSD"}
    assert preview["attachments"] == []

    assert list(_previewed(JobRunner(store), run_id, ["glyph[1]", rejected["operation_id"]])) == [
        "glyph[1]"]
    with pytest.raises(JobNotFoundError):
        JobRunner(store).repair_preview(run_id, ["missing[0]"])
    with pytest.raises(JobNotFoundError):
        JobRunner(store).repair_preview("absent")
    with pytest.raises(TypeError, match="operation key"):
        JobRunner(store).repair_preview(run_id, "glyph[1]")

    # After a budget overrun, nothing more is admitted under the run's approval.
    data["operations"][0]["count"] = 2
    data["execution"]["max_budget_usd"] = "0.40"
    overrun_plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    overrun = store.create_run(overrun_plan, make_approval(overrun_plan), manifest_root=tmp_path)
    first_operation = store.pending_operations(overrun)[0]
    attempt = store.begin_attempt(overrun, first_operation["operation_id"])
    permit = store.prepare_call(attempt["attempt_id"], 100_000)
    store.mark_call_dispatched(permit.call_id)
    store.complete_call(permit.call_id, cost_microusd=150_000, cost_is_complete=True,
                        cost_is_estimate=False, artifacts=[], result_text="none", accepted=False)
    latched = JobRunner(store).repair_preview(overrun)
    assert latched["budget"]["overrun"] is True and latched["status"] == RunStatus.BUDGET_OVERRUN.value
    assert [(item["action"], item["new_plan_required"]) for item in latched["operations"]] == [
        ("regenerate", True), ("regenerate", True)]


def test_a_repair_preview_offers_a_recheck_for_a_changed_validator(tmp_path, store_factory):
    store = store_factory(tmp_path / "jobs.db")
    _, first, _ = _validated_run(tmp_path, store, _PaletteValidator())
    run_id = first["run_id"]
    bumped = JobRunner(store, validators=[_PaletteValidator(version="2")])

    assert list(_previewed(bumped, run_id).values()) == [
        ("succeeded", "recheck", ["validator_set_changed"], 1, False)]
    assert list(_previewed(JobRunner(store, validators=[_PaletteValidator()]), run_id).values()) == [
        ("succeeded", "reuse", [], 1, False)]

    asyncio.run(bumped.recheck(run_id, [first["operations"][0]["operation_key"]], reason="v2"))
    assert list(_previewed(bumped, run_id).values()) == [("succeeded", "reuse", [], 1, False)]
    stricter = JobRunner(store, validators=[_PaletteValidator(_off_palette, version="3")])
    asyncio.run(stricter.recheck(run_id, [first["operations"][0]["operation_key"]], reason="v3"))
    # A rejection of verified bytes is for a person: no repair in the run can replace them.
    assert list(_previewed(stricter, run_id).values()) == [
        ("succeeded", "review", ["decision_not_accepted"], 1, False)]


def test_a_repair_preview_never_offers_another_runs_identical_output(tmp_path, store_factory):
    from smythe.jobs.preflight import operation_input_digest

    store = store_factory(tmp_path / "jobs.db")
    runner = JobRunner(store)
    results, digests = {}, set()
    # Another attempt allowance makes another plan and approval with the same input digest.
    for run_id, attempts in (("intact", 1), ("damaged", 2)):
        plan = preflight_job(_manifest(count=1, attempts=attempts), manifest_root=tmp_path)
        digests.add(operation_input_digest(plan.operations[0], plan.attachments))
        results[run_id] = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path,
                                                   run_id=run_id))
    assert len(digests) == 1
    [artifact] = results["damaged"]["artifacts"]
    _accepted_file(tmp_path, results["damaged"], artifact).unlink()
    intact_rows = store._call_rows("intact")

    # The intact run holds an accepted output for the same input, but no run reuses
    # another's: the damaged operation regenerates in its own run, under its approval.
    assert list(_previewed(runner, "damaged").values()) == [
        ("succeeded", "regenerate", ["artifact_missing"], 1, False)]
    repaired = asyncio.run(runner.reroll("damaged", ["glyph[0]"], reason="accepted file deleted",
                                         repair_integrity=True))
    assert len(store._call_rows("damaged")) == 2 and store._call_rows("intact") == intact_rows
    [_lost, new] = repaired["artifacts"]
    assert "/attempt-0002/" in new["relative_path"]
    assert _accepted_file(tmp_path, repaired, new).is_file()
