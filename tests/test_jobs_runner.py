"""End-to-end offline execution tests for Jobs v1."""

from __future__ import annotations

import asyncio
import io
import os
import threading
from contextlib import ExitStack

import pytest
import smythe.jobs.runner as runner_module

from smythe.jobs import ApprovalError, JobManifestV1, PreflightError, make_approval, preflight_job
from smythe.jobs.models import (
    DEFAULT_CALL_TIMEOUT_S,
    DEFAULT_MAX_WALL_SECONDS,
    MAX_CALL_TIMEOUT_S,
)
from smythe.jobs.providers import ProviderPool
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import (
    InvalidTransitionError, OperationStatus, RunLeaseError, RunStatus, SQLiteRunStore,
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
