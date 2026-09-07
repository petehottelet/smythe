"""End-to-end offline execution tests for Jobs v1."""

from __future__ import annotations

import asyncio
import io
import os
import threading

import pytest
import smythe.jobs.runner as runner_module

from smythe.jobs import JobManifestV1, make_approval, preflight_job
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


def test_runner_executes_bounded_offline_job_and_persists_artifacts(tmp_path):
    plan = preflight_job(_manifest(), manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")
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
def test_runner_accepts_equivalent_windows_extended_path_namespace(tmp_path, monkeypatch):
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
    store = SQLiteRunStore(tmp_path / "jobs.db")

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


def test_internal_provider_transport_timeout_covers_approved_contract_max(tmp_path):
    runner = JobRunner(SQLiteRunStore(tmp_path / "jobs.db"))

    assert runner.providers._request_timeout_s == MAX_CALL_TIMEOUT_S
    assert runner.call_timeout_s is None
    assert runner.max_wall_seconds is None


def test_operator_deadlines_can_only_tighten_the_approved_plan(tmp_path):
    data = _manifest(count=1).to_dict()
    data["execution"]["call_timeout_s"] = 900
    data["execution"]["max_wall_seconds"] = 7200
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)

    default_runner = JobRunner(SQLiteRunStore(tmp_path / "default.db"))
    strict_runner = JobRunner(
        SQLiteRunStore(tmp_path / "strict.db"),
        call_timeout_s=120,
        max_wall_seconds=600,
    )
    loose_runner = JobRunner(
        SQLiteRunStore(tmp_path / "loose.db"),
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


def test_selective_reroll_replaces_only_rejected_operation(tmp_path):
    manifest = _manifest(count=1, attempts=2)
    plan = preflight_job(manifest, manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")
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


class _AmbiguousProvider(Provider):
    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        raise ConnectionError("response lost")


def test_provider_exception_after_dispatch_becomes_unknown_outcome(tmp_path):
    plan = preflight_job(_manifest(count=1, attempts=2), manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")
    runner = JobRunner(store, provider_pool=_FixedPool(_AmbiguousProvider()))

    result = asyncio.run(runner.start(plan, make_approval(plan), manifest_root=tmp_path))

    assert result["status"] == RunStatus.NEEDS_ATTENTION.value
    assert result["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}
    assert store.pending_operations(result["run_id"]) == []


def test_runner_rejects_attachment_changed_after_approval(tmp_path):
    attachment = tmp_path / "input.png"
    attachment.write_bytes(b"approved-input")
    data = _manifest(count=1).to_dict()
    data["operations"][0]["attachments"] = ["input.png"]
    manifest = JobManifestV1.from_dict(data)
    plan = preflight_job(manifest, manifest_root=tmp_path)
    approval = make_approval(plan)
    attachment.write_bytes(b"different-input")
    store = SQLiteRunStore(tmp_path / "jobs.db")

    with pytest.raises(ValueError, match="changed after preflight"):
        asyncio.run(JobRunner(store).start(plan, approval, manifest_root=tmp_path))


@pytest.mark.parametrize(
    "run_id",
    ["../outside", "..\\outside", ".", "", "name/child", "name\\child"],
)
def test_runner_rejects_unsafe_custom_run_id(tmp_path, run_id):
    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")

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


def test_never_returning_provider_hits_call_deadline_and_becomes_unknown(tmp_path):
    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")
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


def test_whole_run_deadline_cancels_dispatched_call_conservatively(tmp_path, monkeypatch):
    class EnteredProvider(_NeverReturningProvider):
        def __init__(self):
            self.entered = asyncio.Event()

        async def complete(self, system, prompt, model):
            self.entered.set()
            return await super().complete(system, prompt, model)

    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")
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


def test_whole_run_deadline_before_dispatch_leaves_work_pending(tmp_path, monkeypatch):
    class CountingProvider(_ImmediateArtifactProvider):
        calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            return await super().complete(system, prompt, model)

    release = threading.Event()
    plan = preflight_job(_manifest(count=1), manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")
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


def test_invalid_provider_cost_is_immediately_journaled_unknown(tmp_path):
    plan = preflight_job(_priced_manifest(), manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")
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
    tmp_path, mutated, field, value,
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
    store = SQLiteRunStore(tmp_path / "jobs.db")
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


def test_invalid_accounting_requires_explicit_reroll_acknowledgement(tmp_path):
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
    store = SQLiteRunStore(tmp_path / "jobs.db")
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


def test_invalid_accounting_preserves_already_dispatched_sibling_cost(tmp_path, monkeypatch):
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
    store = SQLiteRunStore(tmp_path / "jobs.db")
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


def test_arbitrary_provider_message_cannot_impersonate_accounting_classification(tmp_path):
    class MisleadingFailure(Provider):
        def __init__(self):
            self.calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            raise ConnectionError("invalid provider accounting: arbitrary remote message")

    data = _priced_manifest(count=2).to_dict()
    data["execution"]["max_concurrency"] = 1
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")
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


def test_estimated_provider_cost_remains_exposure_not_confirmed(tmp_path):
    plan = preflight_job(_priced_manifest(), manifest_root=tmp_path)
    store = SQLiteRunStore(tmp_path / "jobs.db")
    runner = JobRunner(store, provider_pool=_FixedPool(_EstimatedCostProvider()))

    result = asyncio.run(
        runner.start(plan, make_approval(plan), manifest_root=tmp_path)
    )

    assert result["status"] == RunStatus.COMPLETED.value
    assert result["cost"]["confirmed_microusd"] == 0
    assert result["cost"]["exposure_microusd"] == 100_000
    assert not result["cost"]["cost_is_complete"]
    assert result["cost"]["cost_contains_estimates"]


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
