"""Immutable logical calls backed by the durable workflow journal.

Only the caller assigns call keys. A claimed dispatch is never retried; saved
responses and accepted results are recovered locally. The journal owns money,
while ``CompletionResult.cost_usd`` remains a compatibility projection.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING

from smythe.provider import (
    CompletionResult, OfflineProvider, Provider, ProviderAccountingCancelledError,
    ProviderAccountingError, ProviderResponseError,
)
from smythe.provider_responses import (
    OpenAIResponsesProvider, PreparedRequest, RawResponseEnvelope, ResponseQuoteError,
    _json_dump,
)
from smythe.tools import ChatMessage, ToolSpec
from smythe.workflow_store import WorkflowError

if TYPE_CHECKING:
    from smythe.workflow_store import CallKey, LeaseToken, SQLiteWorkflowStore


class WorkflowQuoteError(ResponseQuoteError, WorkflowError):
    """A managed quote failed; resume the same logical call explicitly.

    Raw quote evidence remains on ``envelope`` and in the journal. This is
    terminal to node retry/skip policies, because advancing the attempt would
    abandon a prepared call without a durable disposition.
    """


def describe_provider(provider: Provider) -> dict:
    """Preflight an exact supported built-in without SDK, network or secrets."""
    if type(provider) not in (OpenAIResponsesProvider, OfflineProvider):
        raise ValueError("Durable workflows require native Responses or stateless OfflineProvider")
    return json.loads(_json_dump(provider.workflow_descriptor()))


def snapshot_provider(provider: Provider) -> Provider:
    """Detach supported configuration; scripts and custom subclasses are rejected."""
    describe_provider(provider)
    return provider.snapshot_for_workflow()


def validate_workflow_model(provider: Provider, model: str) -> None:
    """Reject unsupported phase models during preflight, before any paid call."""
    descriptor = describe_provider(provider)
    supported = descriptor["supported_models"]
    if type(model) is not str or not model or (supported is not None and model not in supported):
        raise ValueError("Model is not supported by this durable provider")


class WorkflowProviderContext:
    """One run's detached providers and lazily pooled, same-loop native clients.

    Replaying saved calls does not construct an SDK client. Small journal
    transactions run synchronously: there is no cancellable gap between raw
    response persistence, billing settlement and local result acceptance.
    """

    def __init__(self, store: SQLiteWorkflowStore, lease: LeaseToken,
                 providers: Mapping[str, Provider]) -> None:
        from smythe.workflow_store import LeaseToken

        if not isinstance(lease, LeaseToken):
            raise TypeError("Expected a workflow LeaseToken")
        if not isinstance(providers, Mapping) or not providers:
            raise ValueError("Provide a nonempty provider registry")
        if any(type(name) is not str or not name for name in providers):
            raise ValueError("Provider IDs must be nonempty strings")
        self.store = store
        self._lease = lease
        self._providers = {name: snapshot_provider(source) for name, source in providers.items()}
        self._descriptors = {name: describe_provider(source) for name, source in self._providers.items()}
        self._active = False
        self._closed = False
        self._loop = None
        self._stack = AsyncExitStack()
        self._pooled = {}
        self._pool_lock = asyncio.Lock()
        self._call_locks = {}
        self._failed = False

    @property
    def lease(self) -> LeaseToken:
        return self._lease

    def update_lease(self, lease: LeaseToken) -> None:
        from smythe.workflow_store import LeaseToken, WorkflowLeaseError

        if (not isinstance(lease, LeaseToken)
                or (lease.run_id, lease.owner_id, lease.epoch) != (
                    self._lease.run_id, self._lease.owner_id, self._lease.epoch)):
            raise WorkflowLeaseError("A context cannot change its fenced owner or epoch")
        self._lease = lease

    async def __aenter__(self):
        if self._active or self._closed:
            raise RuntimeError("Workflow provider context cannot be entered twice")
        self._loop = asyncio.get_running_loop()
        await self._stack.__aenter__()
        self._active = True
        return self

    async def __aexit__(self, *exc):
        self._active = False
        self._closed = True
        return await self._stack.__aexit__(*exc)

    def _check_active(self) -> None:
        if not self._active or asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Use the workflow provider inside its owning async context")
        if self._failed:
            from smythe.workflow_store import WorkflowStateError

            raise WorkflowStateError("This provider context stopped after a journal persistence failure")

    def for_call(self, provider_id: str, key: CallKey) -> JournaledProvider:
        from smythe.workflow_store import CallKey

        if provider_id not in self._providers:
            raise ValueError("Unknown workflow provider ID")
        if not isinstance(key, CallKey):
            raise TypeError("An explicit CallKey is required")
        return JournaledProvider(self, provider_id, key)

    async def _backend(self, provider_id):
        async with self._pool_lock:
            if provider_id not in self._pooled:
                source = self._providers[provider_id]
                self._pooled[provider_id] = (
                    await self._stack.enter_async_context(source.session())
                    if type(source) is OpenAIResponsesProvider else source
                )
            return self._pooled[provider_id]


@dataclass(frozen=True)
class JournaledProvider(Provider):
    """A provider bound to one explicit immutable logical call identity."""

    _context: WorkflowProviderContext = field(repr=False)
    provider_id: str
    key: CallKey
    workflow_managed = True

    def budget_estimate_usd(self, model: str) -> None:
        return None  # The durable quote and reservation belong to the journal.

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return await self.chat(system, [ChatMessage("user", prompt)], model)

    async def chat(self, system: str, messages: list[ChatMessage], model: str,
                   tools: list[ToolSpec] | None = None) -> CompletionResult:
        context = self._context
        context._check_active()
        source = context._providers[self.provider_id]
        validate_workflow_model(source, model)
        if (tools or type(system) is not str or type(messages) is not list or not messages
                or any(not isinstance(message, ChatMessage)
                       or message.role not in {"user", "assistant"}
                       or type(message.content) is not str or message.attachments
                       or message.tool_calls or message.tool_results
                       or message.provider_continuation is not None for message in messages)):
            raise ValueError("Durable workflows currently accept plain text messages without tools")
        if type(source) is OpenAIResponsesProvider:
            prepared = source.prepare(system, messages, model)
        else:
            if len(messages) != 1 or messages[0].role != "user":
                raise ValueError("Durable OfflineProvider accepts one user prompt")
            prepared = PreparedRequest(_json_dump({
                "provider_kind": "offline", "version": 1, "model": model,
                "system": system, "prompt": messages[0].content,
            }))
        # Lock by logical key, not provider ID: a changed provider under the
        # same key must meet the store's immutable-binding check.
        lock = context._call_locks.setdefault(self.key, asyncio.Lock())
        async with lock:
            context._check_active()
            return await self._execute(prepared)

    def _managed_receipt(self, receipt, record, *, replayed):
        value = json.loads(_json_dump(receipt or {}))
        value.update(workflow_run_id=record["run_id"], workflow_call_id=record["call_id"],
                     workflow_charge_recorded=True, workflow_replayed=replayed)
        return value

    def _managed_result(self, result, record, *, replayed):
        result.native_receipt = self._managed_receipt(result.native_receipt, record, replayed=replayed)
        return result

    def _managed_error(self, error, record, *, replayed):
        error.receipt = self._managed_receipt(error.receipt or record.get("receipt"), record,
                                              replayed=replayed)
        if error.billing_result is not None:
            self._managed_result(error.billing_result, record, replayed=replayed)
        return error

    @staticmethod
    def _envelope(prepared, evidence):
        return RawResponseEnvelope(
            prepared, evidence["body"], evidence["operation"], evidence["status_code"],
            evidence["request_id"], evidence["transport_error"],
        )

    @staticmethod
    def _serialize(result):
        # Raw bytes and encrypted reasoning remain only in the evidence table.
        return json.loads(_json_dump({
            "text": result.text, "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens, "stop_reason": result.stop_reason,
            "cost_usd": result.cost_usd, "cost_usd_is_estimate": result.cost_usd_is_estimate,
            "cost_usd_unknown": result.cost_usd_unknown, "native_receipt": result.native_receipt,
        }))

    async def _execute(self, prepared):
        context, store = self._context, self._context.store
        descriptor = context._descriptors[self.provider_id]
        record = store.prepare_call(
            context.lease, self.key, request_json=prepared.payload_json,
            tool_names_json=prepared.tool_names_json, provider=descriptor,
            price_version=descriptor["price_version"],
        )
        call_id = record["call_id"]
        if record["result_state"] in {"accepted", "applied"}:
            replay = store.load_replay(call_id)
            result = CompletionResult(**replay["decoded_result"])
            result.response_envelope = self._envelope(prepared, replay["evidence"])
            return self._managed_result(result, record, replayed=True)
        if record["evidence_id"] is not None:
            replay = store.load_replay(call_id)
            envelope = self._envelope(prepared, replay["evidence"])
            return self._finish(record, envelope, replayed=True)
        if record["state"] in {"dispatched", "unknown"}:
            raise self._managed_error(ProviderAccountingError(
                "Claimed workflow call has no settled response; it cannot be resent",
                receipt=record.get("receipt"),
            ), record, replayed=True)

        if record["quote_id"] is None:
            if descriptor["kind"] == "offline":
                evidence = RawResponseEnvelope(prepared, _json_dump({
                    "provider_kind": "offline", "version": 1, "input_tokens": 0,
                }).encode(), "input_tokens", 200)
            else:
                backend = await context._backend(self.provider_id)
                try:
                    evidence = (await backend.quote(prepared)).envelope
                except ResponseQuoteError as error:
                    store.append_quote_evidence(context.lease, call_id, error.envelope)
                    raise WorkflowQuoteError(
                        "Managed input counting did not establish a usable quote", error.envelope,
                    ) from error
            evidence_id = store.append_quote_evidence(context.lease, call_id, evidence)
            record.update(store.accept_quote(context.lease, call_id, evidence_id))
        # Configuration/SDK/session failures are still pre-dispatch failures.
        # Do not claim a one-shot attempt until its transport is ready.
        backend = await context._backend(self.provider_id)
        store.reserve_call(context.lease, call_id, record["quote_id"])
        permit = store.claim_dispatch(context.lease, call_id, prepared.request_sha256)
        try:
            if descriptor["kind"] == "offline":
                request = prepared.payload
                result = await backend.complete(request["system"], request["prompt"], request["model"])
                envelope = RawResponseEnvelope(prepared, _json_dump({
                    "provider_kind": "offline", "version": 1, "text": result.text,
                }).encode(), status_code=200)
            else:
                envelope = await backend.dispatch(prepared)
        except BaseException as error:
            cancelled = isinstance(error, asyncio.CancelledError)
            envelope = getattr(error, "envelope", None)
            if not isinstance(envelope, RawResponseEnvelope):
                envelope = RawResponseEnvelope(prepared, b"", transport_error=type(error).__name__)
            try:
                evidence_id = store.append_response(permit, envelope)
                settled = store.settle_call(context.lease, call_id, evidence_id)
            except BaseException as persistence_error:
                context._failed = True
                settled = record
                error = persistence_error
            kind = (ProviderAccountingCancelledError if cancelled
                    else ProviderAccountingError)
            raise self._managed_error(kind(
                "Workflow dispatch ended without an accepted response",
                envelope=envelope, receipt=settled.get("receipt"),
            ), settled, replayed=False) from error
        try:
            evidence_id = store.append_response(permit, envelope)
            record["evidence_id"] = evidence_id
            return self._finish(record, envelope, replayed=False)
        except ProviderResponseError:
            raise
        except BaseException as error:
            context._failed = True
            raise self._managed_error(ProviderAccountingError(
                "Raw workflow response could not complete its journal transition",
                envelope=envelope, receipt=record.get("receipt"),
            ), record, replayed=False) from error

    def _finish(self, record, envelope, *, replayed):
        context, store = self._context, self._context.store
        record = store.settle_call(context.lease, record["call_id"], record["evidence_id"])
        if record["billing_state"] != "known":
            raise self._managed_error(ProviderAccountingError(
                "Workflow response has unresolved billing evidence", envelope=envelope,
                receipt=record.get("receipt"),
            ), record, replayed=replayed)
        source = context._providers[self.provider_id]
        if record.get("admission_closed"):
            # A billing latch is not a semantic rejection. Leave saved output
            # pending so resolving another call can unlock local decoding.
            try:
                if type(source) is OpenAIResponsesProvider:
                    native = source.extract_receipt(envelope)
                    billing = CompletionResult(
                        "", prompt_tokens=native.input_tokens, completion_tokens=native.output_tokens,
                        cost_usd=float(native.cost_usd), native_receipt=record["receipt"],
                        response_envelope=envelope,
                    )
                else:
                    billing = CompletionResult("", cost_usd=0.0, native_receipt=record["receipt"],
                                               response_envelope=envelope)
            except (ValueError, OverflowError) as error:
                raise self._managed_error(ProviderAccountingError(
                    "Exact journal charge cannot be projected to the legacy result",
                    envelope=envelope, receipt=record["receipt"],
                ), record, replayed=replayed) from error
            raise self._managed_error(ProviderResponseError(
                "Workflow admission closed after settling this response: "
                + str(record.get("blocked_reason")), envelope=envelope,
                receipt=record["receipt"], billing_result=billing,
            ), record, replayed=replayed)
        try:
            if type(source) is OpenAIResponsesProvider:
                result = source.decode(envelope)
            else:
                raw = envelope.json()
                if type(raw.get("text")) is not str:
                    raise ProviderResponseError("Offline response has no text", envelope=envelope,
                                                receipt=record["receipt"])
                result = CompletionResult(raw["text"], cost_usd=0.0,
                                          native_receipt=record["receipt"], response_envelope=envelope)
        except ProviderResponseError as error:
            store.reject_result(context.lease, record["call_id"], str(error))
            raise self._managed_error(error, record, replayed=replayed)
        store.accept_result(context.lease, record["call_id"], self._serialize(result),
                            decoder_version=context._descriptors[self.provider_id]["decoder_version"])
        # Managed text workflows do not need opaque continuation in graph state.
        result.provider_continuation = None
        return self._managed_result(result, record, replayed=replayed)
