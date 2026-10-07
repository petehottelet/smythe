"""Publish an accepted document to an external service under a verified approval.

    python examples/19_governed_publication.py

A durable run writes a release announcement, and a strict acceptance policy
accepts the published document's exact bytes. The run's journal then records
an external action that delivers exactly those bytes to a publishing service:
its intent, the one key every attempt sends, and the connector's declared
guarantees. Nothing is sent until an approval verified by the host's identity
service binds a principal to that one action and its exact intent, with a
ceiling and an expiry.

The first submission commits on the service, but its answer is lost, so the
action's outcome is unknown and `smythe runs inspect` exits with 12. The
runner never sends it again blindly. Reconciliation looks the key up, reads
the committed post back and checks its bytes, so the action is reconciled
with one effect on the service. Inspection shows the attempt, the approval,
the lookup and readback records and the action's nanoUSD balances, never the
assertion or the document.

Everything is offline and keyless. The publishing service is an in-memory
stand-in, the approval host signs assertions with a random local key, and a
deterministic local synthesizer writes the announcement, declared with
LocalOnly so the run's recipe records its identity and version. In a live
deployment your host supplies the connector and the approval verifier.

Docs: docs/actions.md
"""

from __future__ import annotations

import base64
from contextlib import redirect_stdout
import hashlib
import hmac
import io
import json
from pathlib import Path
import secrets
import sys
from tempfile import TemporaryDirectory
import time

from smythe import (
    AcceptancePolicy, LocalOnly, MarkdownRenderer, OfflineProvider, SimpleArchitect,
    SQLiteWorkflowStore, Swarm, Task, TextValidator,
)
from smythe.actions import (
    ActionIntent, ActionRunner, ActionTarget, ApprovalRejected, ConnectorAmbiguous,
    ConnectorProfile, ConnectorReceipt, ContentRef, ObservedObject, Principal, approve,
    inspect_actions, prepare_action,
)
from smythe.cli import main as smythe_cli
from smythe.workflow_artifact import current_artifact

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

HOUR_NS = 3600 * 10**9
ANNOUNCEMENT = """# Release 2.4

## What changed

- Exports finish in half the time on large projects.
- Shared folders keep their permissions when they are moved.

## Upgrade

Update from the settings page. Existing projects open without changes.
"""


class ReleaseNote:
    """Writes the announcement; deterministic local code standing in for a model's draft."""

    def synthesize(self, graph) -> str:
        return ANNOUNCEMENT


# ---------------------------------------------------------------------------
# A publishing service and its connector
# ---------------------------------------------------------------------------


class PostService:
    """An in-memory publishing API: it commits each key once and answers a repeat from that commit.

    ``lose_answer(key)`` commits the next submission of ``key`` and then
    loses its answer, as a dropped connection does. ``effects`` counts the
    posts actually written.
    """

    def __init__(self) -> None:
        self.posts: dict[str, dict] = {}
        self.keys: dict[tuple[str, str], dict] = {}
        self.submissions = 0
        self._lose: set[str] = set()

    @property
    def effects(self) -> int:
        return len(self.posts)

    def lose_answer(self, key: str) -> None:
        self._lose.add(key)

    def publish(self, key: str, tenant: str, collection: str, body: bytes, media_type: str) -> dict:
        self.submissions += 1
        if (tenant, key) in self.keys:
            return {**self.keys[tenant, key], "deduplicated": True}
        post_id = f"post-{len(self.posts) + 1}"
        self.posts[post_id] = {"tenant": tenant, "collection": collection, "body": body,
                               "media_type": media_type, "version": 1, "key": key}
        receipt = {"key": key, "tenant": tenant, "collection": collection, "post_id": post_id,
                   "version": 1, "cost_nanousd": 1_500_000, "deduplicated": False}
        self.keys[tenant, key] = receipt
        if key in self._lose:
            self._lose.discard(key)
            raise ConnectionError("the connection closed before the answer arrived")
        return receipt

    def find(self, tenant: str, key: str) -> dict | None:
        return self.keys.get((tenant, key))

    def get(self, tenant: str, post_id: str) -> dict | None:
        post = self.posts.get(post_id)
        return post if post is not None and post["tenant"] == tenant else None


def _receipt(answer: dict) -> ConnectorReceipt:
    return ConnectorReceipt(answer["key"], ActionTarget(answer["tenant"], answer["collection"], answer["post_id"]),
                            object_version=answer["version"], cost_nanousd=answer["cost_nanousd"],
                            deduplicated=answer["deduplicated"])


class PostConnector:
    """A ``WriteConnector``: one request per call, no retries of its own.

    Its profile declares what the service guarantees: it deduplicates keys
    without expiry, finds a submission by key, reads a post's bytes back and
    shows a commit at once. Ceilings are whole nanoUSD.
    """

    profile = ConnectorProfile(
        "example-posts", "1", ["publish"], deduplicates=True, lookup_by_key=True,
        lookup_by_object=True, visibility_bound_s=0, write_ceiling_nanousd=2_000_000,
        read_ceiling_nanousd=10_000, max_attempts=2, readback_reads=2, timeout_s=10,
    )

    def __init__(self, service: PostService) -> None:
        self.service = service

    def submit(self, key, intent, payload, timeout_s):
        target = intent.target
        try:
            answer = self.service.publish(key, target.tenant, target.collection, payload,
                                          intent.content.media_type)
        except ConnectionError as error:
            raise ConnectorAmbiguous(str(error)) from error  # the post may have been written
        return _receipt(answer)

    def lookup(self, key, target, timeout_s):
        answer = self.service.find(target.tenant, key)
        return None if answer is None else _receipt(answer)

    def read(self, target, timeout_s):
        post = self.service.get(target.tenant, target.object_id)
        if post is None:
            return None
        return ObservedObject(ActionTarget(post["tenant"], post["collection"], target.object_id),
                              post["body"], post["media_type"], version=post["version"], key=post["key"])


# ---------------------------------------------------------------------------
# The host's approval verifier
# ---------------------------------------------------------------------------


class ApprovalHost:
    """A stand-in for your identity service: signed approvals of one action's subject digest, with expiry.

    ``issue`` signs an assertion that ``subject`` of ``tenant`` approves
    exactly one digest, such as ``PreparedAction.approval_sha256``, until
    ``expires_at_ns``; ``verify`` accepts only this host's signature for that
    digest before the expiry.
    """

    identity = "example-approvals/1"
    issuer = "example-identity-host"

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)

    def _sign(self, claims: str) -> str:
        return hmac.new(self._key, claims.encode("ascii"), hashlib.sha256).hexdigest()

    def issue(self, subject_sha256: str, *, subject: str, tenant: str, expires_at_ns: int) -> str:
        claims = base64.urlsafe_b64encode(json.dumps({
            "subject_sha256": subject_sha256, "subject": subject, "tenant": tenant,
            "expires_at_ns": expires_at_ns, "nonce": secrets.token_hex(16)}).encode()).decode("ascii")
        return f"{claims}.{self._sign(claims)}"

    def verify(self, assertion: str, subject_sha256: str, now_ns: int) -> Principal:
        claims, _, signature = assertion.partition(".")
        if not hmac.compare_digest(signature, self._sign(claims)):
            raise ApprovalRejected("The assertion is not signed by this host")
        data = json.loads(base64.urlsafe_b64decode(claims))
        if data["subject_sha256"] != subject_sha256:
            raise ApprovalRejected("The assertion approves something else")
        if now_ns >= data["expires_at_ns"]:
            raise ApprovalRejected("The assertion has expired")
        return Principal(self.issuer, data["subject"], data["tenant"])

    def still_authorized(self, principal: Principal, intent: ActionIntent, now_ns: int) -> bool:
        return principal.issuer == self.issuer and principal.tenant == intent.target.tenant


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def inspect_exit_code(journal: Path, run_id: str) -> int:
    """What ``smythe runs --journal JOURNAL inspect RUN_ID`` exits with, its report set aside."""
    with redirect_stdout(io.StringIO()):
        return smythe_cli(["runs", "--journal", str(journal), "inspect", run_id])


def main() -> None:
    print("Running offline: OfflineProvider, an in-memory publishing service and a local "
          "approval host; no keys.\n")
    service, host = PostService(), ApprovalHost()
    connector = PostConnector(service)
    with TemporaryDirectory(prefix="smythe-") as work:
        journal = Path(work) / "runs.db"
        with SQLiteWorkflowStore(journal) as store:
            swarm = Swarm(
                model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                run_store=store, max_budget_usd=0.10,
                synthesizer=LocalOnly(ReleaseNote, "release-note", "1", role="synthesizer"),
                renderer=MarkdownRenderer(), artifact_dir=Path(work) / "artifacts",
                acceptance=AcceptancePolicy(mode="strict", final_validators=[
                    TextValidator({"sections": ["What changed", "Upgrade"]})]),
            )

            print("=== 1. An accepted document ===")
            result = swarm.execute(Task("Announce release 2.4 to customers"))
            run_id = result.execution_id
            document = current_artifact(store, run_id).publication.path.read_bytes()
            print(f"  run {run_id}: {result.acceptance.decision.value}")
            print(f"  document: {len(document)} bytes, sha256 {hashlib.sha256(document).hexdigest()[:16]}\n")

            print("=== 2. Prepare the publication ===")
            action = prepare_action(store, run_id, slot="announcement", profile=connector.profile, intent=(
                ActionIntent("publish", ActionTarget("acme", "news"), parameters={"title": "Release 2.4"},
                             content=ContentRef.for_bytes(document, "text/markdown"),
                             connector=connector.profile.name,
                             connector_version=connector.profile.version)))
            (item,) = inspect_actions(store, run_id)["items"]
            print(f"  action {action.slot}/{action.sequence}: key {action.action_key[:16]}, "
                  f"intent {action.intent.digest()[:16]}, approval subject {action.approval_sha256[:16]}")
            print(f"  ceiling {action.ceiling_nanousd} nanoUSD; state {action.state}; "
                  f"next {item['next_action']}\n")

            print("=== 3. Approve ===")
            expires = time.time_ns() + HOUR_NS
            # The intent digest alone is not an approval subject: it would fit an identical intent elsewhere.
            other = host.issue(action.intent.digest(), subject="editor-1", tenant="acme", expires_at_ns=expires)
            try:
                approve(store, action.action_id, other, host, ceiling_nanousd=action.ceiling_nanousd,
                        expires_at_ns=expires)
            except ApprovalRejected as refused:
                print(f"  an assertion for the bare intent digest is refused: {refused}")
            assertion = host.issue(action.approval_sha256, subject="editor-1", tenant="acme",
                                   expires_at_ns=expires)
            approval = approve(store, action.action_id, assertion, host,
                               ceiling_nanousd=action.ceiling_nanousd, expires_at_ns=expires)
            principal = approval.principal
            print(f"  approved by {principal.subject} of {principal.tenant} ({principal.issuer}), "
                  f"up to {approval.ceiling_nanousd} nanoUSD for one hour\n")

            print("=== 4. Dispatch; the service commits but the answer is lost ===")
            service.lose_answer(action.action_key)
            runner = ActionRunner(store, connector, host)
            progress = runner.dispatch(run_id, action.action_id)
            print(f"  state {progress.state}; next {progress.next_action}")
            print(f"  service: {service.submissions} submission, {service.effects} post written")
            unresolved = inspect_exit_code(journal, run_id)
            print(f"  smythe runs inspect exits with {unresolved}: "
                  "accepted output with an unresolved action\n")
            assert unresolved == 12

            print("=== 5. Reconcile ===")
            progress = runner.reconcile(run_id, action.action_id)
            print(f"  state {progress.state}; readback {progress.report.outcome.value} "
                  f"({progress.report.validator.name} {progress.report.validator.version})")
            print(f"  service: {service.submissions} submission, {service.effects} post written; "
                  "nothing was sent again\n")
            assert (progress.state, service.submissions, service.effects) == ("reconciled", 1, 1)

            print("=== 6. Inspect ===")
            block = inspect_actions(store, run_id)
            (item,) = block["items"]
            (attempt,) = item["attempts"]
            (shown,) = item["approvals"]
            print(f"  attempt {attempt['number']}: {attempt['state']}, cost {attempt['cost_nanousd']} nanoUSD "
                  "confirmed by the receipt the lookup found")
            print(f"  approval by {shown['principal']['subject']} ({shown['verifier']['identity']}); "
                  f"the {item['state']} action admits no other attempt")
            print(f"  records: {', '.join(record['kind'] for record in item['records'])}")
            print(f"  balances: {block['confirmed_nanousd']} confirmed, {block['reserved_nanousd']} reserved, "
                  f"{block['unknown_nanousd']} unknown nanoUSD; unresolved {block['unresolved']}")
            shown_text = json.dumps(block)
            assert assertion not in shown_text and ANNOUNCEMENT.splitlines()[4] not in shown_text

        print("\n=== 7. smythe runs --journal runs.db inspect RUN_ID ===")
        code = smythe_cli(["runs", "--journal", str(journal), "inspect", run_id])
        print(f"(exit code {code})")
        assert code == 0


if __name__ == "__main__":
    main()
