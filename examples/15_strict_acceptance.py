"""A passing draft does not make a defective deliverable acceptable.

    python examples/15_strict_acceptance.py

A gate judges the draft and passes it. A later step rewrites the draft into
the final brief and drops its Sources section. Without a policy, the run
returns that defective brief as its output. A strict acceptance policy
decides on exactly the text the run returns: a final validator, local code
that makes no provider call, finds the missing section, and the run returns
normally without accepting the brief. The rejected text stays inspectable.

Without a final validator, nothing could check the final step, so the same
strict run is refused before any provider call.

The run is offline and deterministic: OfflineProvider answers each step from
a script, even when an API key is set.

Docs: docs/verifier.md#strict-delivery-acceptance
"""

import sys

from smythe import AcceptancePolicy, OfflineProvider, Swarm, ValidationReport, ValidatorIdentity
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.validation import hard_finding

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")


class RequiredSection:
    """A final validator: the delivered text must contain a heading."""

    def __init__(self, heading: str) -> None:
        self.heading = heading
        self.identity = ValidatorIdentity.for_config("required-section", "1", {"heading": heading})

    def validate(self, subject):
        if self.heading in subject.text():
            return ValidationReport("output", validator=self.identity)
        missing = hard_finding("section_missing", f"no {self.heading!r} section")
        return ValidationReport("output", (missing,), validator=self.identity)


def brief_graph() -> ExecutionGraph:
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="Draft the launch brief with a Sources section"),
        Node(id="check", label="Check that every claim cites a source. Reply PASS or FAIL.",
             depends_on=["draft"], verifies="draft", max_regenerations=1),
        Node(id="final", label="Rewrite the draft as the final brief",
             depends_on=["draft", "check"]),
    ])


def scripted() -> OfflineProvider:
    """The draft is sound and passes its gate; the final step drops the sources."""
    return OfflineProvider(responses=[
        "## Launch brief\nThe charger ships in May [1].\n\n## Sources\n[1] Supplier schedule.",
        "PASS",
        "## Launch brief\nThe charger ships in May.",
    ])


def swarm(provider: OfflineProvider, acceptance: AcceptancePolicy | None = None) -> Swarm:
    return Swarm(provider=provider, model="demo-model", artifact_dir=None, acceptance=acceptance)


def shown(text: str) -> str:
    return " / ".join(line for line in text.splitlines() if line)


print("Running offline with smythe's built-in OfflineProvider (no keys, no cost).\n")

print("=== Without a policy ===")
result = swarm(scripted()).execute(brief_graph())
print(f"  output:     {shown(result.output)}")
print(f"  acceptance: {result.acceptance}")

print("\n=== Strict policy with a final validator ===")
policy = AcceptancePolicy(mode="strict", final_validators=[RequiredSection("## Sources")])
result = swarm(scripted(), policy).execute(brief_graph())
receipt = result.graph.nodes[1].metadata["verification_receipt"]
print(f"  gate on the draft: {receipt['outcome']}")
print(f"  decision:          {result.acceptance.decision.value}")
for reason in result.acceptance.reasons:
    print(f"  reason:            {reason}")
print(f"  accepted:          {result.accepted}")
print(f"  output:            {result.output!r}")
print(f"  rejected_output:   {shown(result.rejected_output)}")

print("\n=== Strict policy without a final validator ===")
provider = scripted()
try:
    swarm(provider, AcceptancePolicy(mode="strict")).execute(brief_graph())
except ValueError as exc:
    print(f"  refused before any provider call ({len(provider.calls)} calls):")
    print(f"  {exc}")
