"""Fixture data for the acquisition-diligence demo.

FixtureProvider plays back a committed Architect plan and per-node
specialist outputs, keyed by node label so results are deterministic
even under parallel execution. All company details are fictional.
"""

from __future__ import annotations

from smythe.provider import CompletionResult, OfflineProvider

DILIGENCE_PLAN = {
    "topology": ["fork_join", "adversarial", "serial"],
    "nodes": [
        {
            "id": "financial",
            "label": (
                "Analyze Acme Corp's revenue model, margins, burn rate, "
                "and comparable valuations"
            ),
            "depends_on": [],
            "agent": {
                "name": "FinancialAnalyst",
                "persona": (
                    "You are a buy-side financial analyst. You quantify "
                    "everything and flag what the numbers don't support."
                ),
                "capabilities": ["financial-analysis"],
            },
        },
        {
            "id": "technical",
            "label": (
                "Assess Acme Corp's IP portfolio, tech debt signals, "
                "and key-person dependencies"
            ),
            "depends_on": [],
            "agent": {
                "name": "TechDiligenceAgent",
                "persona": (
                    "You are a technical due-diligence lead. You evaluate "
                    "IP defensibility, architecture risk, and team risk."
                ),
                "capabilities": ["technical-diligence"],
            },
        },
        {
            "id": "regulatory",
            "label": (
                "Review Acme Corp's SEC filings, antitrust exposure, "
                "and pending litigation"
            ),
            "depends_on": [],
            "agent": {
                "name": "RegulatoryAgent",
                "persona": (
                    "You are a regulatory counsel. You surface filing, "
                    "antitrust, and litigation risk without overstating it."
                ),
                "capabilities": ["regulatory-review"],
            },
        },
        {
            "id": "draft",
            "label": "Merge the specialist findings into a draft diligence report",
            "depends_on": ["financial", "technical", "regulatory"],
            "agent": {
                "name": "DiligenceEditor",
                "persona": (
                    "You merge specialist findings into one coherent draft, "
                    "preserving every material claim and its source."
                ),
                "capabilities": ["synthesis"],
            },
        },
        {
            "id": "redteam",
            "label": (
                "Challenge every bullish claim in the draft report; "
                "stress-test projections and surface contradictions"
            ),
            "depends_on": ["draft"],
            "metadata": {"role": "adversarial"},
            "agent": {
                "name": "RedTeamAgent",
                "persona": (
                    "You are the red team. Your job is to break the thesis: "
                    "attack assumptions, find contradictions, price the downside."
                ),
                "capabilities": ["critique"],
            },
        },
        {
            "id": "memo",
            "label": (
                "Produce the final structured memo: summary, findings, risks, "
                "recommendation - incorporating the red-team findings"
            ),
            "depends_on": ["draft", "redteam"],
            "agent": {
                "name": "MemoAgent",
                "persona": (
                    "You write decision memos for an investment committee: "
                    "structured, sourced, and explicit about conditions."
                ),
                "capabilities": ["writing"],
            },
        },
    ],
}

_FINANCIAL = """\
FINANCIAL FINDINGS (fictional fixture data)

- Revenue: $48M ARR, 22% YoY growth (down from 41% the prior year).
- Gross margin 71%; net revenue retention 104%.
- Burn $1.1M/month against $21M cash - roughly 19 months of runway.
- Customer concentration: top 3 customers are 38% of ARR, all on
  2-year contracts with renewal windows inside the deal timeline.
- Comparable transactions in the segment closed at 4-6x ARR;
  at the rumored $260M ask, Acme prices at 5.4x - the top of the band
  despite decelerating growth."""

_TECHNICAL = """\
TECHNICAL FINDINGS (fictional fixture data)

- Core IP: the adaptive routing engine - 2 granted patents, 1 pending.
  This is the asset that justifies a strategic premium.
- Architecture: aging monolith plus 2 extracted services; tech debt is
  moderate and paydown is budgeted, not aspirational.
- Key-person risk: 2 staff engineers author ~70% of core-engine commits;
  neither has retention terms surviving an acquisition.
- Security posture: SOC 2 Type II current, no open critical findings."""

_REGULATORY = """\
REGULATORY FINDINGS (fictional fixture data)

- SEC: no enforcement history; S-1 was withdrawn in 2024 for market
  reasons, not disclosure problems.
- Litigation: Novak Systems v. Acme (D. Del., pending) claims the
  routing engine infringes one Novak patent. Motion to dismiss denied;
  discovery underway.
- Antitrust: combined segment share post-close is ~18% - below agency
  screening thresholds, but the deal size triggers an HSR filing with a
  30-day initial waiting period."""

_DRAFT = """\
DRAFT DILIGENCE REPORT

Acme Corp presents as a strategic asset acquisition at a premium price.
The financial profile ($48M ARR, 71% gross margin, 19 months runway) is
sound but decelerating, and the 5.4x ARR ask prices in growth the
current trajectory does not show. The value driver is the patented
adaptive routing engine; the technical review rates it defensible but
concentrated in two engineers with no post-close retention terms. The
regulatory picture is clean except for the Novak patent suit, which
targets the same routing engine that anchors the deal thesis, and a
routine HSR waiting period. Draft view: attractive asset, wrong price,
two unpriced risks (key-person, litigation)."""

_REDTEAM = """\
RED-TEAM FINDINGS

1. The draft calls the product "sticky" while reporting 104% NRR -
   below the 110%+ bar for healthy B2B SaaS. Stickiness is asserted,
   not evidenced. Contradiction stands.
2. Growth deceleration (41% -> 22%) compounds customer concentration:
   if one top-3 account churns in a renewal window, ARR growth goes
   negative. The draft treats these as separate risks; they are one.
3. The Novak suit does not merely add legal cost - it encumbers the
   exact IP justifying the premium. A licensing outcome caps upside;
   an injunction breaks the thesis. The draft underweights this.
4. The 5.4x multiple assumes the routing engine transfers intact, but
   the two engineers who built it can walk at close. Without signed
   retention, the buyer pays a strategic premium for an unsecured asset."""

_MEMO = """\
ACQUISITION DILIGENCE MEMO - ACME CORP

SUMMARY
Acme Corp is a genuine strategic asset - a patented adaptive routing
engine inside a $48M ARR business with 71% gross margins - offered at
5.4x ARR, the top of the comparable band. The thesis survives red-team
review only with conditions attached; the asking price does not.

FINDINGS
- Financials are sound but decelerating: 22% growth (down from 41%),
  104% NRR, 19 months runway, top-3 customer concentration of 38%.
- The routing engine (2 granted patents, 1 pending) is the value
  driver; two engineers author ~70% of its code with no retention terms.
- Regulatory exposure is limited to a routine HSR filing and the Novak
  patent suit - which targets the routing engine itself.

RISKS (red-team confirmed)
1. Concentration x deceleration: one top-3 churn turns growth negative.
2. Novak litigation encumbers the premium-justifying IP; injunction
   scenario breaks the thesis.
3. Key-person flight risk leaves the core asset unsecured at close.
4. "Sticky product" claim is unsupported at 104% NRR.

RECOMMENDATION
Conditional GO. Proceed only if all three conditions are met:
(a) price at or below 4.2x ARR (~$200M) reflecting decelerating growth,
(b) signed retention packages for both core engineers as a closing
    condition, and
(c) an indemnity escrow sized to the Novak litigation range.
If the seller holds at 5.4x, walk away."""

FIXTURE_OUTPUTS = {
    node["label"]: text
    for node, text in zip(
        DILIGENCE_PLAN["nodes"],
        [_FINANCIAL, _TECHNICAL, _REGULATORY, _DRAFT, _REDTEAM, _MEMO],
    )
}


class FixtureProvider(OfflineProvider):
    """Offline provider keyed by node label, not call order.

    Parallel execution makes call order nondeterministic; matching on
    the prompt's first line (always the node label) keeps every run -
    and therefore the committed artifacts - byte-identical.
    """

    def __init__(self) -> None:
        super().__init__(plan=DILIGENCE_PLAN)

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        from smythe.prompts import PLANNING_SYSTEM_PROMPT

        if system == PLANNING_SYSTEM_PROMPT:
            return await super().complete(system, prompt, model)

        first_line = prompt.split("\n")[0]
        self.calls.append(first_line)
        text = FIXTURE_OUTPUTS.get(first_line)
        if text is None:
            raise AssertionError(f"No fixture output for node label: {first_line!r}")
        return CompletionResult(
            text=text,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
        )
