# smythe

**An open-source framework for task-based agent swarms with dynamic parallelization, routing, and execution topology.**

Most agent frameworks make you decide upfront how your agents will work together. Smythe doesn't. It treats the execution graph itself as a generated artifact — letting a planner decide whether a task should run serially, in parallel, or adversarially, and when to recursively decompose work into nested subgraphs, based on the nature of the work and what's been learned from past runs.

---

## The Problem

Today's agent frameworks fall into two camps:

**Personal assistant daemons** (like OpenClaw) give you one persistent agent with many skills. Great for "do this thing for me." Not designed for complex tasks that benefit from multiple specialized agents working in coordination.

**Pipeline frameworks** (like LangGraph, CrewAI, AutoGPT) let you hardcode a topology — chain these agents together in this order. You, the developer, decide how the work gets split up. The framework just executes your decision.

Neither camp asks the more interesting question: *what if the framework could decide how to execute a task based on the task itself?*

---

## What Smythe Does Differently

**1. Execution graphs are generated, not hardcoded.**
Each execution plan is represented as a Directed Acyclic Graph (DAG). A planner — informed by the task's structure and historical execution data — decides the topology: serial, fork-join, broadcast-reduce, or adversarial. For decomposition, the planner can recursively spawn nested subgraphs while keeping each executable graph acyclic. You can override it, but you don't have to specify it.

**2. Agents have persistent identities.**
Each agent carries a capability profile, a persona, episodic memory, and a performance history across task types. Over time, the framework learns which agents are best suited to which work and routes accordingly. You're building a team, not a worker pool.

**3. Synthesis is a first-class tier.**
Merging parallel outputs without losing coherence is hard and almost always an afterthought. Smythe treats synthesis as a dedicated architectural layer with explicit strategies per output type — not a final prompt that hopes for the best.

**4. The planner learns from cold starts.**
The system ships with robust heuristic defaults (e.g., "Research" tasks default to fork-join). As tasks complete, execution history feeds back into the planner. A task that was over-parallelized, or where synthesis failed, teaches the planner how to optimize the next topology.

---

## What It Looks Like

You define the goal; the framework negotiates the path.

### Everyday task — fork-join

```python
from smythe import Swarm, Task

swarm = Swarm(max_budget_usd=0.50, model="claude-mythos")

task = Task(
    goal=(
        "Plan a birthday party for this Friday. I want a strawberry chiffon "
        "cake, a venue that works for ~20 people, and invitations sent out ASAP."
    ),
    constraints=[
        "Budget under $500",
        "Must be within 15 miles of Oakland, CA",
    ],
)

plan = swarm.plan(task)
print(plan)
# TaskGraph(topology="fork-join → serial")
# ├─ fork (parallel):
# │   ├─ BakeryAgent: find bakeries that do strawberry chiffon,
# │   │   check Friday availability, compare pricing
# │   ├─ VenueAgent: find venues for ~20 near Oakland,
# │   │   Friday evening, under budget
# │   └─ InspirationAgent: suggest party themes, decor ideas,
# │       playlist recs based on constraints
# ├─ join: rank options by price/availability/proximity
# └─ serial (depends on join):
#     └─ InvitationAgent: draft invitations with confirmed
#         venue + time, format for email/text
#
# Estimated cost: $0.22 | Depth: 3 | Agents: 4

result = swarm.execute(plan)
```

### Creative task — broadcast-reduce

```python
swarm = Swarm(max_budget_usd=1.50, model="gemini-3-pro-image-preview")

task = Task(
    goal=(
        "Generate a full visual asset package for the launch of 'Solara', "
        "a portable solar-powered phone charger. Every asset must share a "
        "cohesive visual identity — same palette, typography, and tone."
    ),
    constraints=[
        "Brand palette: warm amber, matte black, off-white",
        "Style: clean product photography, natural light, lifestyle context",
        "Assets needed: hero image, 3 social posts, email header, "
        "app store screenshot, OG preview card, print ad",
    ],
)

plan = swarm.plan(task)
print(plan)
# TaskGraph(topology="serial → broadcast-reduce")
# ├─ serial:
# │   └─ StyleDirector: establish visual brief — palette, typography,
# │       mood references, negative-space rules
# ├─ broadcast (parallel, 8 agents):
# │   ├─ ImageAgent-1: hero image — 2400×1200 PNG, product on sunlit trail
# │   ├─ ImageAgent-2: Instagram post — 1080×1080 JPG, lifestyle flat-lay
# │   ├─ ImageAgent-3: X/Twitter banner — 1500×500 JPG, product detail
# │   ├─ ImageAgent-4: Story/Reel card — 1080×1920 PNG, vertical lifestyle
# │   ├─ ImageAgent-5: email header — 600×200 PNG, newsletter announcement
# │   ├─ ImageAgent-6: App Store screenshot — 1290×2796 PNG, feature callout
# │   ├─ ImageAgent-7: OG preview card — 1200×630 PNG, link-share thumbnail
# │   └─ ImageAgent-8: print ad — 8.5×11" 300dpi, magazine full-page bleed
# └─ reduce:
#     └─ ArtDirector: curate for brand consistency, flag off-palette
#         outputs, assemble final asset package with metadata
#
# Estimated cost: $1.12 | Depth: 3 | Agents: 10

result = swarm.execute(plan)
```

### Enterprise task — fork-join with adversarial review

```python
swarm = Swarm(max_budget_usd=2.00, model="claude-mythos")

task = Task(
    goal=(
        "Evaluate whether Acme Corp is a viable acquisition target. "
        "Analyze their financials, technical IP, and regulatory exposure, "
        "then produce a diligence memo with a go/no-go recommendation."
    ),
    constraints=[
        "Red-team every bullish claim before it reaches the memo",
        "Flag any SEC or antitrust risk factors",
        "Final output must be structured: summary, findings, risks, recommendation",
    ],
)

plan = swarm.plan(task)
print(plan)
# TaskGraph(topology="fork-join → adversarial → serial")
# ├─ fork (parallel):
# │   ├─ FinancialAnalyst: revenue model, margins, burn rate,
# │   │   comparable valuations
# │   ├─ TechDiligenceAgent: assess IP portfolio, tech debt signals,
# │   │   key-person dependencies
# │   └─ RegulatoryAgent: SEC filing review, antitrust screen,
# │       pending litigation scan
# ├─ join: merge findings into draft diligence report
# ├─ adversarial:
# │   └─ RedTeamAgent: challenge assumptions, stress-test projections,
# │       surface contradictions across sections
# └─ serial (depends on adversarial):
#     └─ MemoAgent: produce final structured memo incorporating
#         red-team findings and risk flags
#
# Estimated cost: $1.74 | Depth: 4 | Agents: 5

result = swarm.execute(plan)
```

---

## Principles

- **Deterministic guardrails.** Dynamic doesn't mean "out of control." Every execution is constrained by circuit breakers: max depth, token budgets, and cost-aware scheduling.
- **Composable over monolithic.** Use just the DAG engine, just the agent registry, or the full stack.
- **Provider-agnostic.** Abstract over any LLM. Bring your own keys.
- **Observable by default.** Every node execution emits structured traces. The feedback loop is the product.
- **Human oversight is built in.** You can inspect what the planner decided and why before or during execution, and add approval gates for sensitive workflows.

---

## Current Status

Early architecture phase. The core concepts are defined; implementation is beginning.

**What exists:**
- Conceptual architecture and task graph schema (in progress)

**What's next (v0.1):**
- YAML-defined task DAGs, serial/parallel dispatch, per-node agent personas, single LLM provider

---

## License

MIT
