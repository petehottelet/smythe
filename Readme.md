# smythe

**An open-source framework for task-based agent swarms with dynamic parallelization, routing, and execution topology.**

Most agent frameworks make you decide upfront how your agents will work together. Smythe doesn't. It treats the execution graph itself as a generated artifact — letting an Architect decide whether a task should run serially, in parallel, or adversarially, and when to recursively decompose work into nested subgraphs, based on the nature of the work and what's been learned from past runs.

---

## The Problem

Today's agent frameworks fall into two camps:

**Personal assistant daemons** (like OpenClaw) give you one persistent agent with many skills. Great for "do this thing for me." Not designed for complex tasks that benefit from multiple specialized agents working in coordination.

**Pipeline frameworks** (like LangGraph, CrewAI, AutoGPT) let you hardcode a topology — chain these agents together in this order. You, the developer, decide how the work gets split up. The framework just executes your decision.

Neither camp asks the more interesting question: *what if the framework could decide how to execute a task based on the task itself?*

---

## What Smythe Does Differently

**1. Execution graphs are generated, not hardcoded.**
Each execution plan is represented as a Directed Acyclic Graph (DAG). An Architect — informed by the task's structure and historical execution data — decides the topology: serial, fork-join, broadcast-reduce, or adversarial. For decomposition, the Architect can recursively spawn nested subgraphs while keeping each executable graph acyclic. You can override it, but you don't have to specify it.

**2. Agents have persistent identities.**
Each agent carries a capability profile, a persona, episodic memory, and a performance history across task types. Over time, the framework learns which agents are best suited to which work and routes accordingly. You're building a team, not a worker pool.

**3. Synthesis is a first-class tier.**
Merging parallel outputs without losing coherence is hard and almost always an afterthought. Smythe treats synthesis as a dedicated architectural layer with explicit strategies per output type — not a final prompt that hopes for the best.

**4. The Architect learns from cold starts.**
The system ships with robust heuristic defaults (e.g., "Research" tasks default to fork-join). As tasks complete, execution history feeds back into the Architect. A task that was over-parallelized, or where synthesis failed, teaches the Architect how to optimize the next topology.

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
- **Human oversight is built in.** You can inspect what the Architect decided and why before or during execution, and add approval gates for sensitive workflows.

---

## Architecture

```
Task → Architect → ExecutionGraph (DAG) → Executor → Synthesizer → SwarmResult
          │                                   │             │
      WhiteRabbit                          Sentinel        Tracer
      (optional)
```

### Architect tiers

Smythe ships with three Architect strategies, plus optional routing via the WhiteRabbit:

| Tier | Class | Description |
|---|---|---|
| **Deterministic** | `DeterministicArchitect` | Pure Python DAG construction. Zero LLM cost, zero latency. Subclass and override `plan()`. |
| **Constrained** | `ConstrainedArchitect` | LLM selects from a menu of pre-built `SubGraphTemplate`s. Dramatically smaller failure space than fully autonomous planning. |
| **Autonomous** | `LLMArchitect` | LLM builds bespoke DAGs from scratch. Maximum flexibility. Context-preserving retries on malformed output. |

Pass any Architect explicitly via `Swarm(architect=...)`, or use the `WhiteRabbit` for classifier-based routing:

```python
from smythe import Swarm, WhiteRabbit, SimpleArchitect, LLMArchitect

router = WhiteRabbit(
    deterministic={"etl-pipeline": MyETLArchitect()},
    constrained=my_constrained_architect,
    autonomous=LLMArchitect(provider=my_provider),
    classifier_provider=my_provider,
)
swarm = Swarm(router=router)
```

When no classifier provider is set, the WhiteRabbit falls back to the autonomous Architect.

### Node failure policies

Each node can declare how failures are handled:

| Policy | Behavior |
|---|---|
| `HALT` (default) | Propagate the exception; stop execution. |
| `SKIP` | Mark the node as `SKIPPED` and let dependents continue. |
| `RETRY` | Retry up to `max_retries` times before failing. |

Set policies in YAML or when constructing nodes programmatically:

```yaml
nodes:
  - id: flaky-api
    label: "Call external service"
    failure_policy: retry
    max_retries: 3
  - id: optional-enrichment
    label: "Nice-to-have step"
    failure_policy: skip
    depends_on: [flaky-api]
```

### Synthesis strategies

The synthesizer merges parallel execution outputs into a single result:

| Strategy | Description |
|---|---|
| `CONCATENATE` (default) | Join results with newlines. Zero cost. |
| `LLM_MERGE` | Send all results to an LLM for intelligent merging. Budget-tracked and traced. |
| `STRUCTURED` | Parse each result as JSON and shallow-merge into a single object. |

```python
from smythe import Swarm, SynthesisStrategy
from smythe.synthesizer import Synthesizer

swarm = Swarm(
    synthesizer=Synthesizer(
        strategy=SynthesisStrategy.LLM_MERGE,
        provider=my_provider,
        model="claude-mythos",
    ),
)
```

### Capability-aware agent assignment

Nodes can declare `required_capabilities`. The registry matches agents whose capabilities are a superset of the required set, preferring the tightest match with alphabetical tie-breaking:

```python
from smythe.agent import Agent, AgentProfile
from smythe.graph import Node

agent = Agent(profile=AgentProfile(
    name="researcher",
    capabilities=["research", "summarize"],
))
registry.register(agent)

node = Node(label="Research task", required_capabilities=["research"])
registry.assign(graph)  # assigns the researcher agent
```

### Skill-based capability profiles

Agent capabilities can be derived from external skill systems like [OpenClaw AgentSkills](https://docs.openclaw.ai/skills/) instead of (or in addition to) static tags. The registry hydrates each agent's capabilities at assignment time, caches the results, and falls back to static capabilities if the skill provider is unavailable.

```python
from smythe import Swarm
from smythe.registry import Registry
from smythe.openclaw_adapter import OpenClawSkillProvider
from smythe.skills import DefaultCapabilityMapper, CapabilityHydrationMode

registry = Registry(
    skill_provider=OpenClawSkillProvider(),
    capability_mapper=DefaultCapabilityMapper(
        aliases={"search": "research", "summarize-text": "summarize"}
    ),
    hydration_mode=CapabilityHydrationMode.MERGE,
    capability_cache_ttl_seconds=300,
)

swarm = Swarm(registry=registry, provider=my_provider)
```

Hydration modes:

| Mode | Behavior |
|---|---|
| `MERGE` (default) | Union of static profile capabilities and skill-derived capabilities. |
| `REPLACE` | Skill-derived capabilities only; static profile is ignored. |
| `STATIC_ONLY` | Ignore the skill provider entirely. |

Cache entries expire after the configured TTL. Force a refresh with `registry.refresh_agent_capabilities(agent_id)` or `registry.refresh_all_capabilities()`.

### Budget enforcement

Set a USD spending cap that is enforced at every execution step. Parallel execution uses a reservation protocol to prevent concurrent nodes from collectively exceeding the budget:

```python
swarm = Swarm(max_budget_usd=0.50)
result = swarm.execute(task)
print(result.total_cost_usd)  # actual cost
```

### YAML-defined DAGs

Define execution graphs declaratively. Load and execute without writing Python:

```yaml
topology: fork_join
nodes:
  - id: research
    label: "Research the topic"
    agent:
      name: Researcher
      persona: "You are a thorough researcher."
      capabilities: [research]
  - id: summarize
    label: "Summarize findings"
    depends_on: [research]
    failure_policy: retry
    max_retries: 2
```

```python
swarm = Swarm.from_yaml("pipeline.yaml", provider=my_provider)
result = swarm.execute()
```

### Observability

Every node execution emits structured trace spans. The Architect's `PlannerMemory` persists execution outcomes as JSONL for learning-informed future planning.

---

## Installation

```bash
pip install -e .
```

Optional extras for LLM providers and skill integration:

```bash
pip install -e ".[anthropic]"    # Anthropic Claude models
pip install -e ".[openai]"       # OpenAI GPT models
pip install -e ".[openclaw]"     # OpenClaw AgentSkills integration
pip install -e ".[all]"          # all of the above
pip install -e ".[dev]"          # pytest, ruff, dev tooling
```

Requires Python 3.11+. Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for the respective providers.

---

## Current Status

The core framework is implemented and tested. **189 tests passing.**

**What's shipped:**
- Three-tier Architect hierarchy (Deterministic, Constrained, Autonomous LLM)
- Classifier-based WhiteRabbit router with deterministic fallback
- Serial and async parallel executors with shared base class
- Node failure policies (HALT, SKIP, RETRY)
- Capability-aware agent assignment with deterministic tie-breaking
- Skill-based capability hydration (OpenClaw AgentSkills adapter) with caching and fallback
- Synthesis strategies (CONCATENATE, LLM_MERGE, STRUCTURED) with budget/trace accounting
- Budget enforcement with reservation protocol for parallel safety
- YAML-defined DAGs with failure policy and capabilities support
- Context-preserving Architect retries
- Persistent execution memory (JSONL) for learning Architect
- Provider abstraction (Anthropic, OpenAI) with defensive response parsing
- Structured observability traces

**What's next:**
- Recursive subgraph decomposition
- Approval gates for human-in-the-loop workflows
- Performance history-based agent routing
- Additional providers (Gemini, local models)

---

## License

MIT
