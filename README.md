# smythe

[![PyPI](https://img.shields.io/pypi/v/smythe)](https://pypi.org/project/smythe/)
[![CI](https://github.com/petehottelet/smythe/actions/workflows/ci.yml/badge.svg)](https://github.com/petehottelet/smythe/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**An open-source framework for task-based agent swarms with dynamic parallelization, routing, and execution topology.**

Most agent frameworks make you decide upfront how your agents will work together. Smythe doesn't. It treats the execution graph itself as a generated artifact — letting an Architect decide whether a task should run serially, in parallel, or adversarially, based on the nature of the work and what's been learned from past runs.

## Install

```bash
pip install smythe
```

Python 3.11+. Provider extras (`smythe[anthropic]`, `[openai]`, `[gemini]`, `[mcp]`, `[all]`) are covered under [Installation](#installation).

## 60-second quickstart

With an API key set (`ANTHROPIC_API_KEY` here), you hand the Swarm a goal; the Architect designs the execution graph, and you inspect it before anything runs:

```python
from smythe import Swarm, Task

swarm = Swarm(model="claude-opus-4-8", max_budget_usd=0.50)

task = Task(
    goal=(
        "Produce a competitive brief on portable solar phone chargers: "
        "market landscape, top competitors, and a one-page summary."
    ),
    constraints=["Keep the final brief under 400 words"],
)

graph = swarm.plan(task)   # the generated DAG — inspect it (or reject it)
print(graph)

result = swarm.execute(graph)
print(result.output)
print(f"cost: ${result.total_cost_usd:.4f}")
```

No API key? Clone the repo and every example — including the flagship demo below — runs offline against deterministic fixtures, for free.

## See it run

The flagship demo hands Smythe one goal — *evaluate whether MetaCortex Corp is a viable acquisition target* — and the Architect answers with a topology, not a transcript:

```bash
python examples/acquisition_diligence/run.py
```

```text
=== The Architect's plan ===
TaskGraph(topology="fork-join → adversarial → serial")
├─ fork (parallel):
│   ├─ FinancialAnalyst: Analyze MetaCortex Corp's revenue model, margins, burn rate, and comparable valuations
│   ├─ TechDiligenceAgent: Assess MetaCortex Corp's IP portfolio, tech debt signals, and key-person dependencies
│   └─ RegulatoryAgent: Review MetaCortex Corp's SEC filings, antitrust exposure, and pending litigation
├─ join: DiligenceEditor: Merge the specialist findings into a draft diligence report
├─ adversarial: RedTeamAgent: Challenge every bullish claim in the draft report; stress-test projections and surface contradictions
└─ serial (depends on DiligenceEditor, RedTeamAgent): MemoAgent: Produce the final structured memo
#
# Estimated cost: $0.04 | Depth: 3 | Agents: 6
```

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Roboto Mono, monospace","fontSize":"13px","primaryColor":"#f9fafb","primaryTextColor":"#1f2937","primaryBorderColor":"#9ca3af","lineColor":"#9ca3af"},"flowchart":{"curve":"step","nodeSpacing":50,"rankSpacing":60}}}%%
flowchart TD
    financial("<b>FinancialAnalyst</b><br/>revenue model, margins, burn, comps")
    technical("<b>TechDiligenceAgent</b><br/>IP portfolio, tech debt, key-person risk")
    regulatory("<b>RegulatoryAgent</b><br/>SEC filings, antitrust, litigation")
    draft("<b>DiligenceEditor</b><br/>merge findings into draft report")
    redteam("<b>RedTeamAgent</b><br/>challenge every bullish claim")
    memo("<b>MemoAgent</b><br/>final memo: summary, findings, risks, recommendation")
    financial --> draft
    technical --> draft
    regulatory --> draft
    draft --> redteam
    draft --> memo
    redteam --> memo
    classDef specialist fill:#f9fafb,stroke:#9ca3af,stroke-width:1px,color:#1f2937
    classDef editor fill:#f3f4f6,stroke:#6b7280,stroke-width:1px,color:#1f2937
    classDef adversarial fill:#e5e7eb,stroke:#6b7280,stroke-width:1px,color:#1f2937
    classDef deliverable fill:#374151,stroke:#1f2937,stroke-width:1px,color:#f9fafb
    class financial,technical,regulatory specialist
    class draft editor
    class redteam adversarial
    class memo deliverable
    linkStyle default stroke:#9ca3af,stroke-width:1px
```

Three specialists run in parallel under a budget cap, a red team attacks the draft, and the memo node turns the surviving claims into a conditional go/no-go recommendation. The expected graph, trace, and memo are committed in [examples/acquisition_diligence/expected/](examples/acquisition_diligence/expected/) — a test regenerates them on every CI run, so what you see there is what the code does. Full walkthrough: [examples/acquisition_diligence/](examples/acquisition_diligence/).

---

## The Problem

Today's agent frameworks fall into two camps:

**Personal assistant daemons** (like [OpenClaw](https://github.com/openclaw/openclaw)) give you one persistent agent with many skills. Great for "do this thing for me." Not designed for complex tasks that benefit from multiple specialized agents working in coordination.

**Pipeline frameworks** (like LangGraph, CrewAI, AutoGPT) let you hardcode a topology — chain these agents together in this order. You, the developer, decide how the work gets split up. The framework just executes your decision.

Neither camp asks the more interesting question: *what if the framework could decide how to execute a task based on the task itself?*

---

## What Smythe Does Differently

**1. Execution graphs are generated, not hardcoded.**
Each execution plan is represented as a Directed Acyclic Graph (DAG). An Architect — informed by the task's structure and historical execution data — decides the topology: serial, fork-join, broadcast-reduce, or adversarial. You can override it, but you don't have to specify it. (Recursive decomposition into nested subgraphs is on the roadmap — see "What's next.")

**2. Agents have persistent identities.**
Each agent carries a capability profile and a persona, and the registry matches agents to work by capability. You're building a team, not a worker pool. (Per-agent performance history that influences routing is on the roadmap.)

**3. Synthesis is a first-class tier.**
Merging parallel outputs without losing coherence is hard and almost always an afterthought. Smythe treats synthesis as a dedicated architectural layer with explicit strategies per output type — not a final prompt that hopes for the best.

**4. The Architect remembers past runs.**
As tasks complete, `PlannerMemory` records each outcome (topology, cost, duration, success), and the `LLMArchitect` surfaces the most relevant past outcomes in its planning prompt for similar tasks. The full loop — record → recall → prompt → different plan — is demonstrated end to end in [examples/08_learning_loop.py](examples/08_learning_loop.py) and covered by tests. Quantified evidence that it improves plans — and outcome-weighted agent routing built on it — is roadmap work we intend to publish numbers for, not hand-wave.

---

## What It Looks Like

You define the goal; the framework negotiates the path.

### Everyday task — fork-join

```python
from smythe import Swarm, Task

swarm = Swarm(max_budget_usd=0.50, model="claude-opus-4-8")

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
        "Generate a full visual asset package for the launch of 'Osiris', "
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
swarm = Swarm(max_budget_usd=2.00, model="claude-opus-4-8")

task = Task(
    goal=(
        "Evaluate whether MetaCortex Corp is a viable acquisition target. "
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

This one isn't hypothetical — it's the [flagship demo](examples/acquisition_diligence/), runnable offline with the expected graph, trace, and memo committed.

---

## Principles

- **Deterministic guardrails.** Dynamic doesn't mean "out of control." Every execution is constrained by circuit breakers: USD budget caps, per-node timeouts, bounded concurrency, and node failure policies.
- **Composable over monolithic.** Use just the DAG engine, just the agent registry, or the full stack.
- **Provider-agnostic.** Abstract over any LLM. Bring your own keys.
- **Observable by default.** Every node execution emits structured traces. The feedback loop is the product.
- **Human oversight by design.** `swarm.plan(task)` returns the graph before anything runs — inspect what the Architect decided, then execute (or don't). Approval gates that pause mid-execution are on the roadmap.

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

When no classifier provider is set, the WhiteRabbit falls back to the autonomous Architect (which must be provided via `autonomous=`).

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
    timeout_s: 60
  - id: optional-enrichment
    label: "Nice-to-have step"
    failure_policy: skip
    depends_on: [flaky-api]
```

`timeout_s` caps the wall-clock time of a single execution attempt; a timed-out attempt fails and is handled by the node's failure policy like any other error.

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
        model="claude-opus-4-8",
    ),
)
```

### Capability-aware agent assignment

Nodes can declare `required_capabilities`. The registry matches agents whose capabilities are a superset of the required set, preferring the tightest match with alphabetical tie-breaking:

```python
from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.registry import Registry

registry = Registry()
agent = Agent(profile=AgentProfile(
    name="researcher",
    capabilities=["research", "summarize"],
))
registry.register(agent)

node = Node(label="Research task", required_capabilities=["research"])
graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
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

### MCP tool use

Agents consume [MCP](https://modelcontextprotocol.io/) servers as tool sources. Declare servers on the agent, pass a tool runtime, and nodes run a bounded tool loop — every call traced, every iteration budgeted:

```python
from smythe import MCPServerSpec, MCPToolRuntime, Swarm
from smythe.agent import Agent, AgentProfile

fs = MCPServerSpec(
    name="fs", transport="stdio",
    command="npx", args=("-y", "@modelcontextprotocol/server-filesystem", "./data"),
    allowed_tools=("read_file", "list_directory"),
)
agent = Agent(profile=AgentProfile(name="Researcher", mcp_servers=[fs]))
swarm = Swarm(tool_runtime=MCPToolRuntime(), ...)
```

Secrets travel by environment-variable *name* (`env_passthrough`) and never touch YAML or checkpoints. Guardrails are on by default: `max_tool_iterations`, mid-loop budget enforcement, per-call timeouts, and `timeout_s` covering the whole loop. Details and threat model: [docs/mcp.md](docs/mcp.md). Install with `pip install smythe[mcp]`.

### Durable, resumable execution

Give the Swarm a checkpoint store and it persists the full execution state — graph, node results, agents, budget consumed — after every node. If the process dies at node 47 of a long run, resume from the last completed node instead of starting over:

```python
from smythe import FileCheckpointStore, Swarm

swarm = Swarm(checkpoint_store=FileCheckpointStore(), parallel=True)
result = swarm.execute(task)          # checkpoints as it goes
print(result.execution_id)

# later — even in a new process:
swarm = Swarm(checkpoint_store=FileCheckpointStore())
result = swarm.resume(execution_id)   # completed nodes are not re-executed
```

Checkpoints are plain JSON (one file per execution, atomic writes) so you can inspect or repair them by hand. After a crash, `FileCheckpointStore().list_ids()` shows what's resumable. Format and resume semantics: [docs/checkpoint-format.md](docs/checkpoint-format.md).

### Concurrency limits

Parallel execution caps in-flight provider calls at `max_concurrency` (default 8), so a wide broadcast doesn't fire every call at once and trip rate limits:

```python
swarm = Swarm(parallel=True, max_concurrency=3)   # at most 3 calls in flight
swarm = Swarm(parallel=True, max_concurrency=None)  # unlimited
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

### Async usage

The sync APIs (`plan`, `execute`, `route`, `synthesize`) use `asyncio.run()` internally and will raise `RuntimeError` if called from within a running event loop (e.g. Jupyter notebooks, ASGI frameworks). In those environments, use the async variants instead:

```python
graph  = await swarm.aplan(task)
result = await swarm.execute_async(task)
```

---

## Installation

```bash
pip install smythe
```

Optional extras for LLM providers and integrations:

```bash
pip install "smythe[anthropic]"    # Anthropic Claude models
pip install "smythe[openai]"       # OpenAI GPT models (and OpenAI-compatible endpoints)
pip install "smythe[gemini]"       # Google Gemini models
pip install "smythe[mcp]"          # MCP tool support
pip install "smythe[openclaw]"     # OpenClaw AgentSkills integration
pip install "smythe[all]"          # all of the above
```

Requires Python 3.11+. Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` for the respective providers — or use the built-in `OfflineProvider` with no keys at all.

Contributing or hacking on smythe itself:

```bash
git clone https://github.com/petehottelet/smythe.git
cd smythe
pip install -e ".[dev]"
```

---

## Examples

The [examples/](examples/) directory has runnable scripts for every major feature — a YAML pipeline quickstart, dynamic LLM planning, a budget-capped parallel run, crash-and-resume, and MCP tool use. Each works offline with a built-in demo provider, so you can see the machinery before spending a token:

```bash
python examples/01_quickstart_yaml.py
```

The flagship demo is [examples/acquisition_diligence/](examples/acquisition_diligence/) — the acquisition-diligence showcase from the topology example above, end to end: parallel specialists, a red-team tier, and a final structured memo, with the expected graph, trace, and memo committed so you know what success looks like:

```bash
python examples/acquisition_diligence/run.py
```

---

## Current Status

The core framework is implemented and tested. **386 tests passing.**

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
- Persistent execution memory (JSONL) with recall into planning prompts
- Per-node timeouts and bounded parallel concurrency
- MCP tool support — agents use MCP servers (stdio + HTTP) through a bounded,
  budget-enforced tool loop, with capability hydration and planner tool awareness
- Durable execution — per-node checkpointing and `swarm.resume()` with a pluggable store
- Provider abstraction (Anthropic, OpenAI, Gemini) with defensive response parsing
- Structured observability traces
- Runnable examples that work offline
- Flagship demo — the acquisition-diligence showcase with committed
  expected artifacts ([examples/acquisition_diligence/](examples/acquisition_diligence/))

**What's next:** see [ROADMAP.md](ROADMAP.md) — currently published benchmarks, then recursive subgraph decomposition and the trace inspector.

---

## License

MIT
