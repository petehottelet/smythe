# smythe

**An open-source framework for task-based personalized agent swarms with dynamic execution topology.**

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

Smythe focuses on intent-based execution. You define the goal; the framework negotiates the path.

```python
from smythe import Swarm, Task

# Initialize with a hard cost-cap guardrail
swarm = Swarm(max_budget_usd=1.00, model="gpt-4o")

# Define a complex, multi-stage intent
task = Task(
    goal=(
        "Analyze the impact of solid-state batteries on EV range. "
        "Compare three manufacturers and synthesize a technical report."
    ),
    constraints=["Must include adversarial review of manufacturer claims"]
)

# The Planner generates the DAG (Fork-Join -> Adversarial Review -> Synthesis)
# then executes via the Agent Registry.
result = swarm.execute(task)
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
