# PRD — OpenClaw AgentSkills Capability Integration

Enable Smythe to use OpenClaw AgentSkills as the source of truth for per-agent capability profiles, so planner and registry assignment decisions reflect real, installed agent skills.

---

## 1) Problem Statement

Smythe currently relies on `AgentProfile.capabilities` values provided at creation time or YAML/planner output. This works, but capability data can drift from the actual skillset available to each agent at runtime.

OpenClaw AgentSkills already represent concrete, executable abilities. We should leverage those skills directly so:

- assignment quality improves,
- capability data stays current,
- and orchestration is grounded in verifiable tooling rather than static tags.

---

## 2) Goals

1. **Skill-grounded capability profiles**
   - Populate/refresh each agent's capability profile from OpenClaw skills.
2. **Backwards-compatible behavior**
   - Existing projects with static capabilities continue working unchanged.
3. **Optional integration**
   - OpenClaw integration is adapter-based and does not hard-require OpenClaw.
4. **Deterministic assignment**
   - Registry selection remains deterministic and testable.
5. **Production-safe operation**
   - Support caching, timeouts, and graceful fallback when skill provider is unavailable.

---

## 3) Non-Goals

- Building a full OpenClaw runtime inside Smythe.
- Executing skills directly from Smythe executor (this PRD is about capability profiling and assignment).
- Replacing existing planner tiers or router logic.
- Introducing network-dependent tests in CI.

---

## 4) User Stories

1. As a developer, I want capabilities auto-derived from installed AgentSkills so I do not manually maintain capability tags.
2. As an operator, I want fallback to static capabilities if OpenClaw is unavailable.
3. As a planner author, I want node capability requirements to route to agents with matching real skills.
4. As a maintainer, I want deterministic tie-breaking and clear observability for why an agent was chosen.

---

## 5) Functional Requirements

### FR-1: Skill Provider Abstraction

Add an adapter interface for external skill systems:

```python
@dataclass(frozen=True)
class SkillRef:
    name: str
    version: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillProvider(Protocol):
    def list_agent_skills(self, agent_id: str) -> list[SkillRef]:
        ...
```

Notes:
- Keep interface sync for now (consistent with `Registry.assign()` sync path).
- Future async extension is acceptable but not required in v1.

### FR-2: Skill-to-Capability Mapping

Provide a mapper layer:

```python
class CapabilityMapper(Protocol):
    def map_skills(self, skills: list[SkillRef]) -> set[str]:
        ...
```

Default implementation:
- map by normalized skill name (`"web-search"` -> `"web-search"`),
- include alias map support (configurable dictionary),
- optional static per-skill metadata override.

### FR-3: Capability Hydration

Add optional hydration in registry:

- On registration or assignment, if `skill_provider` is configured:
  1. fetch skill refs for the candidate agent,
  2. map to capabilities,
  3. merge or replace `agent.profile.capabilities` per policy.

Policy enum:

```python
class CapabilityHydrationMode(Enum):
    MERGE = "merge"      # static + derived
    REPLACE = "replace"  # derived only
    STATIC_ONLY = "static_only"
```

Default: `MERGE`.

### FR-4: Caching and Refresh

Add cache to avoid repeated provider calls during assignment:

- key: `agent_id`
- value: hydrated capabilities + timestamp
- TTL default: 300 seconds
- manual invalidation API:
  - `Registry.refresh_agent_capabilities(agent_id: str) -> None`
  - `Registry.refresh_all_capabilities() -> None`

### FR-5: Assignment Integration

`Registry.find_by_capabilities()` must use hydrated capabilities when integration is enabled.

Deterministic behavior preserved:
- exact/superset matching as existing,
- tightest superset preference,
- stable tie-break rule (alphabetical/ID deterministic).

### FR-6: Observability

For each assignment decision, capture metadata (debug-level or tracer annotation):

- required capabilities,
- candidate agents considered,
- source of capabilities (static, hydrated, merged),
- selected agent and reason (exact/tightest/tiebreak/fallback).

### FR-7: Failure Handling

If skill provider fails or times out:
- log warning,
- continue with static capabilities,
- never crash assignment flow.

---

## 6) API / Code Surface Changes

### New files

- `smythe/skills.py`
  - `SkillRef`
  - `SkillProvider` protocol
  - `CapabilityMapper` protocol + default implementation
  - hydration policy enum
- `smythe/openclaw_adapter.py`
  - `OpenClawSkillProvider` adapter (best-effort import; optional dependency)

### Modified files

- `smythe/registry.py`
  - accept optional `skill_provider`, `capability_mapper`, hydration mode, cache TTL
  - apply hydration in assignment/matching path
  - expose refresh methods
- `smythe/agent.py` (only if needed)
  - optional metadata fields for skill provenance
- `smythe/__init__.py`
  - export new integration types
- `Readme.md`
  - add configuration and examples

---

## 7) Configuration

Swarm/Registry configuration examples:

```python
from smythe import Swarm
from smythe.registry import Registry
from smythe.openclaw_adapter import OpenClawSkillProvider
from smythe.skills import DefaultCapabilityMapper, CapabilityHydrationMode

registry = Registry(
    skill_provider=OpenClawSkillProvider(...),
    capability_mapper=DefaultCapabilityMapper(
        aliases={"search": "research", "summarize-text": "summarize"}
    ),
    hydration_mode=CapabilityHydrationMode.MERGE,
    capability_cache_ttl_seconds=300,
)

swarm = Swarm(registry=registry, ...)
```

---

## 8) Test Plan

### Unit Tests

1. `test_skill_mapper_default_name_mapping`
2. `test_skill_mapper_alias_mapping`
3. `test_registry_hydrates_capabilities_merge_mode`
4. `test_registry_hydrates_capabilities_replace_mode`
5. `test_registry_uses_cache_until_ttl_expired`
6. `test_registry_refresh_agent_capabilities_forces_reload`
7. `test_assignment_uses_hydrated_capabilities`
8. `test_skill_provider_failure_falls_back_to_static`
9. `test_deterministic_tiebreak_with_hydrated_caps`

### Integration-ish (mocked adapter)

10. `test_openclaw_adapter_translates_skills_to_skillrefs` (with mocked OpenClaw client)

No live OpenClaw dependency in CI.

---

## 9) Acceptance Criteria

1. With OpenClaw adapter enabled, assignment outcomes change to reflect installed skills (verified in tests).
2. With adapter disabled or failing, behavior matches current static-capability assignment.
3. All existing tests pass.
4. New tests pass and cover hydration/caching/fallback paths.
5. README documents usage, fallback semantics, and examples.

---

## 10) Rollout Plan

### Phase A: Core Abstractions
- Add `skills.py` protocols/types and default mapper.

### Phase B: Registry Integration
- Add hydration logic + cache + fallback.

### Phase C: OpenClaw Adapter
- Add optional adapter module with defensive imports.

### Phase D: Docs + Hardening
- README updates, examples, and logging/trace polish.

Feature flag default:
- Integration disabled unless `skill_provider` explicitly configured.

---

## 11) Risks and Mitigations

1. **Capability drift due to stale cache**
   - Mitigation: TTL + explicit refresh APIs.
2. **Mapping ambiguity**
   - Mitigation: alias map and deterministic normalization.
3. **Adapter dependency instability**
   - Mitigation: optional import + graceful fallback.
4. **Assignment latency**
   - Mitigation: cache, batched refresh in future iteration.

---

## 12) Open Questions

1. Should hydration run only during assignment, or also at registration time by default?
2. Do we want tenant/workspace scoping for skill sources?
3. Should capability mapping live in config file (YAML) in addition to Python API?
4. Is an async skill provider interface needed immediately for remote skill catalogs?

---

## 13) Success Metrics

1. Reduced unassigned/fallback agent creations in capability-constrained graphs.
2. Higher exact-match rate in `find_by_capabilities()`.
3. No increase in executor failure rate due to assignment mismatch.
4. Negligible assignment latency overhead under cache-hit conditions.

