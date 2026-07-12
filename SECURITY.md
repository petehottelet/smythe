# Security Policy

## Supported versions

While **smythe** is on a `0.x` line the API is not stable, and only the latest
minor release receives security fixes. Once a `1.0.0` release ships, this
section will be updated with a longer support window.

| Version  | Supported          |
| -------- | ------------------ |
| 0.6.x    | Yes                |
| < 0.6    | No                 |

## Reporting a vulnerability

**Please do not report security issues through public GitHub issues, pull
requests, or discussions.**

Instead, use GitHub's private security advisory flow:

1. Go to the
   [Security Advisories page](https://github.com/petehottelet/smythe/security/advisories/new)
   for this repository.
2. Click **Report a vulnerability**.
3. Describe the issue, including:
   - The version of smythe affected.
   - A minimal reproduction (if available).
   - Whether the reproduction uses serial or parallel execution, tools,
     artifacts, checkpoints, or resume.
   - The impact you believe the vulnerability has (e.g. credential leak,
     unbounded resource use, prompt-injection-driven exfiltration).
   - Any mitigations or workarounds you are aware of.

Remove API keys, access tokens, private prompts, customer data, and sensitive
generated artifacts from the report. If a secret is required to reproduce the
issue, coordinate a secure transfer with the maintainer rather than pasting it
into an advisory comment.

You should receive an acknowledgement within **5 business days**. If you do
not, please ping [@petehottelet](https://github.com/petehottelet) on the
repository (without disclosing details) so the report can be re-routed.

## Disclosure timeline

- **Acknowledgement:** within 5 business days.
- **Initial assessment and severity classification:** within 10 business days
  of acknowledgement.
- **Fix and coordinated disclosure:** target 30-90 days from acknowledgement,
  depending on severity and complexity. We will keep you updated.

We will credit reporters in the release notes for the fixing version unless
you request otherwise.

## Scope and threat model

The threat model for smythe is currently **single-tenant, trusted-operator**:
a developer runs a swarm against LLM provider APIs they control. The most
relevant security concerns within that model:

- **Prompt injection.** Untrusted task descriptions or tool outputs that
  manipulate the Architect or an agent into making unintended calls, invoking
  an allowed tool unsafely, or exfiltrating data. Treat model output as
  untrusted input and keep tool allowlists narrow.
- **Secrets and sensitive outputs at rest.** Traces, `PlannerMemory` JSONL,
  benchmark records, generated artifacts, and checkpoint JSON can contain
  prompts, tool results, file paths, or customer data. Store these outside
  shared repositories, restrict access, and sanitize them before reporting a
  bug. MCP `env_passthrough` stores variable names in configuration, but the
  launched tool process still receives their values.
- **Tool authority.** MCP servers and other tool runtimes execute with the
  permissions of the operator who starts them. Smythe provides allowlists,
  timeouts, and bounded tool loops; it does not sandbox the server process or
  make an unsafe tool safe.
- **Cost and resource exhaustion.** Smythe caps concurrency, tool iterations,
  and per-node time, and budgeted image calls now fail closed without a
  conservative whole-call estimate. Provider pricing and caller-supplied
  `max_cost_per_call_usd` values can still become stale. Start paid runs with a
  small cap, monitor provider-side usage limits, and treat resumed
  non-idempotent calls as potentially duplicating spend or side effects.
- **Artifact and checkpoint integrity.** Files are atomically finalized, but
  local users with write access can replace or edit persisted evidence. Do not
  treat an untrusted artifact directory or checkpoint store as authenticated.
- **Dependency vulnerabilities.** smythe pins minimum versions of provider
  SDKs but does not vendor them. Provider-side issues should be reported
  upstream.

Out of scope:

- Multi-tenant isolation (smythe is not currently designed to be safe across
  mutually distrusting tenants in a single process).
- Sandboxing of arbitrary tool or provider execution.
- Security guarantees for third-party MCP servers, model providers, or files
  supplied by an untrusted checkpoint/artifact backend.

Security hardening priorities are tracked in [ROADMAP.md](ROADMAP.md). Pull
requests are welcome, but please coordinate privately first when a proposed
test would disclose an exploitable weakness.
