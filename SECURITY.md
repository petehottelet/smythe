# Security Policy

## Supported versions

While **smythe** is on a `0.x` line the API is not stable, and only the latest
minor release receives security fixes. Once a `1.0.0` release ships, this
section will be updated with a longer support window.

| Version  | Supported          |
| -------- | ------------------ |
| 0.1.x    | Yes                |
| < 0.1    | No                 |

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
   - The impact you believe the vulnerability has (e.g. credential leak,
     unbounded resource use, prompt-injection-driven exfiltration).
   - Any mitigations or workarounds you are aware of.

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
  manipulate the Architect into emitting unintended agent calls or
  exfiltrating system prompts. Hardening is on the roadmap; please report
  reproductions.
- **Secrets in traces.** API keys or PII captured in `Tracer` spans or
  `PlannerMemory` JSONL. Redaction hooks are on the roadmap.
- **Resource exhaustion.** Unbounded fan-out, missing timeouts, or unbounded
  memory growth. Concurrency caps and per-node timeouts are on the roadmap.
- **Dependency vulnerabilities.** smythe pins minimum versions of provider
  SDKs but does not vendor them. Provider-side issues should be reported
  upstream.

Out of scope:

- Multi-tenant isolation (smythe is not currently designed to be safe across
  mutually distrusting tenants in a single process).
- Sandboxing of arbitrary tool execution (no tool-execution sandbox is
  shipped today).

These items are on the public roadmap; pull requests welcome.
