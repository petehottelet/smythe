# Security policy

## Supported code

Security work targets the **Rust 0.9.0 source preview on main**. Report the exact
commit because the preview version number also covers subsequent fixes. APIs
and journal formats remain experimental; stable Rust 1.0.0 is still ahead.

| Code line | Status |
|---|---|
| Rust preview on main | Active development and security fixes |
| Original Rust preview tags | Historical snapshots; use main for subsequent fixes |
| Python releases through 0.10.0 | Retained history; not the active implementation |

See [source builds](docs/distribution.md) for version identities and
[release verification](RELEASING.md) for the current checks.

## Reporting a vulnerability

Please report vulnerabilities through the repository's
[private security advisory form](https://github.com/petehottelet/smythe/security/advisories/new).
Keep exploit details out of public issues, pull requests and discussions.

Include:

- The commit, operating system and Rust toolchain used.
- A minimal graph or command that reproduces the problem.
- Whether the failure concerns admission, accounting, journal recovery, output
  loading, controls or Noumenon exports.
- The impact and any known workaround.

Remove credentials, private prompts, customer data, sensitive output and local
account paths from the report. A sanitized graph and relevant error class are
preferable to an entire run store. If sensitive material is essential,
coordinate its transfer privately before including it.

You should receive an acknowledgement within **5 business days**. If you do
not, contact [@petehottelet](https://github.com/petehottelet) on the repository
without disclosing the vulnerability details.

## Disclosure timeline

- **Acknowledgement:** within 5 business days.
- **Initial assessment:** within 10 business days of acknowledgement.
- **Fix and coordinated disclosure:** target 30–90 days from acknowledgement,
  depending on severity and complexity.

Reporters are credited in the fixing release's notes unless they request
otherwise. Coordinate privately before submitting a test that discloses an
exploitable weakness.

## Native threat model

The runtime is **single-tenant, trusted-operator** software. Local provider
callbacks execute with the caller's permissions. A run store and the process
that owns it must be trusted.

| Boundary | Current behavior and limit |
|---|---|
| Execution | Graph, concurrency, attempt, deadline and output limits constrain the runtime; they cannot sandbox or forcibly terminate arbitrary callback code |
| Accounting | Checked integer nanoUSD, reservations before dispatch and retained known spend; the preview does not implement paid remote providers |
| Stored evidence | Hash and length verification detect changed output bytes; SHA-256 is not authentication against a writer who controls the store |
| Ownership | Local process locks and generation/sequence checks reject stale writes; this is not a distributed or multi-tenant isolation boundary |
| Recovery | SQLite WAL with synchronous NORMAL covers application-process crashes; power loss, OS crashes and network filesystems are outside the guarantee |
| External effects | The runtime refuses external-effect execution; its pure retry contract does not establish safe replay of remote actions |

Keep run stores and exported artifacts outside shared source repositories.
Restrict access to their directories and inspect records before sharing them:
task input, outputs, diagnostics and journals can contain sensitive material.
Deleting a source checkout does not remove run stores elsewhere.

## Future integrations and historical behavior

Model-driven planning, paid providers, tools, MCP, supervision and Jobs are not
shipped native integrations. The [roadmap](ROADMAP.md) requires explicit tool
authority, bounded calls, untrusted-source handling and reconciliation before
those integrations can establish their own guarantees.

The [historical Python security policy](https://github.com/petehottelet/smythe/blob/12f14704beff344228fffffce40d995daa1f7cce/SECURITY.md)
describes the retired provider and tool paths. Its protections and APIs should
not be inferred from the Rust preview.

Multi-tenant isolation, callback sandboxing and security guarantees for
third-party services are outside the current scope. Report dependency issues
that affect Smythe, including a minimal reproduction where possible.
