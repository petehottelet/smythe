# Repo Doctor Scoring Rubric

Use a 10-point score, but make it hard to get a 9 or 10.

## Overall score guide

| Score | Meaning |
|---:|---|
| 1-2 | Bare idea, broken, or not meaningfully reviewable. |
| 3-4 | Prototype exists but lacks basic onboarding, structure, or runnable proof. |
| 5-6 | Usable early project with clear gaps in tests, docs, packaging, or release flow. |
| 7 | Credible project with good structure, but missing proof, polish, or adoption readiness. |
| 8 | Publishable and useful, with solid open-source hygiene and a clear next milestone. |
| 9 | Strong project with docs, tests, CI, release discipline, examples, benchmarks, and external validation. |
| 10 | Exceptional: production-grade, trusted by users, benchmarked, documented, stable, and differentiated. |

## Category weights

| Category | Weight |
|---|---:|
| Positioning and purpose | 10% |
| README and onboarding | 15% |
| Installability and quickstart | 10% |
| Code structure | 10% |
| Tests and CI | 15% |
| Packaging and release readiness | 15% |
| Docs and examples | 10% |
| Security/license/community hygiene | 10% |
| Proof/benchmarks/adoption | 5% |

## Hard caps

Apply these hard caps unless the final synthesis explicitly explains an exception:

- No README: max score 5.
- Not installable/runnable: max score 6.
- No license for an open-source project: max score 6.
- No tests for a package/framework: max score 7.
- No CI for a package/framework: max score 8.
- Big technical claims with no demo/benchmark: max score 8.
- Cannot determine what the project does in under 60 seconds: max score 6.
