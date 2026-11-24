# Security policy

## Supported versions

NCLY is pre-1.0 (`0.1.x`). Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use
[GitHub's private vulnerability reporting](https://github.com/zomop/ncly/security/advisories/new)
and include:

- affected version or commit,
- a minimal reproduction,
- the impact you believe it has,
- any suggested mitigation.

You can expect an initial response within a week. If the report is confirmed, a
fix and an advisory are prepared before public disclosure.

## Threat model

NCLY is a numerical library. Its security-relevant surface is small, but it is not
empty:

| Area | Consideration |
| --- | --- |
| `unsafe` code | Confined to SIMD kernels and slice construction. Every `unsafe` block documents the invariant it relies on; a soundness bug here is a security bug. |
| Resource exhaustion | Shapes and sizes come from the caller. `Shape::size` is checked for overflow, but a caller can still request an allocation larger than memory; OOM aborts rather than corrupting data. |
| Panics | The public API returns `Result` instead of panicking on bad shapes and singular matrices. A panic on caller-controlled input is a bug; report it. |
| Python bindings | Conversions between Python objects and Rust memory are checked; indices are bounds-checked before use. Exceptions map to Python error types rather than aborting the interpreter. |
| Untrusted serialised data | NCLY does not implement deserialisation of matrices from untrusted input. If you add such a format, treat it as a parsing target and fuzz it. |
| Numerical denial of service | Pathological inputs can trigger slow convergence. `Convergence` errors are returned with the iteration count instead of hanging forever. |

## Dependencies

The core crates have no external dependencies; `ncly-python` depends on PyO3.
Dependency advisories are handled by updating PyO3 when a relevant advisory is
published.
