## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Evidence

<!-- Numerical changes require evidence; delete the rows that do not apply. -->

- [ ] Residual test (`‖A − reconstruction‖`, `‖Av − λv‖`, `‖Ax − b‖`) added
- [ ] Finite-difference check added or updated
- [ ] Forward-vs-reverse agreement checked
- [ ] Benchmark numbers attached (machine named, `ncly-benchmarks` output pasted)
- [ ] No numerical claim is made without a measurement

## Checklist

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- [ ] `pytest -q` in `python/` if the bindings changed
- [ ] README / `ROADMAP.md` updated if user-visible state changed
- [ ] No stub modules, no invented metadata, no unverified badges
