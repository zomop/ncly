# Contributing to NCLY

Thanks for taking the time to contribute. This document describes what the
project expects from a change, and why the bar for numerical code is higher than
for most application code.

## Ground rules

1. **No copied code.** NCLY is an independent implementation. Do not paste code,
   documentation or branding from ndarray, faer, nalgebra, burn, candle, PyO3,
   maturin, BLAS or LAPACK. Re-implementing a published algorithm from its
   mathematical description is fine and expected; copying an implementation is
   not.
2. **Only document what works.** If a feature is partial, say so in the README,
   the crate docs and `ROADMAP.md`.
3. **No unmeasured performance claims.** If a change claims a speed-up, it must
   come with before/after numbers from `cargo run --release -p ncly-benchmarks`
   on a named machine.
4. **No fabricated metadata.** Do not add badges, star counts, download counts or
   version claims that are not literally true.

## Workflow

```bash
git clone https://github.com/zomop/ncly
cd ncly
cargo build --workspace --all-targets
cargo test --workspace
```

Before opening a pull request:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

For changes that touch the Python bindings:

```bash
cd python
maturin build --release
pip install --force-reinstall target/wheels/ncly-*.whl
pytest -q
```

## What a numerical change must include

This is the important part. Numerical bugs are silent: the code compiles, the
tests pass, and the answer is wrong in the third digit. Every numerical change
therefore needs **evidence**, not just code:

| Change | Required evidence |
| --- | --- |
| New linalg routine | A residual test: `‖A − reconstruction‖`, `‖Av − λv‖`, or `‖Ax − b‖` at a stated tolerance |
| New gradient rule | A finite-difference comparison *and* a forward-vs-reverse comparison |
| Kernel/vectorisation change | Bit-exactness against the scalar path, plus a benchmark number |
| Layout/broadcast change | A test against an independent reference implementation of broadcasting |
| New error path | A test that asserts the specific error variant is returned |

Tolerances must be justified. "Loosen the tolerance until it passes" is not a
fix; if a residual is `1e-6` and you expected `1e-12`, the algorithm has a bug,
a pivot test is wrong, or the conditioning is bad — find out which.

## Code style

- `rustfmt` defaults, no custom configuration.
- Clippy runs with `-D warnings`; if a lint is genuinely wrong for the code,
  write `#[allow(clippy::…)]` with a comment explaining why.
- Public items need doc comments: what it computes, its complexity, and the error
  conditions. Doc comments are compiled (`RUSTDOCFLAGS="-D warnings"`).
- Keep functions small enough that the mathematical structure is visible.

## Commit messages

Conventional Commits, scoped by crate:

```text
fix(qr): apply reflectors in the correct order for Q
feat(autodiff): add a matmul adjoint rule
docs(architecture): explain loop collapsing
test(sparse): cover duplicate coordinates
```

A commit should leave the workspace building and testing green.

## Reporting bugs

Please include:

- the smallest input that reproduces the problem,
- what NCLY returns and what you expected,
- the Rust or Python version and `uname -a`,
- for numerical issues, the condition number or scale of the input if you know it.

Accuracy reports are especially welcome. If a result is wrong in the last few
digits, that is a bug worth filing.

## Pull request review

A reviewer will check:

- [ ] the change is in scope and does not add a stub module,
- [ ] evidence for numerical correctness is present,
- [ ] public API changes are documented and reflected in the README if user-facing,
- [ ] `ROADMAP.md` is updated if a milestone moved,
- [ ] no fabricated claims were added.
