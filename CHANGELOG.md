# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Operator fusion in `ncly-compiler` (needs a fused kernel per pattern, or a JIT).
- Multithreaded kernels.
- GPU backend (deferred until the CPU path is stable).

## [0.1.0] — unreleased

First functional release. Everything below is implemented and covered by tests.

### Added

**Core (`ncly-core`)**

- `Shape`, `Strides`, `Layout` with overflow-checked sizes, canonical row-major
  strides, `span()` bounds and `MAX_RANK = 8`.
- Broadcasting by zero-stride insertion, computed from an operand's actual
  strides (`Broadcast::plan_with_strides`).
- `NclyError` covering shape, index, singular, convergence and graph failures.
- Neumaier compensated summation and relative pivot tests (`scalar`).

**SIMD (`ncly-simd`)**

- Scalar reference kernels plus AVX2 kernels selected once per process by runtime
  CPU feature detection; strided kernels so broadcasts need no copy.

**Tensors (`ncly-tensor`)**

- `Tensor` as `Arc<Vec<f64>>` plus `Layout`; `O(rank)` views for `transpose`,
  `permute`, `swap_axes`, `narrow`, `unsqueeze`, `squeeze`, `broadcast_to`,
  `reshape`.
- Element-wise `add`, `sub`, `mul`, `div`, `powf`, `exp`, `ln`, `sqrt`, `sin`,
  `cos`, `tanh`, `abs`, `maximum`, with loop collapsing between broadcast plans.
- Reductions `sum`, `mean`, `max`, `min`, `dot`, `sum_axis`, `max_axis`;
  matmul for rank-2 and batched rank-3.

**Autodiff (`ncly-autodiff`)**

- Forward mode: `Dual` with `+ − × ÷`, `exp`, `ln`, `sqrt`, `sin`, `cos`, `tanh`,
  `powi`, `powf`, `abs`, `recip`.
- Reverse mode: `Tape` + `Variable` with a linear Wengert list, thread-local tape
  guards, leaf-gradient accumulation and intermediate reset per sweep.
- Local-gradient enum (`Elementwise`, `Permute`, `Reshape`, `MatmulLeft`,
  `MatmulRight`, `FixedIndex`) so that matmul, view and reduction adjoints are
  structurally distinct from element-wise masks.

**Linear algebra (`ncly-linalg`)**

- LU with partial pivoting and relative pivot rejection, `solve`, `inverse`,
  `determinant`, `solve_matrix`.
- Householder QR, `Q`/`R` extraction, `lstsq` and `residual`.
- Cholesky with indefiniteness detection.
- Symmetric eigen-decomposition (cyclic Jacobi) and general eigenvalues
  (Hessenberg + shifted QR) returning real/complex pairs.
- `matrix_norm_inf`.

**Sparse (`ncly-sparse`)**

- CSR and CSC with `O(nnz)` conversion, duplicate summation, preserved explicit
  zeros and sorted rows/columns.
- `spmv`, `spmv_transpose`, `spmm`, sparse addition, scaling, dense conversion.

**Graph and compiler (`ncly-graph`, `ncly-compiler`)**

- Append-only DAG whose node ids are a topological order; shape inference at
  `add_node`, so an ill-typed graph cannot be built.
- Execution in dependency levels, each node's produced shape verified.
- Constant folding (with baked leaf values), common-subexpression elimination and
  dead-node elimination. Operator fusion is *not* implemented and is documented as
  such.

**Python (`ncly-python`, `python/`)**

- PyO3 0.23 (`abi3-py38`) module `ncly` with `Tensor`, `GradTensor`, `CsrMatrix`
  and module-level `solve`, `lstsq`, `inverse`, `eigvalsh`.
- Error mapping to `ValueError` / `IndexError` / `ZeroDivisionError` /
  `ArithmeticError`.
- maturin packaging and a pytest suite, both exercised in CI.

**Tooling**

- Integration test suite (`tests/integration`) covering gradients against finite
  differences, tensor shape validation, linalg residuals and sparse invariants.
- Dependency-free benchmark runner (`ncly-benchmarks`) reporting median/best time
  and derived GFLOP/s or GiB/s.
- GitHub Actions: build/test/doc check and Python build/test, plus a benchmark job
  that records measured numbers on the commit.

### Fixed during development

- `matmul` and `reshape` adjoints were element-wise, producing wrong gradients for
  non-symmetric operands; replaced with dedicated rules.
- `apply_q`/`apply_qt` applied Householder reflectors in the wrong order, so
  least-squares solutions were wrong even though `Q·R = A` held.
- The symmetric Jacobi rotation used `a_pp − a_qq` instead of `a_qq − a_pp`,
  which prevented convergence.
- The finite-difference test step was derived from the comparison tolerance
  instead of machine epsilon, leaving a truncation error larger than the
  tolerance.
- `Tensor::as_slice`/`as_slice_unchecked` ignored the layout offset, so a narrowed
  but contiguous view read the wrong elements.
- `eigen::eigenvalues` returned `n + 1` values: an unconditional final push
  appended a phantom eigenvalue after the 1×1 and 2×2 deflation branches had
  already produced every one.
- `eigen::eigenvalues` never converged for a non-symmetric matrix: the first QR
  rotation was built from the unshifted pencil (making it the identity), the bulge
  chase ran over every row (re-filling the sub-diagonal it had just cleared), and
  deflation tested a super-diagonal entry instead of the sub-diagonal. The shifted
  QR step is now validated against `numpy.linalg.eigvals` for rotated, triangular
  and random 5×5/8×8 matrices.
- The general 2×2 eigenvalue solver had the sign of the quadratic root wrong, so
  real eigenvalues came back negated whenever the trace was negative.
- `usize::is_multiple_of` (Rust 1.87) was used in the LU determinant sign, which
  broke the declared 1.75 MSRV; parity is now `% 2 == 0`, with the
  `manual_is_multiple_of` lint allowed and the conflict documented in the crate.
- Clippy was wired as a log-only diagnostic in the development workflow rather
  than a gate, so two lints (a 101-line function and a by-reference 3-byte
  argument) reached the pull request. It is now a gate, and every CI job posts
  its log to the commit and emits check annotations on failure.
- Combining two `Variable`s from different tapes silently produced a wrong
  gradient; it is now a panic with an explanatory message.
- Common-subexpression elimination merged distinct `Input` nodes: two leaves with
  the same shape but different names hash to the same `(op, inputs)` signature,
  which produced a graph that could not be executed (it asked for one input's
  value in order to compute another). Leaves are now keyed by name.
- Constant folding left the folded subgraph's inputs behind as unreachable nodes,
  so a caller was still expected to supply values that nothing consumed. The
  rebuilt graph now stops at nodes whose value is known, and caller-supplied
  constants are baked in as leaves too, so an optimised graph needs no constant
  inputs.
- The graph example re-keyed only some of its inputs and failed at run time; every
  example is now executed in CI, which is how that (and the two bugs above) were
  found.
- A `lib` name collision between `ncly` and the Python extension produced
  duplicate `libncly.rlib` output.

### Known limitations

- Single-threaded; parallelism is not implemented.
- `f64` only; no generic scalar type.
- No complex support in tensors (general eigenvalues report complex pairs but do
  not return complex vectors).
- Operator fusion is not implemented; only folding, CSE and dead-node removal are.
- `ncly-graph` has no device or memory-planning model.
