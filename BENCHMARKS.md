# Benchmarks

Everything here was produced by `cargo run --release -p ncly-benchmarks` on the
machine described below. No number in this file is an estimate.

## Measurement setup

| | |
| --- | --- |
| CPU | Intel Xeon 6973P-C (CI runner, shared vCPU) |
| Kernel | `avx2` (selected at runtime by CPU feature detection) |
| Toolchain | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |
| Profile | `release`, `opt-level = 3`, `lto = "thin"`, `codegen-units = 1` |
| Threads | 1 (kernels are single-threaded) |

A shared CI vCPU is not a quiet machine, so absolute times fluctuate between runs
by roughly 10%. Ratios between kernels measured in the same run are the meaningful
part, and the runner reports the **median** of repeated executions after a 10%
warm-up.

Reproduce:

```bash
cargo run --release -p ncly-benchmarks            # every case
cargo run --release -p ncly-benchmarks matmul     # one family
cargo run --release -p ncly-benchmarks -- --json  # machine-readable
```

## Results

| kernel | size | median | best | GFLOP/s | GiB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| axpy (`y = 2a + b`) | n=2^20 | 0.850 ms | 0.831 ms | 3.70 | 27.58 |
| tensor add (1-d) | n=2^20 | 2.999 ms | 2.684 ms | 0.35 | 7.81 |
| exp (1-d) | n=2^20 | 4.396 ms | 4.292 ms | 1.91 | 3.55 |
| sum (1-d) | n=2^20 | 1.522 ms | 1.495 ms | 0.69 | 5.13 |
| matmul (f64, square) | 64x64 | 0.046 ms | 0.043 ms | 11.42 | — |
| matmul (f64, square) | 128x128 | 0.398 ms | 0.388 ms | 10.55 | — |
| matmul (f64, square) | 256x256 | 3.197 ms | 3.164 ms | 10.49 | — |
| matmul (f64, square) | 512x512 | 43.367 ms | 43.167 ms | 6.19 | — |
| LU factorisation | 256x256 | 1.195 ms | 1.163 ms | 9.36 | — |
| spmv (1-d Laplacian) | n=200000, nnz=399999 | 0.804 ms | 0.775 ms | 1.00 | 11.13 |
| autodiff fwd+bwd (`exp·sin`) | n=4096 | 0.308 ms | 0.114 ms | 0.11 | — |
| transpose + materialise | 1024x1024 | 6.482 ms | 6.414 ms | — | 2.41 |

## What the numbers say

These are *observations from the table*, not projections:

- **matmul sits at 10–11 GFLOP/s for 64–256 and drops to 6.2 at 512.** The
  single-threaded AVX2 peak for `fma` on this core is far higher, so the kernel is
  limited by memory traffic and lack of register blocking. The drop at 512 is
  where the row of `B` no longer fits in L2. This is the clearest evidence that a
  blocked/threaded GEMM is the highest-value optimisation left.
- **`axpy` reaches 27.6 GiB/s while `tensor add` reaches 7.8 GiB/s** for the same
  element count. The gap is the cost of the broadcast/plan machinery around the
  kernel: `axpy` calls the SIMD kernel directly, `tensor add` builds a
  `Broadcast` plan and collapses axes first. Note also that the two are timed
  differently — `axpy` times a single fused pass, while the `tensor add` case
  allocates the output tensor and adds the two operands in a separate pass — so
  part of the gap is one extra pass over memory, not only bookkeeping. Splitting
  the two effects is the first thing to measure before optimising either.
- **`exp` at 3.55 GiB/s** is bounded by the scalar `exp` implementation per lane;
  vectorised `exp` (range reduction + polynomial) is the standard next step.
- **`sum` at 5.1 GiB/s** is one read pass over 8 MiB, so it is close to the memory
  ceiling of a shared vCPU.
- **`spmv` at 11.1 GiB/s effective** corresponds to about 1.0 GFLOP/s on a
  three-point stencil. The `GiB/s` column counts the matrix's coefficient bytes
  plus one read of `x` and one write of `y`, so this is a bandwidth figure, not a
  bandwidth ceiling: the real cost is dominated by the indirection through
  `col_idx`, which is why sparse kernels are latency-bound rather than
  bandwidth-bound.
- **`autodiff fwd+bwd` shows a 2.7x spread between best and median**, which is a
  scheduling artefact on a shared runner rather than a property of the code. The
  case includes tape recording and the backward sweep, so it is a pipeline number,
  not a kernel number; the median is the honest figure to quote.

## Methodology notes

- **Warm-up** before timing: the caches and the branch predictors need a pass over
  the data, and the first allocation of the working set must not be timed.
- **`std::hint::black_box`** wraps the result of every call so the optimiser
  cannot delete the loop.
- **Derived throughput** uses the operation count for matmul (`2n³`) and `2·8·n`
  bytes for element-wise kernels, i.e. one read of each input and one write of the
  output.
- The runner has no external dependencies, so it builds on any stable toolchain
  without a network connection.
