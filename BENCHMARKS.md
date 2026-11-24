# Benchmarks

Everything here was produced by `cargo run --release -p ncly-benchmarks` on the
machine described below. No number in this file is an estimate.

## Measurement setup

| | |
| --- | --- |
| CPU | AMD EPYC 7763 64-Core (CI runner, shared vCPU) |
| Kernel | `avx2` (selected at runtime by CPU feature detection) |
| Host | `Linux 7.0.0-1020-azure x86_64` |
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
| axpy (`y = 2a + b`) | n=2^20 | 0.453 ms | 0.402 ms | 6.95 | 51.79 |
| tensor add (1-d) | n=2^20 | 2.411 ms | 2.346 ms | 0.43 | 9.72 |
| exp (1-d) | n=2^20 | 6.215 ms | 6.184 ms | 1.35 | 2.51 |
| sum (1-d) | n=2^20 | 1.750 ms | 1.738 ms | 0.60 | 4.46 |
| matmul (f64, square) | 64x64 | 0.061 ms | 0.061 ms | 8.54 | — |
| matmul (f64, square) | 128x128 | 0.487 ms | 0.482 ms | 8.62 | — |
| matmul (f64, square) | 256x256 | 4.552 ms | 4.518 ms | 7.37 | — |
| matmul (f64, square) | 512x512 | 57.855 ms | 57.615 ms | 4.64 | — |
| LU factorisation | 256x256 | 1.595 ms | 1.585 ms | 7.01 | — |
| spmv (1-d Laplacian) | n=200000, nnz=399999 | 1.027 ms | 1.019 ms | 0.78 | 8.71 |
| autodiff fwd+bwd (`exp·sin`) | n=4096 | 0.426 ms | 0.180 ms | 0.08 | — |
| transpose + materialise | 1024x1024 | 7.252 ms | 6.357 ms | — | 2.15 |
| graph: unoptimised run | n=2^16, 6 nodes | 1.564 ms | 1.551 ms | — | — |
| graph: after folding/CSE/DCE | n=2^16, 5 nodes, folded=2, cse=1, dce=0 | 1.175 ms | 1.167 ms | — | — |

## What the numbers say

These are *observations from the table*, not projections:

- **matmul sits at 7.4–8.6 GFLOP/s for 64–256 and drops to 4.6 at 512.** The
  single-threaded AVX2 peak for `fma` on this core is far higher, so the kernel is
  limited by memory traffic and lack of register blocking. The drop at 512 is
  where the row of `B` no longer fits in L2. This is the clearest evidence that a
  blocked/threaded GEMM is the highest-value optimisation left.
- **`axpy` reaches 51.8 GiB/s while `tensor add` reaches 9.7 GiB/s** for the same
  element count. The gap is the cost of the broadcast/plan machinery around the
  kernel: `axpy` calls the SIMD kernel directly, `tensor add` builds a
  `Broadcast` plan and collapses axes first. Note also that the two are timed
  differently — `axpy` times a single fused pass, while the `tensor add` case
  allocates the output tensor and adds the two operands in a separate pass — so
  part of the gap is one extra pass over memory, not only bookkeeping. Splitting
  the two effects is the first thing to measure before optimising either.
- **`exp` at 2.5 GiB/s** is bounded by the scalar `exp` implementation per lane;
  vectorised `exp` (range reduction + polynomial) is the standard next step.
- **`sum` at 4.5 GiB/s** is one read pass over 8 MiB, so it is close to the memory
  ceiling of a shared vCPU.
- **`spmv` at 8.7 GiB/s effective** corresponds to about 0.8 GFLOP/s on a
  three-point stencil. The `GiB/s` column counts the matrix's coefficient bytes
  plus one read of `x` and one write of `y`, so this is a bandwidth figure, not a
  bandwidth ceiling: the real cost is dominated by the indirection through
  `col_idx`, which is why sparse kernels are latency-bound rather than
  bandwidth-bound.
- **The optimiser pays off, modestly, on a graph it can fold.** The same graph
  runs in 1.564 ms unoptimised and 1.175 ms after folding/CSE — about 25% faster —
  and it no longer needs the constant `c` supplied at run time, because folding
  baked it into a leaf. On this graph, with `n = 2^16`, the run is dominated by
  `tanh` and the element-wise add over the data, so 25% is the *upper* bound of
  what the passes can buy here; a graph with more repeated constant arithmetic
  would gain more. This is the only measured claim about the optimiser, and it is
  one graph on one machine.
- **`autodiff fwd+bwd` shows a 2.4x spread between best and median**, which is a
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
- **The graph cases** include graph execution overhead (a `HashMap` of values and
  the per-node shape checks), because that is what a caller pays. Both graphs
  produce identical output, which the benchmark asserts before reporting a timing;
  a run that disagrees panics, and the CI job fails rather than publishing the
  number.
- The runner has no external dependencies, so it builds on any stable toolchain
  without a network connection.
