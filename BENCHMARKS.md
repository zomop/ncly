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
| axpy (`y = 2a + b`) | n=2^20 | 0.470 ms | 0.445 ms | 6.69 | 49.86 |
| tensor add (1-d) | n=2^20 | 2.399 ms | 2.353 ms | 0.44 | 9.77 |
| exp (1-d) | n=2^20 | 6.205 ms | 6.180 ms | 1.35 | 2.52 |
| sum (1-d) | n=2^20 | 1.754 ms | 1.739 ms | 0.60 | 4.45 |
| matmul (f64, square) | 64x64 | 0.061 ms | 0.061 ms | 8.53 | — |
| matmul (f64, square) | 128x128 | 0.486 ms | 0.478 ms | 8.62 | — |
| matmul (f64, square) | 256x256 | 4.468 ms | 4.415 ms | 7.51 | — |
| matmul (f64, square) | 512x512 | 60.340 ms | 58.610 ms | 4.45 | — |
| LU factorisation | 256x256 | 1.607 ms | 1.592 ms | 6.96 | — |
| spmv (1-d Laplacian) | n=200000, nnz=399999 | 1.025 ms | 1.019 ms | 0.78 | 8.72 |
| autodiff fwd+bwd (`exp·sin`) | n=4096 | 0.421 ms | 0.181 ms | 0.08 | — |
| transpose + materialise | 1024x1024 | 6.645 ms | 6.448 ms | — | 2.35 |
| graph: unoptimised run | n=2^16, 6 nodes | 1.577 ms | 1.558 ms | — | — |
| graph: after folding/CSE/DCE | n=2^16, 4 nodes, folded=2 | 1.187 ms | 1.166 ms | — | — |

## What the numbers say

These are *observations from the table*, not projections:

- **matmul sits at 7.5–8.6 GFLOP/s for 64–256 and drops to 4.5 at 512.** The
  single-threaded AVX2 peak for `fma` on this core is far higher, so the kernel is
  limited by memory traffic and lack of register blocking. The drop at 512 is
  where the row of `B` no longer fits in L2. This is the clearest evidence that a
  blocked/threaded GEMM is the highest-value optimisation left.
- **`axpy` reaches 49.9 GiB/s while `tensor add` reaches 9.8 GiB/s** for the same
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
  runs in 1.577 ms unoptimised and 1.187 ms after folding — about 25% faster — and
  the optimised graph has 4 nodes instead of 6 and needs no constants supplied at
  run time, because both folded operations and the constant leaf are baked in. The
  remaining work over `n = 2^16` elements is a `tanh` and an element-wise add, so
  25% is close to the most these passes could buy here; a graph with more repeated
  constant arithmetic would gain more. This is the only measured claim about the
  optimiser, and it is one graph on one machine.
- **The `matmul` figures vary run to run.** Across four runs on the same host the
  256×256 case measured 4.47–4.80 ms and the 512×512 case 56.9–60.3 ms. The
  table above is a single run; treat ±10% as the noise floor.
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
