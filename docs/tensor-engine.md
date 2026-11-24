# Tensor engine internals

This document explains how `ncly-tensor` represents arrays, how the hot loops are
organised, and which invariants the kernels rely on.

## Representation

```rust
pub struct Tensor {
    storage: Arc<Vec<f64>>,   // shared, immutable buffer
    layout:  Layout,          // shape + strides + offset
}
```

Three consequences follow directly:

1. **Views are free.** `transpose`, `narrow`, `broadcast_to`, `reshape` of a
   contiguous tensor only change the `Layout`, in `O(rank)`.
2. **Cloning is cheap.** `Tensor: Clone` bumps an `Arc`; it does not copy data.
   Mutation goes through `Arc::make_mut`, so copy-on-write happens only when the
   buffer is actually shared.
3. **Every kernel is a strided loop.** There is no separate implementation for
   contiguous and non-contiguous data — only a fast path when all strides are 1.

### Layout and strides

Row-major canonical strides for shape `(d₀, d₁, …, dₙ₋₁)`:

$$
s_k = \prod_{j>k} d_j, \qquad s_{n-1} = 1 .
$$

The flat offset of logical index `(i₀, …, iₙ₋₁)` is

$$
\text{offset} = \text{base} + \sum_{k} i_k \, s_k .
$$

`Layout::span()` returns an exclusive upper bound on every address the layout can
touch, which is what makes it safe to build a `&[f64]` sub-slice for the innermost
loop without a per-element bounds check.

## Broadcasting

Broadcasting is implemented **only** by inserting zero strides:

```text
a: shape (3, 1),  strides (1, 1)
b: shape (2, 3, 4)
      │
      ▼  broadcast plan for a
   shape (2, 3, 4), strides (0, 1, 0)
```

A zero stride means "read the same element regardless of this index", which turns
any broadcast kernel into a plain strided loop with no branching. The plan is
computed from the operand's **actual** strides (`Broadcast::plan_with_strides`),
so transposed and narrowed views broadcast correctly.

## Loop collapsing

Naively, a binary operation over a rank-3 broadcast result is a triple loop. Two
adjacent axes can be merged when the merger preserves the address arithmetic, i.e.

$$
s_k = d_{k+1} \cdot s_{k+1}
\quad\text{for every operand.}
$$

When that holds, index `i·d_{k+1} + j` of the fused axis maps to the same memory as
the pair `(i, j)`, and the **fused axis keeps the stride of the inner axis**:

```text
(2, 3, 4) with strides (12, 4, 1)
  fuse axes 1,2 → 12 elements with stride 1
  fuse axes 0,1 → 24 elements with stride 1
result: one axis of length 24, stride 1
```

This is what lets a contiguous tensor run at the speed of a flat `f64` slice.
Length-1 axes are dropped before fusing, so patterns like `(3, 1, 5)` broadcast
against `(4, 5)` collapse to a single loop.

The tests in `crates/ncly-tensor/src/ops.rs` assert the retained stride directly,
because keeping the *outer* stride instead of the inner one is a plausible
"optimisation" that silently reads out of bounds.

## Kernel structure

```text
binary(lhs, rhs, op)
  ├── out_shape = broadcast_with(lhs.shape, rhs.shape)
  ├── plan = layout.broadcast_to(out_shape)            (zero strides)
  ├── axes  = collapse(out_shape, strides…)             (few loop axes)
  └── loop
        ├── inner slice → ncly_simd::add/sub/mul/div
        └── odometer over outer axes
```

The odometer avoids divisions: it keeps the running offset per operand and
subtracts `len · stride` when an axis wraps.

## Reductions

Reductions normalise to a contiguous buffer for the reduced axis, then run a
compensated (Neumaier) sum:

$$
\text{sum} = \sum_i x_i + \sum_i c_i, \qquad
c_i \approx (x_i - \text{fl}(x_i + \text{fl}(\text{sum})))
$$

Naive accumulation of `n` terms has worst-case relative error `n·ε`; compensated
accumulation holds it near `ε` for the *result*, regardless of cancellation. The
cost is one extra addition and one comparison per element, which is why it is used
for `sum`, `mean`, `dot` and the matmul accumulator, but not for the elementwise
kernels where it would be pure overhead.

## Matrix multiplication

`matmul` handles rank-2 and rank-3 (batched) operands, broadcasting a rank-2
right-hand operand across the batch. The kernel is

```text
for i in rows(A):
  for p in inner:
    for j in cols(B):
      C[i][j] += A[i][p] * B[p][j]
```

with the `(p, j)` inner part being an AXPY over contiguous memory. A column block
of `B` is chosen so that a row of `B` stays in L1 for all `i`, giving streaming
access to `A` and blocked reuse of `B`.

Arithmetic intensity is `O(1)` per flop, so this is a *memory-bandwidth* problem
for small matrices and a *compute* problem for large ones. The benchmark runner in
`benchmarks/` reports measured GFLOP/s for `64…512` square matrices so that any
future blocking change can be justified with numbers.

## Numerical conventions

| Situation | Behaviour | Reason |
| --- | --- | --- |
| `0/0` | `NaN` | IEEE-754; the gradient is undefined, hiding it is worse |
| `1/0` | `+inf` | IEEE-754 |
| `max`/`min` with `NaN` | `NaN` | Ignores NaN silently would hide data corruption |
| empty `sum` | `0` | Identity element |
| empty `max` | `-inf` | Identity element |
| `mean` of empty | `NaN` | There is no mean of nothing |

## Limits

* `MAX_RANK = 8`. Broadcasting beyond that returns `AxisError::RankOverflow`.
* Storage is `f64` only; there is no generic scalar parameter yet. Generic
  element types would multiply the SIMD surface, so they are deliberately deferred.
* No parallel execution yet: kernels are single-threaded and rely on SIMD. The
  benchmark numbers in `BENCHMARKS.md` are per-core.
