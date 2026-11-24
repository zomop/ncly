# Roadmap

Ordered by dependency, not by appeal. A milestone is "done" only when it is
implemented, tested and documented — the state column reflects that.

| # | Milestone | State |
| --- | --- | --- |
| 1 | Workspace, core types, tensors | done |
| 2 | Tensor operations and validation | done |
| 3 | Autodiff: dual numbers and tape | done |
| 4 | Dense linear algebra | done |
| 5 | Sparse CSR/CSC | done |
| 6 | SIMD kernels and benchmark harness | done |
| 7 | Python bindings | done |
| 8 | Documentation, examples, CI | done |
| 9 | Computational graph (`ncly-graph`) | done |
| 10 | Optimising passes (`ncly-compiler`) | done, except fusion |
| 11 | Multithreaded kernels | planned |
| 12 | GPU backend | deferred |

## 9. Computational graph — done

The tape in `ncly-autodiff` is a *linear* list, sufficient for gradients but not
for optimisation. `ncly-graph` is an append-only DAG: a node id is always larger
than the ids of its inputs, so the id order *is* a topological order and cycles
are impossible by construction rather than by detection.

Implemented surface (`crates/ncly-graph/src/lib.rs`):

```text
Graph::new() -> Graph
Graph::input(shape, name) -> NodeId          // unvalidated shape is impossible
Graph::add_node(op, &[NodeId]) -> NodeId     // shape inferred eagerly, parents validated
Graph::plan() -> Schedule                    // dependency levels + descending cost
Graph::run(outputs, values) -> Execution     // verifies every produced shape
Graph::folded_leaf(shape, value, name)       // a constant baked into the node
```

`Node.value` bakes a constant into the graph, so an optimised graph runs with an
empty value map. Exit criteria met: shape inference for every op, deterministic
scheduling, no cycles by construction, and the compiler passes below consuming a
real graph. What is *not* there: any notion of a device, streams, or memory
planning — nodes own their outputs.

## 10. Optimising passes — done, except fusion

Each pass is semantics-preserving by construction; none of them reassociates
floating-point arithmetic, because `(a + b) + c` is not the same `f64` as
`a + (b + c)`.

| Pass | Status | What it does |
| --- | --- | --- |
| Dead-node elimination | done | keeps only nodes reachable from the requested outputs |
| Common-subexpression elimination | done | hash-cons identical `(op, inputs)` signatures |
| Constant folding | done | evaluates a node whose operands are all compile-time constants and replaces the subgraph with one baked leaf |
| Fusion | **not implemented** | would need a fused kernel per pattern or a JIT; see below |

Folding is conservative: it applies only when every operand is known, and it
never reorders arithmetic. `Optimized::removed_dead` counts dead nodes in the
*input* graph; a constant that disappears because the subgraph consuming it was
folded is reported through `folded`, not through `removed_dead`.

**Fusion is out of scope for now.** Emitting `exp → mul → sum` as one pass over
memory requires either a fused kernel in `ncly-simd` for each supported pattern
or a just-in-time code generator. Writing a pass that *claims* to fuse without
one would be exactly the kind of unbacked claim this project avoids, so the
documentation says plainly that it is missing.

Exit criterion: a benchmark showing the passes reduce time, with the numbers
recorded in `BENCHMARKS.md`. Met as far as the current graph allows: on a graph
with a repeated constant subexpression the optimised run is **1.175 ms against
1.564 ms** unoptimised (AMD EPYC 7763, `n = 2^16`), and it no longer requires the
constant as a run-time input. That is one graph on one machine; DCE and CSE are
covered by unit tests rather than by a timing, and no timing is claimed for them.

## 11. Multithreading

Row-block parallel matmul and parallel reductions, using `std::thread::scope`
rather than a runtime dependency. Deterministic reduction order is required so
results do not depend on the thread count.

## 12. GPU backend

Deferred. A GPU backend is only worth adding once the CPU path is stable and the
graph layer can express what would be offloaded. No timeline is claimed.

## Not planned

- Copying ndarray/faer/nalgebra APIs wholesale.
- A "compiler" that does not compile anything.
- Performance claims without measurements.
