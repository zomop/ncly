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
| 9 | Computational graph (`ncly-graph`) | planned |
| 10 | Optimising passes (`ncly-compiler`) | planned |
| 11 | Multithreaded kernels | planned |
| 12 | GPU backend | deferred |

## 9. Computational graph

The tape in `ncly-autodiff` is a *linear* list, sufficient for gradients. A DAG
is needed for optimisation: shared subexpressions must be identified, the nodes
scheduled into levels, and the schedule validated against the shape rules.

Planned surface:

```text
Graph::new() -> Graph
Graph::input(shape, name) -> Node
Graph::op(kind, inputs) -> Node       // validated shape inference
Graph::levels() -> Vec<Vec<Node>>     // topological levels
Graph::validate() -> Result<Schedule> // dead nodes, cycles, shape mismatches
```

Exit criteria: shape inference for every op, deterministic scheduling, cycle
detection, and the compiler passes below consuming a real graph.

## 10. Optimising passes

Each pass must be provably semantics-preserving and measured:

- **Dead-op elimination** — drop nodes with no path to an output.
- **Common-subexpression elimination** — hash-cons identical `(op, inputs)`.
- **Fusion** — emit `exp → mul → sum` as one pass over memory.
- **Constant folding** — evaluate subgraphs with only constant inputs.

Exit criteria: a benchmark where each pass reduces either time or allocations, with
numbers recorded in `BENCHMARKS.md`.

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
