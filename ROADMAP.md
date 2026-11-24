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
never reorders arithmetic. A folded subgraph collapses to a **single** baked leaf
— if `exp(c)` only feeds `mul(exp(c), exp(c))` and that product is the folded
value, the intermediate leaf is not rebuilt.

`constants` means "known at compile time", not "a value the caller still has to
supply": every node with a known value is rebuilt as a leaf with that value baked
in, so the optimised graph runs without those inputs at all.

Two details are easy to get wrong and are covered by regression tests:

- **Leaves are not interchangeable.** Two `Input` nodes with the same shape but
  different names carry different caller-supplied values, so CSE keys them by name.
  Merging them produced a graph that could not be executed.
- **Folding removes its own inputs.** After a subgraph is baked into a leaf,
  whatever used to feed it is unreachable and is dropped from the rebuilt graph;
  a constant whose every consumer folded disappears completely, rather than
  lingering as a node nobody reads.

`Optimized::removed_dead` counts nodes that were already unreachable in the
*input* graph. A constant that disappears because every consumer was folded is a
consequence of folding, not of dead-code elimination, so it is not counted there.

**Fusion is out of scope for now.** Emitting `exp → mul → sum` as one pass over
memory requires either a fused kernel in `ncly-simd` for each supported pattern
or a just-in-time code generator. Writing a pass that *claims* to fuse without
one would be exactly the kind of unbacked claim this project avoids, so the
documentation says plainly that it is missing.

Exit criterion: a benchmark showing the passes reduce time, with the numbers
recorded in `BENCHMARKS.md`. Met as far as the current graph allows: on a graph
with a repeated constant subexpression the optimised run is **1.187 ms against
1.577 ms** unoptimised (AMD EPYC 7763, `n = 2^16`), and it no longer requires any
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
