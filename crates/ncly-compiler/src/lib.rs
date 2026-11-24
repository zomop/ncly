//! Optimising passes over [`ncly_graph`] graphs.
//!
//! Three passes, in the order they are applied by [`optimize`]:
//!
//! | Pass | What it removes | Why it is safe |
//! | --- | --- | --- |
//! | constant folding | subgraphs with only constant inputs | floating-point evaluation is deterministic for a fixed order of operations |
//! | common-subexpression elimination | repeated `(op, operands)` pairs | every operation is pure: same operands, same result |
//! | dead-node elimination | nodes with no path to an output | they cannot affect the result |
//!
//! Every pass is **conservative**: a rewrite is applied only when it is exactly
//! semantics-preserving. In particular no pass reassociates floating-point
//! arithmetic, because `(a + b) + c` and `a + (b + c)` are not the same `f64`.
//!
//! ```
//! use ncly_compiler::{optimize, Passes};
//! use ncly_graph::{Graph, NodeId, Op};
//!
//! let mut graph = Graph::new();
//! let a = graph.input([2], "a");
//! let one = graph.input([2], "one");       // constant, supplied at run time
//! let exp1 = graph.add_node(Op::Exp, &[a])?;
//! let exp2 = graph.add_node(Op::Exp, &[a])?;   // duplicate of exp1
//! let sum = graph.add_node(Op::Add, &[exp1, exp2])?;
//! let unused = graph.add_node(Op::Neg, &[one])?; // dead
//!
//! let result = optimize(&graph, &[sum], &Passes::all());
//! // `one` and `neg` are not reachable from the output.
//! assert_eq!(result.removed_dead, 2);
//! assert_eq!(result.removed_duplicates, 1);
//! assert!(result
//!     .graph
//!     .node(NodeId(result.remap[&sum.0]))
//!     .is_ok());
//! assert!(!result.remap.contains_key(&unused.0));
//! # Ok::<(), ncly_core::NclyError>(())
//! ```
//!
//! # What is *not* here
//!
//! Operator fusion is not implemented. Fusing `exp → mul → sum` into one kernel
//! requires either a fused kernel in `ncly-simd` for each pattern or a
//! just-in-time code generator, and claiming fusion without one would be
//! dishonest. The roadmap entry stays open until a fused kernel exists and is
//! measured.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::hash::BuildHasher;

use ncly_core::{NclyError, Result};
use ncly_graph::{Graph, Node, NodeId, Op};
use ncly_tensor::Tensor;

/// Which passes to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Passes {
    /// Evaluate constant subgraphs and replace them with leaves.
    pub constant_folding: bool,
    /// Merge nodes with identical operations and operands.
    pub common_subexpressions: bool,
    /// Drop nodes that no output depends on.
    pub dead_code: bool,
}

impl Passes {
    /// No pass.
    pub fn none() -> Self {
        Self {
            constant_folding: false,
            common_subexpressions: false,
            dead_code: false,
        }
    }

    /// Every implemented pass.
    pub fn all() -> Self {
        Self {
            constant_folding: true,
            common_subexpressions: true,
            dead_code: true,
        }
    }
}

impl Default for Passes {
    fn default() -> Self {
        Self::all()
    }
}

/// Statistics and the rewritten graph produced by [`optimize`].
#[derive(Debug)]
pub struct Optimized {
    /// The rewritten graph.
    pub graph: Graph,
    /// Old node index to new node index. Nodes removed by a pass are absent.
    pub remap: HashMap<usize, usize>,
    /// Number of nodes that were unreachable in the input graph and were thus
    /// excluded from the rebuilt one.
    ///
    /// Folding can also drop a node (a constant whose only consumers were folded),
    /// and that is *not* counted here: it is a consequence of folding, and folding
    /// is reported through [`Optimized::folded`].
    pub removed_dead: usize,
    /// Number of nodes merged into an existing equivalent node.
    pub removed_duplicates: usize,
    /// Number of operations that were evaluated at compile time.
    ///
    /// Caller-supplied constants are not operations and are not counted; they are
    /// baked into leaves all the same.
    pub folded: usize,
}

impl Optimized {
    /// Outputs translated to the rewritten graph.
    ///
    /// An output that was folded into a constant is still a node in the new
    /// graph, so this never fails for a valid input graph.
    pub fn remap_outputs(&self, outputs: &[NodeId]) -> Result<Vec<NodeId>> {
        outputs
            .iter()
            .map(|id| {
                self.remap
                    .get(&id.0)
                    .map(|&new| NodeId(new))
                    .ok_or_else(|| NclyError::Graph {
                        reason: format!("output {id} was eliminated by the optimiser"),
                    })
            })
            .collect()
    }

    /// How many nodes the graph lost to dead-code elimination and CSE.
    ///
    /// Folding also shrinks the graph, but it replaces a subgraph rather than
    /// deleting nodes one by one, so it is not included: compare
    /// `graph.len()` with the input graph if you need the total.
    pub fn nodes_removed(&self) -> usize {
        self.removed_dead + self.removed_duplicates
    }
}

/// Values discovered for constant nodes, and which of them were computed here.
#[derive(Default)]
struct Folding {
    /// Every constant node's value, including caller-supplied leaves.
    known: HashMap<NodeId, Tensor>,
    /// The nodes whose value was *computed* by this pass; these become leaves.
    folded: HashSet<NodeId>,
}

/// Evaluate every subgraph whose operands are all known at compile time.
///
/// A node folds only when each of its operands is already constant, so an
/// input-fed operand breaks the chain exactly as it should.
fn fold_constants<S: BuildHasher>(
    graph: &Graph,
    live: &HashSet<NodeId>,
    constants: &HashMap<NodeId, Tensor, S>,
) -> Folding {
    let mut folding = Folding::default();
    // Nodes whose value is known before execution: a caller-supplied constant, or
    // an operation whose operands are all in this set.
    let mut constant: HashSet<NodeId> = HashSet::new();
    for index in 0..graph.len() {
        let id = NodeId(index);
        if !live.contains(&id) {
            continue;
        }
        let node = &graph.nodes()[index];
        if let Some(value) = constants.get(&id) {
            folding.known.insert(id, value.clone());
            constant.insert(id);
            continue;
        }
        if node.op == Op::Input {
            continue;
        }
        if !node.inputs.iter().all(|parent| constant.contains(parent)) {
            continue;
        }
        let operands: Option<Vec<Tensor>> = node
            .inputs
            .iter()
            .map(|parent| folding.known.get(parent).cloned())
            .collect();
        let Some(operands) = operands else { continue };
        if let Ok(value) = evaluate_constant(node, &operands) {
            folding.known.insert(id, value);
            constant.insert(id);
            folding.folded.insert(id);
        }
    }
    folding
}

/// Nodes still needed in the *rebuilt* graph.
///
/// Reachability from the outputs, but stopping at nodes whose value is already
/// known — those are rebuilt as leaves with the value baked in, so whatever used
/// to feed them is no longer needed. Without this, a constant that only existed to
/// feed a folded subgraph would survive as an unreachable node and the caller
/// would still have to supply a value for it.
fn needed_after_folding(
    graph: &Graph,
    outputs: &[NodeId],
    known: &HashSet<NodeId>,
) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut stack: Vec<NodeId> = outputs.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if known.contains(&id) {
            continue;
        }
        let Some(node) = graph.nodes().get(id.0) else {
            continue;
        };
        for &parent in &node.inputs {
            stack.push(parent);
        }
    }
    seen
}

/// Apply the requested passes to `graph` and return the optimised graph.
///
/// The output list defines liveness: anything not reachable from an output is
/// subject to dead-code elimination. Constant folding uses `constants`, a map
/// from input node to value, for the leaves that are known at compile time.
pub fn optimize(graph: &Graph, outputs: &[NodeId], passes: &Passes) -> Optimized {
    optimize_with_constants(graph, outputs, passes, &HashMap::new())
}

/// Like [`optimize`], but with known constant leaves.
///
/// A node is constant when every reachable leaf below it is present in
/// `constants`. Such a subgraph is evaluated once and replaced by a new leaf
/// holding the result, which removes both the arithmetic and the temporaries.
///
/// Every node with a known value is rebuilt as a leaf with that value baked in —
/// including the leaves listed in `constants` themselves. `constants` therefore
/// means "known at compile time", not "a value the caller still has to supply":
/// the optimised graph runs with no input map for them at all.
pub fn optimize_with_constants<S: BuildHasher>(
    graph: &Graph,
    outputs: &[NodeId],
    passes: &Passes,
    constants: &HashMap<NodeId, Tensor, S>,
) -> Optimized {
    // --- 1. Liveness ------------------------------------------------------
    let live: HashSet<NodeId> = graph
        .reachable(outputs)
        .unwrap_or_else(|_| (0..graph.len()).map(NodeId).collect());
    let removed_dead = if passes.dead_code {
        graph.len() - live.len()
    } else {
        0
    };

    // --- 2. Constant folding ---------------------------------------------
    let folding = if passes.constant_folding {
        fold_constants(graph, &live, constants)
    } else {
        Folding::default()
    };
    let folded_count = folding.folded.len();
    let (known, folded) = (folding.known, folding.folded);
    // `folded` first: this decides which nodes are still needed in the rebuilt
    // graph, which is *not* the same as the pre-fold liveness set.
    let needed = if passes.constant_folding {
        let known_ids: HashSet<NodeId> = known.keys().copied().collect();
        needed_after_folding(graph, outputs, &known_ids)
    } else {
        live.clone()
    };

    // --- 3. Rebuild with CSE ---------------------------------------------
    let mut rebuilt = Graph::new();
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut deduplicated: BTreeMap<Signature, NodeId> = BTreeMap::new();
    let mut removed_duplicates = 0usize;

    for index in 0..graph.len() {
        let id = NodeId(index);
        if !live.contains(&id) && passes.dead_code {
            continue;
        }
        // Dropped because folding made it unreachable; it is not a dead-code
        // removal, which is why it is not counted in `removed_dead`.
        if passes.constant_folding && !needed.contains(&id) {
            continue;
        }
        let node = &graph.nodes()[index];

        // Every node with a known value becomes a leaf with that value baked in:
        // both the operations evaluated here and the caller's constant leaves.
        // Baking the leaves too is what makes `constants` mean "compile-time
        // constant" rather than "value the caller must remember to pass again".
        if known.contains_key(&id) {
            let value = known[&id].clone();
            let shape = value.shape().dims().to_vec();
            let name = if folded.contains(&id) {
                match node.name.as_deref() {
                    Some(name) => format!("{name}[folded]"),
                    None => format!("{}[folded]", node.op.name()),
                }
            } else {
                // A caller-supplied constant keeps its name; it is the same value.
                match node.name.as_deref() {
                    Some(name) => name.to_string(),
                    None => format!("{}[constant]", node.op.name()),
                }
            };
            let new_id = rebuilt.folded_leaf(shape, value, name);
            remap.insert(index, new_id.0);
            continue;
        }

        let mut operands = Vec::with_capacity(node.inputs.len());
        let mut missing = false;
        for parent in &node.inputs {
            if let Some(&new_parent) = remap.get(&parent.0) {
                operands.push(NodeId(new_parent));
            } else {
                missing = true;
                break;
            }
        }
        if missing {
            continue;
        }

        let signature = Signature {
            op: node.op.clone(),
            inputs: operands.iter().map(|id| id.0).collect(),
            leaf: if node.op == Op::Input {
                node.name.clone()
            } else {
                None
            },
        };
        if passes.common_subexpressions {
            if let Some(&existing) = deduplicated.get(&signature) {
                remap.insert(index, existing.0);
                removed_duplicates += 1;
                continue;
            }
        }

        let new_id = if node.op == Op::Input {
            match node.name.as_deref() {
                Some(name) => rebuilt.input(node.shape.clone(), name),
                None => rebuilt.input(node.shape.clone(), format!("input%{index}")),
            }
        } else {
            rebuilt
                .add_node(node.op.clone(), &operands)
                .unwrap_or_else(|_| rebuilt.input(node.shape.clone(), format!("invalid%{index}")))
        };
        deduplicated.insert(signature, new_id);
        remap.insert(index, new_id.0);
    }

    Optimized {
        graph: rebuilt,
        remap,
        removed_dead,
        removed_duplicates,
        folded: folded_count,
    }
}

/// Structural identity of a node, used for common-subexpression elimination.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Signature {
    op: Op,
    inputs: Vec<usize>,
    /// Name of the node, for `Input` only.
    ///
    /// Two `Input` nodes with the same shape are *not* interchangeable: each one
    /// carries a distinct value supplied by the caller. Hashing them together
    /// merged `w`, `b` and `bias` into a single leaf and produced a graph that
    /// could not be executed at all. Every other op is a pure function of its
    /// operands, so its name is irrelevant.
    leaf: Option<String>,
}

fn evaluate_constant(node: &Node, operands: &[Tensor]) -> Result<Tensor> {
    let value = match &node.op {
        Op::Input => {
            return Err(NclyError::Graph {
                reason: "an input node is never a constant".into(),
            })
        }
        Op::Add => operands[0].add(&operands[1])?,
        Op::Sub => operands[0].sub(&operands[1])?,
        Op::Mul => operands[0].mul(&operands[1])?,
        Op::Div => operands[0].div(&operands[1])?,
        Op::Matmul => operands[0].matmul(&operands[1])?,
        Op::Exp => operands[0].exp(),
        Op::Ln => operands[0].ln(),
        Op::Tanh => operands[0].tanh(),
        Op::Neg => operands[0].neg(),
        Op::Sum => Tensor::scalar(operands[0].sum()),
        Op::SumAxis(axis) => operands[0].sum_axis(*axis)?,
        Op::Broadcast(target) => operands[0].broadcast_to(target.clone())?.to_contiguous(),
        Op::Transpose => operands[0].transpose().to_contiguous(),
        Op::Reshape(target) => operands[0].reshape(target.clone())?,
    };
    Ok(value)
}

/// A printable description of what a pass set would do, without running it.
#[derive(Clone, Debug)]
pub struct Report {
    /// Nodes in the input graph.
    pub nodes_before: usize,
    /// Nodes the optimised graph would have.
    pub nodes_after: usize,
    /// Per-pass counts.
    pub passes: Passes,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} -> {} nodes (folding: {}, cse: {}, dce: {})",
            self.nodes_before,
            self.nodes_after,
            self.passes.constant_folding,
            self.passes.common_subexpressions,
            self.passes.dead_code
        )
    }
}

/// Summarise the effect of a pass set on a graph.
pub fn report(graph: &Graph, outputs: &[NodeId], passes: &Passes) -> Report {
    let optimised = optimize(graph, outputs, passes);
    Report {
        nodes_before: graph.len(),
        nodes_after: optimised.graph.len(),
        passes: *passes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(execution: &ncly_graph::Execution, id: NodeId) -> Vec<f64> {
        execution.get(id).unwrap().to_vec()
    }

    #[test]
    fn dead_nodes_are_removed() {
        let mut graph = Graph::new();
        let a = graph.input([2], "a");
        let dead = graph.add_node(Op::Exp, &[a]).unwrap();
        let kept = graph.add_node(Op::Neg, &[a]).unwrap();
        let result = optimize(&graph, &[kept], &Passes::all());
        assert_eq!(result.removed_dead, 1);
        assert!(!result.remap.contains_key(&dead.0));
        assert!(result.remap.contains_key(&kept.0));
    }

    #[test]
    fn distinct_inputs_are_never_merged() {
        // Two leaves with the same shape but different names carry different
        // caller-supplied values. Hashing them as one `(op, inputs)` signature
        // merged them into a single node, which made the optimised graph
        // impossible to execute: it asked for `w`'s value to compute `b`.
        let mut graph = Graph::new();
        let a = graph.input([2], "a");
        let b = graph.input([2], "b");
        let sum = graph.add_node(Op::Add, &[a, b]).unwrap();

        let result = optimize(&graph, &[sum], &Passes::all());
        assert_eq!(result.removed_duplicates, 0, "{}", result.graph.describe());
        assert_eq!(result.graph.len(), 3, "{}", result.graph.describe());
        assert_ne!(result.remap[&a.0], result.remap[&b.0]);

        // Both inputs survive as distinct nodes in the rewritten graph.
        let leaves: Vec<&str> = result
            .graph
            .nodes()
            .iter()
            .filter(|node| node.op == Op::Input)
            .filter_map(|node| node.name.as_deref())
            .collect();
        assert_eq!(leaves, vec!["a", "b"]);

        // And it executes with both values supplied, giving the right answer.
        let mut values = HashMap::new();
        values.insert(a, Tensor::from_vec(vec![1.0, 2.0], [2]).unwrap());
        values.insert(b, Tensor::from_vec(vec![10.0, 20.0], [2]).unwrap());
        let out = NodeId(result.remap[&sum.0]);
        let execution = result.graph.run(&[out], &values).unwrap();
        assert_eq!(execution.get(out).unwrap().to_vec(), vec![11.0, 22.0]);
    }

    #[test]
    fn duplicates_are_merged_and_the_result_is_unchanged() {
        let mut graph = Graph::new();
        let a = graph.input([3], "a");
        let first = graph.add_node(Op::Exp, &[a]).unwrap();
        let second = graph.add_node(Op::Exp, &[a]).unwrap();
        let sum = graph.add_node(Op::Add, &[first, second]).unwrap();

        let result = optimize(&graph, &[sum], &Passes::all());
        assert_eq!(result.removed_duplicates, 1);
        // Both old ids map to the same new node, so the rewritten add still
        // reads the value twice instead of reading an uninitialised buffer.
        assert_eq!(result.remap[&first.0], result.remap[&second.0]);

        // Numerical equivalence on a concrete input.
        let mut values = HashMap::new();
        values.insert(a, Tensor::from_vec(vec![0.0, 1.0, 2.0], [3]).unwrap());
        let before = graph.run(&[sum], &values).unwrap();
        let rewritten_sum = NodeId(result.remap[&sum.0]);
        let after_input = &result.graph.nodes()[0];
        assert_eq!(after_input.op, Op::Input);
        let after = result
            .graph
            .run(&[rewritten_sum], &{
                let mut map = HashMap::new();
                map.insert(NodeId(0), values[&a].clone());
                map
            })
            .unwrap();
        assert_eq!(value(&before, sum), value(&after, rewritten_sum));
    }

    #[test]
    fn constant_subgraphs_are_folded() {
        let mut graph = Graph::new();
        let a = graph.input([2], "a");
        let constant = graph.input([2], "c");
        // exp(c) + exp(c) has a constant subgraph; the add is constant too.
        let exp = graph.add_node(Op::Exp, &[constant]).unwrap();
        let double = graph.add_node(Op::Mul, &[exp, exp]).unwrap();
        let output = graph.add_node(Op::Add, &[a, double]).unwrap();

        let mut constants = HashMap::new();
        constants.insert(constant, Tensor::from_vec(vec![0.0, 1.0], [2]).unwrap());
        let result = optimize_with_constants(&graph, &[output], &Passes::all(), &constants);
        // `exp` and `mul` fold; the add cannot, because `a` is an input.
        assert_eq!(result.folded, 2);
        // Nodes: `a`, one baked leaf holding `exp(c)·exp(c)`, and the add.
        //
        // Both folded operations land in a single leaf, because the inner `exp` is
        // only reachable through the `mul`. `c` itself is gone: the only node that
        // consumed it was folded away, so nothing needs its value. No constant has
        // to be supplied at run time.
        assert_eq!(
            result.graph.len(),
            graph.len() - 2,
            "{}",
            result.graph.describe()
        );
        assert_eq!(result.removed_dead, 0, "{}", result.graph.describe());
        // Exactly one node carries a baked value, and it is the folded product.
        let baked: Vec<&str> = result
            .graph
            .nodes()
            .iter()
            .filter(|node| node.value.is_some())
            .filter_map(|node| node.name.as_deref())
            .collect();
        assert_eq!(baked, vec!["mul[folded]"], "{}", result.graph.describe());
        // `c` is not in the optimised graph at all: its only consumer was folded.
        assert!(
            !result
                .graph
                .nodes()
                .iter()
                .any(|node| node.name.as_deref() == Some("c")),
            "{}",
            result.graph.describe()
        );

        // Numerical check: the folded leaf equals exp(c) * exp(c).
        let mut values = HashMap::new();
        values.insert(a, Tensor::from_vec(vec![1.0, 2.0], [2]).unwrap());
        let expected = 1.0 + (0.0f64.exp() * 0.0f64.exp());
        let out = NodeId(result.remap[&output.0]);
        // Only `a` needs a value: the folded leaf carries its own.
        let execution = result.graph.run(&[out], &values).unwrap();
        let actual = execution.get(out).unwrap().get(&[0]).unwrap();
        assert!((actual - expected).abs() < 1e-15, "{actual} vs {expected}");
    }

    #[test]
    fn folding_matches_execution() {
        let mut graph = Graph::new();
        let c = graph.input([3], "c");
        let exp = graph.add_node(Op::Exp, &[c]).unwrap();
        let sum = graph.add_node(Op::Sum, &[exp]).unwrap();

        let mut constants = HashMap::new();
        constants.insert(c, Tensor::from_vec(vec![0.0, 1.0, 2.0], [3]).unwrap());
        let result = optimize_with_constants(&graph, &[sum], &Passes::all(), &constants);

        let mut values = HashMap::new();
        values.insert(c, Tensor::from_vec(vec![0.0, 1.0, 2.0], [3]).unwrap());
        let reference = graph.run(&[sum], &values).unwrap();
        let expected = reference.get(sum).unwrap().scalar_value().unwrap();

        let folded_output = NodeId(result.remap[&sum.0]);
        let folded = result.graph.run(&[folded_output], &HashMap::new()).unwrap();
        let actual = folded.get(folded_output).unwrap().scalar_value().unwrap();
        assert!((expected - actual).abs() < 1e-15, "{expected} vs {actual}");
    }

    #[test]
    fn disabling_passes_leaves_the_graph_intact() {
        let mut graph = Graph::new();
        let a = graph.input([2], "a");
        let first = graph.add_node(Op::Exp, &[a]).unwrap();
        let second = graph.add_node(Op::Exp, &[a]).unwrap();
        let result = optimize(&graph, &[first, second], &Passes::none());
        assert_eq!(result.graph.len(), graph.len());
        assert_eq!(result.nodes_removed(), 0);
    }

    #[test]
    fn remap_outputs_translates_ids() {
        let mut graph = Graph::new();
        let a = graph.input([2], "a");
        let output = graph.add_node(Op::Neg, &[a]).unwrap();
        let result = optimize(&graph, &[output], &Passes::all());
        let remapped = result.remap_outputs(&[output]).unwrap();
        assert_eq!(remapped.len(), 1);
        assert_eq!(remapped[0].0, result.remap[&output.0]);
    }

    #[test]
    fn report_describes_the_reduction() {
        let mut graph = Graph::new();
        let a = graph.input([2], "a");
        let keep = graph.add_node(Op::Neg, &[a]).unwrap();
        graph.add_node(Op::Exp, &[a]).unwrap();
        let summary = report(&graph, &[keep], &Passes::all());
        assert_eq!(summary.nodes_before, 3);
        assert_eq!(summary.nodes_after, 2);
        assert!(summary.to_string().contains("3 -> 2 nodes"));
    }
}
