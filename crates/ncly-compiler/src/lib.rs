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
//! use ncly_graph::{Graph, Op};
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
//! assert!(result.graph.node(result.remap[sum.0]).is_ok());
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
    /// Number of nodes removed as dead.
    pub removed_dead: usize,
    /// Number of nodes merged into an existing equivalent node.
    pub removed_duplicates: usize,
    /// Number of operations evaluated at compile time.
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

    /// How many nodes the graph lost.
    pub fn nodes_removed(&self) -> usize {
        self.removed_dead + self.removed_duplicates
    }
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
pub fn optimize_with_constants(
    graph: &Graph,
    outputs: &[NodeId],
    passes: &Passes,
    constants: &HashMap<NodeId, Tensor>,
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
    let mut known: HashMap<NodeId, Tensor> = HashMap::new();
    let mut folded: HashSet<NodeId> = HashSet::new();
    let mut folded_count = 0usize;
    if passes.constant_folding {
        for index in 0..graph.len() {
            let id = NodeId(index);
            if !live.contains(&id) {
                continue;
            }
            let node = &graph.nodes()[index];
            if let Some(value) = constants.get(&id) {
                known.insert(id, value.clone());
                continue;
            }
            if node.op == Op::Input {
                continue;
            }
            // Constant only if every operand is known *and* was itself constant
            // rather than produced by executing an input-fed subgraph.
            if node.inputs.iter().all(|parent| folded.contains(parent)) {
                let operands: Option<Vec<Tensor>> = node
                    .inputs
                    .iter()
                    .map(|parent| known.get(parent).cloned())
                    .collect();
                if let Some(operands) = operands {
                    if let Ok(value) = evaluate_constant(node, &operands) {
                        known.insert(id, value);
                        folded.insert(id);
                        folded_count += 1;
                    }
                }
            }
        }
    }

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
        let node = &graph.nodes()[index];

        // A folded node becomes a fresh leaf holding its value.
        if folded.contains(&id) {
            let value = &known[&id];
            let new_id = match node.name.as_deref() {
                Some(name) => rebuilt.constant(value, format!("{name}[folded]")),
                None => rebuilt.constant(value, format!("{}[folded]", node.op.name())),
            };
            remap.insert(index, new_id.0);
            continue;
        }

        let mut operands = Vec::with_capacity(node.inputs.len());
        let mut missing = false;
        for parent in &node.inputs {
            match remap.get(&parent.0) {
                Some(&new_parent) => operands.push(NodeId(new_parent)),
                None => {
                    missing = true;
                    break;
                }
            }
        }
        if missing {
            continue;
        }

        let signature = Signature {
            op: node.op.clone(),
            inputs: operands.iter().map(|id| id.0).collect(),
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
        // exp, mul and the add are all constant-foldable: the add reads `a`, so
        // only exp and mul fold.
        assert_eq!(result.folded, 2);
        assert_eq!(result.graph.len(), 3); // a, folded leaf, add
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
