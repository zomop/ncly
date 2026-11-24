//! Explicit dataflow graphs: shape inference, validation and execution.
//!
//! The tape in `ncly-autodiff` answers "what is the gradient?". This crate
//! answers a different question: "in what order do I run these operations, and
//! which of them can be skipped?".
//!
//! A [`Graph`] is a DAG of operations over tensors. Building one performs **shape
//! inference immediately**, so a graph that cannot be executed cannot be
//! constructed. That invariant is what the scheduler and, later, the optimiser
//! rely on.
//!
//! ```
//! use ncly_graph::{Graph, Op};
//!
//! // A rank-1 linear layer: y = tanh(x·w + b)
//! let mut g = Graph::new();
//! let x = g.input([4, 3], "x");
//! let w = g.input([3, 2], "w");
//! let b = g.input([2], "b");
//! let product = g.add_node(Op::Matmul, &[x, w])?;
//! let shifted = g.add_node(Op::Add, &[product, b])?;
//! let y = g.add_node(Op::Tanh, &[shifted])?;
//!
//! assert_eq!(g.shape_of(y), &[4, 2]);
//! let schedule = g.plan(&[y])?;
//! // Only the nodes the output depends on are executed.
//! assert_eq!(schedule.order.len(), 6);
//! # Ok::<(), ncly_core::NclyError>(())
//! ```
//!
//! # Execution
//!
//! [`Graph::run`] evaluates a schedule with a caller-supplied tensor map. The
//! evaluation itself is delegated to the kernels in `ncly-tensor`; this crate
//! owns ordering, validation and the topological structure, not the arithmetic.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

use std::collections::{HashMap, HashSet};
use std::fmt;

use ncly_core::{NclyError, Result, Shape};
use ncly_tensor::Tensor;

/// Identifier of a node inside a [`Graph`].
///
/// Ids are indices into the graph's node list, which makes them cheap to store
/// and lets the scheduler use plain `Vec` lookups.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeId(pub usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// An operation in the graph.
///
/// Every variant has a closed-form shape rule, applied by [`Graph::add_node`].
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Op {
    /// A leaf value supplied by the caller, or — after constant folding — a
    /// leaf whose `value` is baked into the node.
    Input,
    /// Element-wise addition with broadcasting.
    Add,
    /// Element-wise subtraction with broadcasting.
    Sub,
    /// Element-wise multiplication with broadcasting.
    Mul,
    /// Element-wise division with broadcasting.
    Div,
    /// Element-wise `exp`.
    Exp,
    /// Element-wise `ln`.
    Ln,
    /// Element-wise `tanh`.
    Tanh,
    /// Element-wise negation.
    Neg,
    /// Matrix product of the last two axes.
    Matmul,
    /// Sum of every element, producing a scalar.
    Sum,
    /// Sum along one axis.
    SumAxis(usize),
    /// Broadcast to a larger shape.
    Broadcast(Vec<usize>),
    /// Transpose (reverse the axes).
    Transpose,
    /// Reshape to a new shape.
    Reshape(Vec<usize>),
}

impl Op {
    /// Number of operands the operation takes.
    ///
    /// Zero for `Input`, two for the binary operations, one otherwise.
    pub fn arity(&self) -> usize {
        match self {
            Op::Input => 0,
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Matmul => 2,
            _ => 1,
        }
    }

    /// A short name used in diagnostics and in the debug dump.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Input => "input",
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::Div => "div",
            Op::Exp => "exp",
            Op::Ln => "ln",
            Op::Tanh => "tanh",
            Op::Neg => "neg",
            Op::Matmul => "matmul",
            Op::Sum => "sum",
            Op::SumAxis(_) => "sum_axis",
            Op::Broadcast(_) => "broadcast",
            Op::Transpose => "transpose",
            Op::Reshape(_) => "reshape",
        }
    }

    /// True when the operation reads more than one element per output element, so
    /// that a fused implementation would need more than one pass.
    fn is_reduction(&self) -> bool {
        matches!(self, Op::Sum | Op::SumAxis(_) | Op::Matmul)
    }

    /// Cost model used by the scheduler's tie-breaking, in abstract operations.
    fn cost(&self, shape: &[usize]) -> u64 {
        let elements = shape.iter().product::<usize>().max(1) as u64;
        match self {
            Op::Matmul => {
                // 2n³ for a square product; for non-square use the inner extent.
                elements.saturating_mul(2)
            }
            Op::Input => 0,
            op if op.is_reduction() => elements,
            _ => elements,
        }
    }
}

/// A node in the graph: an operation, its operands and its inferred shape.
#[derive(Clone, Debug)]
pub struct Node {
    /// The operation.
    pub op: Op,
    /// Operand ids, in argument order.
    pub inputs: Vec<NodeId>,
    /// Output shape, inferred when the node was added.
    pub shape: Vec<usize>,
    /// Optional name, set for graph inputs and used in dumps.
    pub name: Option<String>,
    /// A value that is known without executing anything.
    ///
    /// Only [`Op::Input`] nodes carry one: the optimiser (`ncly-compiler`) folds
    /// constant subgraphs by replacing them with such a leaf.
    pub value: Option<Tensor>,
}

impl Node {
    /// Number of output elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// True when the output has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A DAG of tensor operations with inferred shapes.
///
/// Nodes are stored in append order, and an edge always points from a lower id to
/// a higher id, so the node list is already a topological order.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Borrow a node.
    pub fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes.get(id.0).ok_or_else(|| NclyError::Graph {
            reason: format!("node {id} does not exist"),
        })
    }

    /// The inferred output shape of a node.
    pub fn shape_of(&self, id: NodeId) -> &[usize] {
        self.nodes
            .get(id.0)
            .map_or(&[], |node| node.shape.as_slice())
    }

    /// Every node, in append (topological) order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Add a graph input with a name.
    pub fn input(&mut self, shape: impl Into<Vec<usize>>, name: impl Into<String>) -> NodeId {
        let shape: Vec<usize> = shape.into();
        self.nodes.push(Node {
            op: Op::Input,
            inputs: Vec::new(),
            shape,
            name: Some(name.into()),
            value: None,
        });
        NodeId(self.nodes.len() - 1)
    }

    /// Add a leaf whose value is stored in the node.
    ///
    /// Used by the optimiser to replace a folded subgraph; `run` then supplies the
    /// value automatically, so no entry in the value map is needed.
    pub fn folded_leaf(
        &mut self,
        shape: impl Into<Vec<usize>>,
        value: Tensor,
        name: impl Into<String>,
    ) -> NodeId {
        let shape: Vec<usize> = shape.into();
        self.nodes.push(Node {
            op: Op::Input,
            inputs: Vec::new(),
            shape,
            name: Some(name.into()),
            value: Some(value),
        });
        NodeId(self.nodes.len() - 1)
    }

    /// Add a constant leaf whose value is supplied at execution time.
    ///
    /// This is plain [`Graph::input`]; the name exists to make intent explicit at
    /// call sites that assemble a graph from known constants.
    pub fn constant(&mut self, tensor: &Tensor, name: impl Into<String>) -> NodeId {
        self.input(tensor.shape().dims().to_vec(), name)
    }

    /// Add an operation node, checking arity and inferring the output shape.
    ///
    /// Fails with [`NclyError::Graph`] when the arity is wrong and with the
    /// underlying shape error when the operands are incompatible, so an
    /// unexecutable graph cannot be built.
    pub fn add_node(&mut self, op: Op, inputs: &[NodeId]) -> Result<NodeId> {
        if inputs.len() != op.arity() {
            return Err(NclyError::Graph {
                reason: format!(
                    "{} expects {} operand(s) but {} were given",
                    op.name(),
                    op.arity(),
                    inputs.len()
                ),
            });
        }
        let mut operand_shapes = Vec::with_capacity(inputs.len());
        for &id in inputs {
            if id.0 >= self.nodes.len() {
                return Err(NclyError::Graph {
                    reason: format!("operand {id} does not exist"),
                });
            }
            operand_shapes.push(self.nodes[id.0].shape.clone());
        }
        let shape = infer_shape(&op, &operand_shapes)?;
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            shape,
            name: None,
            value: None,
        });
        Ok(NodeId(self.nodes.len() - 1))
    }

    /// Nodes reachable from `outputs` (the transitive closure of the operands).
    ///
    /// This is what makes dead-node elimination possible: nodes outside the
    /// closure cannot influence any output.
    pub fn reachable(&self, outputs: &[NodeId]) -> Result<HashSet<NodeId>> {
        let mut seen = HashSet::new();
        let mut stack: Vec<NodeId> = outputs.to_vec();
        for &id in outputs {
            self.node(id)?;
        }
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            for &parent in &self.nodes[id.0].inputs {
                stack.push(parent);
            }
        }
        Ok(seen)
    }

    /// Validate that every output is reachable from a graph input and that no
    /// node is used before it is produced.
    ///
    /// Cycles cannot be constructed through the public API (an edge always points
    /// to a lower id), but the check is kept as an assertion about the invariant.
    pub fn validate(&self, outputs: &[NodeId]) -> Result<()> {
        for (index, node) in self.nodes.iter().enumerate() {
            for (position, &parent) in node.inputs.iter().enumerate() {
                if parent.0 >= index {
                    return Err(NclyError::Graph {
                        reason: format!(
                            "node %{index} reads operand {position} = {parent}, which is not defined earlier"
                        ),
                    });
                }
            }
        }
        let reachable = self.reachable(outputs)?;
        for &id in &reachable {
            let node = self.node(id)?;
            if node.op == Op::Input {
                continue;
            }
            if node.inputs.iter().any(|parent| !reachable.contains(parent)) {
                return Err(NclyError::Graph {
                    reason: format!("node {id} has an operand outside the reachable set"),
                });
            }
        }
        Ok(())
    }

    /// Produce an execution plan for `outputs`.
    ///
    /// The plan is a topological order over the reachable subgraph with dead nodes
    /// removed, plus the buffer requirements of each step. Within a level,
    /// operations are ordered by descending cost so that expensive work starts
    /// first, which matters once execution is parallel.
    pub fn plan(&self, outputs: &[NodeId]) -> Result<Schedule> {
        self.validate(outputs)?;
        let reachable = self.reachable(outputs)?;

        let mut order: Vec<NodeId> = reachable.iter().copied().collect();
        // Nodes were appended in topological order, so sorting by id is a valid
        // schedule.
        order.sort();

        // Group into levels: a node's level is one past the maximum level of its
        // operands, so all operands of a level are produced before it runs.
        let mut levels: HashMap<NodeId, usize> = HashMap::new();
        let mut depth = 0usize;
        for &id in &order {
            let node = self.node(id)?;
            let level = node
                .inputs
                .iter()
                .filter_map(|parent| levels.get(parent))
                .max()
                .map_or(0, |parent_level| parent_level + 1);
            levels.insert(id, level);
            depth = depth.max(level + 1);
        }

        let mut staged: Vec<Vec<NodeId>> = vec![Vec::new(); depth];
        for &id in &order {
            let level = levels[&id];
            staged[level].push(id);
        }
        for level in &mut staged {
            level.sort_by_key(|id| {
                let node = &self.nodes[id.0];
                std::cmp::Reverse(node.op.cost(&node.shape))
            });
        }

        let live: Vec<NodeId> = staged.iter().flatten().copied().collect();
        let dead = self.nodes.len() - live.len();
        Ok(Schedule {
            order: live,
            levels: staged,
            outputs: outputs.to_vec(),
            dead_nodes: dead,
        })
    }

    /// Evaluate the graph for the given outputs.
    ///
    /// `values` must supply a tensor for every `Input` node that is reachable from
    /// `outputs`. The returned map holds every executed node.
    pub fn run(&self, outputs: &[NodeId], values: &HashMap<NodeId, Tensor>) -> Result<Execution> {
        let schedule = self.plan(outputs)?;
        let mut environment: HashMap<NodeId, Tensor> = HashMap::with_capacity(schedule.order.len());
        for &id in &schedule.order {
            let node = self.node(id)?;
            let output = if node.op == Op::Input {
                match node.value.clone().or_else(|| values.get(&id).cloned()) {
                    Some(value) => value,
                    None => {
                        return Err(NclyError::Graph {
                            reason: format!(
                                "no value supplied for input {}",
                                node.name.as_deref().unwrap_or("(unnamed)")
                            ),
                        })
                    }
                }
            } else {
                evaluate(&node.op, &node.inputs, &environment)?
            };
            if output.shape().dims() != node.shape.as_slice() {
                return Err(NclyError::Graph {
                    reason: format!(
                        "node {id} inferred {:?} but produced {:?}",
                        node.shape,
                        output.shape().dims()
                    ),
                });
            }
            environment.insert(id, output);
        }
        Ok(Execution {
            schedule,
            values: environment,
        })
    }

    /// The value baked into a leaf, if any.
    pub fn baked_value(&self, id: NodeId) -> Option<&Tensor> {
        self.nodes.get(id.0).and_then(|node| node.value.as_ref())
    }

    /// A human-readable dump, one node per line, in topological order.
    pub fn describe(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        for (index, node) in self.nodes.iter().enumerate() {
            let name = node.name.as_deref().unwrap_or("");
            let _ = writeln!(
                out,
                "%{index} {}({:?}) -> {:?} {name}",
                node.op.name(),
                node.inputs,
                node.shape
            );
        }
        out
    }
}

/// A validated execution plan.
#[derive(Clone, Debug)]
pub struct Schedule {
    /// Nodes to execute, in topological order, with dead nodes removed.
    pub order: Vec<NodeId>,
    /// The same nodes grouped into dependency levels.
    pub levels: Vec<Vec<NodeId>>,
    /// The outputs the plan was built for.
    pub outputs: Vec<NodeId>,
    /// How many nodes were skipped as dead.
    pub dead_nodes: usize,
}

impl Schedule {
    /// Number of execution levels; the critical path length of the graph.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Number of operations to execute.
    pub fn steps(&self) -> usize {
        self.order.len()
    }
}

/// The result of [`Graph::run`]: the schedule and every computed buffer.
#[derive(Clone, Debug)]
pub struct Execution {
    /// The plan that was executed.
    pub schedule: Schedule,
    /// Values of every executed node.
    pub values: HashMap<NodeId, Tensor>,
}

impl Execution {
    /// Borrow the value of one node.
    pub fn get(&self, id: NodeId) -> Result<&Tensor> {
        self.values.get(&id).ok_or_else(|| NclyError::Graph {
            reason: format!("node {id} was not executed"),
        })
    }
}

fn broadcast_binary(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    Ok(Shape::new(lhs.to_vec())
        .broadcast_with(&Shape::new(rhs.to_vec()))?
        .dims()
        .to_vec())
}

fn infer_shape(op: &Op, operands: &[Vec<usize>]) -> Result<Vec<usize>> {
    let first = operands.first().map_or(&[][..], Vec::as_slice);
    match op {
        Op::Input | Op::Exp | Op::Ln | Op::Tanh | Op::Neg => Ok(first.to_vec()),
        Op::Add | Op::Sub | Op::Mul | Op::Div => broadcast_binary(first, &operands[1]),
        Op::Matmul => {
            let lhs = Shape::new(operands[0].clone());
            let rhs = Shape::new(operands[1].clone());
            if lhs.rank() == 0 || rhs.rank() == 0 {
                return Err(NclyError::InvalidShape {
                    op: "matmul",
                    reason: "operands must have at least one axis".into(),
                });
            }
            let (rows, inner) = match lhs.rank() {
                1 => (1, lhs.dims()[0]),
                2 => (lhs.dims()[0], lhs.dims()[1]),
                _ => {
                    return Err(NclyError::InvalidShape {
                        op: "matmul",
                        reason: "left operand must be rank 1 or 2".into(),
                    })
                }
            };
            let (rhs_inner, cols) = match rhs.rank() {
                1 => (rhs.dims()[0], 1),
                2 => (rhs.dims()[0], rhs.dims()[1]),
                _ => {
                    return Err(NclyError::InvalidShape {
                        op: "matmul",
                        reason: "right operand must be rank 1 or 2".into(),
                    })
                }
            };
            if inner != rhs_inner {
                return Err(NclyError::shape_mismatch(
                    "matmul",
                    format!("inner extent {inner}"),
                    format!("inner extent {rhs_inner}"),
                ));
            }
            Ok(match (lhs.rank(), rhs.rank()) {
                (2, 2) => vec![rows, cols],
                (1, 2) => vec![cols],
                (2, 1) => vec![rows],
                _ => vec![1],
            })
        }
        Op::Sum => Ok(Vec::new()),
        Op::SumAxis(axis) => {
            let axis = *axis;
            if axis >= first.len() {
                return Err(NclyError::InvalidShape {
                    op: "sum_axis",
                    reason: format!("axis {axis} is out of range for rank {}", first.len()),
                });
            }
            let mut shape = first.to_vec();
            shape.remove(axis);
            Ok(shape)
        }
        Op::Broadcast(target) => {
            let broadcast =
                Shape::new(first.to_vec()).broadcast_with(&Shape::new(target.clone()))?;
            if broadcast.dims() != target.as_slice() {
                return Err(NclyError::InvalidShape {
                    op: "broadcast",
                    reason: format!("{first:?} cannot broadcast to {target:?}"),
                });
            }
            Ok(target.clone())
        }
        Op::Transpose => {
            let mut shape = first.to_vec();
            shape.reverse();
            Ok(shape)
        }
        Op::Reshape(target) => {
            let source_size: usize = first.iter().product();
            let target_size: usize = target.iter().product();
            if source_size != target_size {
                return Err(NclyError::InvalidShape {
                    op: "reshape",
                    reason: format!(
                        "cannot reshape {first:?} ({source_size}) to {target:?} ({target_size})"
                    ),
                });
            }
            Ok(target.clone())
        }
    }
}

fn argument(environment: &HashMap<NodeId, Tensor>, id: NodeId) -> Result<&Tensor> {
    environment.get(&id).ok_or_else(|| NclyError::Graph {
        reason: format!("operand {id} has no value at execution time"),
    })
}

fn evaluate(op: &Op, inputs: &[NodeId], environment: &HashMap<NodeId, Tensor>) -> Result<Tensor> {
    Ok(match op {
        Op::Input => {
            return Err(NclyError::Graph {
                reason: "input nodes are supplied by the caller".into(),
            })
        }
        Op::Add => argument(environment, inputs[0])?.add(argument(environment, inputs[1])?)?,
        Op::Sub => argument(environment, inputs[0])?.sub(argument(environment, inputs[1])?)?,
        Op::Mul => argument(environment, inputs[0])?.mul(argument(environment, inputs[1])?)?,
        Op::Div => argument(environment, inputs[0])?.div(argument(environment, inputs[1])?)?,
        Op::Matmul => {
            argument(environment, inputs[0])?.matmul(argument(environment, inputs[1])?)?
        }
        Op::Exp => argument(environment, inputs[0])?.exp(),
        Op::Ln => argument(environment, inputs[0])?.ln(),
        Op::Tanh => argument(environment, inputs[0])?.tanh(),
        Op::Neg => argument(environment, inputs[0])?.neg(),
        Op::Sum => Tensor::scalar(argument(environment, inputs[0])?.sum()),
        Op::SumAxis(axis) => argument(environment, inputs[0])?.sum_axis(*axis)?,
        Op::Broadcast(target) => argument(environment, inputs[0])?
            .broadcast_to(target.clone())?
            .to_contiguous(),
        Op::Transpose => argument(environment, inputs[0])?
            .transpose()
            .to_contiguous(),
        Op::Reshape(target) => argument(environment, inputs[0])?.reshape(target.clone())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond() -> (Graph, NodeId, NodeId) {
        // y = (a + b) * (a + b) — a shared subexpression.
        let mut graph = Graph::new();
        let a = graph.input([2, 3], "a");
        let b = graph.input([2, 3], "b");
        let sum = graph.add_node(Op::Add, &[a, b]).unwrap();
        let product = graph.add_node(Op::Mul, &[sum, sum]).unwrap();
        (graph, sum, product)
    }

    #[test]
    fn shapes_are_inferred_on_construction() {
        let (graph, sum, product) = diamond();
        assert_eq!(graph.shape_of(sum), &[2, 3]);
        assert_eq!(graph.shape_of(product), &[2, 3]);
    }

    #[test]
    fn broadcast_shapes_follow_numpy_rules() {
        let mut graph = Graph::new();
        let matrix = graph.input([4, 3], "m");
        let row = graph.input([3], "r");
        let sum = graph.add_node(Op::Add, &[matrix, row]).unwrap();
        assert_eq!(graph.shape_of(sum), &[4, 3]);

        let column = graph.input([4, 1], "c");
        let sum = graph.add_node(Op::Add, &[matrix, column]).unwrap();
        assert_eq!(graph.shape_of(sum), &[4, 3]);

        let incompatible = graph.input([5], "bad");
        assert!(graph.add_node(Op::Add, &[matrix, incompatible]).is_err());
    }

    #[test]
    fn matmul_shape_rules_cover_vector_cases() {
        let mut graph = Graph::new();
        let m = graph.input([2, 3], "m");
        let v = graph.input([3], "v");
        let w = graph.input([3, 4], "w");
        let matrix_vector = graph.add_node(Op::Matmul, &[m, v]).unwrap();
        assert_eq!(graph.shape_of(matrix_vector), &[2]);
        let product = graph.add_node(Op::Matmul, &[m, w]).unwrap();
        assert_eq!(graph.shape_of(product), &[2, 4]);

        let mismatched = graph.input([5], "bad");
        assert!(graph.add_node(Op::Matmul, &[m, mismatched]).is_err());
    }

    #[test]
    fn arity_is_checked() {
        let mut graph = Graph::new();
        let a = graph.input([2], "a");
        let error = graph.add_node(Op::Add, &[a]).unwrap_err();
        assert!(matches!(error, NclyError::Graph { .. }));
    }

    #[test]
    fn shared_subexpressions_are_executed_once() {
        let (graph, _, product) = diamond();
        let schedule = graph.plan(&[product]).unwrap();
        // a, b, add, mul
        assert_eq!(schedule.steps(), 4);
        let adds = schedule
            .order
            .iter()
            .filter(|id| graph.node(**id).unwrap().op == Op::Add)
            .count();
        assert_eq!(adds, 1);
    }

    #[test]
    fn dead_nodes_are_removed_from_the_schedule() {
        let (mut graph, _, product) = diamond();
        let unused = graph.input([2, 3], "unused");
        let dead = graph.add_node(Op::Exp, &[unused]).unwrap();
        let schedule = graph.plan(&[product]).unwrap();
        assert_eq!(schedule.steps(), 4);
        assert_eq!(schedule.dead_nodes, 2);
        assert!(!schedule.order.contains(&dead));
    }

    #[test]
    fn levels_respect_dependencies() {
        let (graph, sum, product) = diamond();
        let schedule = graph.plan(&[product]).unwrap();
        // inputs | add | mul
        assert_eq!(schedule.depth(), 3);
        let position = |id: NodeId| {
            schedule
                .levels
                .iter()
                .position(|level| level.contains(&id))
                .unwrap()
        };
        assert!(position(sum) < position(product));
    }

    #[test]
    fn run_evaluates_the_graph() {
        let mut graph = Graph::new();
        let a = graph.input([2], "a");
        let b = graph.input([2], "b");
        let product = graph.add_node(Op::Mul, &[a, b]).unwrap();
        let total = graph.add_node(Op::Sum, &[product]).unwrap();

        let mut values = HashMap::new();
        values.insert(a, Tensor::from_vec(vec![1.0, 2.0], [2]).unwrap());
        values.insert(b, Tensor::from_vec(vec![3.0, 4.0], [2]).unwrap());

        let execution = graph.run(&[total], &values).unwrap();
        assert_eq!(execution.get(total).unwrap().scalar_value().unwrap(), 11.0);
        // Intermediate buffers are kept for inspection.
        assert_eq!(execution.get(product).unwrap().to_vec(), vec![3.0, 8.0]);
    }

    #[test]
    fn missing_input_values_are_reported() {
        let (graph, _, product) = diamond();
        let error = graph.run(&[product], &HashMap::new()).unwrap_err();
        match error {
            NclyError::Graph { reason } => assert!(reason.contains("no value supplied")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn plan_rejects_unknown_outputs() {
        let graph = Graph::new();
        assert!(graph.plan(&[NodeId(7)]).is_err());
    }

    #[test]
    fn describe_lists_nodes_in_topological_order() {
        let (graph, _, _) = diamond();
        let dump = graph.describe();
        assert_eq!(dump.lines().count(), 4);
        assert!(dump.contains("%2 add([NodeId(0), NodeId(1)])"));
    }

    #[test]
    fn reshape_and_transpose_shapes() {
        let mut graph = Graph::new();
        let m = graph.input([2, 3], "m");
        let flat = graph.add_node(Op::Reshape(vec![6]), &[m]).unwrap();
        assert_eq!(graph.shape_of(flat), &[6]);
        let transposed = graph.add_node(Op::Transpose, &[m]).unwrap();
        assert_eq!(graph.shape_of(transposed), &[3, 2]);
        let bad = graph.add_node(Op::Reshape(vec![5]), &[m]);
        assert!(bad.is_err());
    }
}
