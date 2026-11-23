//! Reverse-mode automatic differentiation on a dynamically grown tape.
//!
//! # Design
//!
//! The tape is a `Vec<Node>`; node `i` is created after every node it depends
//! on, so the tape indices form a valid topological order and a backward pass
//! is a single reverse iteration — no recursion, no explicit graph traversal.
//!
//! ```text
//! node 0: leaf x = 2          parents []
//! node 1: leaf y = 3          parents []
//! node 2: mul(0, 1)  = 6      parents [(0, 3.0), (1, 2.0)]   ← local partials
//! node 3: powi(2, 2) = 36     parents [(2, 12.0)]
//! node 4: sum(3)     = 36     parents [(3, 1.0)]
//! ```
//!
//! Backward starts with `adjoint[4] = 1` and visits nodes in decreasing order:
//! `adjoint[parent] += adjoint[child] * partial`. Each parent's local partial is
//! evaluated **during the forward pass**, when the primal values are still
//! cheap to read, which is why the tape stores `(parent, partial)` pairs rather
//! than closures.
//!
//! # Semantics
//!
//! * Gradients **accumulate**: calling `backward` twice doubles the gradient,
//!   exactly like PyTorch. Call [`Variable::zero_grad`]/[`Tape::zero_grad`] to
//!   reset.
//! * Operations are symbolic over [`Variable`] handles; nothing is evaluated
//!   in reverse mode. [`Variable::to_tensor`] materialises the primal value.
//! * Broadcasting is handled by [`Variable::sum_to`], which collapses the
//!   adjoint of the broadcast result back onto the operand's shape.

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};
use ncly_core::{Layout, NclyError, Result, Shape};
use ncly_tensor::Tensor;
use std::sync::{Arc, Mutex, MutexGuard};

/// How an adjoint flows from a node back to one of its parents.
#[derive(Debug, Clone)]
pub enum LocalGrad {
    /// Multiply the adjoint by a local partial tensor, then reduce any
    /// broadcasting back onto the parent's shape.
    Elementwise(Tensor),
    /// Permute the adjoint's axes, used by nodes that rearrange data
    /// (transpose). The permutation is an axis order: `order[i]` is the axis of
    /// the parent that output axis `i` came from.
    Permute(Vec<usize>),
}

/// One recorded operation.
#[derive(Debug, Clone)]
pub struct VariableNode {
    /// Primitive operation name, for diagnostics and graph dumps.
    pub op: &'static str,
    /// Primal value of this node.
    pub data: Tensor,
    /// `(parent index, local gradient rule)` pairs.
    pub parents: Vec<(usize, LocalGrad)>,
    /// Accumulated adjoint, written during the backward sweep.
    pub adjoint: Tensor,
    /// Whether this node is a leaf the caller can read a gradient from.
    pub is_leaf: bool,
    /// Optional name for debugging.
    pub name: Option<String>,
}

impl VariableNode {
    fn new(op: &'static str, data: Tensor, parents: Vec<(usize, LocalGrad)>) -> Self {
        let adjoint = Tensor::zeros(data.shape().dims().to_vec());
        Self {
            op,
            data,
            parents,
            adjoint,
            is_leaf: false,
            name: None,
        }
    }
}

/// An append-only recording of operations.
///
/// A tape is owned by the [`Variable`]s it produced (via `Arc<Mutex<_>>`), so
/// building an expression tree keeps every intermediate alive. Tapes are
/// `Send + Sync`; gradients can be computed from any thread.
#[derive(Debug, Default, Clone)]
pub struct Tape {
    nodes: Arc<Mutex<Vec<VariableNode>>>,
}

impl Tape {
    /// A fresh tape.
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Vec<VariableNode>> {
        match self.nodes.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Number of recorded nodes.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// True when nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Allocate a leaf node and return its index.
    pub fn leaf(&self, data: Tensor, name: Option<String>) -> usize {
        let mut nodes = self.lock();
        let index = nodes.len();
        let mut node = VariableNode::new("leaf", data, Vec::new());
        node.is_leaf = true;
        node.name = name;
        nodes.push(node);
        index
    }

    /// Record an operation and return the new node index.
    pub fn record(&self, op: &'static str, data: Tensor, parents: Vec<(usize, LocalGrad)>) -> usize {
        let mut nodes = self.lock();
        let index = nodes.len();
        // Parents must already exist; a violation would make the reverse sweep
        // silently wrong, so it is checked in release builds too.
        for (parent, _) in &parents {
            assert!(*parent < index, "autodiff: parent recorded after child");
        }
        nodes.push(VariableNode::new(op, data, parents));
        index
    }

    /// Primitive value of a node.
    pub fn data(&self, index: usize) -> Tensor {
        self.lock()[index].data.clone()
    }

    /// Operation name of a node.
    pub fn op(&self, index: usize) -> &'static str {
        self.lock()[index].op
    }

    /// Parents recorded for a node.
    pub fn parents(&self, index: usize) -> Vec<usize> {
        self.lock()[index].parents.iter().map(|&(p, _)| p).collect()
    }

    /// Current adjoint of a node.
    pub fn adjoint(&self, index: usize) -> Tensor {
        self.lock()[index].adjoint.clone()
    }

    /// Clear every adjoint.
    pub fn zero_grad(&self) {
        for node in self.lock().iter_mut() {
            node.adjoint = Tensor::zeros(node.data.shape().dims().to_vec());
        }
    }

    /// Seed `index` with ones and run the reverse sweep.
    ///
    /// # Errors
    ///
    /// Returns [`NclyError::Graph`] when `index` does not exist.
    pub fn backward(&self, index: usize) -> Result<()> {
        let mut nodes = self.lock();
        if index >= nodes.len() {
            return Err(NclyError::Graph {
                reason: format!("backward called on unknown node {index}"),
            });
        }
        let shape = nodes[index].data.shape().clone();
        nodes[index].adjoint = Tensor::ones(shape.dims().to_vec());

        for cursor in (0..nodes.len()).rev() {
            // Nodes downstream of the seed start at zero, so they contribute
            // nothing. Skipping them also avoids *materialising* zero adjoints,
            // which is what makes a backward sweep over a large tape cheap.
            if nodes[cursor].adjoint.is_zero() {
                continue;
            }
            let adjoint = nodes[cursor].adjoint.clone();
            if adjoint.len() == 0 {
                continue;
            }
            // Collect first to avoid holding a borrow while indexing parents.
            let parents = nodes[cursor].parents.clone();
            for (parent, rule) in parents {
                let target = nodes[parent].data.shape().clone();
                let reduced = match rule {
                    LocalGrad::Elementwise(partial) => {
                        let contribution = adjoint.mul(&partial)?;
                        sum_to_shape(&contribution, &target)?
                    }
                    LocalGrad::Permute(order) => {
                        let permuted = permute_axes(&adjoint, &order);
                        if permuted.shape() != &target {
                            return Err(NclyError::Graph {
                                reason: format!(
                                    "permuted adjoint has shape {} but parent expects {}",
                                    permuted.shape(),
                                    target
                                ),
                            });
                        }
                        permuted
                    }
                };
                let updated = nodes[parent].adjoint.add(&reduced)?;
                nodes[parent].adjoint = updated;
            }
        }
        Ok(())
    }

    /// Render the tape as a list of `index: op(shape) <- parents` lines.
    pub fn describe(&self) -> String {
        let nodes = self.lock();
        let mut out = String::new();
        for (index, node) in nodes.iter().enumerate() {
            let parents: Vec<String> = node.parents.iter().map(|&(p, _)| p.to_string()).collect();
            out.push_str(&format!(
                "{index}: {}({}){}{} <- [{}]\n",
                node.op,
                node.data.shape(),
                if node.is_leaf { " leaf" } else { "" },
                node.name
                    .as_ref()
                    .map(|n| format!(" {n}"))
                    .unwrap_or_default(),
                parents.join(", ")
            ));
        }
        out
    }
}

/// Reorder the axes of `value` and materialise the result.
fn permute_axes(value: &Tensor, order: &[usize]) -> Tensor {
    let mut dims = Vec::with_capacity(order.len());
    let mut strides = Vec::with_capacity(order.len());
    for &axis in order {
        dims.push(value.shape().dims()[axis]);
        strides.push(value.strides().get(axis).unwrap_or(0));
    }
    Tensor::from_parts(
        Arc::clone(value.storage()),
        Layout::new(Shape::new(dims), Strides::new(strides), value.layout().offset()),
    )
    .to_contiguous()
}

/// Sum a broadcast contribution back down to `target`.
fn sum_to_shape(value: &Tensor, target: &Shape) -> Result<Tensor> {
    if value.shape() == target {
        return Ok(value.clone());
    }
    if value.rank() < target.rank() {
        return Err(NclyError::Graph {
            reason: format!(
                "cannot reduce shape {} to higher-rank shape {}",
                value.shape(),
                target
            ),
        });
    }
    let pad = value.rank() - target.rank();
    let mut current = value.clone();
    // Leading axes added by broadcasting are summed away first.
    for axis in (0..pad).rev() {
        current = current.sum_axis(axis)?;
    }
    // Then axes the operand had with length 1 are collapsed with keepdim so
    // that the remaining axis numbering stays stable.
    for axis in 0..target.rank() {
        if target.dims()[axis] == 1 && current.shape().dims()[axis] != 1 {
            current = current.sum_axis_keepdim(axis)?;
        }
    }
    if current.shape() != target {
        return Err(NclyError::Graph {
            reason: format!(
                "gradient reduction produced {} instead of {}",
                current.shape(),
                target
            ),
        });
    }
    Ok(current)
}

/// A handle to a differentiable value.
///
/// Cloning a `Variable` clones the handle (same tape node), not the value.
#[derive(Clone)]
pub struct Variable {
    tape: Tape,
    index: usize,
}

impl Variable {
    /// Create a leaf variable on a fresh tape.
    pub fn new(value: f64, name: impl Into<String>) -> Self {
        let tape = Tape::new();
        let index = tape.leaf(Tensor::scalar(value), Some(name.into()));
        Self { tape, index }
    }

    /// Create a leaf variable from a tensor on a fresh tape.
    pub fn from_tensor(value: Tensor, name: impl Into<String>) -> Self {
        let tape = Tape::new();
        let index = tape.leaf(value, Some(name.into()));
        Self { tape, index }
    }

    /// Create a leaf variable on an existing tape (for shared recordings).
    pub fn on_tape(tape: &Tape, value: Tensor, name: impl Into<String>) -> Self {
        let index = tape.leaf(value, Some(name.into()));
        Self {
            tape: tape.clone(),
            index,
        }
    }

    /// The tape this variable lives on.
    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// This variable's node index.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The primal value.
    pub fn value(&self) -> Tensor {
        self.tape.data(self.index)
    }

    /// The primal value as a scalar.
    ///
    /// # Errors
    ///
    /// Returns an error unless the variable holds exactly one element.
    pub fn scalar(&self) -> Result<f64> {
        self.value().scalar_value()
    }

    /// The accumulated gradient (zeros before `backward`).
    pub fn grad(&self) -> Tensor {
        self.tape.adjoint(self.index)
    }

    /// The accumulated gradient as a scalar.
    pub fn grad_scalar(&self) -> Result<f64> {
        self.grad().scalar_value()
    }

    /// Run the reverse sweep from this variable.
    pub fn backward(&self) {
        // A failure here means the tape is corrupt; surfacing it as a panic is
        // preferable to silently producing wrong gradients.
        if let Err(err) = self.tape.backward(self.index) {
            panic!("ncly-autodiff: backward failed: {err}");
        }
    }

    /// Reset every adjoint on the tape to zero.
    pub fn zero_grad(&self) {
        self.tape.zero_grad();
    }

    fn record(&self, op: &'static str, data: Tensor, parents: Vec<(usize, Tensor)>) -> Variable {
        let index = self.tape.record(op, data, parents);
        Variable {
            tape: self.tape.clone(),
            index,
        }
    }

    // -----------------------------------------------------------------
    // Differentiable operations
    // -----------------------------------------------------------------

    /// Element-wise sum of two variables (with broadcasting).
    pub fn add(&self, rhs: &Variable) -> Variable {
        let data = self
            .value()
            .add(&rhs.value())
            .expect("broadcast failed in add");
        let left = Tensor::ones(self.value().shape().dims().to_vec());
        let right = Tensor::ones(rhs.value().shape().dims().to_vec());
        self.record("add", data, vec![(self.index, LocalGrad::Elementwise(left)), (rhs.index, LocalGrad::Elementwise(right))])
    }

    /// Element-wise subtraction (with broadcasting).
    pub fn sub(&self, rhs: &Variable) -> Variable {
        let data = self
            .value()
            .sub(&rhs.value())
            .expect("broadcast failed in sub");
        let left = Tensor::ones(self.value().shape().dims().to_vec());
        let right = Tensor::ones(rhs.value().shape().dims().to_vec()).neg();
        self.record("sub", data, vec![(self.index, LocalGrad::Elementwise(left)), (rhs.index, LocalGrad::Elementwise(right))])
    }

    /// Element-wise product (with broadcasting).
    pub fn mul(&self, rhs: &Variable) -> Variable {
        let (a, b) = (self.value(), rhs.value());
        let data = a.mul(&b).expect("broadcast failed in mul");
        // ∂(ab)/∂a = b ; ∂(ab)/∂b = a
        self.record(
            "mul",
            data,
            vec![(self.index, LocalGrad::Elementwise(b)), (rhs.index, LocalGrad::Elementwise(a))],
        )
    }

    /// Element-wise division (with broadcasting).
    pub fn div(&self, rhs: &Variable) -> Variable {
        let (a, b) = (self.value(), rhs.value());
        let data = a.div(&b).expect("broadcast failed in div");
        // ∂(a/b)/∂a = 1/b ; ∂(a/b)/∂b = -a/b²
        let inv_b = b.unary(ncly_tensor::UnaryOp::Recip);
        let d_b = a.div(&b.mul(&b).expect("square")).expect("square");
        self.record(
            "div",
            data,
            vec![
                (self.index, LocalGrad::Elementwise(inv_b)),
                (rhs.index, LocalGrad::Elementwise(d_b.neg_in_place())),
            ],
        )
    }

    /// `e^x`.
    pub fn exp(&self) -> Variable {
        let data = self.value().exp();
        let local = data.clone();
        self.record("exp", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// `ln(x)`.
    pub fn ln(&self) -> Variable {
        let value = self.value();
        let data = value.ln();
        let local = value.unary(ncly_tensor::UnaryOp::Recip);
        self.record("ln", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// `sqrt(x)`.
    pub fn sqrt(&self) -> Variable {
        let data = self.value().sqrt();
        let local = data.muls(2.0).unary(ncly_tensor::UnaryOp::Recip);
        self.record("sqrt", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// `sin(x)`.
    pub fn sin(&self) -> Variable {
        let data = self.value().sin();
        let local = self.value().cos();
        self.record("sin", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// `cos(x)`.
    pub fn cos(&self) -> Variable {
        let data = self.value().cos();
        let local = self.value().sin().neg();
        self.record("cos", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// `tanh(x)`.
    pub fn tanh(&self) -> Variable {
        let data = self.value().tanh();
        let local = data.mul(&data).expect("square").neg().adds(1.0);
        self.record("tanh", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// `x^n`.
    pub fn powi(&self, n: i32) -> Variable {
        let value = self.value();
        let data = value.powf(n as f64);
        let local = value.powf((n - 1) as f64).muls(n as f64);
        self.record("powi", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// `x^p` for real `p`.
    pub fn powf(&self, p: f64) -> Variable {
        let value = self.value();
        let data = value.powf(p);
        let local = value.powf(p - 1.0).muls(p);
        self.record("powf", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// `|x|`.
    pub fn abs(&self) -> Variable {
        let value = self.value();
        let data = value.abs();
        let local = Tensor::from_fn(value.shape().dims().to_vec(), |index| {
            let x = value.get_unchecked(index);
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        });
        self.record("abs", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Variable {
        let data = self.value().neg();
        let local = Tensor::full(self.value().shape().dims().to_vec(), -1.0);
        self.record("neg", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// Multiply by a constant.
    pub fn scale(&self, scalar: f64) -> Variable {
        let data = self.value().muls(scalar);
        let local = Tensor::full(self.value().shape().dims().to_vec(), scalar);
        self.record("scale", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// Sum over every axis, producing a 0-d variable.
    pub fn sum(&self) -> Variable {
        let value = self.value();
        let total = value.sum();
        let data = Tensor::scalar(total);
        let local = Tensor::ones(value.shape().dims().to_vec());
        self.record("sum", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// Sum one axis (removed from the result).
    pub fn sum_axis(&self, axis: usize) -> Result<Variable> {
        let value = self.value();
        let local = Tensor::ones(value.shape().dims().to_vec());
        let data = value.sum_axis(axis)?;
        Ok(self.record("sum_axis", data, vec![(self.index, LocalGrad::Elementwise(local))]))
    }

    /// Sum several axes, keeping the listed ones.
    pub fn sum_axes(&self, axes: &[usize]) -> Result<Variable> {
        let value = self.value();
        let local = Tensor::ones(value.shape().dims().to_vec());
        let data = value.sum_axes(axes)?;
        Ok(self.record("sum_axes", data, vec![(self.index, LocalGrad::Elementwise(local))]))
    }

    /// Mean over every axis.
    pub fn mean(&self) -> Variable {
        let value = self.value();
        let count = value.len();
        let data = Tensor::scalar(value.mean());
        let local = Tensor::full(value.shape().dims().to_vec(), 1.0 / count as f64);
        self.record("mean", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// Reshape.
    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Result<Variable> {
        let value = self.value();
        let target: Vec<usize> = shape.into();
        let data = value
            .to_contiguous()
            .reshape(target.clone())?;
        let local = Tensor::ones(value.shape().dims().to_vec());
        Ok(self.record("reshape", data, vec![(self.index, local)]))
    }

    /// Transpose (reverses axes).
    pub fn transpose(&self) -> Result<Variable> {
        let value = self.value();
        let data = value.transpose().to_contiguous();
        let local = Tensor::ones(value.shape().dims().to_vec());
        if data.len() != local.len() {
            return Err(NclyError::Graph {
                reason: "transpose changed the element count".into(),
            });
        }
        Ok(self.record("transpose", data, vec![(self.index, local)]))
    }

    /// Broadcast to a larger shape.
    ///
    /// The adjoint is summed back down to the operand shape by
    /// [`sum_to_shape`], which is exactly the transpose of broadcasting.
    pub fn broadcast_to(&self, shape: impl Into<Vec<usize>>) -> Result<Variable> {
        let value = self.value();
        let target = Shape::new(shape.into());
        let data = value.broadcast_to(target.dims().to_vec())?.to_contiguous();
        let local = Tensor::ones(target.dims().to_vec());
        Ok(self.record("broadcast_to", data, vec![(self.index, LocalGrad::Elementwise(local))]))
    }

    /// Matrix product of two 2-d variables.
    pub fn matmul(&self, rhs: &Variable) -> Result<Variable> {
        let (a, b) = (self.value(), rhs.value());
        let data = a.matmul(&b)?;
        // ∂L/∂A = G·Bᵀ ; ∂L/∂B = Aᵀ·G, materialised lazily as functions.
        let a_t = a.transpose().to_contiguous();
        let b_t = b.transpose().to_contiguous();
        Ok(self.record(
            "matmul",
            data,
            vec![
                (self.index, LocalGrad::Elementwise(b_t)),
                (rhs.index, LocalGrad::Elementwise(a_t)),
            ],
        ))
    }

    /// Element-wise natural log of the sum of exponentials over all axes.
    pub fn logsumexp(&self) -> Variable {
        let value = self.value();
        let shift = value.max();
        let shifted = value.adds(-shift);
        let data = Tensor::scalar(shift + shifted.exp().sum().ln());
        let local = value.adds(-shift).exp();
        let total: f64 = local.sum();
        let local = local.muls(1.0 / total);
        self.record("logsumexp", data, vec![(self.index, LocalGrad::Elementwise(local))])
    }

    /// Materialise the primal value.
    pub fn to_tensor(&self) -> Tensor {
        self.value()
    }

    /// Describe this variable and its ancestors.
    pub fn describe(&self) -> String {
        self.tape.describe()
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("index", &self.index)
            .field("value", &self.value())
            .field("op", &self.tape.op(self.index))
            .finish()
    }
}

impl Add for &Variable {
    type Output = Variable;
    fn add(self, rhs: &Variable) -> Variable {
        Variable::add(self, rhs)
    }
}

impl Sub for &Variable {
    type Output = Variable;
    fn sub(self, rhs: &Variable) -> Variable {
        Variable::sub(self, rhs)
    }
}

impl Mul for &Variable {
    type Output = Variable;
    fn mul(self, rhs: &Variable) -> Variable {
        Variable::mul(self, rhs)
    }
}

impl Div for &Variable {
    type Output = Variable;
    fn div(self, rhs: &Variable) -> Variable {
        Variable::div(self, rhs)
    }
}

impl Add for Variable {
    type Output = Variable;
    fn add(self, rhs: Variable) -> Variable {
        Variable::add(&self, &rhs)
    }
}

impl Sub for Variable {
    type Output = Variable;
    fn sub(self, rhs: Variable) -> Variable {
        Variable::sub(&self, &rhs)
    }
}

impl Mul for Variable {
    type Output = Variable;
    fn mul(self, rhs: Variable) -> Variable {
        Variable::mul(&self, &rhs)
    }
}

impl Div for Variable {
    type Output = Variable;
    fn div(self, rhs: Variable) -> Variable {
        Variable::div(&self, &rhs)
    }
}

impl Neg for &Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        Variable::neg(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_chain_rule() {
        let x = Variable::new(2.0, "x");
        // f = (x * x + 3x) * 2 at x=2 → 10 * 2 = 20 ; f' = 4x + 6 = 14
        let x2 = &x * &x;
        let three_x = x.scale(3.0);
        let sum = &x2 + &three_x;
        let f = sum.scale(2.0);
        f.backward();
        assert!((x.grad_scalar().unwrap() - 14.0).abs() < 1e-12);
    }

    #[test]
    fn product_rule_two_variables() {
        let x = Variable::new(2.0, "x");
        let y = Variable::new(3.0, "y");
        let f = (&x * &y).powi(2).sum();
        f.backward();
        assert!((x.grad_scalar().unwrap() - 36.0).abs() < 1e-12);
        assert!((y.grad_scalar().unwrap() - 24.0).abs() < 1e-12);
    }

    #[test]
    fn elementary_gradients() {
        let cases: Vec<(&str, fn(&Variable) -> Variable, f64, f64)> = vec![
            ("exp", |v| v.exp(), 1.0, std::f64::consts::E),
            ("ln", |v| v.ln(), 2.0, 0.5),
            ("sqrt", |v| v.sqrt(), 4.0, 0.25),
            ("sin", |v| v.sin(), 0.0, 1.0),
            ("cos", |v| v.cos(), 0.0, 0.0),
            ("tanh", |v| v.tanh(), 0.0, 1.0),
            ("powi", |v| v.powi(3), 2.0, 12.0),
            ("abs", |v| v.abs(), -3.0, -1.0),
            ("neg", |v| v.neg(), 5.0, -1.0),
            ("recip", |v| v.scale(1.0).div(v), 2.0, -0.25),
        ];
        for (name, build, x0, expected) in cases {
            let x = Variable::new(x0, name);
            let y = build(&x).sum();
            y.backward();
            let got = x.grad_scalar().unwrap();
            assert!(
                (got - expected).abs() < 1e-10,
                "{name}: gradient {got} != {expected}"
            );
        }
    }

    #[test]
    fn broadcasting_backpropagates_correctly() {
        let x = Variable::from_tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]).unwrap(), "x");
        let y = Variable::from_tensor(Tensor::from_vec(vec![10.0, 20.0], [2, 1]).unwrap(), "y");
        let z = (&x + &y).sum();
        z.backward();
        assert_eq!(x.grad().to_vec(), vec![1.0, 1.0, 1.0]);
        assert_eq!(y.grad().to_vec(), vec![3.0, 3.0]);
    }

    #[test]
    fn mean_gradient_is_one_over_n() {
        let x = Variable::from_tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [4]).unwrap(), "x");
        x.mean().backward();
        assert_eq!(x.grad().to_vec(), vec![0.25; 4]);
    }

    #[test]
    fn matmul_gradient() {
        let a = Variable::from_tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap(), "a");
        let b = Variable::from_tensor(Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap(), "b");
        let c = a.matmul(&b).unwrap().sum();
        c.backward();
        // d/dA sum(A·I) = ones(2,2) ; d/dB = ones(2,2)
        assert_eq!(a.grad().to_vec(), vec![1.0; 4]);
        assert_eq!(b.grad().to_vec(), vec![1.0; 4]);
    }

    #[test]
    fn logsumexp_is_numerically_stable() {
        let x = Variable::from_tensor(Tensor::from_vec(vec![1000.0, 1000.0], [2]).unwrap(), "x");
        let y = x.logsumexp();
        y.backward();
        let value = y.scalar().unwrap();
        assert!(value.is_finite());
        assert!((value - (1000.0 + 2.0f64.ln())).abs() < 1e-9);
        let grad = x.grad().to_vec();
        assert!((grad[0] - 0.5).abs() < 1e-12 && (grad[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gradients_accumulate_like_pytorch() {
        let x = Variable::new(3.0, "x");
        let y = (&x * &x).sum();
        y.backward();
        assert!((x.grad_scalar().unwrap() - 6.0).abs() < 1e-12);
        y.backward();
        assert!((x.grad_scalar().unwrap() - 12.0).abs() < 1e-12);
        x.zero_grad();
        assert_eq!(x.grad_scalar().unwrap(), 0.0);
    }

    #[test]
    fn shared_subexpressions_accumulate() {
        // y = x*x + x*x = 2x² ; dy/dx = 4x
        let x = Variable::new(1.5, "x");
        let square = &x * &x;
        let y = (square.clone() + square).sum();
        y.backward();
        assert!((x.grad_scalar().unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn reshape_and_transpose_preserve_gradient_mass() {
        let x = Variable::from_tensor(
            Tensor::from_vec((0..6).map(f64::from).collect(), [2, 3]).unwrap(),
            "x",
        );
        let y = x.reshape([3, 2]).unwrap().transpose().unwrap().sum();
        y.backward();
        assert_eq!(x.grad().to_vec(), vec![1.0; 6]);
    }

    #[test]
    fn tape_describe_lists_operations() {
        let x = Variable::new(1.0, "x");
        let y = x.exp().sum();
        y.backward();
        let text = x.describe();
        assert!(text.contains("leaf"));
        assert!(text.contains("exp"));
        assert!(text.contains("sum"));
    }
}
